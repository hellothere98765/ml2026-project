import torch
import torch.nn as nn
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence
import random
import time
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import os 

torch.manual_seed(10701)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1)

        _2i = torch.arange(0, dim, step=2).float()

        pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / dim)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / dim)))
        self.pe = pe.unsqueeze(0)
    
    def forward(self, input_tens):
        input_tens = input_tens + self.pe[:, : input_tens.size(1)].to(input_tens.device)
        return self.dropout(input_tens)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout = .1):
        super().__init__()

        self.d_model = d_model 
        self.n_heads = num_heads
        self.d_k = d_model//num_heads

        self.W_q = nn.Linear(d_model, d_model, bias = True)
        self.W_k = nn.Linear(d_model, d_model, bias = True)
        self.W_v = nn.Linear(d_model, d_model, bias = True)
        self.W_o = nn.Linear(d_model, d_model, bias = True)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask):
        B, Sq, _ = q.shape
        Sk = k.shape[1]

        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        Q = Q.view(B, Sq, self.n_heads, self.d_k).transpose(1,2)
        K = K.view(B, Sk, self.n_heads, self.d_k).transpose(1,2)
        V = V.view(B, Sk, self.n_heads, self.d_k).transpose(1,2)

        scores = (Q @ K.transpose(-2, -1))/(self.d_k**.5)

        if mask is not None:
            scores = scores.masked_fill(mask ==0, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))

        attended = attn @ V 
        output = attended.transpose(1,2).reshape(B, -1, self.d_model)

        return self.W_o(output)

class Residual(nn.Module):
    def __init__(self, module, d_model, drop_p=0.1):
        super().__init__()
        self.module = module
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, input_val, *inp):
        return input_val +  self.dropout(self.module(self.norm(input_val), *inp))
    
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0):
        super().__init__()
        self.sequence = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, d_model))
    
    def forward(self, x):
        return self.sequence(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=.1):
        super().__init__()
        self.attn = Residual(MultiHeadAttention(d_model, num_heads, dropout), d_model, dropout)
        self.ff = Residual(FeedForward(d_model, hidden_dim), d_model, dropout) #Add dropout?
    
    def forward(self, input_val, mask):
        m = self.attn(input_val,input_val,input_val, mask)
        m = self.ff(m)

        return m

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout = 0.1):
        super().__init__()
        self.f_attn = Residual(MultiHeadAttention(d_model, num_heads, dropout), d_model, dropout)
        self.c_attn = Residual(MultiHeadAttention(d_model, num_heads, dropout), d_model, dropout)
        self.ff = Residual(FeedForward(d_model, hidden_dim), d_model, dropout)

    def forward(self, inp, out, out_mask = None, inp_mask = None):
        inp = self.f_attn(inp, inp, inp, inp_mask)
        inp = self.c_attn(inp, out, out, out_mask)
        return self.ff(inp)
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, hidden_dim, num_layers, dropout, max_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, inp, inp_mask = None):
        inp = self.pe(self.embedding(inp) * (self.d_model**.5))
        for layer in self.layers:
            inp = layer(inp, inp_mask)
        
        return self.norm(inp)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, hidden_dim, num_layers, dropout, max_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc, out, inp_mask, out_mask):
        k = self.pe(self.embedding(enc)*(self.d_model**.5))
        for layer in self.layers:
            k = layer(k, out, inp_mask, out_mask)
        return self.norm(k)

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, num_heads, hidden_dim, num_layers, dropout = .1, max_len = 1000):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_heads, hidden_dim, num_layers, dropout, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, num_heads, hidden_dim, num_layers, dropout, max_len)
        self.proj = nn.Linear(d_model, tgt_vocab, bias = True)
        self.initialize()
    
    def initialize(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'W_o' in name or 'proj' in name:
                    nn.init.normal_(p, mean=0, std=0.05 / (2 * self.encoder.layers.__len__()) ** 0.5)
                else:
                    nn.init.normal_(p, mean=0, std=0.05)

    def encode(self, inp, inp_mask):
        return self.encoder(inp, inp_mask)

    def decode(self, enc, out, inp_mask, out_mask):
        return self.decoder(enc, out, inp_mask, out_mask)
    
    def forward(self, inp, out, pad_id):
        inp_mask = (inp != pad_id).unsqueeze(1).unsqueeze(2)
        out_mask = (out != pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(out.size(1), out.size(1), device = out.device)).bool()
        out_mask = out_mask & causal_mask 

        enc = self.encode(inp, inp_mask)
        dec = self.decode(out, enc, inp_mask, out_mask)

        return self.proj(dec)

class TranslationDataset(IterableDataset):
    def __init__(self, sp, max_len = 1000, chunksize = 10000, train=True):
        self.sp = sp 
        self.csv_path = r"/home/parth/.cache/kagglehub/datasets/dhruvildave/en-fr-translation-dataset/versions/2/en-fr.csv"
        self.max_len = max_len
        self.chunksize = chunksize 
        self.train = train
        self.start = 1
        self.stop = 3157959
        if(not train):
            self.start = 3157959
            self.stop = 3508844
        #Actual size of pd array seems to be 3508844
        """self.start = random.randint(0, 2807075)
        self.stop = self.start + 50000
        if(not self.train):
            self.start = random.randint(2857075, 2907075)
            self.stop = self.start + 5000"""
        
        
    def __iter__(self):
        csv_reader = pd.read_csv(self.csv_path, chunksize = self.chunksize, skiprows = self.start, nrows = (self.stop-self.start), header = None, names = ['en', 'fr'])
        for chunk in csv_reader:
            for _, row in chunk.iterrows():
                eng, frc = row['en'], row['fr']

                if pd.isna(eng) or pd.isna(frc):
                    continue

                eng_ids = [self.sp.bos_id()]+self.sp.encode(str(eng), out_type = int)+[self.sp.eos_id()]
                frc_ids = [self.sp.bos_id()]+self.sp.encode(str(frc), out_type = int)+[self.sp.eos_id()]

                eng_ids=eng_ids[:self.max_len]
                frc_ids=frc_ids[:self.max_len]

                yield(torch.tensor(eng_ids), torch.tensor(frc_ids))

def collate(batch):
    eng, frc = zip(*batch)

    eng_pad = pad_sequence(eng, batch_first = True, padding_value = 0)
    frc_pad = pad_sequence(frc, batch_first = True, padding_value = 0)

    return eng_pad, frc_pad

#def train(save_dir="versions", save_every=1, d_model = 256, num_heads = 8, hidden_dim = 512, num_layers = 12, dropout = .1, max_len = 256, epochs = 30, batch_size = 32, device = None):

def train(save_dir="versions", save_every=1, d_model = 256, num_heads = 8, hidden_dim = 512, num_layers = 12, dropout = .1, max_len = 256, epochs = 50, batch_size = 32, device = None, path = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    sp = spm.SentencePieceProcessor()
    sp.Load("en_fr.model")

    vocab_size = sp.GetPieceSize() # Should be 16000

    pad_id = 0

    translator = Transformer(src_vocab=vocab_size, tgt_vocab = vocab_size, d_model = d_model, num_heads = num_heads, hidden_dim = hidden_dim, num_layers = num_layers, dropout=dropout, max_len = max_len).to(device)

    translator = torch.compile(translator)

    optimizer = torch.optim.Adam(translator.parameters(), lr = .0001)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing = .1)

    step = 0
    for epoch in range(1, epochs+1):


        train_dataset = TranslationDataset(sp, max_len = max_len, chunksize= 10000, train=True)

        train_loader = DataLoader(train_dataset, batch_size = batch_size, collate_fn = collate, num_workers = 4, pin_memory = True)
    
        test_dataset = TranslationDataset(sp, max_len = max_len, chunksize= 10000, train=False)
    
        test_loader = DataLoader(test_dataset, batch_size = batch_size, collate_fn = collate, num_workers = 4, pin_memory = True)

        translator.train()
        epoch_loss = 0
        epoch_tokens = 0
        start_time = time.time()

        translator.train()
        for batch_id, (eng, frc) in enumerate(train_loader):
            eng = eng.to(device)
            frc = frc.to(device)

            frc_in = frc[:, :-1]
            frc_out = frc[:, 1:]

            with torch.autocast(device_type = device):
                logits = translator(eng, frc_in, pad_id)
                loss = criterion(logits.reshape(-1, vocab_size), frc_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
            optimizer.step()


            non_pad = (frc_out != pad_id).sum().item()
            epoch_loss += loss.item() * non_pad
            epoch_tokens += non_pad
            step+=1

            if (batch_id + 1) % 500 == 0:
                avg = epoch_loss / max(epoch_tokens, 1)
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch:>3} | Step {step:>7} | "
                    f"Batch {batch_id+1:>6} | Loss {avg:.4f} | "
                    f" {elapsed:.1f}s"
                )


        train_loss = epoch_loss/max(epoch_tokens, 1)
        elapsed = time.time() - start_time

        epoch_loss = 0
        epoch_tokens = 0

        translator.eval()
        with torch.no_grad():
            for batch_id, (eng, frc) in enumerate(test_loader):
                eng = eng.to(device)
                frc = frc.to(device)
    
                frc_in = frc[:, :-1]
                frc_out = frc[:, 1:]
    
                logits = translator(eng, frc_in, pad_id)
                loss = criterion(logits.reshape(-1, vocab_size), frc_out.reshape(-1))
    
                non_pad = (frc_out != pad_id).sum().item()
                epoch_loss += loss.item() * non_pad
                epoch_tokens += non_pad

        test_loss = epoch_loss/max(epoch_tokens, 1)
         
        with open("benchmarks.txt", "a") as f:
            f.write(f"Epoch:{epoch}, step:{step}, train loss:{train_loss:.4f}, test_loss:{test_loss:.4f}, train time: {elapsed}\n")
            f.flush()


        if epoch % save_every == 0:
            save_path = os.path.join(save_dir, f"Model{epoch:03d}.pt")
            torch.save({"epoch":epoch, "step":step, "model_state_dict":translator.state_dict(), "optimizer_state_dict":optimizer.state_dict(), "train_loss":train_loss, "test_loss":test_loss,
                            "hparams":dict(
                                vocab_size=vocab_size,
                                d_model=d_model,
                                num_heads=num_heads,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers,
                                dropout=dropout,
                                max_len=max_len,
                                )}, save_path)

        del train_dataset, train_loader, test_dataset, test_loader
        


if __name__ == "__main__":
    train()