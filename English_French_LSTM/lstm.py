import math
import torch
import torch.nn as nn
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence
import time
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import os

torch.manual_seed(10701)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.LSTM(embed_dim if i == 0 else hidden_dim, hidden_dim, num_layers=1, batch_first=True)
            for i in range(num_layers)
        ])
        self.layer_drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers - 1)])

    def forward(self, src):
        x = self.embed_drop(self.embedding(src))
        hs, cs = [], []
        for i, lstm in enumerate(self.layers):
            x, (h, c) = lstm(x)
            hs.append(h)
            cs.append(c)
            if i < self.num_layers - 1:
                x = self.layer_drops[i](x)
        return torch.cat(hs, dim=0), torch.cat(cs, dim=0)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.LSTM(embed_dim if i == 0 else hidden_dim, hidden_dim, num_layers=1, batch_first=True)
            for i in range(num_layers)
        ])
        self.layer_drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers - 1)])
        self.out_down = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))

    def _project(self, x):
        x = self.out_down(x)
        return x @ self.embedding.weight.T + self.out_bias

    def forward(self, tgt_in, h0, c0):
        x = self.embed_drop(self.embedding(tgt_in))
        for i, lstm in enumerate(self.layers):
            h_i = h0[i:i + 1].contiguous()
            c_i = c0[i:i + 1].contiguous()
            x, _ = lstm(x, (h_i, c_i))
            if i < self.num_layers - 1:
                x = self.layer_drops[i](x)
        return self._project(x)

    def step(self, tok, h, c):
        x = self.embedding(tok)
        new_h, new_c = [], []
        for i, lstm in enumerate(self.layers):
            h_i = h[i:i + 1].contiguous()
            c_i = c[i:i + 1].contiguous()
            x, (h_out, c_out) = lstm(x, (h_i, c_i))
            new_h.append(h_out)
            new_c.append(c_out)
        return self._project(x), torch.cat(new_h, dim=0), torch.cat(new_c, dim=0)


class Seq2SeqLSTM(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_dim, hidden_dim, num_layers, dropout=.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab, embed_dim, hidden_dim, num_layers, dropout)
        self.initialize()

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0, std=.05)

    def forward(self, src, tgt_in):
        h, c = self.encoder(src)
        return self.decoder(tgt_in, h, c)

    @torch.no_grad()
    def translate(self, src, bos_id, eos_id, max_len):
        self.eval()
        src_rev = torch.flip(src, dims=[1])
        h, c = self.encoder(src_rev)
        B = src.size(0)
        tok = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        out = []
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits, h, c = self.decoder.step(tok, h, c)
            tok = logits.argmax(dim=-1)
            out.append(tok)
            finished = finished | (tok.squeeze(1) == eos_id)
            if finished.all():
                break
        return torch.cat(out, dim=1)


class TranslationDataset(IterableDataset):
    def __init__(self, sp, csv_path, max_len=64, chunksize=10000, train=True,
                 max_samples=None, start=None, stop=None, filter_max_len=None):
        self.sp = sp
        self.csv_path = csv_path
        self.max_len = max_len
        self.chunksize = chunksize
        if start is not None and stop is not None:
            self.start = start
            self.stop = stop
        elif train:
            self.start = 0
            self.stop = 20268340
        else:
            self.start = 20268340
            self.stop = 22520376
        self.max_samples = max_samples
        self.filter_max_len = filter_max_len

    def __iter__(self):
        nrows = self.stop - self.start
        csv_reader = pd.read_csv(self.csv_path, chunksize=self.chunksize,
                                 skiprows=range(1, self.start + 1), nrows=nrows)
        count = 0
        for chunk in csv_reader:
            for _, row in chunk.iterrows():
                eng, frc = row.iloc[0], row.iloc[1]
                if pd.isna(eng) or pd.isna(frc):
                    continue
                eng_ids = [self.sp.bos_id()] + self.sp.encode(str(eng), out_type=int) + [self.sp.eos_id()]
                frc_ids = [self.sp.bos_id()] + self.sp.encode(str(frc), out_type=int) + [self.sp.eos_id()]
                if self.filter_max_len is not None and (
                        len(eng_ids) > self.filter_max_len or len(frc_ids) > self.filter_max_len):
                    continue
                eng_ids = eng_ids[:self.max_len]
                frc_ids = frc_ids[:self.max_len]
                yield (torch.tensor(eng_ids), torch.tensor(frc_ids))
                count += 1
                if self.max_samples is not None and count >= self.max_samples:
                    return


def collate(batch):
    eng, frc = zip(*batch)
    eng_pad = pad_sequence(list(eng), batch_first=True, padding_value=0)
    frc_pad = pad_sequence(list(frc), batch_first=True, padding_value=0)
    eng_pad = torch.flip(eng_pad, dims=[1])
    return eng_pad, frc_pad


def train(save_dir="versions_lstm", save_every=5, embed_dim=256, hidden_dim=512, num_layers=2,
          dropout=.1, max_len=64, epochs=30, batch_size=16,
          max_train_samples=None, max_test_samples=None,
          train_range=None, test_range=None,
          filter_max_len=None, label_smoothing=0.0, device=None,
          lr=1e-3, warmup_steps=0, min_lr_ratio=1.0):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)

    csv_path = os.environ.get("EN_FR_CSV", "en-fr.csv")
    spm_path = os.environ.get("EN_FR_SPM_MODEL", "en_fr.model")

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    vocab_size = sp.GetPieceSize()
    pad_id = 0

    tr_start, tr_stop = train_range if train_range else (None, None)
    te_start, te_stop = test_range if test_range else (None, None)

    train_dataset = TranslationDataset(sp, csv_path, max_len=max_len, chunksize=batch_size,
                                       train=True, max_samples=max_train_samples,
                                       start=tr_start, stop=tr_stop,
                                       filter_max_len=filter_max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)

    test_dataset = TranslationDataset(sp, csv_path, max_len=max_len, chunksize=10000,
                                      train=False, max_samples=max_test_samples,
                                      start=te_start, stop=te_stop,
                                      filter_max_len=filter_max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)

    model = Seq2SeqLSTM(src_vocab=vocab_size, tgt_vocab=vocab_size, embed_dim=embed_dim,
                        hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)

    total_steps = None
    if max_train_samples is not None:
        total_steps = epochs * max(1, max_train_samples // batch_size)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps is None or total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(save_dir, exist_ok=True)

    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_tokens = 0
        start_time = time.time()

        for batch_id, (eng, frc) in enumerate(train_loader):
            eng = eng.to(device)
            frc = frc.to(device)

            frc_in = frc[:, :-1]
            frc_out = frc[:, 1:]

            logits = model(eng, frc_in)
            loss = criterion(logits.reshape(-1, vocab_size), frc_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            non_pad = (frc_out != pad_id).sum().item()
            epoch_loss += loss.item() * non_pad
            epoch_tokens += non_pad
            step += 1

        train_loss = epoch_loss / max(epoch_tokens, 1)
        elapsed = time.time() - start_time

        epoch_loss = 0
        epoch_tokens = 0

        model.eval()
        with torch.no_grad():
            for batch_id, (eng, frc) in enumerate(test_loader):
                eng = eng.to(device)
                frc = frc.to(device)

                frc_in = frc[:, :-1]
                frc_out = frc[:, 1:]

                logits = model(eng, frc_in)
                loss = criterion(logits.reshape(-1, vocab_size), frc_out.reshape(-1))

                non_pad = (frc_out != pad_id).sum().item()
                epoch_loss += loss.item() * non_pad
                epoch_tokens += non_pad

        test_loss = epoch_loss / max(epoch_tokens, 1)

        cur_lr = optimizer.param_groups[0]["lr"]
        with open("benchmarks_lstm.txt", "a") as f:
            f.write(f"Epoch:{epoch}, step:{step}, lr:{cur_lr:.2e}, train loss:{train_loss:.4f}, test_loss:{test_loss:.4f}, train time: {elapsed}\n")
            f.flush()

        if epoch % save_every == 0:
            save_path = os.path.join(save_dir, f"Model{epoch:03d}.pt")
            torch.save({"epoch": epoch, "step": step, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "train_loss": train_loss,
                        "test_loss": test_loss,
                        "hparams": dict(
                            vocab_size=vocab_size,
                            embed_dim=embed_dim,
                            hidden_dim=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            max_len=max_len,
                        )}, save_path)


if __name__ == "__main__":
    train(save_dir="versions_lstm_overnight",
          embed_dim=512, hidden_dim=768, num_layers=3,
          dropout=.2, max_len=64,
          max_train_samples=1_000_000, max_test_samples=5_000,
          train_range=(0, 5_000_000), test_range=(20_000_000, 20_020_000),
          filter_max_len=40, label_smoothing=0.1,
          epochs=5, save_every=1, batch_size=32,
          lr=1e-3, warmup_steps=2000, min_lr_ratio=0.05)

