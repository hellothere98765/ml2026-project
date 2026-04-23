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
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, embedding=None):
        super().__init__()
        self.embedding = embedding if embedding is not None else nn.Embedding(vocab_size, embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, src):
        x = self.embed_drop(self.embedding(src))
        outputs, (h, c) = self.lstm(x)
        return outputs, h, c


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, embedding=None):
        super().__init__()
        self.embedding = embedding if embedding is not None else nn.Embedding(vocab_size, embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.scale = hidden_dim ** 0.5
        self.combine = nn.Linear(2 * hidden_dim, embed_dim, bias=False)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))

    def _attend(self, query, enc_out, enc_mask):
        scores = torch.bmm(query, enc_out.transpose(1, 2)) / self.scale
        if enc_mask is not None:
            scores = scores.masked_fill(~enc_mask.unsqueeze(1), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return torch.bmm(attn, enc_out)

    def _project(self, x):
        return x @ self.embedding.weight.T + self.out_bias

    def forward(self, tgt_in, enc_out, enc_mask, h, c):
        x = self.embed_drop(self.embedding(tgt_in))
        lstm_out, _ = self.lstm(x, (h, c))
        ctx = self._attend(lstm_out, enc_out, enc_mask)
        return self._project(self.combine(torch.cat([lstm_out, ctx], dim=-1)))

    def step(self, tok, enc_out, enc_mask, h, c):
        x = self.embedding(tok)
        lstm_out, (h, c) = self.lstm(x, (h, c))
        ctx = self._attend(lstm_out, enc_out, enc_mask)
        logits = self._project(self.combine(torch.cat([lstm_out, ctx], dim=-1)))
        return logits, h, c


class Seq2SeqAttnLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.1, pad_id=0):
        super().__init__()
        shared_emb = nn.Embedding(vocab_size, embed_dim)
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout, embedding=shared_emb)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout, embedding=shared_emb)
        self.pad_id = pad_id
        self.initialize()

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0, std=0.05)

    def _encode(self, src):
        src_rev = torch.flip(src, dims=[1])
        enc_out, h, c = self.encoder(src_rev)
        mask = src_rev != self.pad_id
        return enc_out, mask, h, c

    def forward(self, src, tgt_in):
        enc_out, mask, h, c = self._encode(src)
        return self.decoder(tgt_in, enc_out, mask, h, c)

    @torch.no_grad()
    def translate(self, src, bos_id, eos_id, max_len):
        self.eval()
        enc_out, mask, h, c = self._encode(src)
        B = src.size(0)
        tok = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        out = []
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits, h, c = self.decoder.step(tok, enc_out, mask, h, c)
            tok = logits.argmax(dim=-1)
            out.append(tok)
            finished = finished | (tok.squeeze(1) == eos_id)
            if finished.all():
                break
        return torch.cat(out, dim=1)


class RefinementDataset(IterableDataset):
    """Streams (src, tgt) text pairs from a CSV. Column 0 is the space-joined
    per-word French gloss produced by the stage-1 translator; column 1 is the
    fluent French reference. Tokenized with the shared joint BPE model."""

    def __init__(self, sp, csv_path, max_len=64, chunksize=10000,
                 start=0, stop=None, max_samples=None, filter_max_len=None,
                 src_col=0, tgt_col=1):
        self.sp = sp
        self.csv_path = csv_path
        self.max_len = max_len
        self.chunksize = chunksize
        self.start = start
        self.stop = stop
        self.max_samples = max_samples
        self.filter_max_len = filter_max_len
        self.src_col = src_col
        self.tgt_col = tgt_col

    def __iter__(self):
        nrows = (self.stop - self.start) if self.stop is not None else None
        csv_reader = pd.read_csv(self.csv_path, chunksize=self.chunksize,
                                 skiprows=range(1, self.start + 1), nrows=nrows)
        count = 0
        for chunk in csv_reader:
            for _, row in chunk.iterrows():
                src, tgt = row.iloc[self.src_col], row.iloc[self.tgt_col]
                if pd.isna(src) or pd.isna(tgt):
                    continue
                src_ids = [self.sp.bos_id()] + self.sp.encode(str(src), out_type=int) + [self.sp.eos_id()]
                tgt_ids = [self.sp.bos_id()] + self.sp.encode(str(tgt), out_type=int) + [self.sp.eos_id()]
                if self.filter_max_len is not None and (
                        len(src_ids) > self.filter_max_len or len(tgt_ids) > self.filter_max_len):
                    continue
                src_ids = src_ids[:self.max_len]
                tgt_ids = tgt_ids[:self.max_len]
                yield torch.tensor(src_ids), torch.tensor(tgt_ids)
                count += 1
                if self.max_samples is not None and count >= self.max_samples:
                    return


def collate(batch):
    src, tgt = zip(*batch)
    src_pad = pad_sequence(list(src), batch_first=True, padding_value=0)
    tgt_pad = pad_sequence(list(tgt), batch_first=True, padding_value=0)
    return src_pad, tgt_pad


def train(save_dir="versions_refinement_lstm", save_every=1, embed_dim=256, hidden_dim=512,
          num_layers=2, dropout=0.2, max_len=64, epochs=4, batch_size=32,
          max_train_samples=None, max_test_samples=None,
          train_range=None, test_range=None,
          filter_max_len=None, label_smoothing=0.1, device=None,
          lr=1e-3, warmup_steps=1000, min_lr_ratio=0.1,
          csv_path=None, spm_path=None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)

    csv_path = csv_path or os.environ.get("REFINE_CSV", "fr_gloss-fr.csv")
    spm_path = spm_path or os.environ.get("EN_FR_SPM_MODEL", "en_fr.model")

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    vocab_size = sp.GetPieceSize()
    pad_id = 0

    tr_start, tr_stop = train_range if train_range else (0, None)
    te_start, te_stop = test_range if test_range else (0, None)

    train_dataset = RefinementDataset(sp, csv_path, max_len=max_len, chunksize=batch_size,
                                      max_samples=max_train_samples,
                                      start=tr_start, stop=tr_stop,
                                      filter_max_len=filter_max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)

    test_dataset = RefinementDataset(sp, csv_path, max_len=max_len, chunksize=10000,
                                     max_samples=max_test_samples,
                                     start=te_start, stop=te_stop,
                                     filter_max_len=filter_max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)

    model = Seq2SeqAttnLSTM(vocab_size=vocab_size, embed_dim=embed_dim,
                            hidden_dim=hidden_dim, num_layers=num_layers,
                            dropout=dropout, pad_id=pad_id).to(device)

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

        for batch_id, (src, tgt) in enumerate(train_loader):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            non_pad = (tgt_out != pad_id).sum().item()
            epoch_loss += loss.item() * non_pad
            epoch_tokens += non_pad
            step += 1

        train_loss = epoch_loss / max(epoch_tokens, 1)
        elapsed = time.time() - start_time

        epoch_loss = 0
        epoch_tokens = 0

        model.eval()
        with torch.no_grad():
            for batch_id, (src, tgt) in enumerate(test_loader):
                src = src.to(device)
                tgt = tgt.to(device)

                tgt_in = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))

                non_pad = (tgt_out != pad_id).sum().item()
                epoch_loss += loss.item() * non_pad
                epoch_tokens += non_pad

        test_loss = epoch_loss / max(epoch_tokens, 1)

        cur_lr = optimizer.param_groups[0]["lr"]
        with open("benchmarks_refinement_lstm.txt", "a") as f:
            f.write(f"Epoch:{epoch}, step:{step}, lr:{cur_lr:.2e}, "
                    f"train loss:{train_loss:.4f}, test_loss:{test_loss:.4f}, "
                    f"train time: {elapsed}\n")
            f.flush()

        if epoch % save_every == 0:
            save_path = os.path.join(save_dir, f"Model{epoch:03d}.pt")
            torch.save({"epoch": epoch, "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss, "test_loss": test_loss,
                        "hparams": dict(
                            vocab_size=vocab_size,
                            embed_dim=embed_dim,
                            hidden_dim=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            max_len=max_len,
                        )}, save_path)


if __name__ == "__main__":
    # REFINE_CSV expects at a CSV whose column 0 is the space-joined per-word French gloss from stage 1 and column 1 is the fluent French reference.
    train(save_dir="versions_refinement_lstm",
          embed_dim=256, hidden_dim=512, num_layers=2,
          dropout=0.2, max_len=64,
          max_train_samples=400_000, max_test_samples=4_000,
          train_range=(0, 2_000_000), test_range=(2_000_000, 2_020_000),
          filter_max_len=50, label_smoothing=0.1,
          epochs=4, save_every=1, batch_size=32,
          lr=1e-3, warmup_steps=1000, min_lr_ratio=0.1)
