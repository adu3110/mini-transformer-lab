"""
Training loop: AdamW optimizer, cosine LR with warmup, periodic checkpointing and sampling.

Usage:
    python src/train.py --data data/train.txt
"""

import argparse
import math
import os
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from model import Transformer, TransformerConfig
from tokenizer import CharTokenizer


def cosine_lr(step: int, warmup_steps: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def make_batches(tokens: list[int], batch_size: int, seq_len: int, device) -> tuple:
    """Return (inputs, targets) as random chunks of length seq_len."""
    n = len(tokens) - seq_len
    starts = torch.randint(0, n, (batch_size,))
    x = torch.stack([torch.tensor(tokens[s : s + seq_len]) for s in starts]).to(device)
    y = torch.stack([torch.tensor(tokens[s + 1 : s + seq_len + 1]) for s in starts]).to(device)
    return x, y


def sample(model: Transformer, prompt_ids: list[int], max_new: int, temperature: float, device) -> str:
    model.eval()
    tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(tokens[:, -model.config.max_seq_len:])
            logits = logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_tok.unsqueeze(0)], dim=1)
    model.train()
    return tokens[0].tolist()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.data) as f:
        text = f.read()

    tok = CharTokenizer()
    tok.fit(text)
    tokens = tok.encode(text)
    split = int(0.9 * len(tokens))
    train_tokens, val_tokens = tokens[:split], tokens[split:]
    print(f"Vocabulary size: {tok.vocab_size}, train tokens: {len(train_tokens):,}")

    config = TransformerConfig(
        vocab_size=tok.vocab_size,
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.seq_len,
    )
    model = Transformer(config).to(device)
    print(f"Parameters: {model.num_parameters():,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    os.makedirs("checkpoints", exist_ok=True)
    log = []

    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.lr / 10)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = make_batches(train_tokens, args.batch_size, args.seq_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            # validation loss
            with torch.no_grad():
                vx, vy = make_batches(val_tokens, args.batch_size, args.seq_len, device)
                val_loss = F.cross_entropy(model(vx).view(-1, config.vocab_size), vy.view(-1))
            log.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss.item()})
            print(f"step {step:5d} | train {loss.item():.4f} | val {val_loss.item():.4f} | lr {lr:.2e} | gnorm {grad_norm:.3f}")

        if step % args.sample_every == 0:
            prompt = tok.encode(args.sample_prompt)
            ids = sample(model, prompt, max_new=200, temperature=0.8, device=device)
            print("\n--- sample ---")
            print(tok.decode(ids))
            print("---\n")

        if step % args.save_every == 0:
            torch.save({"step": step, "model": model.state_dict(), "config": config}, f"checkpoints/step_{step}.pt")

    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--sample_prompt", default="The ")
    train(parser.parse_args())
