"""
Load a checkpoint and generate text with temperature + top-p sampling.

Usage:
    python src/generate.py --checkpoint checkpoints/step_5000.pt --prompt "The model"
"""

import argparse
import torch
import torch.nn.functional as F

from model import Transformer, TransformerConfig
from tokenizer import CharTokenizer


def top_p_sample(logits: torch.Tensor, top_p: float) -> int:
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # keep tokens until cumulative probability exceeds top_p
    mask = (cumulative - sorted_probs) < top_p
    sorted_probs[~mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    chosen = torch.multinomial(sorted_probs, 1)
    return sorted_ids[chosen].item()


def generate(model: Transformer, tok: CharTokenizer, prompt: str, max_new: int, temperature: float, top_p: float, device):
    model.eval()
    ids = tok.encode(prompt)
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    kv_caches = [{} for _ in range(model.config.n_layers)]

    with torch.no_grad():
        for _ in range(max_new):
            # only pass the last token when using KV cache
            inp = tokens[:, -1:]
            logits = model(inp, kv_caches=kv_caches)
            logits = logits[0, -1, :] / temperature
            next_id = top_p_sample(logits, top_p)
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)

    return tok.decode(tokens[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/train.txt", help="needed to rebuild tokenizer vocab")
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--max_new", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    with open(args.data) as f:
        text = f.read()
    tok = CharTokenizer()
    tok.fit(text)

    config = ckpt["config"]
    model = Transformer(config).to(device)
    model.load_state_dict(ckpt["model"])

    output = generate(model, tok, args.prompt, args.max_new, args.temperature, args.top_p, device)
    print(output)
