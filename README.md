# mini-transformer-lab

> **RMSNorm + RoPE + GQA + SwiGLU + KV cache — every modern LLM component, in one inspectable codebase.**  
> Not a competitive model. A decoder-only transformer small enough to fully understand.

The goal is not to match GPT. It is to build something small enough to trace — attention weights, residual stream norms, loss curves, and KV cache — and understand every design choice by building it yourself.

```bash
pip install -r requirements.txt
python src/train.py --data data/train.txt --model_dim 256 --n_layers 6 --n_heads 8
```

---

## What's here

| File | What you learn |
|------|----------------|
| `src/tokenizer.py` | Character-level tokenizer vs BPE — why subword matters |
| `src/model.py` | RMSNorm, RoPE, Grouped-Query Attention, SwiGLU activation |
| `src/train.py` | AdamW + cosine LR schedule + gradient clipping |
| `src/generate.py` | Temperature sampling + top-p (nucleus) sampling |
| `src/kv_cache.py` | Why inference is O(T) not O(T²) with caching |

---

## Architecture choices (and why)

| Choice | This repo | Vanilla transformer |
|--------|-----------|-------------------|
| Norm | RMSNorm (no centering) | LayerNorm |
| Position | RoPE (rotary, no learned params) | Learned absolute |
| Attention | GQA (fewer KV heads) | MHA (n_kv = n_q) |
| Activation | SwiGLU | ReLU / GELU |

Each of these is a decision from a production model (LLaMA, Mistral, Gemma). This lab lets you swap them and observe the difference.

---

## Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model_dim` | 256 | Hidden dimension |
| `n_layers` | 6 | Transformer blocks |
| `n_heads` | 8 | Attention heads |
| `n_kv_heads` | 2 | KV heads (GQA: 4 query heads per KV head) |
| `ffn_mult` | 4 | MLP expansion ratio |
| `max_seq_len` | 512 | Context length |
| `vocab_size` | 256 | Character-level default |
| `lr` | 3e-4 | Peak learning rate |

---

## Experiments to run

1. Train with and without RoPE — compare position-sensitivity at long context
2. Vary KV heads: MHA (`n_kv=n_heads`) vs GQA (`n_kv=2`) vs MQA (`n_kv=1`) — measure memory vs quality
3. Compare character-level vs BPE tokenization on the same corpus
4. Insert a fact at different context positions and measure retrieval accuracy (use `long-context-bench`)
5. Visualize attention weights at step 0, 500, 5000 — watch heads specialise

---

## What to watch during training

- **Loss curve**: decreases, plateaus, resumes as model moves from frequency → syntax → semantics
- **Attention weights**: visualize with `notebooks/attention_viz.ipynb` — watch for head collapse early in training
- **Gradient norm**: should stay stable; spikes indicate LR too high or corrupted batches
- **Sample quality**: inspect every 500 steps; loss alone is not sufficient

---

## Related

- [quantization](https://github.com/adu3110/quantization) — what to do after training: 4× smaller at 20 dB SNR cost
- [microssm](https://github.com/adu3110/microssm) — the O(T) alternative to O(T²) attention
- Article: [Building a Tiny Transformer from Scratch](https://aditichatterji.com/articles#building-tiny-transformer-from-scratch)

---

## License

MIT
