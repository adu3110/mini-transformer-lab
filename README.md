# mini-transformer-lab

A small decoder-only transformer trained from scratch in PyTorch.

The goal is not to train a competitive model. It is to build something small enough to fully inspect — attention weights, residual stream norms, loss curves, KV cache — and learn by doing.

## What's here

| File | Purpose |
|------|---------|
| `src/tokenizer.py` | Character-level and BPE tokenizer |
| `src/model.py` | Decoder-only transformer: RMSNorm, RoPE, GQA, SwiGLU |
| `src/train.py` | Training loop with AdamW, cosine LR schedule, logging |
| `src/generate.py` | Sampling with temperature and top-p |
| `src/kv_cache.py` | KV cache for efficient autoregressive inference |

## Quick start

```bash
pip install -r requirements.txt

# Train on a small text file
python src/train.py --data data/train.txt --model_dim 256 --n_layers 6 --n_heads 8

# Generate from a checkpoint
python src/generate.py --checkpoint checkpoints/step_5000.pt --prompt "The transformer"
```

## Configuration (defaults)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model_dim` | 256 | Hidden dimension |
| `n_layers` | 6 | Transformer blocks |
| `n_heads` | 8 | Attention heads |
| `n_kv_heads` | 2 | KV heads (GQA) |
| `ffn_mult` | 4 | MLP expansion ratio |
| `max_seq_len` | 512 | Context length |
| `vocab_size` | 256 | Character-level default |
| `batch_size` | 32 | Training batch size |
| `lr` | 3e-4 | Peak learning rate |

## What to watch

- **Loss curve**: should decrease, plateau, then resume as the model moves from frequency patterns to syntax to semantics
- **Attention weights**: visualize with `notebooks/attention_viz.ipynb` — watch for head collapse early in training
- **Gradient norm**: should stay stable; spikes indicate LR too high or bad batches
- **Sample quality**: inspect every 500 steps; loss alone is not enough

## Experiments to run

1. Train with and without RoPE — compare position-sensitivity
2. Vary KV heads (MHA vs GQA vs MQA) and measure memory vs quality trade-off
3. Compare character-level vs BPE tokenization on the same corpus
4. Insert a fact at different context positions and measure retrieval accuracy

## Related

- Article: [Building a Tiny Transformer from Scratch](https://aditichatterji.com/articles#building-tiny-transformer-from-scratch)
- Article: [What Does a Language Model Learn First?](https://aditichatterji.com/articles#what-does-language-model-learn-first)
# mini-transformer-lab
