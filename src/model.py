"""
Decoder-only transformer: RMSNorm, RoPE, grouped-query attention, SwiGLU MLP.

Architecture decisions follow the 2023-2026 open-weight consensus:
  - Pre-norm residual (RMSNorm before attention and MLP)
  - RoPE for position encoding
  - GQA for inference efficiency
  - SwiGLU activation in the MLP
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    vocab_size: int = 256        # character-level default
    model_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2          # GQA: fewer KV heads than query heads
    ffn_mult: int = 4
    max_seq_len: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0  # RoPE base frequency


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


def build_rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0, device=None):
    """Precompute RoPE sin/cos cache for a given sequence length."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, freqs)          # (seq_len, half)
    angles = torch.cat([angles, angles], dim=-1)    # (seq_len, head_dim)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to x of shape (B, n_heads, T, head_dim)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.model_dim // config.n_heads
        self.groups = config.n_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.model_dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.model_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.model_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.model_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # apply RoPE to queries and keys
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        # KV cache: append new K/V and use full history
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"] = k
            kv_cache["v"] = v

        # expand KV heads to match query heads (GQA)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores + mask

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: two parallel projections, one gates the other."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        hidden = config.model_dim * config.ffn_mult
        self.gate = nn.Linear(config.model_dim, hidden, bias=False)
        self.up = nn.Linear(config.model_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, config.model_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.model_dim)
        self.attn = GroupedQueryAttention(config)
        self.mlp_norm = RMSNorm(config.model_dim)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, mask, kv_cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.model_dim)
        self.head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.embed.weight = self.head.weight  # weight tying

        # precompute RoPE cache
        cos, sin = build_rope_cache(config.max_seq_len, config.model_dim // config.n_heads, config.rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        kv_caches: Optional[list] = None,
    ) -> torch.Tensor:
        B, T = tokens.shape
        x = self.embed(tokens)

        # causal mask: -inf for future positions
        mask = torch.full((T, T), float("-inf"), device=tokens.device).triu(diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, self.rope_cos, self.rope_sin, mask, cache)

        x = self.norm(x)
        return self.head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
