"""Gemma decode slice — a miniature decoder-only transformer for e2e testing.

Represents a single decode step of a Gemma-like model with KV-cache,
grouped-query attention, and RMSNorm.  Designed to be torch.export-friendly.

See ``examples/real_models/gemma2b_compile.py`` for the real Gemma-2B path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GemmaDecodeSlice(nn.Module):
    """Single-step decoder slice: one transformer layer with GQA + SwiGLU FFN."""

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        ffn_dim: int = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads

        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.input_norm = nn.RMSNorm(embed_dim, eps=1e-6)
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)
        self.post_attn_norm = nn.RMSNorm(embed_dim, eps=1e-6)
        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.final_norm = nn.RMSNorm(embed_dim, eps=1e-6)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = input_ids.shape[0]
        x = self.text_embed(input_ids)
        residual = x
        x = self.input_norm(x)

        q = self.q_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        new_keys = torch.cat([past_keys, k], dim=2)
        new_values = torch.cat([past_values, v], dim=2)

        n_rep = self.num_heads // self.num_kv_heads
        k_expanded = new_keys.repeat_interleave(n_rep, dim=1)
        v_expanded = new_values.repeat_interleave(n_rep, dim=1)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v_expanded)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, self.embed_dim)
        x = residual + self.o_proj(attn_out)

        residual = x
        x = self.post_attn_norm(x)
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        x = residual + self.down_proj(gate * up)

        logits = self.lm_head(self.final_norm(x))
        return logits, new_keys, new_values


@dataclass
class _Bundle:
    model: nn.Module
    sample_inputs: tuple[Any, ...]
    source: str
    capture_mode: str
    notes: str
    extra: dict[str, Any] = field(default_factory=dict)


def load() -> _Bundle:
    torch.manual_seed(0)
    model = GemmaDecodeSlice()
    model.eval()
    sample_ids = torch.randint(0, 256, (1, 1))
    past_keys = torch.zeros(1, model.num_kv_heads, 0, model.head_dim)
    past_values = torch.zeros(1, model.num_kv_heads, 0, model.head_dim)
    return _Bundle(
        model=model,
        sample_inputs=(sample_ids, past_keys, past_values),
        source="user_perspective/models/gemma_decode_slice.py",
        capture_mode="torch_export",
        notes="Gemma decode slice miniature — single decoder step with KV-cache, GQA, RMSNorm",
        extra={"param_count": sum(p.numel() for p in model.parameters())},
    )
