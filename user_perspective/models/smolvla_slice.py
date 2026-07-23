"""SmolVLA slice — a miniature Vision-Language-Action model for e2e pipeline testing.

Represents a small robotics model: vision encoder + lightweight
language backbone + action head.  Designed to be torch.export-friendly
so the full capture -> compile pipeline can be exercised on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class SmolVLA(nn.Module):
    """Miniature VLA: conv visionstem + tiny transformer + action mlp."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        action_dim: int = 7,
        vocab_size: int = 256,
        max_seq_len: int = 16,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Vision stem: simple conv patch embed
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Position embeddings for patches + text tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + max_seq_len, embed_dim) * 0.02)

        # Lightweight transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, activation=F.gelu,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Text token embedding
        self.text_embed = nn.Embedding(vocab_size, embed_dim)

        # Action head
        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, action_dim),
        )

    def forward(self, image: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        vis_tokens = self.patch_embed(image).flatten(2).transpose(1, 2)
        txt_tokens = self.text_embed(text_tokens)
        combined = torch.cat([vis_tokens, txt_tokens], dim=1)
        combined = combined + self.pos_embed[:, :combined.shape[1], :]
        encoded = self.transformer(combined)
        pooled = encoded.mean(dim=1)
        return self.action_head(pooled)


# ---------------------------------------------------------------------------
# Bundle / loader
# ---------------------------------------------------------------------------

@dataclass
class _Bundle:
    model: nn.Module
    sample_inputs: tuple[Any, ...]
    source: str
    capture_mode: str
    notes: str
    extra: dict[str, Any] = field(default_factory=dict)
    num_cams: int | None = None


def load(mode: str = "auto") -> _Bundle:
    torch.manual_seed(0)
    model = SmolVLA()
    model.eval()
    sample_image = torch.randn(1, 3, model.img_size, model.img_size)
    sample_text = torch.randint(0, 256, (1, 8))
    return _Bundle(
        model=model,
        sample_inputs=(sample_image, sample_text),
        source="user_perspective/models/smolvla_slice.py",
        capture_mode="torch_export",
        notes="SmolVLA miniature — vision encoder + transformer + action head",
        extra={"param_count": sum(p.numel() for p in model.parameters())},
        num_cams=1,
    )
