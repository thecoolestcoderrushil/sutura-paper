"""
Sutura — cross-slice attention (B -> A).

Each moving-slice (B) spot queries all reference-slice (A) spots by embedding
similarity. The attention-weighted average of A's coordinates is a coarse
soft-correspondence estimate of where that B spot lands in A's frame, and the
attention-weighted average of A's value embeddings is fed to the residual head.

The module exposes both read-outs needed downstream:
  * the raw scaled dot-product attention logits (q . k * scale), the distribution
    actually used for the coarse prediction; and
  * the L2-normalized cosine similarity (q_hat . k_hat), used by the canonical
    temperature-scaled InfoNCE contrastive read-out (see model/contrastive.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Scaled dot-product attention from B queries to A keys/values."""

    def __init__(self, hidden: int, attn_dim: int):
        super().__init__()
        self.q = nn.Linear(hidden, attn_dim)
        self.k = nn.Linear(hidden, attn_dim)
        self.v = nn.Linear(hidden, hidden)
        self.scale = attn_dim ** -0.5

    def forward(self, z_b, z_a, a_coords_norm):
        """z_b: (n_B, H)  z_a: (n_A, H)  a_coords_norm: (n_A, 2) pitch units.

        Returns:
          coarse       (n_B, 2)  soft-correspondence coordinate in A's frame
          attended_za  (n_B, H)  attention-weighted A value embeddings
          match        dict with raw attention logits and cosine similarity
        """
        qb, ka = self.q(z_b), self.k(z_a)                    # (n_B,d), (n_A,d)
        scores = (qb @ ka.T) * self.scale                    # (n_B, n_A)
        attn = torch.softmax(scores, dim=1)
        coarse = attn @ a_coords_norm                        # (n_B, 2)
        attended_za = attn @ self.v(z_a)                     # (n_B, H)
        qn = F.normalize(qb, dim=1)
        kn = F.normalize(ka, dim=1)
        match = {"attn_logits": scores, "cos_sim": qn @ kn.T}
        return coarse, attended_za, match
