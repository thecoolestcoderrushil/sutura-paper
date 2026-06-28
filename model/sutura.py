"""
Sutura — two-slice CROSS deformation model.

For each spot in the moving slice B, predict its location in the reference slice
A's coordinate frame (pitch units). Architecture:

  1. A shared graph encoder (model/encoder.py) runs on each slice's own kNN graph
     (A on its fixed coords, B on its warped coords; edge feature = relative
     position) -> per-spot embeddings z_A, z_B.
  2. Cross-slice attention (model/attention.py): each B spot queries all A spots
     by embedding similarity; the attention-weighted average of A's coordinates is
     a COARSE soft-correspondence estimate of the B spot's A-frame location.
  3. Residual head: MLP([z_B, attended z_A, coarse coord]) -> residual
     displacement. pred_A = coarse + residual. (Soft correspondence + a learned
     deformation field; the residual is where tear discontinuities get modeled.)

Supervision (see train.py): loss = || pred_A - gt_A || over array-matched B spots
(pitch units). Because Visium array positions don't move under a warp, gt_A is
fixed across severities; only B's input graph/coords change.

CPU-only. Runnable: `python -c "import torch; from model.sutura import SuturaNet"`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import Encoder
from .attention import CrossAttention


class SuturaNet(nn.Module):
    """Two-slice cross model: B spots -> predicted A-frame coordinates (pitch units)."""

    def __init__(self, feat_dim: int, hidden: int = 64, layers: int = 3,
                 attn_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(feat_dim, hidden, layers)   # shared across slices
        self.attn = CrossAttention(hidden, attn_dim)
        # refine head: [z_B, attended z_A, coarse coord(2)] -> residual displacement(2)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 2))

    def forward(self, ga, gb, a_coords_norm, return_match=False):
        """ga/gb: dicts with x, edge_index, edge_attr. a_coords_norm: (n_A, 2)
        in pitch units. Returns predicted A-frame coords for every B spot, in
        pitch units.

        If return_match=True, also returns a dict of cross-slice match logits over
        every (B spot, A spot) pair for the contrastive correspondence loss — both
        the raw scaled attention logits (q.k*scale, the distribution actually used
        for the coarse prediction) and the L2-normalized cosine similarity (for the
        canonical temperature-scaled InfoNCE read-out). The inference path
        (return_match=False) is byte-for-byte unchanged.
        """
        z_a = self.encoder(ga["x"], ga["edge_index"], ga["edge_attr"])   # (n_A,H)
        z_b = self.encoder(gb["x"], gb["edge_index"], gb["edge_attr"])   # (n_B,H)

        coarse, attended_za, match = self.attn(z_b, z_a, a_coords_norm)
        residual = self.head(torch.cat([z_b, attended_za, coarse], dim=-1))
        pred = coarse + residual                                         # (n_B,2)
        if not return_match:
            return pred
        return pred, match


# Backwards-compatible alias: checkpoints in checkpoints/ were trained under the
# original class name. The state_dict keys are identical (encoder.*, attn.q/k/v,
# head.*), so this alias loads them directly.
ARCACrossNet = SuturaNet
