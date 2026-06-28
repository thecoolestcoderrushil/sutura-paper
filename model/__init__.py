"""Sutura model package.

Public API:
    SuturaNet        the two-slice cross-attention registration model
    Encoder          shared graph encoder (residual DeformConv stack)
    CrossAttention   B -> A scaled dot-product cross attention
    knn_edges        scipy kNN graph builder for the spatial graphs
    load_checkpoint  load a published checkpoint into a SuturaNet instance
"""

from __future__ import annotations

import torch

from .encoder import Encoder, DeformConv, knn_edges
from .attention import CrossAttention
from .sutura import SuturaNet, ARCACrossNet

__all__ = [
    "SuturaNet", "ARCACrossNet", "Encoder", "DeformConv",
    "CrossAttention", "knn_edges", "load_checkpoint",
]


def _remap_state_dict(sd):
    """Map legacy flat attention keys (q/k/v.*) onto the nested CrossAttention
    submodule (attn.q/k/v.*). Checkpoints published with this release were trained
    with the attention projections defined directly on the model; the refactored
    model nests them under `self.attn`, but the weights are identical."""
    out = {}
    for key, val in sd.items():
        if key.split(".")[0] in ("q", "k", "v"):
            out["attn." + key] = val
        else:
            out[key] = val
    return out


def load_checkpoint(path, map_location="cpu"):
    """Build a SuturaNet from a checkpoint dict and return (model, ckpt).

    The checkpoint stores `state_dict`, `args` (training hyperparameters), and
    `pitch`. The model is reconstructed with the saved feat/hidden/layer/attn-dim
    so the state_dict loads cleanly."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    a = ckpt.get("args", {})
    model = SuturaNet(
        feat_dim=a.get("pca_dim", 50),
        hidden=a.get("hidden", 64),
        layers=a.get("layers", 3),
        attn_dim=a.get("attn_dim", 64),
    )
    model.load_state_dict(_remap_state_dict(ckpt["state_dict"]))
    model.eval()
    return model, ckpt
