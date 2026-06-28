"""
Sutura — shared graph encoder.

A residual stack of DeformConv message-passing layers reads each slice's own kNN
spatial graph (relative-position edge features) and produces per-spot embeddings.
The SAME encoder weights are applied to both the reference slice A and the moving
slice B, so their embeddings live in a comparable space for cross-slice matching.

CPU-only and dependency-light: the kNN graph is built with scipy (no
torch_cluster); message passing uses PyG's native scatter fallback (no
torch_scatter).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from torch_geometric.nn import MessagePassing


def knn_edges(coords: np.ndarray, k: int):
    """Symmetric kNN graph -> (edge_index [2, E], dst, src).

    Returns the directed edge_index plus the destination/source index arrays so
    callers can build relative-position edge features (coords[dst] - coords[src]).
    """
    _, idx = cKDTree(coords).query(coords, k=k + 1)   # includes self at col 0
    src = np.repeat(np.arange(coords.shape[0]), k)
    dst = idx[:, 1:].reshape(-1)
    # symmetrize
    s = np.concatenate([src, dst])
    d = np.concatenate([dst, src])
    edge_index = np.stack([s, d])
    return edge_index, d, s


class DeformConv(MessagePassing):
    """Message passing with relative-position edge features."""

    def __init__(self, dim: int):
        super().__init__(aggr="mean")
        self.msg = nn.Sequential(nn.Linear(2 * dim + 2, dim), nn.ReLU())
        self.upd = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU())

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.upd(torch.cat([x, out], dim=-1))

    def message(self, x_i, x_j, edge_attr):
        return self.msg(torch.cat([x_i, x_j, edge_attr], dim=-1))


class Encoder(nn.Module):
    """Shared GNN encoder: feat -> hidden embeddings (residual DeformConv stack)."""

    def __init__(self, feat_dim: int, hidden: int, layers: int):
        super().__init__()
        self.enc = nn.Linear(feat_dim, hidden)
        self.convs = nn.ModuleList([DeformConv(hidden) for _ in range(layers)])

    def forward(self, x, edge_index, edge_attr):
        h = torch.relu(self.enc(x))
        for conv in self.convs:
            h = h + conv(h, edge_index, edge_attr)      # residual
        return h
