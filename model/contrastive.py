"""
Sutura — donor-invariant contrastive correspondence loss.

Supervises the cross-slice SIMILARITY GEOMETRY (not just per-spot coordinates) so
the learned matching can transfer to unseen donors, where coordinate regression
alone did not. The full training objective is

    loss = L_reg + lambda * L_contrastive

  L_reg : the coordinate loss ||pred_A - gt_A|| (pitch units), defined in train.py.
  L_c   : a SOFT cross-entropy of the cross-slice match distribution against a
          Gaussian target centered on each B spot's array-bridge A-partner p(j):
              t[j, i] proportional to exp(-||A[i] - A[p(j)]||^2 / 2 sigma^2)
          so near-true neighbours share mass (no false-negative penalty on
          continuous tissue). The denominator (logsumexp) spans ALL A spots, so
          every other A spot is a negative. Target support is sparse (A spots
          within 4 sigma of the partner), precomputed once per pair (warp-
          invariant, like gt_A).

Two read-outs for the match logits ell[j, i] (--readout), ablated in the paper:
  cosine : L2-normalized shared q/k projections, ell = (q_hat . k_hat) / temp
           (canonical temperature-scaled InfoNCE).
  attn   : the model's raw scaled attention logits ell = (q . k) * scale
           (the distribution actually used for the coarse prediction; no norm).

Diagnostic: held-out top-1 MATCH ACCURACY = fraction of bridged B spots whose
argmax_i ell[j, i] equals p(j) — measures whether correspondence transfers,
independent of the coordinate head.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree


def partner_index(A, B):
    """p[j] = index of the A spot at B spot j's Visium array position (-1 if none).
    Warp-invariant (array positions don't move), so computed once per pair."""
    key = {(int(r), int(c)): i for i, (r, c) in
           enumerate(zip(A.obs["array_row"], A.obs["array_col"]))}
    p = np.full(B.n_obs, -1, np.int64)
    for j, (r, c) in enumerate(zip(B.obs["array_row"], B.obs["array_col"])):
        i = key.get((int(r), int(c)))
        if i is not None:
            p[j] = i
    return p


def soft_target(a_coords, partner, pitch, sigma_pitch):
    """Sparse Gaussian target over A spots around each bridged B spot's partner.

    Returns torch COO pieces (tj, ti, tw) + the bridged-B index list. tw rows sum
    to 1 over each j, so L_c is a proper soft cross-entropy."""
    sigma = sigma_pitch * pitch
    radius = 4.0 * sigma
    tree = cKDTree(a_coords)
    bridged = np.where(partner >= 0)[0]
    tj, ti, tw = [], [], []
    for j in bridged:
        p = int(partner[j])
        nb = tree.query_ball_point(a_coords[p], radius)        # incl. p itself
        d2 = ((a_coords[nb] - a_coords[p]) ** 2).sum(1)
        w = np.exp(-d2 / (2.0 * sigma * sigma))
        w /= w.sum()
        tj.extend([j] * len(nb)); ti.extend(nb); tw.extend(w)
    return {
        "tj": torch.tensor(tj, dtype=torch.long),
        "ti": torch.tensor(ti, dtype=torch.long),
        "tw": torch.tensor(tw, dtype=torch.float32),
        "bridged": torch.tensor(bridged, dtype=torch.long),
        "partner": partner,
    }


def match_logits(match, readout, temp):
    """Pick the (n_B, n_A) match-logit matrix for the chosen read-out.

    `match` is the dict returned by SuturaNet(..., return_match=True)."""
    if readout == "cosine":
        return match["cos_sim"] / temp
    return match["attn_logits"]


def contrastive_loss(logits, ct):
    """Soft cross-entropy: mean_j ( logsumexp_i ell[j,i] - sum_i t[j,i] ell[j,i] ),
    averaged over bridged B spots (rows whose target mass sums to 1)."""
    lse = torch.logsumexp(logits, dim=1)                       # (n_B,)
    gathered = logits[ct["tj"], ct["ti"]]                      # (nnz,)
    wsum = torch.zeros(logits.shape[0]).index_add(
        0, ct["tj"], ct["tw"] * gathered)                      # (n_B,)
    return (lse - wsum)[ct["bridged"]].mean()


def match_top1(logits, ct):
    """Fraction of bridged B spots whose argmax A partner is the true one."""
    pi = logits.argmax(dim=1).cpu().numpy()
    b = ct["bridged"].cpu().numpy()
    return float((pi[b] == ct["partner"][b]).mean())
