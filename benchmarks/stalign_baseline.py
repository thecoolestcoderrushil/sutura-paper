"""
Sutura benchmark — STalign (diffeomorphic LDDMM) baseline.

STalign aligns two slices by diffeomorphic metric mapping of their (rasterized)
spatial density. We align source = warped-moving to target = reference, transform
the warped B spots into A's frame, and score against the SAME array-bridge ground
truth used everywhere else. This directly tests the thesis that diffeomorphic
methods structurally cannot represent a tear (a diffeomorphism is invertible and
continuous, so it cannot tear tissue apart). Writes results/stalign_sweep.csv.

STalign is an OPTIONAL dependency; if it is missing this wrapper fails with a
clear message rather than at import time.

Usage:
    python -m benchmarks.stalign_baseline --severities 0,4,8 --niter 1500   # full
    python -m benchmarks.stalign_baseline --severities 0,8 --niter 500      # smoke
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy.spatial import cKDTree

from benchmarks.deformation import apply_warp
from model.contrastive import partner_index
from metrics import registration_error_stats

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def _import_stalign():
    try:
        from STalign import STalign as ST
    except ImportError as exc:  # pragma: no cover - optional dependency
        sys.exit(
            "STalign is not installed. The STalign baseline is an optional "
            "dependency; install it (see requirements.txt) to run this "
            f"benchmark.\n  underlying error: {exc}")
    return ST


def array_bridge(A, B):
    """gt_A[j] = A coordinate at B spot j's Visium array position; have[j] flag."""
    p = partner_index(A, B)
    acoord = np.asarray(A.obsm["spatial"], np.float32)
    gt = np.full((B.n_obs, 2), np.nan, np.float32)
    have = p >= 0
    gt[have] = acoord[p[have]]
    return gt, have


def run(args):
    ST = _import_stalign()
    data_dir, results_dir = Path(args.data_dir), Path(args.results_dir)
    A = ad.read_h5ad(data_dir / f"DLPFC_{args.reference}.h5ad")
    B = ad.read_h5ad(data_dir / f"DLPFC_{args.sample}.h5ad")
    cA = np.asarray(A.obsm["spatial"], float)            # [x,y]
    pitch = float(np.median(cKDTree(cA).query(cA, k=2)[0][:, 1]))
    gt_A, have = array_bridge(A, B)                       # A-frame [x,y] per B spot
    dx = args.dx if args.dx > 0 else pitch / 2.0

    # rasterize TARGET (A) once
    xJ, yJ = cA[:, 0], cA[:, 1]
    XJ, YJ, J = ST.rasterize(xJ, yJ, dx=dx, draw=False)
    J = torch.tensor(ST.normalize(J), dtype=torch.float64)
    xvJ = [torch.tensor(YJ, dtype=torch.float64),
           torch.tensor(XJ, dtype=torch.float64)]

    rows = []
    for sev in [float(s) for s in args.severities.split(",")]:
        t0 = time.time()
        w, _ = apply_warp(B, sev, seed=args.seed, tear=True)
        cB = np.asarray(w.obsm["spatial"], float)        # warped [x,y]
        xI, yI = cB[:, 0], cB[:, 1]
        XI, YI, I = ST.rasterize(xI, yI, dx=dx, draw=False)
        I = torch.tensor(ST.normalize(I), dtype=torch.float64)
        xvI = [torch.tensor(YI, dtype=torch.float64),
               torch.tensor(XI, dtype=torch.float64)]

        out = ST.LDDMM(xvI, I, xvJ, J, niter=args.niter, device="cpu",
                       dtype=torch.float64, a=args.a * dx)
        Amat, v, xv = out["A"], out["v"], out["xv"]

        pts = np.stack([yI, xI], -1)                     # SOURCE points [row,col]
        tpts = ST.transform_points_source_to_target(xv, v, Amat, pts)
        tpts = tpts.detach().numpy() if hasattr(tpts, "detach") else np.asarray(tpts)
        pred = np.stack([tpts[:, 1], tpts[:, 0]], -1)    # back to [x,y]

        st = registration_error_stats(pred, gt_A, mask=have)
        dt = time.time() - t0
        rows.append(dict(severity=sev, reg_err_median=st["median"],
                         reg_err_mean=st["mean"], reg_err_p90=st["p90"],
                         n=st["n"], runtime_s=dt))
        print(f"  sev{sev:>3.0f}: median {st['median']:7.1f}px  "
              f"mean {st['mean']:7.1f}px  p90 {st['p90']:7.1f}px  "
              f"(n={st['n']}, {dt:.0f}s)")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"stalign_sweep{args.suffix}.csv"
    with open(out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"wrote {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="151507")
    p.add_argument("--sample", default="151508")
    p.add_argument("--severities", default="0,4,8")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--niter", type=int, default=1500)
    p.add_argument("--dx", type=float, default=0)     # 0 -> pitch/2
    p.add_argument("--a", type=float, default=5.0)    # smoothness scale (units of dx)
    p.add_argument("--suffix", default="")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    run(p.parse_args())


if __name__ == "__main__":
    main()
