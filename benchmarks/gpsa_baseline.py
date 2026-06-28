"""
Sutura benchmark — GPSA (Gaussian-Process Spatial Alignment) baseline.

Mirrors benchmarks/paste2_baseline.py (same slices, same apply_warp tear at
severities 0-8 seed 0, same array-bridge GT, same registration_error_stats) but
uses GPSA (Jones et al. 2023) as the aligner. View 0 = A (fixed), view 1 =
warped B; the model's aligned coords (G_means) for view-1 spots are B mapped into
A's frame. Writes results/gpsa_sweep.csv.

GPSA is an OPTIONAL dependency.

NOTE on installing GPSA: GPSA's own pinned requirements conflict with the
torch>=2.2 stack used here, so install it WITHOUT its dependencies and rely on
this repo's environment for torch/numpy:

    pip install gpsa --no-deps        # or: pip install -e <gpsa-checkout> --no-deps

The symbols (VariationalGPSA, rbf_kernel, LossNotDecreasingChecker) are imported
lazily via importlib so the wrapper imports cleanly even when GPSA is absent.

Usage:
    python -m benchmarks.gpsa_baseline --severities 0,1,2,3,4,6,8       # full
    python -m benchmarks.gpsa_baseline --severities 8 --subsample 400 --epochs 300
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from benchmarks.deformation import apply_warp
from metrics import registration_error_stats

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DEVICE = "cpu"


def _import_gpsa():
    # importlib so a missing GPSA never breaks `import benchmarks.gpsa_baseline`.
    try:
        gpsa = importlib.import_module("gpsa")
        return (gpsa.VariationalGPSA, gpsa.rbf_kernel,
                gpsa.LossNotDecreasingChecker)
    except ImportError as exc:  # pragma: no cover - optional dependency
        sys.exit(
            "GPSA is not installed. Install it WITHOUT its dependencies "
            "(`pip install gpsa --no-deps`) so it uses this repo's torch>=2.2 "
            f"stack, then re-run.\n  underlying error: {exc}")


def load(s, data_dir):
    return ad.read_h5ad(data_dir / f"DLPFC_{s}.h5ad")


def common_genes(a, b):
    g = a.var_names.intersection(b.var_names)
    return a[:, g].copy(), b[:, g].copy()


def expr_pcs(A, B, k=10):
    def ln(X):
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, float)
        s = X.sum(1, keepdims=True); s[s == 0] = 1.0
        return np.log1p(X / s * 1e4)
    Xa, Xb = ln(A.X), ln(B.X)
    Z = np.vstack([Xa, Xb]); mu = Z.mean(0)
    U, S, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    comp = Vt[: min(k, Vt.shape[0])]
    fa = (Xa - mu) @ comp.T; fb = (Xb - mu) @ comp.T
    both = np.vstack([fa, fb]); m, sd = both.mean(0), both.std(0) + 1e-8
    return (fa - m) / sd, (fb - m) / sd


def subset_common(ref, tgt, n, seed=0):
    def keys(a):
        return list(zip(a.obs.array_row.astype(int), a.obs.array_col.astype(int)))
    rk = {k: i for i, k in enumerate(keys(ref))}
    tk = {k: j for j, k in enumerate(keys(tgt))}
    common = sorted(set(rk) & set(tk)); rng = np.random.default_rng(seed)
    if n < len(common):
        common = [common[i] for i in np.sort(rng.choice(len(common), n, replace=False))]
    return ref[[rk[k] for k in common]].copy(), tgt[[tk[k] for k in common]].copy()


def gt_targets(ref, warped):
    rk = {k: i for i, k in enumerate(
        zip(ref.obs.array_row.astype(int), ref.obs.array_col.astype(int)))}
    rc = np.asarray(ref.obsm["spatial"], float)
    gt = np.full((warped.n_obs, 2), np.nan); have = np.zeros(warped.n_obs, bool)
    for j, k in enumerate(zip(warped.obs.array_row.astype(int),
                              warped.obs.array_col.astype(int))):
        i = rk.get(k)
        if i is not None:
            gt[j] = rc[i]; have[j] = True
    return gt, have


def align_gpsa(A_xy, B_xy, A_feat, B_feat, m_inducing, epochs, lr=1e-2,
               verbose=False):
    VariationalGPSA, rbf_kernel, LossNotDecreasingChecker = _import_gpsa()
    nA, nB = A_xy.shape[0], B_xy.shape[0]
    X = np.vstack([A_xy, B_xy]).astype(np.float32)
    Y = np.vstack([A_feat, B_feat]).astype(np.float32)
    x = torch.from_numpy(X).float().to(DEVICE)
    y = torch.from_numpy(Y).float().to(DEVICE)
    data_dict = {"expression": {"spatial_coords": x, "outputs": y,
                                "n_samples_list": [nA, nB]}}
    model = VariationalGPSA(
        data_dict, n_spatial_dims=2,
        m_X_per_view=m_inducing, m_G=m_inducing,
        data_init=True, minmax_init=False, grid_init=False,
        n_latent_gps={"expression": None},
        mean_function="identity_fixed",
        kernel_func_warp=rbf_kernel, kernel_func_data=rbf_kernel,
        fixed_view_idx=0,
    ).to(DEVICE)
    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    checker = LossNotDecreasingChecker(max_epochs=epochs, atol=1e-4, window_size=50)
    loss_trace = np.zeros(epochs)
    for t in range(epochs):
        model.train()
        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            {"expression": x}, view_idx=view_idx, Ns=Ns, S=3)
        loss = model.loss_fn(data_dict, F_samples)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_trace[t] = loss.item()
        if verbose and t % 100 == 0:
            print(f"    iter {t:4d} LL {-loss.item():.3e}")
        if t > 100 and checker.check_loss(t, loss_trace):
            if verbose:
                print(f"    converged at iter {t}")
            break
    model.eval()
    with torch.no_grad():
        G_means, _, _, _ = model.forward({"expression": x}, view_idx=view_idx, Ns=Ns)
    G = G_means["expression"] if isinstance(G_means, dict) else G_means
    G = G.detach().cpu().numpy()
    return G[nA:]   # aligned coords for view 1 (B) in the common (~A) frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="151507")
    ap.add_argument("--sample", default="151508")
    ap.add_argument("--severities", default="0,1,2,3,4,6,8")
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--inducing", type=int, default=150)
    ap.add_argument("--out", default="gpsa_sweep.csv")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    sevs = [float(s) for s in args.severities.split(",")]
    data_dir, results_dir = Path(args.data_dir), Path(args.results_dir)

    ref = load(args.reference, data_dir); tgt = load(args.sample, data_dir)
    rows = []
    for sev in sevs:
        warped, _ = apply_warp(tgt, sev, seed=0, tear=True)
        A, B = ref, warped
        if args.subsample:
            A, B = subset_common(ref, warped, args.subsample, seed=0)
        Ac, Bc = common_genes(A, B)
        fa, fb = expr_pcs(Ac, Bc, k=10)
        t0 = time.time()
        pred = align_gpsa(np.asarray(A.obsm["spatial"], float),
                          np.asarray(B.obsm["spatial"], float),
                          fa, fb, args.inducing, args.epochs, verbose=args.verbose)
        dt = time.time() - t0
        gt, have = gt_targets(A, B)
        st = registration_error_stats(pred, gt, mask=have)
        rows.append({"severity": sev, "reg_err_median": st["median"],
                     "reg_err_mean": st["mean"], "reg_err_p90": st["p90"],
                     "n": st["n"]})
        print(f"  sev={sev:>4}: GPSA median={st['median']:8.1f}px "
              f"mean={st['mean']:8.1f} p90={st['p90']:8.1f} n={st['n']} ({dt:.0f}s)")

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / args.out
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["severity", "reg_err_median",
                                           "reg_err_mean", "reg_err_p90", "n"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
