"""
Sutura benchmark — PASTE2 partial fused Gromov-Wasserstein baseline.

For a sweep of warp severities, warp the moving slice with a known displacement
field (benchmarks.deformation.apply_warp), run PASTE2 partial alignment of
(reference, warped-moving), and measure how badly registration degrades:

  * registration error (px) — map each warped spot back into the reference frame
    via the transport plan (barycentric and argmax projections) and compare to the
    array-bridge ground-truth target. mean / median / p90 reported.
  * layer-label transfer accuracy + random floor (from metrics.py).

This is the optimal-transport baseline Sutura is compared against. PASTE2 is
unsupervised, so it has no train/test gap (the same curve applies in-sample and
held-out).

PASTE2 is an OPTIONAL dependency. Install it from the PASTE2 distribution; if it
is missing, this wrapper fails with a clear message rather than at import time.

Usage:
    python -m benchmarks.paste2_baseline --severities 0,1,2,3,4,6,8 --tear
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np

from benchmarks.deformation import apply_warp, sev_tag
from metrics import (argmax_projection, barycentric_projection,
                     registration_error_stats, report)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def _import_paste2():
    try:
        from paste2.PASTE2 import partial_pairwise_align
        from paste2.helper import filter_for_common_genes
    except ImportError as exc:  # pragma: no cover - optional dependency
        sys.exit(
            "PASTE2 is not installed. The PASTE2 baseline is an optional "
            "dependency; install it (see requirements.txt 'optional baselines') "
            f"to run this benchmark.\n  underlying error: {exc}")
    return partial_pairwise_align, filter_for_common_genes


def load_slice(sample_id: str, data_dir: Path) -> ad.AnnData:
    path = data_dir / f"DLPFC_{sample_id}.h5ad"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run python -m data.load_spatiallibd first.")
    return ad.read_h5ad(path)


def array_keys(adata):
    return list(zip(adata.obs["array_row"].astype(int),
                    adata.obs["array_col"].astype(int)))


def gt_targets(ref, warped_target, mode):
    """Ground-truth reference-frame coordinate for each warped target spot."""
    if mode == "self":
        return (np.asarray(warped_target.obsm["spatial_original"], dtype=float),
                np.ones(warped_target.n_obs, dtype=bool))
    rk = {k: i for i, k in enumerate(array_keys(ref))}
    rcoord = np.asarray(ref.obsm["spatial"], dtype=float)
    gt = np.full((warped_target.n_obs, 2), np.nan)
    have = np.zeros(warped_target.n_obs, dtype=bool)
    for j, k in enumerate(array_keys(warped_target)):
        i = rk.get(k)
        if i is not None:
            gt[j] = rcoord[i]
            have[j] = True
    return gt, have


def run_one(ref, target_orig, severity, *, seed, tear, mode, s, alpha,
            dissimilarity, align_fn, filter_fn):
    """Warp, align, and score a single severity level (barycentric + argmax)."""
    warped, _ = apply_warp(target_orig, severity, seed=seed, tear=tear)

    A = ref.copy()
    B = warped.copy()
    filter_fn([A, B])

    t0 = time.time()
    pi = np.asarray(align_fn(
        A, B, s=s, alpha=alpha, dissimilarity=dissimilarity, verbose=False))
    dt = time.time() - t0

    gt, have = gt_targets(A, B, mode)
    pred_b, col_mass = barycentric_projection(pi, A.obsm["spatial"])
    pred_a, _ = argmax_projection(pi, A.obsm["spatial"])
    mask = have & (col_mass > 0)
    reg_bary = registration_error_stats(pred_b, gt, mask=mask)
    reg_arg = registration_error_stats(pred_a, gt, mask=mask)

    lab = report(pi, A.obs["layer"].astype(str).values,
                 B.obs["layer"].astype(str).values)

    warp_meta = warped.uns["warp"]
    return {
        "severity": severity,
        "max_disp_px": warp_meta["max_disp_px"],
        "mean_disp_px": warp_meta["mean_disp_px"],
        "reg_err_mean": reg_bary["mean"],
        "reg_err_median": reg_bary["median"],
        "reg_err_p90": reg_bary["p90"],
        "reg_err_mean_argmax": reg_arg["mean"],
        "reg_err_median_argmax": reg_arg["median"],
        "reg_err_p90_argmax": reg_arg["p90"],
        "n_reg_scored": reg_bary["n"],
        "paste2_acc": lab["model"]["accuracy"],
        "random_floor": lab["random_floor"]["accuracy_mean"],
        "n_label_scored": lab["model"]["n_scored"],
        "runtime_s": dt,
    }


def run_sweep(reference, sample, severities, *, seed=0, tear=True, mode="cross",
              s=0.99, alpha=0.1, dissimilarity="glmpca", data_dir=DATA_DIR,
              results_dir=RESULTS_DIR, suffix="", write_csv=True):
    """Run a full PASTE2 severity sweep and return the rows (also writes a CSV)."""
    align_fn, filter_fn = _import_paste2()
    severities = [float(x) for x in severities]

    ref_full = load_slice(reference if mode == "cross" else sample, data_dir)
    target_full = load_slice(sample, data_dir)
    if mode == "self":
        ref_full = target_full

    print("=" * 70)
    print(f"PASTE2 deformation sweep — mode={mode}  ref={reference} sample={sample}")
    print(f"  severities    : {severities}")
    print(f"  dissimilarity : {dissimilarity}  s={s} alpha={alpha} tear={tear}")
    print("=" * 70)

    rows = []
    for sev in severities:
        r = run_one(ref_full, target_full, sev, seed=seed, tear=tear, mode=mode,
                    s=s, alpha=alpha, dissimilarity=dissimilarity,
                    align_fn=align_fn, filter_fn=filter_fn)
        rows.append(r)
        print(f"  sev={sev:>4}: reg_err median bary={r['reg_err_median']:7.1f}px "
              f"argmax={r['reg_err_median_argmax']:7.1f}px  "
              f"acc={r['paste2_acc']*100:5.1f}% ({r['runtime_s']:.0f}s)")

    if write_csv:
        results_dir.mkdir(parents=True, exist_ok=True)
        suf = f"_{suffix}" if suffix else ""
        csv_path = results_dir / f"paste2_sweep{suf}.csv"
        with open(csv_path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote CSV -> {csv_path}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="151507")
    p.add_argument("--sample", default="151508")
    p.add_argument("--mode", choices=["cross", "self"], default="cross")
    p.add_argument("--severities", default="0,1,2,3,4,6,8")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tear", action="store_true")
    p.add_argument("--s", type=float, default=0.99,
                   help="PASTE2 overlap fraction (0.99 ~ full; s=1.0 trips a "
                        "partial-OT float-rounding infeasibility)")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--dissimilarity", default="glmpca",
                   choices=["glmpca", "pca", "kl", "euclidean"])
    p.add_argument("--suffix", default="")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = p.parse_args()

    run_sweep(args.reference, args.sample,
              args.severities.split(","), seed=args.seed, tear=args.tear,
              mode=args.mode, s=args.s, alpha=args.alpha,
              dissimilarity=args.dissimilarity, suffix=args.suffix,
              data_dir=Path(args.data_dir), results_dir=Path(args.results_dir))


if __name__ == "__main__":
    main()
