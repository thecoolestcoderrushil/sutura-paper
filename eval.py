"""
Sutura — unified evaluation CLI.

Dispatches a severity sweep on the tear benchmark to one of the registration
methods and writes a CSV to results/:

    python eval.py --method {paste2,stalign,gpsa,sutura} \
        --severity-grid 0 1 2 3 4 6 8 [--in-sample] [--lodo] [--contrastive]

  --method paste2 / stalign / gpsa
        runs the corresponding optional-dependency baseline wrapper
        (benchmarks/*_baseline.py). If the dependency is missing the wrapper
        exits with a clear install message.

  --method sutura
        evaluates a trained Sutura checkpoint on the tear sweep. The flags select
        which published checkpoint / regime:
          --in-sample     in-sample checkpoint (checkpoints/sutura_insample.pt),
                          train and eval on the same donor pair (default).
          --lodo          leave-one-donor-out: evaluate on a held-out donor pair.
          --contrastive   use the contrastive checkpoint
                          (checkpoints/sutura_contrastive_S1.pt).
        Use --checkpoint to point at any other .pt; --reference/--sample choose
        the evaluation pair.

The Sutura path is honest: it loads a checkpoint and re-scores it on freshly
generated warps; it does NOT retrain. To produce checkpoints, see train.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"


# --------------------------------------------------------------------------- #
# Sutura model evaluation (no retraining; load checkpoint + re-score warps)
# --------------------------------------------------------------------------- #
def eval_sutura(args, severities):
    import anndata as ad
    import torch
    from scipy.spatial import cKDTree
    from model import load_checkpoint, knn_edges
    from model.contrastive import partner_index
    from benchmarks.deformation import apply_warp
    from metrics import registration_error_stats

    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _default_ckpt(args)
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}\n"
                 "Train one with train.py or point --checkpoint at a .pt file.")
    model, ckpt = load_checkpoint(ckpt_path)
    knn = ckpt.get("args", {}).get("knn", 6)
    pca_dim = ckpt.get("args", {}).get("pca_dim", 50)

    A = ad.read_h5ad(data_dir / f"DLPFC_{args.reference}.h5ad")
    B = ad.read_h5ad(data_dir / f"DLPFC_{args.sample}.h5ad")
    a_coords = np.asarray(A.obsm["spatial"], np.float32)
    pitch = float(np.median(cKDTree(a_coords).query(a_coords, k=2)[0][:, 1]))

    # features: a shared TruncatedSVD basis on the eval pair (per-slice standardized
    # to stay robust to donor batch shift, matching the perslice training mode).
    from train import fit_shared_basis, graph_tensors
    project = fit_shared_basis([A, B], pca_dim, ckpt.get("args", {}).get("seed", 0),
                               feature_mode="perslice")
    Z_A, Z_B = project(A), project(B)
    p = partner_index(A, B)
    gt_A = np.full((B.n_obs, 2), np.nan, np.float32)
    have = p >= 0
    gt_A[have] = a_coords[p[have]]

    ga = graph_tensors(a_coords, Z_A, knn, pitch)
    a_norm = torch.from_numpy(a_coords / pitch)

    rows = []
    for sv in severities:
        w, _ = apply_warp(B, sv, seed=args.eval_seed, tear=not args.smooth)
        gb = graph_tensors(np.asarray(w.obsm["spatial"], np.float32),
                           Z_B, knn, pitch)
        with torch.no_grad():
            pred = model(ga, gb, a_norm).numpy() * pitch
        st = registration_error_stats(pred, gt_A, mask=have)
        rows.append({"severity": sv, "reg_err_median": st["median"],
                     "reg_err_mean": st["mean"], "reg_err_p90": st["p90"],
                     "n": st["n"]})
        print(f"  sev={sv:>4}: Sutura median={st['median']:8.1f}px "
              f"mean={st['mean']:8.1f} p90={st['p90']:8.1f} n={st['n']}")
    return rows, ["severity", "reg_err_median", "reg_err_mean", "reg_err_p90", "n"]


def _default_ckpt(args):
    if args.contrastive:
        return CKPT_DIR / "sutura_contrastive_S1.pt"
    return CKPT_DIR / "sutura_insample.pt"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Sutura tear-benchmark evaluation.")
    ap.add_argument("--method", required=True,
                    choices=["paste2", "stalign", "gpsa", "sutura"])
    ap.add_argument("--severity-grid", nargs="+", type=float,
                    default=[0, 1, 2, 3, 4, 6, 8])
    ap.add_argument("--in-sample", action="store_true",
                    help="Sutura: in-sample checkpoint (default regime)")
    ap.add_argument("--lodo", action="store_true",
                    help="Sutura: leave-one-donor-out (evaluate held-out pair)")
    ap.add_argument("--contrastive", action="store_true",
                    help="Sutura: use the contrastive checkpoint")
    ap.add_argument("--smooth", action="store_true",
                    help="use the smooth (no-tear) warp instead of tears")
    ap.add_argument("--reference", default="151507")
    ap.add_argument("--sample", default="151508")
    ap.add_argument("--checkpoint", default="",
                    help="Sutura: explicit checkpoint .pt (overrides flags)")
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--out", default="",
                    help="output CSV name (default: <method>_eval.csv)")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    severities = [float(s) for s in args.severity_grid]
    results_dir = Path(args.results_dir)
    tear = not args.smooth

    if args.method == "paste2":
        from benchmarks.paste2_baseline import run_sweep
        suffix = args.out or ("paste2_eval" if tear else "paste2_eval_smooth")
        run_sweep(args.reference, args.sample, severities, seed=args.eval_seed,
                  tear=tear, data_dir=Path(args.data_dir), results_dir=results_dir,
                  suffix=suffix.replace("paste2_", "").replace(".csv", "")
                  or "eval")
        return

    if args.method in ("stalign", "gpsa"):
        # delegate to the wrapper's main() via argv so the optional-dep guard +
        # CSV writing live in one place.
        out_name = args.out or f"{args.method}_eval.csv"
        argv = ["--reference", args.reference, "--sample", args.sample,
                "--severities", ",".join(f"{s:g}" for s in severities),
                "--data-dir", args.data_dir, "--results-dir", args.results_dir]
        if args.method == "gpsa":
            from benchmarks.gpsa_baseline import main as gpsa_main
            sys.argv = ["gpsa_baseline", "--out", out_name] + argv
            gpsa_main()
        else:
            from benchmarks.stalign_baseline import main as stalign_main
            sfx = out_name.replace("stalign_", "").replace(".csv", "")
            sys.argv = (["stalign_baseline", "--suffix", "_" + sfx] + argv)
            stalign_main()
        return

    # sutura
    rows, fields = eval_sutura(args, severities)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out or "sutura_eval.csv"
    out_path = results_dir / out_name
    with open(out_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader(); wr.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
