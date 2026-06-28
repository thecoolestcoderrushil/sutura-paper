"""
Sutura — training (in-sample, leave-one-donor-out, and contrastive variants).

One script covers every training regime in the paper:

  * in-sample            train AND evaluate on the same donor pair (only warp
                         seeds held out). Proves Sutura fits torn tissue.
  * leave-one-donor-out  train on one or more donor pairs and evaluate on a
                         DIFFERENT, held-out donor pair (no tissue overlap). The
                         held-out curve is a true cross-sample generalization
                         result. Use --train-pairs / --test-pair.
  * contrastive          add the donor-invariant InfoNCE correspondence loss
                         (model/contrastive.py) on top of the coordinate loss,
                         with --lambda-contrastive > 0.

Supervision: loss = L_reg + lambda * L_contrastive.
  L_reg : ||pred_A - gt_A|| over array-bridge-matched B spots (pitch units).
          Because Visium array positions don't move under a warp, gt_A is fixed
          across severities; only B's input graph/coords change.
  L_c   : soft cross-entropy of the cross-slice match distribution against a
          Gaussian target on each B spot's true A partner (see model/contrastive).

Transferable features: ONE TruncatedSVD basis is fit on the TRAINING slices only,
then every slice (train AND held-out) is projected into it, so "the model never
saw the test tissue" holds for features too. --feature-mode controls how the
embedding is standardized:
  global   : training pooled mean/std (a held-out donor with a batch shift lands
             off-distribution — the diagnosed single-donor LOO failure mode).
  perslice : per-slice mean/std (cheap batch correction; keeps an unseen donor
             in-distribution).

CPU-only.

Usage:
    # in-sample (single pair, warp-seed held out)
    python train.py --train-pairs 151507/151508 --test-pair 151507/151508 \
        --out sutura_insample

    # leave-one-donor-out, perslice features
    python train.py --train-pairs 151507/151508 --test-pair 151669/151670 \
        --feature-mode perslice --out sutura_lodo_S2

    # contrastive (cosine read-out, lambda 0.5)
    python train.py --train-pairs 151507/151508,151669/151670 \
        --test-pair 151673/151674 --readout cosine --lambda-contrastive 0.5 \
        --out sutura_contrastive_S3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy.sparse import issparse, vstack as svstack
from scipy.spatial import cKDTree
from sklearn.decomposition import TruncatedSVD

from model import SuturaNet, knn_edges
from model.contrastive import (partner_index, soft_target, match_logits,
                               contrastive_loss, match_top1)
from benchmarks.deformation import apply_warp
from metrics import registration_error_stats

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def lognorm(adata):
    """Library-size normalize to 1e4 then log1p (keeps sparsity)."""
    X = adata.X
    X = X.tocsr().astype(np.float32) if issparse(X) else np.asarray(X, np.float32)
    counts = np.asarray(X.sum(1)).ravel()
    counts[counts == 0] = 1.0
    if issparse(X):
        X = X.multiply(1e4 / counts[:, None]).tocsr()
        X.data = np.log1p(X.data)
        return X
    return np.log1p(X * (1e4 / counts[:, None]))


def fit_shared_basis(train_slices, dim, seed, feature_mode="global"):
    """Fit ONE TruncatedSVD basis on the lognorm expression of the training
    slices stacked together. Returns a closure that projects any slice (with the
    same gene set) into this fixed basis (see module docstring for feature-mode)."""
    mats = [lognorm(a) for a in train_slices]
    stacked = svstack(mats) if issparse(mats[0]) else np.vstack(mats)
    svd = TruncatedSVD(n_components=dim, random_state=seed).fit(stacked)
    Ztr = svd.transform(stacked)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6

    def project(adata):
        Z = svd.transform(lognorm(adata)).astype(np.float32)
        if feature_mode == "perslice":
            m, s = Z.mean(0), Z.std(0) + 1e-6          # this slice's own stats
            return ((Z - m) / s).astype(np.float32)
        return ((Z - mu) / sd).astype(np.float32)      # pooled training stats

    return project


# --------------------------------------------------------------------------- #
# graph / pair construction
# --------------------------------------------------------------------------- #
def graph_tensors(coords, Z, k, pitch):
    edge_index, d, s = knn_edges(coords, k)
    edge_attr = ((coords[d] - coords[s]) / pitch).astype(np.float32)
    return {"x": torch.from_numpy(Z),
            "edge_index": torch.from_numpy(edge_index).long(),
            "edge_attr": torch.from_numpy(edge_attr)}


def array_bridge(A, B):
    """gt_A[j] = A coordinate at B spot j's Visium array position; have[j] flag."""
    p = partner_index(A, B)
    acoord = np.asarray(A.obsm["spatial"], np.float32)
    gt = np.full((B.n_obs, 2), np.nan, np.float32)
    have = p >= 0
    gt[have] = acoord[p[have]]
    return gt, have


def assert_same_genes(slices):
    """All slices must share var_names in identical order (shared SVD basis)."""
    ref = slices[0].var_names
    for s in slices[1:]:
        if not s.var_names.equals(ref):
            raise ValueError(
                "gene set mismatch: a slice's var_names differ from the training "
                "reference; the shared SVD basis would be invalid.")


def build_pair(ref_id, smp_id, project, knn, data_dir, sigma_pitch):
    """Load a donor pair and assemble everything the model needs for it."""
    A = ad.read_h5ad(data_dir / f"DLPFC_{ref_id}.h5ad")
    B = ad.read_h5ad(data_dir / f"DLPFC_{smp_id}.h5ad")
    a_coords = np.asarray(A.obsm["spatial"], np.float32)
    pitch = float(np.median(cKDTree(a_coords).query(a_coords, k=2)[0][:, 1]))

    Z_A, Z_B = project(A), project(B)
    gt_A, have = array_bridge(A, B)
    ga = graph_tensors(a_coords, Z_A, knn, pitch)
    p = partner_index(A, B)
    return {
        "ref": ref_id, "smp": smp_id, "A": A, "B": B,
        "a_coords": a_coords, "pitch": pitch, "Z_B": Z_B,
        "gt_A": gt_A, "have": have, "ga": ga,
        "a_norm": torch.from_numpy(a_coords / pitch),
        "gt_norm": torch.from_numpy(gt_A / pitch),
        "mask": torch.from_numpy(have),
        "ct": soft_target(a_coords, p, pitch, sigma_pitch),
    }


def warp_graph(pair, sev, seed, tear, knn):
    """Warp pair's B at (sev, seed, tear) and return its input graph tensors."""
    w, _ = apply_warp(pair["B"], sev, seed=seed, tear=tear)
    return graph_tensors(np.asarray(w.obsm["spatial"], np.float32),
                         pair["Z_B"], knn, pair["pitch"])


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def eval_curve(model, pair, severities, seed, tear, knn, readout, temp):
    """Per-severity registration error + held-out top-1 match accuracy."""
    model.eval()
    rows = []
    for sv in severities:
        gb = warp_graph(pair, sv, seed, tear, knn)
        with torch.no_grad():
            pred, match = model(pair["ga"], gb, pair["a_norm"], return_match=True)
        pred = pred.numpy() * pair["pitch"]
        st = registration_error_stats(pred, pair["gt_A"], mask=pair["have"])
        acc = match_top1(match_logits(match, readout, temp), pair["ct"])
        rows.append({"severity": sv, "reg_err_median": st["median"],
                     "reg_err_mean": st["mean"], "reg_err_p90": st["p90"],
                     "match_top1": acc, "n": st["n"]})
    return rows


def write_curve(path, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


def parse_pairs(spec):
    """'151507/151508,151509/151510' -> [('151507','151508'),(...)]"""
    out = []
    for chunk in spec.split(","):
        ref, smp = chunk.split("/")
        out.append((ref.strip(), smp.strip()))
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pairs", default="151507/151508",
                    help="comma list of ref/sample donor pairs to TRAIN on")
    ap.add_argument("--test-pair", default="151669/151670",
                    help="held-out ref/sample pair to EVALUATE on")
    ap.add_argument("--feature-mode", choices=["global", "perslice"],
                    default="global",
                    help="SVD-embedding standardization (see module docstring)")
    ap.add_argument("--readout", choices=["cosine", "attn"], default="cosine")
    ap.add_argument("--lambda-contrastive", type=float, default=0.0,
                    help="weight of the contrastive loss (0 = coordinate only)")
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--target-sigma", type=float, default=1.0,
                    help="Gaussian target width in spot-pitches")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--attn-dim", type=int, default=64)
    ap.add_argument("--knn", type=int, default=6)
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps-per-epoch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-severity", type=float, default=8.0)
    ap.add_argument("--tear-prob", type=float, default=0.5)
    ap.add_argument("--eval-severities", default="0,1,2,3,4,6,8")
    ap.add_argument("--eval-seed", type=int, default=0,
                    help="fixed seed for ALL eval warps (matches the PASTE2 sweep)")
    ap.add_argument("--eval-mode", choices=["tear", "smooth"], default="tear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sutura")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    data_dir, results_dir = Path(args.data_dir), Path(args.results_dir)
    train_pairs = parse_pairs(args.train_pairs)
    test_ref, test_smp = parse_pairs(args.test_pair)[0]

    # load every slice once; fit the shared basis on TRAINING slices only
    train_slices = []
    for ref, smp in train_pairs:
        train_slices.append(ad.read_h5ad(data_dir / f"DLPFC_{ref}.h5ad"))
        train_slices.append(ad.read_h5ad(data_dir / f"DLPFC_{smp}.h5ad"))
    test_slices = [ad.read_h5ad(data_dir / f"DLPFC_{test_ref}.h5ad"),
                   ad.read_h5ad(data_dir / f"DLPFC_{test_smp}.h5ad")]
    assert_same_genes(train_slices + test_slices)

    project = fit_shared_basis(train_slices, args.pca_dim, args.seed,
                               feature_mode=args.feature_mode)
    pairs = [build_pair(r, s, project, args.knn, data_dir, args.target_sigma)
             for r, s in train_pairs]
    test = build_pair(test_ref, test_smp, project, args.knn, data_dir,
                      args.target_sigma)

    model = SuturaNet(args.pca_dim, args.hidden, args.layers, args.attn_dim)
    n_params = sum(q.numel() for q in model.parameters())

    print("=" * 72)
    print("Sutura training")
    print("=" * 72)
    print(f"  feature mode  : {args.feature_mode} (SVD on {len(train_slices)} "
          f"train slices)")
    print(f"  contrastive   : readout={args.readout} lambda={args.lambda_contrastive} "
          f"temp={args.temp} sigma={args.target_sigma} pitch")
    for pr in pairs:
        nb = int(pr["have"].sum())
        print(f"  train pair    : {pr['ref']}/{pr['smp']}  bridge {nb} spots "
              f"(pitch {pr['pitch']:.1f}px)")
    nb = int(test["have"].sum())
    same = (test_ref, test_smp) in train_pairs
    tag = "(in-sample, warp-seed held out)" if same else "<- HELD OUT, unseen donor"
    print(f"  TEST pair     : {test['ref']}/{test['smp']}  bridge {nb} spots {tag}")
    print(f"  params        : {n_params}")
    print("=" * 72)

    eval_sev = [float(x) for x in args.eval_severities.split(",")]
    eval_tear = (args.eval_mode == "tear")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_contrastive = args.lambda_contrastive > 0.0

    for epoch in range(args.epochs):
        model.train()
        tr, tc = 0.0, 0.0
        for _ in range(args.steps_per_epoch):
            pr = pairs[int(rng.integers(0, len(pairs)))]
            sv = float(rng.uniform(0, args.max_severity))
            gb = warp_graph(pr, sv, seed=int(rng.integers(1, 9999)),
                            tear=bool(rng.random() < args.tear_prob), knn=args.knn)
            opt.zero_grad()
            if use_contrastive:
                pred, match = model(pr["ga"], gb, pr["a_norm"], return_match=True)
                l_c = contrastive_loss(
                    match_logits(match, args.readout, args.temp), pr["ct"])
            else:
                pred = model(pr["ga"], gb, pr["a_norm"])
                l_c = torch.zeros(())
            l_reg = (pred - pr["gt_norm"])[pr["mask"]].norm(dim=1).mean()
            loss = l_reg + args.lambda_contrastive * l_c
            loss.backward()
            opt.step()
            tr += l_reg.item(); tc += float(l_c)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            te = eval_curve(model, test, [eval_sev[0], eval_sev[-1]],
                            args.eval_seed, eval_tear, args.knn,
                            args.readout, args.temp)
            cells = "  ".join(
                f"sev{r['severity']:g}:{r['reg_err_median']:5.0f}px"
                f"/acc{r['match_top1']:.2f}" for r in te)
            print(f"  ep {epoch:3d} | Lreg={tr/args.steps_per_epoch:.3f} "
                  f"Lc={tc/args.steps_per_epoch:.3f} | TEST {cells}")

    results_dir.mkdir(parents=True, exist_ok=True)
    test_rows = eval_curve(model, test, eval_sev, args.eval_seed, eval_tear,
                           args.knn, args.readout, args.temp)
    train_rows = eval_curve(model, pairs[0], eval_sev, args.eval_seed, eval_tear,
                            args.knn, args.readout, args.temp)
    write_curve(results_dir / f"{args.out}_test_curve.csv", test_rows)
    write_curve(results_dir / f"{args.out}_train_curve.csv", train_rows)
    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "pitch": {f"{pr['ref']}/{pr['smp']}": pr["pitch"] for pr in pairs}
                | {f"{test['ref']}/{test['smp']}": test["pitch"]}},
               results_dir / f"{args.out}.pt")

    print("\nTEST curve (px median, tear regime):")
    for r in test_rows:
        print(f"  sev{r['severity']:g}: median {r['reg_err_median']:6.1f}  "
              f"match_top1 {r['match_top1']:.3f}  (n={r['n']})")
    print(f"wrote {args.out}_test_curve.csv, {args.out}_train_curve.csv, "
          f"{args.out}.pt")


if __name__ == "__main__":
    main()
