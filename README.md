# Sutura Genomics: Spatial Transcriptomics Tear Benchmark

Code accompanying the preprint **"Tissue tearing degrades optimal-transport and
diffeomorphic registration of spatial transcriptomics, and a graph
cross-attention model that fits torn tissue."**

Physical tissue **tears** introduce displacement *discontinuities* that
optimal-transport (PASTE2), diffeomorphic (STalign / LDDMM), and Gaussian-process
warp (GPSA) aligners cannot represent — a diffeomorphism is, by construction,
continuous and invertible. This repository provides a controlled tear benchmark on
the spatialLIBD DLPFC Visium dataset (known ground-truth correspondence via the
Visium array bridge), wrappers to score the three baselines on it, and **Sutura**,
a graph cross-attention registration model that learns the tear discontinuity.

## Results summary

Sutura fits torn tissue **in-sample** (median registration error ~99 -> 106 px
across tear severities 0 -> 8, well under the 137 px spot pitch), while PASTE2,
STalign, and GPSA degrade to many hundreds of pixels. **Cross-donor
generalisation remains open**: a contrastive correspondence loss roughly halves
the held-out gap on 2 of 3 donors but does not beat PASTE2 on unseen tissue. We
report this honestly — the in-sample result is the contribution; cross-donor
transfer is future work.

## Repository layout

```
model/
  encoder.py        shared graph encoder (residual DeformConv stack) + kNN builder
  attention.py      cross-slice attention B -> A (scaled dot-product, cosine read-out)
  sutura.py         SuturaNet: encoder + cross-attention + barycentric coarse + residual MLP head
  contrastive.py    InfoNCE soft-correspondence loss + cosine/attn read-out variants
  __init__.py       SuturaNet, load_checkpoint, ...
benchmarks/
  deformation.py    synthetic smooth + tear warp generator (apply_warp) and ground truth
  paste2_baseline.py  PASTE2 partial-FGW wrapper          (optional dep: paste2)
  stalign_baseline.py STalign diffeomorphic LDDMM wrapper (optional dep: stalign)
  gpsa_baseline.py    GPSA GP-warp wrapper                (optional dep: gpsa, --no-deps)
data/
  load_spatiallibd.py  DLPFC downloader/standardizer (Figshare article 22004273)
metrics.py          registration error, barycentric/argmax projection, label transfer
train.py            unified training: in-sample / leave-one-donor-out / contrastive
eval.py             evaluation CLI dispatching to any method on the tear severity grid
figures/            paper figures (PNG) + make_figures.py (reads results/, writes figures/)
results/            published result CSVs (all methods, all folds)
checkpoints/        sutura_insample.pt, sutura_contrastive_S1.pt
reproduce_figures.sh, reproduce_figure{1,2,3,4}.sh
requirements.txt, LICENSE, .gitignore
```

## Installation

```bash
git clone https://github.com/thecoolestcoderrushil/sutura-paper
cd sutura-paper
pip install -r requirements.txt
```

The core stack (torch, torch-geometric, anndata, scanpy, numpy, scipy,
scikit-learn, matplotlib, pandas) is enough to train/evaluate **Sutura** and to
regenerate every figure from the shipped CSVs. The competing baselines (PASTE2,
STalign, GPSA) are *optional* — install them only if you want to re-run those
methods (see the commented section of `requirements.txt`). GPSA must be installed
with `pip install gpsa --no-deps` so it uses this repo's torch>=2.2 stack.

## Data download

The benchmark uses the spatialLIBD DLPFC Visium slices (Maynard et al., Nat
Neurosci 2021), distributed as preprocessed `.h5ad` on **Figshare article
22004273** (CC BY 4.0). Download and standardize them into `./data`:

```bash
python -m data.load_spatiallibd                 # default pair 151507 / 151508
python -m data.load_spatiallibd --all           # all 12 samples (3 donors)
```

This writes `data/DLPFC_<sample>.h5ad` with the manual cortical-layer annotation
normalized to `obs["layer"]`.

## Reproducing the paper

**Figures** (from the shipped result CSVs — no GPU, no data download required):

```bash
bash reproduce_figures.sh          # all four figures
bash reproduce_figure1.sh          # registration error vs tear severity
bash reproduce_figure2.sh          # leave-one-donor-out generalization
bash reproduce_figure3.sh          # tear vs smooth magnitude control
bash reproduce_figure4.sh          # architecture schematic
```

**Evaluation** (requires the data download above; baselines require their optional
dependency):

```bash
# Sutura, in-sample tear sweep
python eval.py --method sutura --severity-grid 0 1 2 3 4 6 8 --in-sample

# Sutura, contrastive checkpoint
python eval.py --method sutura --severity-grid 0 1 2 3 4 6 8 --contrastive

# Baselines on the same grid
python eval.py --method paste2  --severity-grid 0 1 2 3 4 6 8
python eval.py --method stalign --severity-grid 0 4 8
python eval.py --method gpsa    --severity-grid 0 1 2 3 4 6 8
```

**Training** (to regenerate checkpoints / curves):

```bash
# in-sample (warp seeds held out)
python train.py --train-pairs 151507/151508 --test-pair 151507/151508 \
    --out sutura_insample

# leave-one-donor-out with batch-robust per-slice features
python train.py --train-pairs 151507/151508 --test-pair 151669/151670 \
    --feature-mode perslice --out sutura_lodo_S2

# contrastive correspondence loss (cosine read-out)
python train.py --train-pairs 151507/151508,151669/151670 \
    --test-pair 151673/151674 --readout cosine --lambda-contrastive 0.5 \
    --out sutura_contrastive_S3
```

Each run writes `results/<out>_test_curve.csv`, `results/<out>_train_curve.csv`,
and `results/<out>.pt`.

## Citation

```bibtex
@article{maniar2026sutura,
  title   = {Tissue tearing degrades optimal-transport and diffeomorphic
             registration of spatial transcriptomics, and a graph cross-attention
             model that fits torn tissue},
  author  = {Maniar, Rushil and Lee, Sean},
  journal = {bioRxiv},
  year    = {2026}
}
```

## License

Released under the MIT License. Copyright (c) 2026 Sutura Genomics, Inc. See
[LICENSE](LICENSE).
