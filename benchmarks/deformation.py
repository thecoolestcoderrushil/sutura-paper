"""
Sutura — synthetic non-rigid warp (smooth + tear) of a spatial slice.

Applies a KNOWN smooth displacement field (sum of Gaussian radial bumps) to a
slice's spot coordinates so we retain exact ground-truth correspondence between
each warped spot and its original location. An optional tear excises a contiguous
region and rigidly displaces it, simulating physical tissue tearing — the
discontinuity that optimal-transport and diffeomorphic registration cannot
represent.

Severity is dimensionless: the field is generated with a fixed shape (by seed),
normalized to unit peak magnitude, then scaled to
    max_displacement = severity * (median spot pitch).
So severity is measured in spot-pitches, and severity=0 reproduces the original
slice exactly. Holding seed fixed while varying severity scales magnitude only
(the field shape is identical), which is what the severity sweep relies on.

The public entry point `apply_warp` returns (warped AnnData, ground-truth dict);
.X is left untouched. Run as a script to write a warped slice + GT to disk.

Usage:
    python -m benchmarks.deformation --sample 151508 --severity 4 --tear --seed 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree

# release-root-relative default data dir (this file is at <root>/benchmarks/)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WARP_DIR = DATA_DIR / "warped"


def median_spot_pitch(coords: np.ndarray) -> float:
    """Median nearest-neighbor distance — the spot-to-spot pitch."""
    d, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(d[:, 1]))


def _gaussian_bump_field(coords, n_bumps, width, rng) -> np.ndarray:
    """Smooth displacement field from summed Gaussian bumps, unit peak norm."""
    lo, hi = coords.min(0), coords.max(0)
    disp = np.zeros_like(coords, dtype=float)
    for _ in range(n_bumps):
        center = lo + rng.random(2) * (hi - lo)
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction) + 1e-12
        amp = rng.uniform(0.5, 1.0)
        r2 = ((coords - center) ** 2).sum(1)
        disp += (amp * np.exp(-r2 / (2.0 * width ** 2)))[:, None] * direction
    peak = np.linalg.norm(disp, axis=1).max()
    if peak > 0:
        disp /= peak                               # unit peak magnitude
    return disp


def _tear_field(coords, rng, pitch, tear_offset_pitch):
    """Excise a contiguous region (one side of a random cut) and translate it."""
    center = coords.mean(0)
    theta = rng.uniform(0, 2 * np.pi)
    axis = np.array([np.cos(theta), np.sin(theta)])
    proj = (coords - center) @ axis
    thr = np.quantile(proj, rng.uniform(0.55, 0.70))   # ~30-45% of tissue
    region = proj > thr
    perp = np.array([-axis[1], axis[0]])
    tdir = perp * (1.0 if rng.random() < 0.5 else -1.0)
    disp = np.zeros_like(coords, dtype=float)
    disp[region] = tdir * pitch * tear_offset_pitch
    return disp, region


def apply_warp(adata: ad.AnnData, severity: float, *, seed: int = 0,
               tear: bool = False, n_bumps: int = 8,
               bump_width_frac: float = 0.18,
               tear_offset_pitch: float = 3.0):
    """Return (warped AnnData, ground-truth dict). .X is left untouched."""
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    pitch = median_spot_pitch(coords)
    extent = float(np.linalg.norm(coords.max(0) - coords.min(0)))
    width = bump_width_frac * extent
    rng = np.random.default_rng(seed)

    unit_field = _gaussian_bump_field(coords, n_bumps, width, rng)
    disp = severity * pitch * unit_field

    region = np.zeros(adata.n_obs, dtype=bool)
    if tear:
        tdisp, region = _tear_field(
            coords, rng, pitch, tear_offset_pitch * max(severity, 1.0))
        disp = disp + tdisp

    warped = coords + disp
    mag = np.linalg.norm(disp, axis=1)

    w = adata.copy()
    w.obsm["spatial_original"] = coords
    w.obsm["spatial"] = warped
    w.obsm["warp_displacement"] = disp
    w.obs["torn"] = region
    w.uns["warp"] = dict(
        severity=float(severity), seed=int(seed), tear=bool(tear),
        pitch=pitch, extent=extent, n_bumps=int(n_bumps),
        bump_width=float(width),
        max_disp_px=float(mag.max()), mean_disp_px=float(mag.mean()),
        max_disp_pitch=float(mag.max() / pitch),
    )

    gt = dict(
        obs_names=adata.obs_names.to_numpy(),
        warped_xy=warped, original_xy=coords, displacement=disp,
        torn=region,
        array_row=np.asarray(adata.obs["array_row"]),
        array_col=np.asarray(adata.obs["array_col"]),
        severity=float(severity), seed=int(seed), tear=bool(tear),
        pitch=pitch,
    )
    return w, gt


def sev_tag(severity: float) -> str:
    return f"{severity:g}".replace(".", "p")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default="151508")
    p.add_argument("--severity", type=float, default=3.0,
                   help="max displacement in spot-pitches (0 = identity)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tear", action="store_true",
                   help="also excise a contiguous region and displace it")
    p.add_argument("--n-bumps", type=int, default=8)
    p.add_argument("--bump-width-frac", type=float, default=0.18)
    p.add_argument("--tear-offset-pitch", type=float, default=3.0)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    src = data_dir / f"DLPFC_{args.sample}.h5ad"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — run python -m data.load_spatiallibd first.")
    adata = ad.read_h5ad(src)

    w, gt = apply_warp(
        adata, args.severity, seed=args.seed, tear=args.tear,
        n_bumps=args.n_bumps, bump_width_frac=args.bump_width_frac,
        tear_offset_pitch=args.tear_offset_pitch)

    warp_dir = data_dir / "warped"
    warp_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.sample}_s{sev_tag(args.severity)}"
    if args.tear:
        tag += "_tear"
    tag += f"_seed{args.seed}"

    h5_path = warp_dir / f"DLPFC_{tag}.h5ad"
    gt_path = warp_dir / f"warpgt_{tag}.npz"
    w.write_h5ad(h5_path)
    np.savez_compressed(gt_path, **gt)

    meta = w.uns["warp"]
    print(f"warped {args.sample}: severity={args.severity} "
          f"(max {meta['max_disp_px']:.1f} px = "
          f"{meta['max_disp_pitch']:.2f} pitches, "
          f"mean {meta['mean_disp_px']:.1f} px), "
          f"tear={args.tear} torn_spots={int(gt['torn'].sum())}")
    print(f"  wrote slice -> {h5_path}")
    print(f"  wrote gt    -> {gt_path}")


if __name__ == "__main__":
    main()
