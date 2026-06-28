"""
Generate the paper figures from the result CSVs shipped in ../results (300 DPI).

All paths are relative to the release root: reads ../results, writes PNGs next to
this script (../figures). No absolute or machine-specific paths.

Figures:
  fig1_benchmark.png        registration error vs tear severity (all methods)
  fig2_lodo.png             leave-one-donor-out generalization (3 folds)
  fig3_magnitude_control.png  tear vs smooth control
  fig_architecture.png      Sutura architecture schematic

Run via ../reproduce_figures.sh, or directly:  python figures/make_figures.py
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
FIGDIR = HERE
FIGDIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colour-blind-safe palette
C = {"sutura": "#0072B2", "paste2": "#009E73", "paste2_smooth": "#D55E00",
     "stalign": "#56B4E9", "gpsa": "#E69F00"}
PITCH = 137.0


def read(path, ycol, xcol="severity"):
    xs, ys = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            xs.append(float(r[xcol])); ys.append(float(r[ycol]))
    return np.array(xs), np.array(ys)


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="0.85", lw=0.8)
    ax.set_axisbelow(True)


def pitch_line(ax):
    ax.axhline(PITCH, ls="--", lw=1.0, color="0.45", zorder=1)
    ax.text(0.02, PITCH + 25, "1 spot pitch (137 px)", color="0.4",
            fontsize=8, transform=ax.get_yaxis_transform())


# ---------------------------------------------------------------- Figure 1 / 3
def fig1(path, add_smooth=False):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    style(ax)
    # Sutura in-sample (5-seed mean +/- CI)
    sx, sy = read(RES / "sutura_multiseed_tear.csv", "median_mean")
    _, sci = read(RES / "sutura_multiseed_tear.csv", "median_ci")
    ax.errorbar(sx, sy, yerr=sci, color=C["sutura"], marker="o", ms=4, lw=2,
                capsize=2, label="Sutura (in-sample, 5-seed)", zorder=5)
    # PASTE2 (OT) tear
    px, py = read(RES / "sweep_deformation_cross_tear.csv", "reg_err_median")
    ax.plot(px, py, color=C["paste2"], marker="s", ms=4, lw=2, label="PASTE2 (OT)")
    # STalign (LDDMM)
    tx, ty = read(RES / "stalign_tear.csv", "reg_err_median")
    ax.plot(tx, ty, color=C["stalign"], marker="^", ms=5, lw=2,
            label="STalign (LDDMM)")
    # GPSA (GP warp)
    gx, gy = read(RES / "gpsa_tear.csv", "reg_err_median")
    ax.plot(gx, gy, color=C["gpsa"], marker="D", ms=4, lw=2, label="GPSA (GP warp)")
    if add_smooth:
        mx, my = read(RES / "sweep_deformation_cross.csv", "reg_err_median")
        ax.plot(mx, my, color=C["paste2_smooth"], marker="s", ms=4, lw=2, ls=":",
                label="PASTE2 (smooth control)")
    pitch_line(ax)
    ax.set_xlabel("Tear Severity (spot pitches)")
    ax.set_ylabel("Median Registration Error (px)")
    ax.set_title("Registration error vs tear severity" if not add_smooth
                 else "Tear vs smooth (magnitude control)", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- Figure 2
def fig2(path):
    # (fold, PASTE2-held-out CSV, best contrastive checkpoint stem, optional CI)
    folds = [("S1", "sweep_deformation_cross_tear.csv",
              "arca_ctr_attn_l2p0_testS1", None),
             ("S2", "sweep_deformation_cross_tear_loo.csv",
              "arca_ctr_attn_l2p0_testS2",
              {0: (528, 7), 4: (580, 12), 8: (697, 18)}),
             ("S3", "sweep_deformation_cross_tear_subj3.csv",
              "arca_ctr_attn_l0p5_testS3",
              {0: (397, 7), 4: (465, 8), 8: (539, 19)})]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, (fold, p2file, best, p2ci) in zip(axes, folds):
        style(ax)
        bx, by = read(RES / f"arca_ctr_cosine_l0_test{fold}_test_curve.csv",
                      "reg_err_median")
        ax.plot(bx, by, color=C["sutura"], marker="o", ms=4, lw=2, ls="--",
                label="Sutura, no contrastive")
        ex, ey = read(RES / f"{best}_test_curve.csv", "reg_err_median")
        ax.plot(ex, ey, color=C["sutura"], marker="o", ms=4, lw=2,
                label="Sutura, best contrastive")
        px, py = read(RES / p2file, "reg_err_median")
        ax.plot(px, py, color=C["paste2"], marker="s", ms=4, lw=2,
                label="PASTE2 (held-out)")
        if p2ci:
            xs = sorted(p2ci); ms = [p2ci[s][0] for s in xs]
            es = [p2ci[s][1] for s in xs]
            ax.errorbar(xs, ms, yerr=es, color=C["paste2"], fmt="none",
                        capsize=3, lw=1.5)
        pitch_line(ax)
        ax.set_title(f"Held-out donor {fold}", fontsize=11)
        ax.set_xlabel("Tear Severity (spot pitches)")
        ax.set_ylim(bottom=0)
        if fold == "S1":
            ax.set_ylabel("Median Registration Error (px)")
            ax.legend(frameon=False, fontsize=8.5, loc="center right")
    fig.suptitle("Leave-one-donor-out generalization (contrastive single-seed; "
                 "PASTE2 5-seed CI on S2/S3)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- Architecture
def fig_architecture(path):
    CIN = "#ededed"; CENC = "#cfe2f3"; CATT = "#ffe0a3"
    CHEAD = "#fff2cc"; COUT = "#d9ead3"; INK = "#222222"

    fig, ax = plt.subplots(figsize=(14.2, 8.0))
    ax.set_xlim(0, 18); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    def box(cx, cy, w, h, text, fc, fs=10, weight="normal"):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     linewidth=1.1, edgecolor="black", facecolor=fc, zorder=3))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                color=INK, weight=weight, zorder=4)

    def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None, fs=9, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=14, lw=1.3,
                     color="#444444", zorder=2,
                     connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text(lx if lx is not None else (x1 + x2) / 2,
                    ly if ly is not None else (y1 + y2) / 2 + 0.22,
                    label, ha="center", va="center", fontsize=fs,
                    color="#333333", style="italic", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))

    def point_panel(x0, y0, w, h, title, warp_tear):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.1",
                     linewidth=1.1, edgecolor="black", facecolor=CIN, zorder=3))
        ax.text(x0 + w / 2, y0 + h + 0.22, title, ha="center", va="bottom",
                fontsize=10.5, weight="bold", color=INK)
        rng = np.random.default_rng(3)
        n = 150
        th = rng.uniform(0, 2 * np.pi, n); r = np.sqrt(rng.uniform(0, 1, n))
        px = r * np.cos(th); py = r * np.sin(th) * 0.85
        colors = np.array(["#34495e"] * n)
        if warp_tear:
            px += 0.10 * np.sin(2.2 * py)
            seam = px > 0.15
            px[seam] += 0.42; py[seam] += 0.18
            colors[seam] = "#c0392b"
        pad = 0.14
        gx = x0 + pad * w + (px - px.min()) / (px.max() - px.min()) * w * (1 - 2 * pad)
        gy = y0 + pad * h + (py - py.min()) / (py.max() - py.min()) * h * (1 - 2 * pad)
        ax.scatter(gx, gy, s=9, c=colors, zorder=4, linewidths=0)

    point_panel(0.3, 4.85, 2.3, 2.7, "Reference slice A", warp_tear=False)
    point_panel(0.3, 0.95, 2.3, 2.7, "Moving slice B (warped)", warp_tear=True)

    box(4.0, 6.2, 2.0, 1.15, "kNN graph\n(k = 6)", CIN, fs=10)
    box(4.0, 2.3, 2.0, 1.15, "kNN graph\n(k = 6)", CIN, fs=10)
    arrow(2.6, 6.2, 3.0, 6.2)
    arrow(2.6, 2.3, 3.0, 2.3)

    box(6.7, 4.25, 2.5, 4.6,
        "Shared Graph\nEncoder\n\nLinear proj (64)\n$\\downarrow$\n3$\\times$ DeformConv\n"
        "(residual)\n\n$\\langle$ shared weights $\\rangle$", CENC, fs=10)
    arrow(5.0, 6.2, 5.45, 5.4)
    arrow(5.0, 2.3, 5.45, 3.1)
    ax.annotate("", xy=(5.3, 5.4), xytext=(5.3, 3.1),
                arrowprops=dict(arrowstyle="<->", color="#1f6fb2", lw=1.2,
                                connectionstyle="arc3,rad=-0.35"), zorder=6)
    ax.text(4.75, 4.25, "same\nweights", ha="center", va="center", fontsize=8,
            color="#1f6fb2", style="italic", weight="bold")

    box(10.0, 4.25, 2.6, 2.0,
        "Cross-Attention (B $\\rightarrow$ A)\n\nscaled dot-product\nsoftmax over A",
        CATT, fs=10)
    arrow(7.95, 5.2, 8.7, 4.7, "$z_A$", lx=8.35, ly=5.25)
    arrow(7.95, 3.3, 8.7, 3.8, "$z_B$", lx=8.35, ly=3.35)

    box(13.1, 5.55, 2.5, 1.5, "Barycentric\n\nattn @ $A_{coords}$", CHEAD, fs=10)
    arrow(11.3, 4.6, 12.1, 5.3, "attention\nweights", lx=11.75, ly=5.05)

    box(13.1, 2.85, 2.6, 1.7,
        "Residual MLP\n\n($z_B$, weighted $z_A$,\ncoarse coord)", CHEAD, fs=9.5)
    arrow(11.3, 3.9, 12.0, 3.2)
    arrow(13.1, 4.8, 13.1, 3.75)

    box(16.4, 2.85, 2.7, 1.9,
        "Predicted\nA-frame coordinates\nper B spot", COUT, fs=10, weight="bold")
    arrow(14.45, 2.85, 15.05, 2.85)

    ax.set_title("Sutura: graph cross-attention registration model",
                 fontsize=13, weight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    fig1(FIGDIR / "fig1_benchmark.png")
    fig2(FIGDIR / "fig2_lodo.png")
    fig1(FIGDIR / "fig3_magnitude_control.png", add_smooth=True)
    fig_architecture(FIGDIR / "fig_architecture.png")
    print("ALL FIGURES DONE ->", FIGDIR)
