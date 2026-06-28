"""
Shared scoring utilities for Sutura alignment evaluation (numpy-only).

Registration error: project each moving-slice (B) spot into the reference (A)
frame and measure Euclidean distance to its ground-truth target (in pixels).

Label-transfer accuracy: map each spot in A to its highest-probability partner in
B (argmax over the transport-matrix row) and measure how often the manual
cortical-layer labels agree.

NA handling: a spot pair contributes to accuracy ONLY if both the A spot and its
matched B spot carry a real layer label. NA-labeled spots are masked out on BOTH
sides before computing accuracy.
"""

from __future__ import annotations

import numpy as np

NA = "NA"


def argmax_partners(pi: np.ndarray) -> np.ndarray:
    """Best partner index in B for each spot in A (argmax over each row)."""
    return np.asarray(pi).argmax(axis=1)


def label_transfer_accuracy(pi, layer_a, layer_b) -> dict:
    """Accuracy of argmax A->B label transfer, masking NA on both slices."""
    pi = np.asarray(pi)
    layer_a = np.asarray(layer_a).astype(str)
    layer_b = np.asarray(layer_b).astype(str)

    j_star = argmax_partners(pi)
    predicted = layer_b[j_star]

    # mask NA on BOTH sides: A spot labeled AND its matched B partner labeled
    valid = (layer_a != NA) & (predicted != NA)
    correct = (predicted == layer_a) & valid

    n_valid = int(valid.sum())
    return {
        "accuracy": float(correct.sum() / max(n_valid, 1)),
        "n_correct": int(correct.sum()),
        "n_scored": n_valid,
        "n_dropped_a_na": int((layer_a == NA).sum()),
        "n_dropped_partner_na": int(((layer_a != NA) & (predicted == NA)).sum()),
    }


def random_mapping_floor(layer_a, layer_b, n_trials: int = 50, seed: int = 0) -> dict:
    """Floor: assign each A spot a uniformly random B partner, masking NA on
    both sides. Averaged over n_trials shuffles for a stable estimate."""
    layer_a = np.asarray(layer_a).astype(str)
    layer_b = np.asarray(layer_b).astype(str)
    rng = np.random.default_rng(seed)
    n_a, n_b = layer_a.shape[0], layer_b.shape[0]

    accs = []
    for _ in range(n_trials):
        j = rng.integers(0, n_b, size=n_a)        # random A->B assignment
        predicted = layer_b[j]
        valid = (layer_a != NA) & (predicted != NA)
        accs.append(((predicted == layer_a) & valid).sum() / max(valid.sum(), 1))
    accs = np.asarray(accs)
    return {
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),
        "n_trials": n_trials,
    }


def barycentric_projection(pi, source_coords):
    """Soft-map each B spot into A's coordinate frame via the transport plan.

    pi has shape (n_a, n_b); source_coords are A's coordinates (n_a, 2).
    Returns (pred_coords (n_b, 2), col_mass (n_b,)). Columns with ~zero mass
    yield NaN predictions (the B spot received no transported mass).
    """
    pi = np.asarray(pi, dtype=float)
    source_coords = np.asarray(source_coords, dtype=float)
    col_mass = pi.sum(axis=0)                       # mass arriving at each B spot
    safe = col_mass > 0
    pred = np.full((pi.shape[1], source_coords.shape[1]), np.nan)
    pred[safe] = (pi[:, safe].T @ source_coords) / col_mass[safe, None]
    return pred, col_mass


def argmax_projection(pi, source_coords):
    """Hard-map each B spot into A's frame: pred[j] = A_coord[argmax_i pi[:,j]].

    Unlike the soft barycentric projection, this takes the single best A partner
    per B spot, so it does not smear toward the tissue centroid when the plan is
    diffuse. Columns with ~zero transported mass yield NaN.
    """
    pi = np.asarray(pi, dtype=float)
    source_coords = np.asarray(source_coords, dtype=float)
    col_mass = pi.sum(axis=0)
    safe = col_mass > 0
    best = pi.argmax(axis=0)                         # best A spot per B column
    pred = np.full((pi.shape[1], source_coords.shape[1]), np.nan)
    pred[safe] = source_coords[best[safe]]
    return pred, col_mass


def registration_error_stats(pred_coords, gt_coords, mask=None) -> dict:
    """Euclidean error between predicted and ground-truth coordinates.

    Reports error in the native units of the coordinates (pixels here). `mask`
    selects which spots are scorable (e.g. those with a GT target and nonzero
    transported mass).
    """
    pred = np.asarray(pred_coords, dtype=float)
    gt = np.asarray(gt_coords, dtype=float)
    err = np.linalg.norm(pred - gt, axis=1)
    valid = np.isfinite(err)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    e = err[valid]
    if e.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "p90": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(e.mean()),
        "median": float(np.median(e)),
        "p90": float(np.percentile(e, 90)),
        "max": float(e.max()),
        "n": int(e.size),
    }


def report(pi, layer_a, layer_b, *, random_trials: int = 50) -> dict:
    """Compute and return both the model accuracy and the random floor."""
    model = label_transfer_accuracy(pi, layer_a, layer_b)
    floor = random_mapping_floor(layer_a, layer_b, n_trials=random_trials)
    return {"model": model, "random_floor": floor}


def print_report(rep: dict, header: str = "layer-label transfer") -> None:
    m, f = rep["model"], rep["random_floor"]
    print("=" * 64)
    print(f"{header}")
    print("=" * 64)
    print(f"  model accuracy  : {m['accuracy']*100:6.2f}%  "
          f"({m['n_correct']}/{m['n_scored']} scored)")
    print(f"  random floor    : {f['accuracy_mean']*100:6.2f}% "
          f"+/- {f['accuracy_std']*100:.2f}%  "
          f"({f['n_trials']} shuffles)")
    print(f"  lift over random: "
          f"{(m['accuracy'] - f['accuracy_mean'])*100:+6.2f} pts")
    print(f"  masked out      : {m['n_dropped_a_na']} NA in A, "
          f"{m['n_dropped_partner_na']} spots whose B partner is NA")
    print("=" * 64)
