"""Compare Harmony donor-batch correction vs the perslice baseline, leave-one-donor-out.
Reads results/sutura_lodo_{perslice,harmony}_test{S1,S2,S3}_test_curve.csv and prints
held-out median (sev0 -> sev8) per fold, against the unsupervised PASTE2 bar."""
import csv
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
FOLDS = {"S1": "151507/151508", "S2": "151669/151670", "S3": "151673/151674"}
# PASTE2 held-out barycentric median (sev0 -> sev8), from the multi-seed LODO work
PASTE2 = {"S1": (658, 838), "S2": (528, 697), "S3": (397, 539)}


def curve(mode, fold):
    f = R / f"sutura_lodo_{mode}_test{fold}_test_curve.csv"
    if not f.exists():
        return None
    rows = {float(r["severity"]): r for r in csv.DictReader(open(f))}
    return float(rows[0.0]["reg_err_median"]), float(rows[8.0]["reg_err_median"])


def main():
    print("=" * 84)
    print("HARMONY (donor-batch correction) vs PERSLICE baseline — held-out LODO, tear")
    print("held-out registration error median px (sev0 -> sev8); lower is better")
    print("=" * 84)
    print(f"{'fold':<5}{'donor (held out)':<20}{'perslice':>16}{'harmony':>16}"
          f"{'PASTE2 (unsup)':>18}")
    print("-" * 84)
    closed = []
    for fold, pair in FOLDS.items():
        ps, hm, pa = curve("perslice", fold), curve("harmony", fold), PASTE2[fold]
        ps_s = f"{ps[0]:.0f}->{ps[1]:.0f}" if ps else "   n/a"
        hm_s = f"{hm[0]:.0f}->{hm[1]:.0f}" if hm else "   n/a"
        pa_s = f"{pa[0]}->{pa[1]}"
        print(f"{fold:<5}{pair:<20}{ps_s:>16}{hm_s:>16}{pa_s:>18}")
        if ps and hm:
            d0 = ps[0] - hm[0]
            beats_paste = hm[0] < pa[0]
            closed.append((fold, d0, beats_paste, ps[0], hm[0], pa[0]))
    print("-" * 84)
    print("\nVERDICT (sev0 held-out median):")
    for fold, d0, beats, p, h, pa in closed:
        verb = "narrows" if d0 > 0 else "WORSENS"
        beat = "  *** BEATS PASTE2 ***" if beats else f"(still {h/pa:.1f}x PASTE2)"
        print(f"  {fold}: harmony {verb} the gap by {d0:+.0f} px vs perslice "
              f"({p:.0f} -> {h:.0f}); {beat}")
    if closed:
        anybeat = any(b for *_, b, _, _, _ in [(f, d, b, p, h, pa) for f, d, b, p, h, pa in closed])
        print("\nBottom line:",
              "Harmony beats unsupervised PASTE2 on >=1 held-out donor — generalization "
              "partially solved." if anybeat else
              "Harmony narrows but does NOT close the gap to PASTE2 — cross-donor transfer "
              "remains open (consistent with the honest negative result).")


if __name__ == "__main__":
    main()
