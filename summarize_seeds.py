"""
Summarise the seed-variance check: mean, standard deviation and range of
relative L2 error across random seeds, for each of the three
configurations that carry the project's central comparison.

A single-seed result cannot distinguish "this architecture is better"
from "this run got a lucky initialisation". Reporting spread across
seeds is what makes the comparison defensible -- and if the spreads
overlap, that is itself the finding, and an honest one.

Seed 0 comes from the main sweep; seeds 1-4 from seed_check.sh.
"""

import glob
import json
import os
import statistics as st

# config label -> (seed-0 run dir from the sweep, glob for the extra seeds)
import sys

# --conv selects the convergence-trained runs; default is the original
# fixed-8000-epoch study, kept so the two can be compared directly.
if "--conv" in sys.argv:
    GROUPS = {
        "QAPINN 4q L=2 strong": (None, "runs/conv_strong_L2_s*"),
        "QAPINN 4q L=2 ring":   (None, "runs/conv_ring_L2_s*"),
        "Classical PINN":       (None, "runs/conv_classical_s*"),
    }
else:
    GROUPS = {
        "QAPINN 4q L=2 strong": ("runs/ansatz_strong_L2_q4", "runs/seed_strong_L2_s*"),
        "QAPINN 4q L=2 ring":   ("runs/reupload_L2_q4",      "runs/seed_ring_L2_s*"),
        "Classical PINN":       ("runs/classical_baseline",  "runs/seed_classical_s*"),
    }


def load(run_dir):
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    print(f"{'configuration':24s} {'n':>2}  {'mean':>7} {'std':>7} {'min':>7} {'max':>7}   "
          f"{'mean shock':>10} {'mean smooth':>11}")
    print("-" * 92)

    table = {}
    for label, (seed0, pattern) in GROUPS.items():
        runs = ([seed0] if seed0 and os.path.exists(seed0) else []) + sorted(glob.glob(pattern))
        ms = [m for m in (load(r) for r in runs) if m]
        if not ms:
            print(f"{label:24s}  0   (no runs found)")
            continue

        errs = [m["rel_l2"] for m in ms]
        shock = [m["rel_l2_shock_region"] for m in ms]
        smooth = [m["rel_l2_smooth_region"] for m in ms]
        sd = st.stdev(errs) if len(errs) > 1 else 0.0
        table[label] = (st.mean(errs), sd, len(errs))

        print(f"{label:24s} {len(errs):>2}  {st.mean(errs):>6.2%} {sd:>6.2%} "
              f"{min(errs):>6.2%} {max(errs):>6.2%}   {st.mean(shock):>9.2%} {st.mean(smooth):>10.2%}")

    # the two comparisons the project actually rests on
    print()
    def verdict(a, b, claim):
        if a not in table or b not in table:
            return
        (ma, sa, na), (mb, sb, nb) = table[a], table[b]
        gap = mb - ma
        spread = sa + sb
        if na < 2 or nb < 2:
            note = "need >=2 seeds each to judge"
        elif gap > spread:
            note = "SEPARATED -- gap exceeds combined spread"
        elif gap > 0:
            note = "overlapping -- gap within combined spread, report as a trend not a result"
        else:
            note = "NOT supported by these runs"
        print(f"{claim}\n    {a} {ma:.2%} vs {b} {mb:.2%}  ->  gap {gap:+.2%}, combined spread {spread:.2%}\n    {note}\n")

    verdict("QAPINN 4q L=2 strong", "QAPINN 4q L=2 ring",
            "Ansatz design rule (strong beats ring at matched depth):")
    verdict("QAPINN 4q L=2 strong", "Classical PINN",
            "QAPINN beats the classical baseline at equal budget:")


if __name__ == "__main__":
    main()
