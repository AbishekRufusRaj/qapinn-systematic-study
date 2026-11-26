"""
Summarise one PDE's sweep, and put both PDEs side by side.

THE CROSS-PDE TABLE IS THE POINT
--------------------------------
A per-PDE table says which model won on that equation. The cross-PDE view
says whether that answer is a fact about the ARCHITECTURE or a fact about
the EQUATION -- which is the whole reason for running a second one.

Three outcomes, all reportable:

  same ordering on both      the design conclusion generalises; say so, and
                             note it was checked rather than assumed
  ordering flips             the conclusion is equation-specific. This is the
                             most valuable outcome and the one no
                             single-PDE study can reach
  ordering holds, margins    the effect is real but its size depends on the
  change a lot               spectral content of the problem

The `hard_easy_ratio` column is deliberately common to both PDEs even
though it means different things (shock/smooth for Burgers', early/late for
heat). Both measure the same underlying quantity: how much worse the model
is where the high-frequency content lives.

Comparability caveat, stated because it would otherwise be a hidden
confound: relative-L2 values are NOT directly comparable across the two
equations -- they are different problems with different difficulty. What
compares is the ORDERING of models within each PDE, and the ratio columns.
"""

import argparse
import glob
import json
import os
import statistics as st

FAIL_LOSS = 1e-2

GROUPS = {
    "burgers": {
        "Classical PINN":  "runs/conv_classical_s*",
        "QAPINN ring L=2": "runs/conv_ring_L2_s*",
        "QAPINN strong+init": "runs/init_strong_L2_s*",
    },
    "heat": {
        "Classical PINN":  "runs/heat_classical_s*",
        "QAPINN ring L=2": "runs/heat_ring_L2_s*",
        "QAPINN strong+init": "runs/heat_strong_L2_s*",
    },
}


def load(d):
    mp, tp = os.path.join(d, "metrics.json"), os.path.join(d, "meta.json")
    if not (os.path.exists(mp) and os.path.exists(tp)):
        return None
    m, t = json.load(open(mp)), json.load(open(tp))
    return {"err": m["rel_l2"] * 100,
            "ratio": m.get("hard_easy_ratio", m.get("spectral_bias_ratio", float("nan"))),
            "loss": t["final_total_loss"],
            "mins": (t.get("total_seconds") or 0) / 60,
            "epochs": t.get("adam_epochs_used"),
            "failed": t["final_total_loss"] > FAIL_LOSS}


def summarize(pattern):
    rows = [r for r in (load(d) for d in sorted(glob.glob(pattern))) if r]
    if not rows:
        return None
    errs = [r["err"] for r in rows]
    ok = [r["err"] for r in rows if not r["failed"]]
    return {"n": len(rows), "nfail": sum(r["failed"] for r in rows),
            "median": st.median(errs),
            "mean": st.mean(ok) if ok else float("nan"),
            "std": st.stdev(ok) if len(ok) > 1 else 0.0,
            "best": min(errs), "worst": max(errs),
            "ratio": st.mean(r["ratio"] for r in rows),
            "mins": st.mean(r["mins"] for r in rows),
            "epochs": st.mean(r["epochs"] for r in rows if r["epochs"])}


def table(pde, stats):
    label = {"burgers": "shock/smooth", "heat": "early/late"}[pde]
    print(f"\n=== {pde.upper()} ===  (hard/easy = {label})")
    print(f"{'model':22s} {'n':>2} {'fail':>5} {'median':>8} {'mean(ok)':>9} {'std':>7} "
          f"{'best':>7} {'worst':>7} {'ratio':>7} {'epochs':>8} {'mins':>6}")
    print("-" * 104)
    for name, s in stats.items():
        print(f"{name:22s} {s['n']:>2} {s['nfail']:>2}/{s['n']:<2} {s['median']:7.2f}% "
              f"{s['mean']:8.2f}% {s['std']:6.2f}% {s['best']:6.2f}% {s['worst']:6.2f}% "
              f"{s['ratio']:7.1f} {s['epochs']:8.0f} {s['mins']:6.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pde", choices=["burgers", "heat", "both"], default="both")
    a = p.parse_args()

    wanted = ["burgers", "heat"] if a.pde == "both" else [a.pde]
    collected = {}

    for pde in wanted:
        stats = {}
        for name, pattern in GROUPS[pde].items():
            s = summarize(pattern)
            if s:
                stats[name] = s
        if stats:
            collected[pde] = stats
            table(pde, stats)
        else:
            print(f"\n=== {pde.upper()} ===  (no runs found)")

    if len(collected) < 2:
        return

    print("\n" + "=" * 104)
    print("DOES THE ORDERING GENERALISE?")
    orders = {}
    for pde, stats in collected.items():
        order = sorted(stats, key=lambda n: stats[n]["mean"])
        orders[pde] = order
        print(f"  {pde:8s} best -> worst: " + "  <  ".join(order))

    # Mean-based orderings are not trustworthy when a group's standard
    # deviation approaches its mean -- which is exactly the Burgers
    # classical case (3.36% +/- 3.35%, one 9.29% seed carrying it). A
    # single high-variance run can invert a mean ordering while the typical
    # run order is unchanged, and reporting that inversion as a finding is
    # the same error this paper documents in other people's work.
    #
    # So the median ordering is computed too, and a flip is only called
    # real when BOTH statistics agree on it.
    med = {pde: sorted(grp, key=lambda n: grp[n]["median"])
           for pde, grp in collected.items()}
    print("\n  by median (robust to a single bad seed):")
    for pde, order in med.items():
        print(f"  {pde:8s} best -> worst: " + "  <  ".join(order))

    skewed = [f"{pde}/{n}" for pde, stt in collected.items() for n, s_ in stt.items()
              if s_["std"] > 0.5 * s_["mean"]]
    if skewed:
        print(f"\n  HIGH SPREAD (std > half the mean): {', '.join(skewed)}")
        print("  Mean orderings involving these groups are unreliable.")

    a_, b_ = orders["burgers"], orders["heat"]
    m_a, m_b = med["burgers"], med["heat"]
    print()
    if a_ != b_ and m_a == m_b:
        print("  The MEAN ordering differs between PDEs but the MEDIAN ordering is")
        print("  IDENTICAL. The apparent flip is driven by spread, not by the")
        print("  equation. Report the medians and state plainly that the mean")
        print("  ordering inverts because of a single high-variance group --")
        print("  that is a finding about statistics, not about architecture, and")
        print("  it is on-thesis for this paper.")
    elif a_ == b_:
        print("  SAME ordering on both PDEs. The conclusion generalises across a")
        print("  shock problem and a smooth one — checked, not assumed.")
    elif a_ != b_ and m_a != m_b:
        print("  ORDERING FLIPS on BOTH mean and median. The design conclusion is")
        print("  equation-specific, which is the strongest available result and one")
        print("  no single-PDE study can reach. Report both orderings; do not")
        print("  average them.")
        moved = [n for i, n in enumerate(m_a) if m_b.index(n) != i]
        print(f"  Models that changed rank: {', '.join(moved)}")
    else:
        print("  Mean and median disagree about whether the ordering flips.")
        print("  Report both and claim neither.")

    print("\n  Reminder: relative-L2 is NOT comparable across equations — different")
    print("  problems, different difficulty. Compare orderings and ratio columns.")


if __name__ == "__main__":
    main()
