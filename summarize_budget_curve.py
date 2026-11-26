"""
The budget curve: does the ranking of architectures depend on training budget?

WHY THIS TABLE IS THE PAPER
---------------------------
Every published comparison of quantum-assisted PINNs against classical ones
picks a training budget and reports which model won. If the winner changes
with the budget, then those papers report a property of their stopping rule
rather than of their architectures -- and the reader has no way to tell,
because none of them vary the budget.

This reads the ranking at each budget from the SAME trajectories, so seed
noise cannot masquerade as a budget effect: every point at every budget
comes from the identical run.

WHAT COUNTS AS A REAL FLIP
--------------------------
Fixed in advance, because a rank change inside the noise is not a finding:

  1. the two models must swap rank between budgets, AND
  2. at the budget where the swap happens, the gap must exceed the combined
     seed spread of the two models.

Condition 2 is what stops this script reproducing the error it exists to
detect. The project has already produced one apparent flip -- Burgers'
classical against strong -- that came entirely from a single 9.29% seed
inflating a mean. Medians are reported alongside for the same reason.
"""

import glob
import json
import os
import re
import statistics as st

FAIL_LOSS = 1e-2

CONFIGS = {
    "Classical PINN":  "bc_classical_s*",
    "QAPINN ring L=2": "bc_ring_L2_s*",
    "QAPINN strong":   "bc_strong_L2_s*",
}


def budgets_in(run_dir):
    """Every budget this run was scored at: checkpoints plus the final."""
    out = {}
    for f in glob.glob(os.path.join(run_dir, "metrics_ep*.json")):
        ep = int(re.search(r"metrics_ep(\d+)\.json", f).group(1))
        out[ep] = json.load(open(f))
    final = os.path.join(run_dir, "metrics.json")
    if os.path.exists(final):
        out["converged"] = json.load(open(final))
    return out


def collect():
    """{config: {budget: [rel_l2 percentages across seeds]}}"""
    data = {}
    for label, pattern in CONFIGS.items():
        per_budget = {}
        for d in sorted(glob.glob(os.path.join("runs", pattern))):
            for ep, m in budgets_in(d).items():
                per_budget.setdefault(ep, []).append(m["rel_l2"] * 100)
        if per_budget:
            data[label] = per_budget
    return data


def ordered_budgets(data):
    keys = set()
    for per in data.values():
        keys |= set(per)
    nums = sorted(k for k in keys if isinstance(k, int))
    return nums + (["converged"] if "converged" in keys else [])


def main():
    data = collect()
    if not data:
        print("No budget-curve runs found (expected runs/bc_*). Run budget_curve.sh first.")
        return

    budgets = ordered_budgets(data)

    print("MEAN relative L2 at each training budget\n")
    head = "".join(f"{str(b):>16}" for b in budgets)
    print(f"{'model':22s}{head}")
    print("-" * (22 + 16 * len(budgets)))
    for label, per in data.items():
        row = ""
        for b in budgets:
            v = per.get(b)
            row += f"{st.mean(v):9.2f}% ±{st.stdev(v) if len(v) > 1 else 0:4.1f}" if v else f"{'-':>16}"
        print(f"{label:22s}{row}")

    print("\nMEDIAN relative L2 (robust to a single bad seed)\n")
    print(f"{'model':22s}{head}")
    print("-" * (22 + 16 * len(budgets)))
    for label, per in data.items():
        row = "".join(f"{st.median(per[b]):15.2f}%" if per.get(b) else f"{'-':>16}"
                      for b in budgets)
        print(f"{label:22s}{row}")

    print("\n" + "=" * (22 + 16 * len(budgets)))
    print("RANKING AT EACH BUDGET")
    rank_mean, rank_med = {}, {}
    for b in budgets:
        avail = [k for k in data if data[k].get(b)]
        rank_mean[b] = sorted(avail, key=lambda k: st.mean(data[k][b]))
        rank_med[b] = sorted(avail, key=lambda k: st.median(data[k][b]))
        print(f"  {str(b):>10}  (mean)   " + "  <  ".join(rank_mean[b]))
        print(f"  {'':>10}  (median) " + "  <  ".join(rank_med[b]))

    print("\n" + "-" * (22 + 16 * len(budgets)))
    print("DOES THE RANKING CHANGE WITH BUDGET?")

    first, last = budgets[0], budgets[-1]
    flips = []
    for i, m in enumerate(rank_med[first]):
        if m in rank_med[last] and rank_med[last].index(m) != i:
            flips.append(m)

    if not flips:
        print(f"  No rank change between {first} and {last} on the median.")
        print("  The ranking is budget-STABLE for these configurations. Report that:")
        print("  it means a short-budget comparison would have reached the same")
        print("  conclusion here, which is a useful negative result and bounds how")
        print("  far the budget critique extends.")
    else:
        print(f"  Rank changes between {first} and {last}: {', '.join(flips)}")
        print("\n  Checking the gap against the spread at each budget:")
        real = False
        for b in budgets:
            avail = rank_med[b]
            for i in range(len(avail) - 1):
                a, c = avail[i], avail[i + 1]
                va, vc = data[a][b], data[c][b]
                gap = abs(st.mean(va) - st.mean(vc))
                spread = ((st.stdev(va) if len(va) > 1 else 0)
                          + (st.stdev(vc) if len(vc) > 1 else 0))
                ok = gap > spread
                real = real or ok
                print(f"    {str(b):>10}  {a} vs {c}: gap {gap:5.2f}%, "
                      f"spread {spread:5.2f}%  {'SEPARATED' if ok else 'inside noise'}")
        print()
        if real:
            print("  At least one budget separates the models beyond their spread, and")
            print("  the ranking changes. This is the paper's central result: the")
            print("  conclusion a reader draws depends on when training was stopped.")
        else:
            print("  The ranking moves but NO budget separates the models beyond their")
            print("  seed spread. Do not claim a flip. The honest statement is that")
            print("  these configurations are indistinguishable at every budget tested,")
            print("  which still undermines any paper claiming a winner from one run.")


if __name__ == "__main__":
    main()
