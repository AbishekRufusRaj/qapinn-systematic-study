"""
floor_analysis.py -- the sample-size arithmetic, done per test statistic.

Why this exists
---------------
`stats_analysis.py` runs its exact permutation test on the difference in
MEDIANS (see its call site: permutation_test(ea, eb, "median")), and Holm is
applied to those p-values. The manuscript, however, quotes the attainable
floor of a MEAN-difference test: 2/252 = 0.0079.

Those are different numbers. A median of five points is the third order
statistic, so many permutations produce the same extreme difference, and the
floor is correspondingly coarser. This script computes both floors exactly, by
enumeration, and reports what each one implies for the required sample size.

Outputs
-------
  floor_table.json  -- floor per (n, statistic), and the smallest n clearing
                       the Holm threshold for a family of `--family` tests
  a printed table suitable for pasting into the manuscript

Nothing here reads or writes any run directory; it is pure combinatorics plus
a re-analysis of stats_report.json.
"""

import argparse
import itertools
import json
from math import comb

import numpy as np


# ---------------------------------------------------------------- floors

def attainable_floor(n, stat):
    """Smallest two-sided p the exact permutation test can ever return, for
    n per arm and the given difference statistic.

    Computed by construction rather than argued: take the maximally separated
    configuration (arm A holds the n smallest observations, arm B the n
    largest), which is the most extreme arrangement the test can be handed,
    and enumerate every split. The observed split is counted in the numerator,
    matching the convention in stats_analysis.permutation_test.
    """
    pooled = np.arange(2 * n, dtype=float)
    f = np.median if stat == "median" else np.mean
    obs = f(pooled[:n]) - f(pooled[n:])
    count = 0
    for c in itertools.combinations(range(2 * n), n):
        mask = np.zeros(2 * n, bool)
        mask[list(c)] = True
        v = f(pooled[mask]) - f(pooled[~mask])
        if abs(v) >= abs(obs) - 1e-15:
            count += 1
    return count, comb(2 * n, n), count / comb(2 * n, n)


def attainable_floor_unequal(n1, n2, stat):
    """The same construction as `attainable_floor`, for arms of UNEQUAL size.

    Needed because the seed extension takes the ten continuous arms to n = 8
    while the three QCPINN arms stay at n = 5, so H10 (`classical_g1.0` vs
    `qc_model2_2751`) becomes an 8-against-5 comparison and neither column of
    the section 7.2 table describes it. There are then three distinct floors
    inside one Holm family.

    `attainable_floor` is deliberately left untouched: it is the function the
    manuscript's table is generated from, and it is enumerated rather than
    closed-form for the reason recorded in this module's docstring. This is an
    extension of it, not a replacement -- for n1 == n2 the two agree exactly,
    which `verify_paper.py` asserts.
    """
    n = n1 + n2
    pooled = np.arange(n, dtype=float)
    f = np.median if stat == "median" else np.mean
    obs = f(pooled[:n1]) - f(pooled[n1:])
    count = 0
    for c in itertools.combinations(range(n), n1):
        mask = np.zeros(n, bool)
        mask[list(c)] = True
        v = f(pooled[mask]) - f(pooled[~mask])
        if abs(v) >= abs(obs) - 1e-15:
            count += 1
    return count, comb(n, n1), count / comb(n, n1)


def holm_smallest_threshold(family):
    """Holm's criterion for the most significant of `family` tests."""
    return 0.05 / family


def smallest_n_clearing(stat, family, nmax=10):
    thr = holm_smallest_threshold(family)
    for n in range(2, nmax + 1):
        if attainable_floor(n, stat)[2] < thr:
            return n
    return None


# ---------------------------------------------------------------- re-analysis

def permutation_p(a, b, stat):
    """Exact two-sided permutation p for the observed data. Same convention as
    stats_analysis.permutation_test, reimplemented here so this script does not
    depend on importing that module."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n1, n = len(a), len(pooled)
    f = np.median if stat == "median" else np.mean
    obs = f(a) - f(b)
    count = total = 0
    for c in itertools.combinations(range(n), n1):
        mask = np.zeros(n, bool)
        mask[list(c)] = True
        v = f(pooled[mask]) - f(pooled[~mask])
        total += 1
        if abs(v) >= abs(obs) - 1e-15:
            count += 1
    return float(obs), count / total, total


def holm(pvals, alpha=0.05):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj, adj <= alpha


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="stats_report.json")
    ap.add_argument("--family", type=int, default=12,
                    help="number of declared comparisons")
    ap.add_argument("--nmax", type=int, default=9)
    ap.add_argument("--out", default="floor_table.json")
    args = ap.parse_args()

    thr = holm_smallest_threshold(args.family)

    # ---- 1. the floor table ------------------------------------------------
    print(f"\nExact attainable two-sided p-floor, by n per arm and statistic.")
    print(f"Holm threshold for the most significant of {args.family} tests: "
          f"{thr:.5f}\n")
    header = (f"{'n':>3} {'splits':>9}  {'median-diff floor':>22}  "
              f"{'mean-diff floor':>22}  {'median OK':>9} {'mean OK':>8}")
    print(header)
    print("-" * len(header))

    table = []
    for n in range(4, args.nmax + 1):
        cm, tot, pm = attainable_floor(n, "median")
        ca, _, pa = attainable_floor(n, "mean")
        row = {"n": n, "splits": tot,
               "median_floor": pm, "median_num": cm,
               "mean_floor": pa, "mean_num": ca,
               "median_clears_holm": bool(pm < thr),
               "mean_clears_holm": bool(pa < thr)}
        table.append(row)
        print(f"{n:>3} {tot:>9}  {f'{cm}/{tot} = {pm:.5f}':>22}  "
              f"{f'{ca}/{tot} = {pa:.5f}':>22}  "
              f"{'yes' if row['median_clears_holm'] else 'no':>9} "
              f"{'yes' if row['mean_clears_holm'] else 'no':>8}")

    n_med = smallest_n_clearing("median", args.family, args.nmax)
    n_mean = smallest_n_clearing("mean", args.family, args.nmax)
    print(f"\nSmallest n per arm whose floor clears Holm:")
    print(f"  difference in medians : n = {n_med}")
    print(f"  difference in means   : n = {n_mean}")
    print(f"\nThese are FLOORS, not power. Reaching the floor requires the "
          f"observed\nseparation to be complete; n is necessary, not sufficient.")

    # ---- 2. re-analyse the declared comparisons under both statistics -------
    out = {"family": args.family, "holm_threshold": thr,
           "floor_table": table,
           "smallest_n_median": n_med, "smallest_n_mean": n_mean}

    try:
        rep = json.load(open(args.report))
    except FileNotFoundError:
        print(f"\n[{args.report} not found -- floor table only]")
        json.dump(out, open(args.out, "w"), indent=2)
        return

    desc = rep["descriptives"]
    comps = rep["comparisons"]

    rows = []
    for c in comps:
        A, B = c["a"], c["b"]
        if A not in desc or B not in desc:
            continue
        ea, eb = desc[A]["values"], desc[B]["values"]
        _, p_med, _ = permutation_p(ea, eb, "median")
        _, p_mean, _ = permutation_p(ea, eb, "mean")
        rows.append({"label": c["label"], "delta": c["cliffs_delta"],
                     "p_median": p_med, "p_mean": p_mean,
                     "p_mannwhitney": c.get("p_mannwhitney")})

    if rows:
        adj_med, sig_med = holm([r["p_median"] for r in rows])
        adj_mean, sig_mean = holm([r["p_mean"] for r in rows])
        for r, am, sm, aa, sa in zip(rows, adj_med, sig_med, adj_mean, sig_mean):
            r["p_median_holm"] = float(am)
            r["p_mean_holm"] = float(aa)
            r["sig_median"] = bool(sm)
            r["sig_mean"] = bool(sa)

        print(f"\n\nThe twelve declared comparisons under both statistics:\n")
        h = (f"{'comparison':46}{'delta':>7}{'p_med':>9}{'p_mean':>9}"
             f"{'p_MWU':>9}{'holm_med':>10}{'holm_mean':>11}")
        print(h)
        print("-" * len(h))
        for r in rows:
            print(f"{r['label'][:45]:46}{r['delta']:>7.2f}"
                  f"{r['p_median']:>9.4f}{r['p_mean']:>9.4f}"
                  f"{(r['p_mannwhitney'] or float('nan')):>9.4f}"
                  f"{r['p_median_holm']:>10.4f}{r['p_mean_holm']:>11.4f}")

        n_at_floor_mean = sum(
            1 for r in rows if abs(r["p_mean"] - 2 / comb(10, 5)) < 1e-9)
        n_at_floor_med = sum(
            1 for r in rows if abs(r["p_median"] - 12 / comb(10, 5)) < 1e-9)
        print(f"\n  comparisons sitting exactly ON their floor: "
              f"{n_at_floor_med} (median stat), {n_at_floor_mean} (mean stat)")
        print(f"  any surviving Holm: "
              f"{'yes' if any(r['sig_median'] for r in rows) else 'no'} (median), "
              f"{'yes' if any(r['sig_mean'] for r in rows) else 'no'} (mean)")

        out["comparisons"] = rows

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
