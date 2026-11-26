#!/usr/bin/env python3
"""
HYPERPARAMETER SENSITIVITY -- Reviewer 2, point 3.

R2 asked whether the reported quantum advantage is sensitive to optimiser
choice, learning rate and batch size. This reads the 30 runs of
run_hyper_sweep.sh plus the baseline cell (the sym4 runs at seeds 0-2, which
protocol5_hyper.py reproduces exactly at its defaults) and reports, per config,
both arms' median and mean.

WHAT THIS DESIGN CAN AND CANNOT SHOW
  Three seeds per arm. The exact two-sided permutation test over C(6,3) = 20
  splits has a floor of 2/20 = 0.10, so NO individual config can reach
  significance here, whatever the effect size. This is a direction-of-effect
  sweep, not a per-config hypothesis test, and the paper must say so. The
  aggregate sign test across configs is the only inferential statement
  available, and it is reported as such.

AGREEMENT RULE
  A cell is only called for an arm when median AND mean agree on the winner
  (CORRECTION_inversion_claim.md). Cells where they disagree are reported as
  split, never as a flip.

USAGE  python3 hyper_analysis.py
"""

import glob
import json
import sys

import numpy as np

from stats_analysis import cliffs_delta, permutation_test

SEEDS = [0, 1, 2]
CONFIGS = ["baseline", "sgd", "rmsprop", "lr0.001", "lr0.01", "batch256"]


def cell(config, arm):
    """err_u_pct for one (config, arm) cell, 3 seeds."""
    if config == "baseline":
        pat = "runs/sym4_%s_s%d/replication.json"
    else:
        pat = "runs/hyp_%s_" + config + "_s%d/replication.json"
    out = []
    for s in SEEDS:
        p = pat % (arm, s)
        out.append(json.load(open(p))["err_u_pct"])
    return np.array(out)


def main():
    rows = []
    for c in CONFIGS:
        q, m = cell(c, "qcpinn"), cell(c, "model2")
        diff, p, nsp = permutation_test(q, m, stat="median")
        d, _ = cliffs_delta(q, m)
        win_med = "quantum" if np.median(q) < np.median(m) else "classical"
        win_mean = "quantum" if q.mean() < m.mean() else "classical"
        agree = win_med == win_mean
        rows.append(dict(config=c, q=q, m=m,
                         qmed=np.median(q), mmed=np.median(m),
                         qmean=q.mean(), mmean=m.mean(),
                         p=p, nsp=nsp, d=d,
                         win_med=win_med, win_mean=win_mean, agree=agree))

    print("=" * 78)
    print("HYPERPARAMETER SENSITIVITY  (3 seeds/arm; permutation floor = 2/20 = 0.10)")
    print("=" * 78)
    hdr = f"{'config':<10} {'quantum med':>12} {'classical med':>14} {'delta':>7} {'p':>7} {'call':>10}"
    print(hdr)
    print("-" * 78)
    for r in rows:
        call = r["win_med"] if r["agree"] else "SPLIT"
        print(f"{r['config']:<10} {r['qmed']:>11.2f}% {r['mmed']:>13.2f}% "
              f"{r['d']:>+7.2f} {r['p']:>7.3f} {call:>10}")
    print()
    print("means (the agreement check):")
    for r in rows:
        flag = "" if r["agree"] else "   <-- MEDIAN AND MEAN DISAGREE"
        print(f"  {r['config']:<10} q {r['qmean']:6.2f}%  c {r['mmean']:6.2f}%  "
              f"median->{r['win_med']:<9} mean->{r['win_mean']:<9}{flag}")
    print()
    print("per-seed values:")
    for r in rows:
        print(f"  {r['config']:<10} q {np.array2string(r['q'], precision=2)}"
              f"   c {np.array2string(r['m'], precision=2)}")

    # ---- aggregate: sign test over configs, median as the statistic -------
    agreed = [r for r in rows if r["agree"]]
    qwin = sum(r["win_med"] == "quantum" for r in agreed)
    n = len(agreed)
    # two-sided exact binomial at p=0.5
    from math import comb
    tail = sum(comb(n, k) for k in range(qwin, n + 1)) / 2 ** n
    pbin = min(1.0, 2 * tail)
    print()
    print("=" * 78)
    print(f"AGGREGATE: quantum wins {qwin} of {n} configs where median and mean agree")
    print(f"           ({len(rows) - n} config(s) split, excluded)")
    print(f"           exact two-sided sign test p = {pbin:.4f}")
    print("=" * 78)

    json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in r.items()} for r in rows],
              open("hyper_report.json", "w"), indent=2)
    print("wrote hyper_report.json")


if __name__ == "__main__":
    main()
