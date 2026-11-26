#!/usr/bin/env python3
"""
VERIFY the hyperparameter-sensitivity paragraph of Section VI.

Same contract as the other suites: re-derive every number from runs/, then
assert the literal string appears in paper_nqcomp/main.tex. Exact tests come
from stats_analysis.py and are never reimplemented here.

USAGE  python3 verify_hyper_section.py
"""

import json
import sys
from math import comb

import numpy as np

from stats_analysis import cliffs_delta, fisher_exact_2x2, permutation_test  # noqa: F401

TEX = "paper_nqcomp/main.tex"
FAIL = 50.0
SEEDS = (0, 1, 2)
CONFIGS = ["baseline", "sgd", "rmsprop", "lr0.001", "lr0.01", "batch256"]
rows = []


def cell(c, arm):
    pat = ("runs/sym4_%s_s%d/replication.json" if c == "baseline"
           else "runs/hyp_%s_" + c + "_s%d/replication.json")
    return np.array([json.load(open(pat % (arm, s)))["err_u_pct"] for s in SEEDS])


def claim(desc, literal, shown, tex):
    rows.append((desc, literal.strip(), shown, literal in tex and shown == literal.strip()))


def main():
    tex = open(TEX).read()

    qa = {c: cell(c, "qcpinn") for c in CONFIGS}
    ma = {c: cell(c, "model2") for c in CONFIGS}

    # --- agreement per cell, and the count the paper quotes -----------------
    agree, split = [], []
    for c in CONFIGS:
        q, m = qa[c], ma[c]
        wmed = np.median(q) < np.median(m)
        wmean = q.mean() < m.mean()
        (agree if wmed == wmean else split).append((c, wmed, wmean))
    qwin = sum(1 for _, wmed, _ in agree if wmed)
    rows.append(("quantum wins all agreeing cells", "five of six",
                 f"{qwin} of {len(agree)} agreeing, {len(split)} split",
                 "five of six" in tex and qwin == 5 and len(agree) == 5 and len(split) == 1))
    rows.append(("the split cell is RMSprop", "RMSprop splits",
                 f"split = {[c for c, _, _ in split]}",
                 "RMSprop splits" in tex and [c for c, _, _ in split] == ["rmsprop"]))

    # --- the pooled figures -------------------------------------------------
    q = np.concatenate([qa[c] for c in CONFIGS])
    m = np.concatenate([ma[c] for c in CONFIGS])
    claim("quantum worst of 18", "worst run of eighteen is 9.71\\%",
          f"worst run of eighteen is {q.max():.2f}\\%", tex)
    claim("classical worst", "classical arm's worst is\n100.7\\%",
          f"classical arm's worst is\n{m.max():.1f}\\%", tex)
    claim("rmsprop dead seed value", "fails outright at 100.7\\%",
          f"fails outright at {ma['rmsprop'].max():.1f}\\%", tex)
    nqf, nmf = int((q > FAIL).sum()), int((m > FAIL).sum())
    rows.append(("dead inits 0/18 vs 2/18", "0/18 against 2/18",
                 f"{nqf}/18 against {nmf}/18",
                 "0/18 against 2/18" in tex and nqf == 0 and nmf == 2))
    fp = fisher_exact_2x2(0, 18, nmf, 18 - nmf)
    fp = float(fp[-1]) if isinstance(fp, (tuple, list)) else float(fp)
    claim("Fisher on pooled failures", "Fisher $p = 0.4857$", f"Fisher $p = {fp:.4f}$", tex)

    # --- the two design-limit statements the paragraph makes ---------------
    floor = 2 / comb(6, 3)
    claim("permutation floor", "$2/\\binom{6}{3} = 0.10$",
          "$2/\\binom{6}{3} = %.2f$" % floor, tex)
    tail = sum(comb(5, k) for k in range(5, 6)) / 2 ** 5
    pbin = min(1.0, 2 * tail)
    claim("sign test over agreeing cells", "$p = 0.0625$", f"$p = {pbin:.4f}$", tex)
    rows.append(("0.0625 IS the n=5 floor", "smallest value five cells admit",
                 f"floor at n=5 is {pbin:.4f}",
                 "smallest value five cells admit" in tex and abs(pbin - 0.0625) < 1e-9))

    # --- run count ----------------------------------------------------------
    rows.append(("thirty runs exist on disk", "thirty runs in all",
                 f"{len(q) + len(m)} runs across both arms",
                 "thirty runs in all" in tex and len(q) + len(m) == 36))
    # 36 = 30 sweep runs + the 6 baseline runs reused from sym4; the paragraph
    # says "five further settings ... thirty runs in all", which is the sweep.
    nsweep = sum(len(qa[c]) + len(ma[c]) for c in CONFIGS if c != "baseline")
    rows.append(("the sweep itself is 30 runs", "five\nfurther settings",
                 f"non-baseline runs = {nsweep}",
                 "five\nfurther settings" in tex and nsweep == 30))

    print(f"{'claim':<36} {'in paper':<34} ok")
    print("-" * 96)
    bad = 0
    for desc, lit, got, ok in rows:
        print(f"{desc:<36} {lit[:33]:<34} {'ok' if ok else 'MISMATCH -> ' + str(got)}")
        bad += (not ok)
    print()
    if bad:
        print(f"{bad} MISMATCH(ES) in the hyperparameter paragraph")
        sys.exit(1)
    print("ALL HYPERPARAMETER NUMBERS VERIFIED against runs/")


if __name__ == "__main__":
    main()
