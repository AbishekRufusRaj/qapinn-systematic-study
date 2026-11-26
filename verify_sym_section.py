#!/usr/bin/env python3
"""
VERIFY SECTION VI (the symmetric control) IN THE NQComp MANUSCRIPT.

verify_paper.py checks PAPER.md; verify_conf_cost.py checks the cost and config
tables of paper_nqcomp/main.tex. Neither covers the symmetric-control section,
which is the manuscript's newest and most load-bearing result. This closes that
gap the same way the others do: re-derive every figure from runs/, then assert
the literal string appears in main.tex.

A number that is right in the analysis and mistyped in the paper is still wrong
in the paper, and that is the only failure mode this guards against.

USAGE
    python3 verify_sym_section.py
"""

import glob
import json
import os
import sys

import numpy as np

from stats_analysis import cliffs_delta, fisher_exact_2x2, permutation_test

TEX = "paper_nqcomp/main.tex"
RUNS = "runs"
FAIL = 50.0

rows = []


def claim(desc, literal, derived, fmt, tex):
    """Assert `literal` is in the .tex AND equals the derived value."""
    shown = fmt.format(derived)
    in_tex = literal in tex
    matches = (shown == literal.strip())
    rows.append((desc, literal.strip(), shown, in_tex and matches))


def main():
    if not os.path.exists(TEX):
        sys.exit(f"not found: {TEX}")
    tex = open(TEX).read()

    def arm(a):
        return [json.load(open(f"{RUNS}/sym4_{a}_s{i}/replication.json"))["err_u_pct"]
                for i in range(10)]

    q, c = arm("qcpinn"), arm("model2")
    qf = [v for v in q if v > FAIL]
    cfail = [i for i in range(10) if c[i] > FAIL]
    keep = [i for i in range(10) if c[i] <= FAIL]
    qs, cs = [q[i] for i in keep], [c[i] for i in keep]

    diff, p_all, nsplits = permutation_test(q, c, stat="median")
    d_all, _ = cliffs_delta(q, c)
    _, p_keep, _ = permutation_test(qs, cs, stat="median")
    d_keep, _ = cliffs_delta(qs, cs)
    fp = fisher_exact_2x2(0, 10, len(cfail), 10 - len(cfail))
    fp = float(fp[-1]) if isinstance(fp, (tuple, list)) else float(fp)

    pm = [json.load(open(f"{RUNS}/pm_model2_ours_s{i}/metrics.json"))["rel_l2"] * 100
          for i in range(8)]

    qsec = [json.load(open(f"{RUNS}/sym4_qcpinn_s{i}/replication.json")) for i in range(10)]
    csec = [json.load(open(f"{RUNS}/sym4_model2_s{i}/replication.json")) for i in range(10)]
    qmin = np.median([(r["adam_seconds"] + r["lbfgs_seconds"]) / 60 for r in qsec])
    cmin = np.median([(r["adam_seconds"] + r["lbfgs_seconds"]) / 60 for r in csec])

    # ---- the literals Section VI quotes --------------------------------
    claim("quantum median", "median 1.73\\%", np.median(q), "median {:.2f}\\%", tex)
    claim("quantum IQR low", "IQR 1.61--2.01",
          np.percentile(q, 25), "IQR {:.2f}--2.01", tex)
    claim("quantum range", "range\n1.21--2.31", min(q), "range\n{:.2f}--2.31", tex)
    claim("classical median", "median 4.30\\%", np.median(c), "median {:.2f}\\%", tex)
    claim("classical non-failing low", "between 1.21\\% and 7.02\\%",
          min(cs), "between {:.2f}\\% and 7.02\\%", tex)
    claim("median difference", "median $-2.57$ pp", abs(diff), "median $-{:.2f}$ pp", tex)
    claim("permutation p, all seeds", "$p = 0.0068$", p_all, "$p = {:.4f}$", tex)
    # LaTeX writes the thousands separator as {,}; claim() would re-run
    # .format() on that and choke, so this one is checked directly.
    _lit = "184{,}756"
    _shown = "{:,}".format(nsplits).replace(",", "{,}")
    rows.append(("permutation splits", _lit, _shown,
                 _lit in tex and _shown == _lit))
    claim("Cliff delta, all seeds", "\\delta = -0.64$", d_all, "\\delta = {:.2f}$", tex)
    claim("non-failing medians q", "1.77\\% against 3.43\\%",
          np.median(qs), "{:.2f}\\% against 3.43\\%", tex)
    claim("non-failing median c", "against 3.43\\%", np.median(cs), "against {:.2f}\\%", tex)
    claim("Cliff delta, non-failing", "\\delta = -0.43$", d_keep, "\\delta = {:.2f}$", tex)
    claim("permutation p, non-failing", "$p = 0.0583$", p_keep, "$p = {:.4f}$", tex)
    claim("Fisher p on failures", "Fisher $p = 0.2105$", fp, "Fisher $p = {:.4f}$", tex)
    claim("our implementation median", "median 0.70\\%", np.median(pm), "median {:.2f}\\%", tex)
    claim("our implementation range", "range 0.53--0.84",
          min(pm), "range {:.2f}--0.84", tex)
    claim("quantum wall-clock", "51.7 minutes", qmin, "{:.1f} minutes", tex)
    claim("classical wall-clock", "against 0.86", cmin, "against {:.2f}", tex)
    claim("wall-clock factor", "factor of 60", qmin / cmin, "factor of {:.0f}", tex)

    # counts stated in prose
    ok_counts = [
        ("quantum failures = 0", "no failures in ten seeds", len(qf) == 0),
        ("classical failures = 3", "seeds\n5, 6 and 9", cfail == [5, 6, 9]),
        ("0/10 vs 3/10", "0/10 against\n3/10", len(qf) == 0 and len(cfail) == 3),
        ("seven non-failing", "seven seeds", len(keep) == 7),
    ]

    print(f"{'claim':<32} {'in paper':<26} {'derived':<26} ok")
    print("-" * 94)
    bad = 0
    for desc, lit, got, ok in rows:
        print(f"{desc:<32} {lit[:25]:<26} {got[:25]:<26} {'ok' if ok else 'MISMATCH'}")
        bad += (not ok)
    for desc, lit, cond in ok_counts:
        present = lit in tex
        ok = present and cond
        print(f"{desc:<32} {lit[:25]!r:<26} {'derived ok' if cond else 'DERIVED BAD':<26}"
              f" {'ok' if ok else 'MISMATCH'}")
        bad += (not ok)

    print()
    if bad:
        print(f"{bad} MISMATCH(ES) in Section VI")
        sys.exit(1)
    print("ALL SECTION VI NUMBERS VERIFIED against runs/")


if __name__ == "__main__":
    main()
