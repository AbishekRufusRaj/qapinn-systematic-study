"""
verify_conf_cost.py -- re-derive every number in the conference paper's cost and
configuration tables from the run artefacts.

verify_paper.py covers PAPER.md, the thesis document. The NQComp submission in
paper_nqcomp/main.tex is not in its scope, so the cost table added for reviewers
4 and 5 would otherwise be unchecked. This reads the medians straight out of
runs/ and compares them against the literals in main.tex.
"""

import glob
import json
import os
import re
import statistics as st
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(ROOT, "paper_nqcomp", "main.tex")


def load(tag, key):
    return [json.load(open(f))
            for f in sorted(glob.glob(os.path.join(ROOT, "runs", f"{tag}_s*", key)))]


def arm(tag, kind):
    """(n, params, median minutes, median error %) for one arm."""
    if kind == "ours":
        r = load(tag, "metrics.json")
        if not r:
            return None
        return (len(r), r[0]["param_counts"]["total"],
                st.median([x["total_seconds"] for x in r]) / 60,
                st.median([x["rel_l2"] * 100 for x in r]))
    r = load(tag, "replication.json")
    if not r:
        return None
    secs = [x.get("seconds", x.get("adam_seconds", 0) + x.get("lbfgs_seconds", 0))
            for x in r]
    return (len(r), r[0]["params"], st.median(secs) / 60,
            st.median([x["err_u_pct"] for x in r]))


# tag, kind, expected (params, minutes, error) as printed in tab:cost
EXPECT = [
    ("rep_qcpinn",     "rep",  771,  71.0, 5.16),
    ("rep_model2",     "rep",  2751, 0.8,  9.88),
    ("frz_qcpinn",     "rep",  771,  58.7, 7.84),
    ("abl_qcpinn",     "rep",  786,  1.1,  1342),
    ("conv_ring_L2",   "ours", 2649, 34.3, 3.73),
    ("conv_strong_L2", "ours", 2665, 35.0, 3.33),
    ("abl_ring",       "ours", 2653, 2.6,  3.23),
    ("abl_strong",     "ours", 2673, 4.1,  2.69),
    ("conv_classical", "ours", 3021, 2.0,  2.21),
    ("hh_strong_ni",   "ours", 2665, 66.9, 7.54),
    ("hh_ring_ni",     "ours", 2649, 52.9, 66.06),
    ("hh_classical",   "ours", 3021, 4.6,  6.90),
]

tex = open(TEX).read()
fails = 0
print(f"{'arm':18}{'n':>3}{'par':>7}{'min':>8}{'err%':>9}   check")
print("-" * 62)
for tag, kind, e_par, e_min, e_err in EXPECT:
    got = arm(tag, kind)
    if got is None:
        print(f"{tag:18}  -- NO RUNS ON DISK --")
        fails += 1
        continue
    n, par, mins, err = got
    ok = []
    ok.append(("par", par == e_par, f"{par} vs {e_par}"))
    ok.append(("min", abs(mins - e_min) <= 0.05 + 0.005 * e_min, f"{mins:.1f} vs {e_min}"))
    tol = 0.5 if e_err > 100 else 0.005 + 0.005 * e_err
    ok.append(("err", abs(err - e_err) <= tol, f"{err:.2f} vs {e_err}"))
    bad = [f"{k} {d}" for k, good, d in ok if not good]
    print(f"{tag:18}{n:>3}{par:>7}{mins:>8.1f}{err:>9.2f}   "
          + ("ok" if not bad else "MISMATCH: " + "; ".join(bad)))
    fails += len(bad)

# every literal in the table must also actually appear in the .tex
print()
for tag, kind, e_par, e_min, e_err in EXPECT:
    for v in (f"{e_min:.1f}", f"{e_err:.2f}" if e_err < 100 else str(int(e_err))):
        if v not in tex:
            print(f"  LITERAL MISSING from main.tex: {v}  (arm {tag})")
            fails += 1

# parameter counts quoted in the configuration table and prose
for lit in ["2{,}649", "2{,}665", "3{,}021", "2{,}751", "771", "15", "8", "24"]:
    if lit not in tex:
        print(f"  CONFIG LITERAL MISSING: {lit}")
        fails += 1

print()
print("ALL COST/CONFIG NUMBERS VERIFIED" if fails == 0
      else f"{fails} PROBLEM(S) -- main.tex disagrees with the artefacts")
sys.exit(1 if fails else 0)
