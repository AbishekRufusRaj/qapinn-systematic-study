#!/usr/bin/env python3
"""
VERIFY the third-equation and circuit-component section (sec:third).

Same contract as verify_sym_section.py: re-derive every number from runs/, then
assert the literal string appears in paper_nqcomp/main.tex. Exact tests are
imported from stats_analysis.py, never reimplemented.

USAGE
    python3 verify_third_section.py
"""

import glob
import json
import os
import sys

import numpy as np

from stats_analysis import cliffs_delta, permutation_test

TEX = "paper_nqcomp/main.tex"
rows = []


def fam(tag):
    v = [json.load(open(p))["rel_l2"] * 100
         for p in sorted(glob.glob(f"runs/{tag}_s*/metrics.json"))]
    if not v:
        sys.exit(f"no runs for {tag}")
    return np.array(v)


def claim(desc, literal, shown, tex):
    rows.append((desc, literal.strip(), shown, literal in tex and shown == literal.strip()))


def main():
    if not os.path.exists(TEX):
        sys.exit(f"not found: {TEX}")
    tex = open(TEX).read()

    hc, hr, hs = fam("heat_classical"), fam("heat_ring_L2"), fam("heat_strong_L2")

    # --- the third equation ------------------------------------------------
    claim("heat classical median", "median 0.33\\%", f"median {np.median(hc):.2f}\\%", tex)
    claim("heat strong median", "strong ansatz 0.48\\%",
          f"strong ansatz {np.median(hs):.2f}\\%", tex)
    claim("heat ring median", "ring ansatz 0.67\\%",
          f"ring ansatz {np.median(hr):.2f}\\%", tex)
    # ordering must hold on the mean too -- the paper says so explicitly, and
    # this project has been burned once by a ranking that held only on one
    # statistic (see CORRECTION_inversion_claim.md).
    order_median = np.median(hc) < np.median(hs) < np.median(hr)
    order_mean = hc.mean() < hs.mean() < hr.mean()
    rows.append(("heat ordering holds on median AND mean",
                 "classical < strong < ring",
                 f"median {order_median}, mean {order_mean}",
                 order_median and order_mean))

    _, p_ring, n_ring = permutation_test(hr, hc, stat="median")
    d_ring, _ = cliffs_delta(hr, hc)
    _, p_strong, _ = permutation_test(hs, hc, stat="median")
    d_strong, _ = cliffs_delta(hs, hc)
    claim("heat ring p = p-floor", "$p = 40/12{,}870 = 0.0031$",
          "$p = 40/12{,}870 = %.4f$" % p_ring, tex)
    claim("heat ring delta", "\\delta = +0.88$", f"\\delta = {d_ring:+.2f}$", tex)
    claim("heat strong p", "$p = 0.0796$", f"$p = {p_strong:.4f}$", tex)
    claim("heat strong delta", "\\delta = +0.66$", f"\\delta = {d_strong:+.2f}$", tex)

    # --- depth -------------------------------------------------------------
    r2, r4 = fam("conv_ring_L2"), fam("conv_ring_L4")
    s2, s4 = fam("conv_strong_L2"), fam("conv_strong_L4")
    _, p_s, _ = permutation_test(s4, s2, stat="median")
    _, p_r, _ = permutation_test(r4, r2, stat="median")
    claim("strong depth L2->L4", "3.33\\% to 3.43\\%",
          f"{np.median(s2):.2f}\\% to {np.median(s4):.2f}\\%", tex)
    claim("strong depth p", "$p = 0.9378$", f"$p = {p_s:.4f}$", tex)
    claim("ring depth L2->L4", "(3.73\\% to 9.44\\%",
          f"({np.median(r2):.2f}\\% to {np.median(r4):.2f}\\%", tex)
    claim("ring depth p", "$p = 0.1064$", f"$p = {p_r:.4f}$", tex)
    rows.append(("ring depth: median AND mean both worsen",
                 "both moving the wrong way",
                 f"median {np.median(r4) > np.median(r2)}, mean {r4.mean() > r2.mean()}",
                 "both moving the wrong way" in tex
                 and np.median(r4) > np.median(r2) and r4.mean() > r2.mean()))

    # --- entanglement structure -------------------------------------------
    bq, bs = fam("conv_ring_L2"), fam("conv_strong_L2")
    hq, hss = fam("hh_ring"), fam("hh_strong")
    claim("burgers ring vs strong", "3.33\\% against 3.73\\%",
          f"{np.median(bs):.2f}\\% against {np.median(bq):.2f}\\%", tex)
    claim("helmholtz ring vs strong", "8.90\\%\nagainst 17.99\\%",
          f"{np.median(hss):.2f}\\%\nagainst {np.median(hq):.2f}\\%", tex)
    claim("heat ring vs strong", "0.48\\% against 0.67\\%",
          f"{np.median(hs):.2f}\\% against {np.median(hr):.2f}\\%", tex)
    for lab, a, b, lit in (("burgers", bs, bq, "$p = 0.79$"),
                           ("helmholtz", hss, hq, "$0.067$"),
                           ("heat", hs, hr, "$0.13$")):
        _, p, _ = permutation_test(a, b, stat="median")
        shown = lit.replace(lit.strip("$").split("=")[-1].strip(),
                            f"{p:.4g}".lstrip("0") if lit.startswith("$0") else f"{p:.2f}")
        rows.append((f"{lab} ring-vs-strong p", lit,
                     f"derived {p:.4f}", lit in tex and abs(p - float(
                         lit.strip("$").split("=")[-1])) < 5e-3))

    # --- qubit count: the paper must NOT quote one -------------------------
    rows.append(("qubit count claimed nowhere",
                 "We do not vary it at a seed count",
                 "single-seed runs only",
                 "We do not vary it at a seed count" in tex))

    print(f"{'claim':<42} {'in paper':<30} ok")
    print("-" * 88)
    bad = 0
    for desc, lit, got, ok in rows:
        print(f"{desc:<42} {lit[:29]:<30} {'ok' if ok else 'MISMATCH -> ' + str(got)}")
        bad += (not ok)
    print()
    if bad:
        print(f"{bad} MISMATCH(ES) in sec:third")
        sys.exit(1)
    print("ALL THIRD-EQUATION / COMPONENT NUMBERS VERIFIED against runs/")


if __name__ == "__main__":
    main()
