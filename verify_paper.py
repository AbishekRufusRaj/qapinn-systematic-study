#!/usr/bin/env python3
"""
VERIFY EVERY NUMBER IN PAPER.md AGAINST THE ARTEFACTS ON DISK.

A paper whose thesis is that under-reported protocol manufactures conclusions
cannot afford a transcription error in its own tables. Every quantitative claim
in PAPER.md is asserted here against runs/, gradient_variance_results.json,
mode_decomposition_summary.json, stats_report.json, the Fourier CSVs and the
XAI metrics. A claim that no longer matches the data fails loudly.

Run this before every submission, and after any re-run of any experiment.

    python3 verify_paper.py            # numpy only; no torch needed

Exit status 0 if every claim holds, 1 otherwise.
"""

import csv
import glob
import json
import os
import sys

import numpy as np

import floor_analysis as fa

RUNS = "runs"
FAIL_LOSS = 1e-2
TOL = 5e-5          # claims are quoted to 2 decimal places as percentages

_checks = []


def check(label, got, want, tol=TOL, unit=""):
    ok = (got is not None) and abs(got - want) <= tol
    _checks.append((label, got, want, ok, unit))
    return ok


def check_int(label, got, want):
    ok = got == want
    _checks.append((label, got, want, ok, ""))
    return ok


# --------------------------------------------------------------------------

def load_cfg(pattern):
    e, f = [], []
    for d in sorted(glob.glob(os.path.join(RUNS, pattern))):
        try:
            m = json.load(open(os.path.join(d, "meta.json")))
            met = json.load(open(os.path.join(d, "metrics.json")))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        e.append(met["rel_l2"])
        f.append(float(m.get("final_total_loss", np.inf)) > FAIL_LOSS)
    return np.array(e), np.array(f, bool)


def med(pat):
    e, _ = load_cfg(pat)
    return float(np.median(e)) if len(e) else None


def mean(pat):
    e, _ = load_cfg(pat)
    return float(e.mean()) if len(e) else None


def sd(pat):
    e, _ = load_cfg(pat)
    return float(e.std(ddof=1)) if len(e) else None


def worst(pat):
    e, _ = load_cfg(pat)
    return float(e.max()) if len(e) else None


def nfail(pat):
    _, f = load_cfg(pat)
    return int(f.sum())


# ==========================================================================

def main():
    print("verifying PAPER.md against the artefacts\n")

    # ---- section 4: parameter counts -------------------------------------
    counts = {}
    for tag, name in [("conv_classical_s0", "classical"), ("conv_ring_L2_s0", "ring"),
                      ("init_strong_L2_s0", "strong"), ("qc_qcpinn_s0", "qcpinn"),
                      ("qc_model1_s0", "model1"), ("qc_model2_s0", "model2")]:
        p = os.path.join(RUNS, tag, "meta.json")
        if os.path.exists(p):
            counts[name] = json.load(open(p))["param_counts"]
    for name, want in [("classical", 3021), ("ring", 2649), ("strong", 2665),
                       ("qcpinn", 771), ("model1", 7851), ("model2", 2751)]:
        if name in counts:
            check_int(f"S4  params {name}", counts[name]["total"], want)

    # ---- section 8.1: fixed-budget study ---------------------------------
    s1 = {"classical": ["classical_baseline"] + [f"seed_classical_s{i}" for i in (1, 2, 3, 4)],
          "ring": ["reupload_L2_q4"] + [f"seed_ring_L2_s{i}" for i in (1, 2, 3, 4)],
          "strong": ["ansatz_strong_L2_q4"] + [f"seed_strong_L2_s{i}" for i in (1, 2, 3, 4)]}
    want_s1 = {"classical": (0.0357, 0.0805, 0.0983), "ring": (0.1953, 0.2427, 0.1128),
               "strong": (0.1094, 0.1747, 0.1443)}
    for k, tags in s1.items():
        vals = []
        for t in tags:
            p = os.path.join(RUNS, t, "metrics.json")
            if os.path.exists(p):
                vals.append(json.load(open(p))["rel_l2"])
        if len(vals) == 5:
            a = np.array(vals)
            m, mu, s = want_s1[k]
            check(f"S8.1 {k} median", float(np.median(a)), m)
            check(f"S8.1 {k} mean", float(a.mean()), mu)
            check(f"S8.1 {k} sd", float(a.std(ddof=1)), s)

    # ---- section 8.2: classical init control -----------------------------
    for pat, lbl, (m, mu, s, w, nf) in [
            ("ctrl_classical_g0.5_s*",    "gain0.5", (0.0990, 0.1996, 0.1864, 0.4501, 3)),
            ("conv_classical_s*",         "gain1.0", (0.0221, 0.0344, 0.0292, 0.0929, 0)),
            ("ctrl_classical_g1.7321_s*", "Xavier",  (0.0140, 0.0153, 0.0040, 0.0227, 0)),
            ("ctrl_classical_g2.0_s*",    "gain2.0", (0.0160, 0.0195, 0.0081, 0.0342, 0)),
            ("ctrl_classical_g2.8868_s*", "Xav*tanh",(0.0159, 0.0161, 0.0039, 0.0223, 0))]:
        check(f"S8.2 {lbl} median", med(pat), m)
        check(f"S8.2 {lbl} mean", mean(pat), mu)
        check(f"S8.2 {lbl} sd", sd(pat), s)
        check(f"S8.2 {lbl} worst", worst(pat), w)
        check_int(f"S8.2 {lbl} failures", nfail(pat), nf)

    # the two ratios the paper quotes
    check("S8.2 sd ratio 1.0/2.0", sd("conv_classical_s*") / sd("ctrl_classical_g2.0_s*"),
          3.63, tol=0.01)
    check("S8.2 sd ratio 0.5/2.0", sd("ctrl_classical_g0.5_s*") / sd("ctrl_classical_g2.0_s*"),
          23.13, tol=0.05)
    check("S8.2 max ratio 0.5/2.0", worst("ctrl_classical_g0.5_s*") / worst("ctrl_classical_g2.0_s*"),
          13.17, tol=0.02)
    # the literature-standard arm against the PyTorch default the study calls "default"
    check("S8.2 sd ratio default/Xavier",
          sd("conv_classical_s*") / sd("ctrl_classical_g1.7321_s*"), 7.27, tol=0.05)
    check_int("S8.2 Xavier beats the PyTorch default on median",
              int(med("ctrl_classical_g1.7321_s*") < med("conv_classical_s*")), 1)

    # ---- section 8.3: the interaction ------------------------------------
    for pat, lbl, (m, mu, s, w, nf) in [
            ("conv_strong_L2_s*", "strong default",  (0.0333, 0.0774, 0.1099, 0.3386, 2)),
            ("init_strong_L2_s*", "strong near-id",  (0.0285, 0.0347, 0.0216, 0.0852, 0)),
            ("conv_ring_L2_s*",   "ring default",    (0.0373, 0.0544, 0.0368, 0.1335, 0)),
            ("init_ring_L2_s*",   "ring near-id",    (0.1442, 0.1548, 0.1194, 0.3745, 3))]:
        check(f"S8.3 {lbl} median", med(pat), m)
        check(f"S8.3 {lbl} mean", mean(pat), mu)
        check(f"S8.3 {lbl} sd", sd(pat), s)
        check(f"S8.3 {lbl} worst", worst(pat), w)
        check_int(f"S8.3 {lbl} failures", nfail(pat), nf)

    # the retracted inversion: strong must lead ring on the MEDIAN at both inits
    check_int("S8.3 strong leads ring on median at default",
              int(med("conv_strong_L2_s*") < med("conv_ring_L2_s*")), 1)
    check_int("S8.3 strong leads ring on median at near-id",
              int(med("init_strong_L2_s*") < med("init_ring_L2_s*")), 1)
    # and the mean must invert at default -- that is the artifact being documented
    check_int("S8.3 mean DOES invert at default (the artifact)",
              int(mean("conv_ring_L2_s*") < mean("conv_strong_L2_s*")), 1)

    # ---- section 8.4: budget curve ---------------------------------------
    want_bc = {"bc_classical_s*": (0.4903, 0.4338, 0.0648, 0.0208),
               "bc_strong_L2_s*": (0.5211, 0.4681, 0.1680, 0.0255),
               "bc_ring_L2_s*":   (0.6587, 0.4874, 0.3811, 0.0543)}
    for pat, wants in want_bc.items():
        for c, wv in zip([800, 2000, 8000, None], wants):
            fn = "metrics.json" if c is None else f"metrics_ep{c}.json"
            vals = [json.load(open(os.path.join(d, fn)))["rel_l2"]
                    for d in sorted(glob.glob(os.path.join(RUNS, pat)))
                    if os.path.exists(os.path.join(d, fn))]
            if vals:
                check(f"S8.4 {pat.split('_s')[0]} @{c or 'conv'}",
                      float(np.median(vals)), wv)

    # The budget-curve convergence column must equal the independent runs.
    # Restricted to seeds 0-4: bc_* was NOT extended by seeds_extend.sh, so once the
    # continuous arms reach n = 8 these compare a 5-sample median against an 8-sample
    # one and differ for a reason that has nothing to do with agreement. The identity
    # being tested is per-seed, so the fix is to compare the seeds both arms have --
    # not to widen the tolerance until it passes.
    check("S8.4 == S8.3 classical (seeds 0-4)",
          med("bc_classical_s[0-4]"), med("conv_classical_s[0-4]"))
    check("S8.4 == S8.3 ring (seeds 0-4)",
          med("bc_ring_L2_s[0-4]"), med("conv_ring_L2_s[0-4]"))
    # tol=1e-12, not the 0.02 this once carried: bc_* comes from pde_train.py and
    # init_*/conv_* from robust_train.py, and the two front-ends are numerically
    # identical -- every seed of all three architectures agrees bit for bit. A
    # loose tolerance here would hide a divergence between them, which is the one
    # thing this cross-check exists to catch.
    check("S8.4 == S8.3 strong (seeds 0-4)",
          med("bc_strong_L2_s[0-4]"), med("init_strong_L2_s[0-4]"), tol=1e-12)

    # ---- section 8.5: heat -----------------------------------------------
    for pat, lbl, (m, mu, s, w) in [
            ("heat_classical_s*", "classical", (0.0033, 0.0033, 0.0012, 0.0057)),
            ("heat_strong_L2_s*", "strong",    (0.0048, 0.0051, 0.0019, 0.0089)),
            ("heat_ring_L2_s*",   "ring",      (0.0067, 0.0072, 0.0029, 0.0118))]:
        check(f"S8.5 heat {lbl} median", med(pat), m)
        check(f"S8.5 heat {lbl} mean", mean(pat), mu)
        check(f"S8.5 heat {lbl} sd", sd(pat), s)
        check(f"S8.5 heat {lbl} worst", worst(pat), w)
        check_int(f"S8.5 heat {lbl} failures", nfail(pat), 0)

    # the ordering must be identical on both PDEs (the section's claim)
    burg = [med("conv_classical_s*"), med("init_strong_L2_s*"), med("conv_ring_L2_s*")]
    heat = [med("heat_classical_s*"), med("heat_strong_L2_s*"), med("heat_ring_L2_s*")]
    check_int("S8.5 median ordering identical on both PDEs",
              int(np.argsort(burg).tolist() == np.argsort(heat).tolist()), 1)

    # ---- section 8.6: QCPINN ---------------------------------------------
    for pat, lbl, (m, mu, s, b, w, nf) in [
            ("qc_qcpinn_s*", "QCPINN",  (0.0815, 0.0853, 0.0641, 0.0243, 0.1898, 2)),
            ("qc_model2_s*", "Model-2", (0.0558, 0.0524, 0.0064, 0.0414, 0.0565, 1)),
            ("qc_model1_s*", "Model-1", (0.0125, 0.0136, 0.0023, 0.0118, 0.0173, 0))]:
        e, _ = load_cfg(pat)
        check(f"S8.6 {lbl} median", med(pat), m)
        check(f"S8.6 {lbl} mean", mean(pat), mu)
        check(f"S8.6 {lbl} sd", sd(pat), s)
        check(f"S8.6 {lbl} best", float(e.min()) if len(e) else None, b)
        check(f"S8.6 {lbl} worst", worst(pat), w)
        check_int(f"S8.6 {lbl} failures", nfail(pat), nf)

    # the 80,000-epoch convergence control of section 8.6
    for pat, lbl, (m, mu, s, b, w, nf) in [
            ("qcc_qcpinn_s*", "ctrl QCPINN",  (0.0535, 0.0698, 0.0370, 0.0303, 0.1119, 2)),
            ("qcc_model2_s*", "ctrl Model-2", (0.0519, 0.0481, 0.0113, 0.0291, 0.0565, 1)),
            ("qcc_model1_s*", "ctrl Model-1", (0.0127, 0.0129, 0.0017, 0.0116, 0.0158, 0))]:
        e_, _ = load_cfg(pat)
        check(f"S8.6 {lbl} median", med(pat), m)
        check(f"S8.6 {lbl} sd", sd(pat), s)
        check(f"S8.6 {lbl} best", float(e_.min()) if len(e_) else None, b)
        check_int(f"S8.6 {lbl} failures", nfail(pat), nf)
    # the claim the withdrawal rests on: still 2/5 at 3.2x the budget, and NOT capped
    check_int("S8.6 QCPINN still 2/5 non-converged at 80k", nfail("qcc_qcpinn_s*"), 2)
    capped = sum(1 for d in sorted(glob.glob(os.path.join(RUNS, "qcc_qcpinn_s*")))
                 if json.load(open(os.path.join(d, "meta.json"))).get("stop_reason")
                 == "max_epochs")
    check_int("S8.6 no control run hit the 80k cap (they plateaued)", capped, 0)

    # all five QCPINN runs must be budget-truncated -- the section's stated caveat
    trunc = sum(1 for d in sorted(glob.glob(os.path.join(RUNS, "qc_qcpinn_s*")))
                if json.load(open(os.path.join(d, "meta.json"))).get("stop_reason")
                == "max_epochs")
    check_int("S8.6 QCPINN runs budget-truncated", trunc, 5)

    # ---- section 5.2: Fourier ceiling ------------------------------------
    if os.path.exists("fourier_entangler_results.csv"):
        rows = list(csv.DictReader(open("fourier_entangler_results.csv")))
        got = {(int(r["reupload_depth"]), r["entangler"]): int(r["measured_cutoff"])
               for r in rows}
        for (L, ent), wv in [((8, "ring"), 9), ((8, "strong"), 15),
                             ((6, "ring"), 8), ((6, "strong"), 12),
                             ((4, "ring"), 6), ((4, "strong"), 8),
                             ((2, "ring"), 4), ((2, "strong"), 4)]:
            if (L, ent) in got:
                check_int(f"S5.2 ceiling L={L} {ent}", got[(L, ent)], wv)

    # ---- section 8.8: gradient variance fits -----------------------------
    if os.path.exists("gradient_variance_results.json"):
        g = json.load(open("gradient_variance_results.json"))
        fits = g.get("global_fits", {})
        for k, (sl, se) in [("global", (-1.414, 0.103)), ("local", (-0.994, 0.060)),
                            ("pde", (-0.891, 0.095)), ("ic", (-0.622, 0.050))]:
            if k in fits:
                check(f"S8.8 {k} slope", fits[k]["slope"], sl, tol=1e-3)
                check(f"S8.8 {k} se", fits[k]["se"], se, tol=1e-3)
        # every global_sweep row must share one settings block
        sigs = {json.dumps(r.get("settings"), sort_keys=True)
                for r in g.get("global_sweep", [])}
        check_int("S8.8 global_sweep settings uniform", len(sigs), 1)
        # the separation claims
        if {"global", "local", "pde", "ic"} <= set(fits):
            def z(a, b):
                return ((fits[a]["slope"] - fits[b]["slope"])
                        / np.hypot(fits[a]["se"], fits[b]["se"]))
            check("S8.8 z(global,pde)", float(z("global", "pde")), -3.75, tol=0.02)
            check("S8.8 z(local,pde)", float(z("local", "pde")), -0.92, tol=0.02)
            check("S8.8 z(pde,ic)", float(z("pde", "ic")), -2.52, tol=0.02)
            check_int("S8.8 pde NOT separated from local",
                      int(abs(z("local", "pde")) < 1.96), 1)
            check_int("S8.8 pde separated from global",
                      int(abs(z("global", "pde")) > 1.96), 1)

    # ---- section 8.7: per-mode -------------------------------------------
    if os.path.exists("mode_decomposition_summary.json"):
        s = json.load(open("mode_decomposition_summary.json"))["summary"]
        want = {"heat_classical": {"1": 0.0011, "3": 0.0020, "5": 0.0038},
                "heat_ring_L2":   {"1": 0.0018, "3": 0.0038, "5": 0.0037},
                "heat_strong_L2": {"1": 0.0010, "3": 0.0036, "5": 0.0031}}
        for cfg, modes in want.items():
            if cfg in s:
                for k, wv in modes.items():
                    check(f"S8.7 {cfg} k={k}", s[cfg]["per_mode"][k]["mean"], wv, tol=1e-4)
        # the section's claim: every mode tracked to better than 1% of a_k
        allmax = max(s[c]["per_mode"][k]["mean"] for c in s for k in s[c]["per_mode"])
        check_int("S8.7 every mode tracked better than 1% of a_k",
                  int(allmax < 0.01), 1)
        # and the negative result: no k5/k1 separation vs classical
        base = s["heat_classical"]["mode5_over_mode1_err"]
        for cfg in ("heat_ring_L2", "heat_strong_L2"):
            r = s[cfg]["mode5_over_mode1_err"]
            gap = abs(r["mean"] - base["mean"])
            check_int(f"S8.7 {cfg} k5/k1 NOT separated",
                      int(gap <= r["std"] + base["std"]), 1)

    # ---- section 7.2: the permutation floor ------------------------------
    # Pure combinatorics, so this is checked by enumeration rather than against
    # a stored artefact. The paper's table is transcribed here; floor_analysis
    # derives it. A disagreement means the table was edited by hand.
    #
    #   n  splits   median-diff numerator   mean-diff numerator
    FLOOR_TABLE = [(4, 70, 4, 2), (5, 252, 12, 2), (6, 924, 12, 2),
                   (7, 3432, 40, 2), (8, 12870, 40, 2)]
    for n, splits, num_med, num_mean in FLOOR_TABLE:
        cm, tot, pm = fa.attainable_floor(n, "median")
        ca, _, pa = fa.attainable_floor(n, "mean")
        check_int(f"S7.2 splits n={n}", tot, splits)
        check_int(f"S7.2 median-diff numerator n={n}", cm, num_med)
        check_int(f"S7.2 mean-diff numerator n={n}", ca, num_mean)
        check(f"S7.2 median-diff floor n={n}", pm, num_med / splits, tol=1e-12)
        check(f"S7.2 mean-diff floor n={n}", pa, num_mean / splits, tol=1e-12)

    # the threshold the table is read against, and the sample sizes it implies
    FAMILY = 12
    thr = fa.holm_smallest_threshold(FAMILY)
    check("S7.2 Holm threshold for the most significant of 12", thr, 0.0042)
    check_int("S7.2 smallest n clearing Holm, median-diff",
              fa.smallest_n_clearing("median", FAMILY), 8)
    check_int("S7.2 smallest n clearing Holm, mean-diff",
              fa.smallest_n_clearing("mean", FAMILY), 6)
    # the claim that carries 7.2: at n = 5 NEITHER statistic clears the
    # threshold, so the null result cannot be attributed to the choice of test
    check_int("S7.2 median-diff floor at n=5 exceeds Holm threshold",
              int(fa.attainable_floor(5, "median")[2] > thr), 1)
    check_int("S7.2 mean-diff floor at n=5 exceeds Holm threshold",
              int(fa.attainable_floor(5, "mean")[2] > thr), 1)
    # and the two floors the abstract and 11 quote to 4 dp
    check("S7.2 primary (median-diff) floor at n=5 quoted as 0.0476",
          fa.attainable_floor(5, "median")[2], 0.0476)
    check("S7.2 secondary (mean-diff) floor at n=5 quoted as 0.0079",
          fa.attainable_floor(5, "mean")[2], 0.0079)
    # the unequal-n extension must reduce to the equal-n function the paper's
    # table is generated from, or the two are not the same construction
    for n in range(4, 9):
        for stat in ("median", "mean"):
            check_int(f"S7.2 unequal-n floor reduces to equal-n, n={n} {stat}",
                      int(fa.attainable_floor_unequal(n, n, stat)
                          == fa.attainable_floor(n, stat)), 1)

    # ---- section 8.9: statistics -----------------------------------------
    if os.path.exists("stats_report.json"):
        st = json.load(open("stats_report.json"))

        # n is read from the data, not assumed. It is also NOT assumed to be
        # the same for every arm: the seed extension takes the ten continuous
        # arms to n = 8 while the three QCPINN arms stay at n = 5, so H10
        # (classical_g1.0 vs qc_model2_2751) becomes an 8-against-5 comparison
        # whose floor, 20/1287 = 0.0155, is neither column of the 7.2 table.
        # After the extension one Holm family therefore spans three floors.
        arm_ns = {k: len(v["values"]) for k, v in st["descriptives"].items()}
        homogeneous = len(set(arm_ns.values())) == 1
        n_arm = next(iter(arm_ns.values())) if homogeneous else None
        floor_med = fa.attainable_floor(n_arm, "median")[2] if homogeneous else None
        floor_mean = fa.attainable_floor(n_arm, "mean")[2] if homogeneous else None

        # stats_analysis.permutation_test is called with stat="median", so the
        # floor governing every p in this report is the median-diff floor.
        # p_floor_n5 stores the mean-diff floor; it is checked here under its
        # real name so that it cannot be mistaken for the operative one again.
        if "p_floor_n5" in st:
            check("S7.2 stats_report p_floor_n5 is the MEAN-diff floor",
                  st["p_floor_n5"], 2 / 252, tol=1e-9)
        if homogeneous and "p_floor_median" in st:
            check("S7.2 stats_report p_floor_median matches enumeration",
                  st["p_floor_median"], floor_med, tol=1e-9)
        if homogeneous and "p_floor_mean" in st:
            check("S7.2 stats_report p_floor_mean matches enumeration",
                  st["p_floor_mean"], floor_mean, tol=1e-9)

        # the consistency requirement: an exact test cannot return a p below
        # its own floor. If this fails, the reported p-values and the quoted
        # floor were produced by different statistics. Computed PER COMPARISON
        # from that comparison's own two arm sizes, so it stays correct when
        # the family spans several n.
        _fcache = {}
        def _floor(n1, n2, stat="median"):
            key = (n1, n2, stat)
            if key not in _fcache:
                _fcache[key] = fa.attainable_floor_unequal(n1, n2, stat)[2]
            return _fcache[key]

        below = []
        for c in st["comparisons"]:
            n1, n2 = arm_ns.get(c["a"]), arm_ns.get(c["b"])
            if n1 is None or n2 is None:
                continue
            if c["p_permutation"] < _floor(n1, n2) - 1e-12:
                below.append(f"{c['label'][:20]} p={c['p_permutation']:.5f} "
                             f"< floor({n1},{n2})={_floor(n1, n2):.5f}")
        check_int("S8.9 no reported p lies below its own comparison's floor",
                  len(below), 0)
        if below:
            print("    " + "; ".join(below))

        check_int("S8.9 declared location comparisons (H8/H9 withdrawn)",
                  len(st["comparisons"]), 10)
        check_int("S8.9 H8 and H9 are absent from the family",
                  sum(1 for c in st["comparisons"]
                      if c["label"].startswith(("H8 ", "H9 "))), 0)

        # THE HEADLINE. The paper declares ONE family of twenty -- the ten location
        # comparisons plus the ten spread tests of 8.2 -- and reports the single
        # survivor. Computed here rather than read from the artefact, because the
        # artefact corrects the location tests alone.
        fam = [c["p_permutation"] for c in st["comparisons"]]
        fam += [r["p_sd"] for r in st["spread"]] + [r["p_max"] for r in st["spread"]]
        check_int("S8.9 declared family is twenty tests", len(fam), 20)
        _adj, _sig = fa.holm(fam)
        check_int("S8.9 exactly one test survives the family of twenty",
                  int(sum(_sig)), 1)
        check("S8.9 survivor Holm-adjusted p", float(min(_adj)), 0.0435, tol=5e-4)
        # and the permissive partition, which the paper also discloses
        _, s_loc = fa.holm([c["p_permutation"] for c in st["comparisons"]])
        check_int("S8.9 two survive if location is its own family of ten",
                  int(sum(s_loc)), 2)

        nsig = sum(1 for c in st["comparisons"] if c["significant_holm_0.05"])
        check_int("S8.9 location-family survivors", nsig, 2)
        byl = {c["label"][:3]: c for c in st["comparisons"]}
        for key, d, p in [("H2 ", 0.91, 0.0031), ("H7 ", -0.59, 0.0909),
                          ("H11", -0.66, 0.0796), ("H12", -0.88, 0.0031)]:
            if key in byl:
                check(f"S8.9 {key.strip()} delta", byl[key]["cliffs_delta"], d, tol=0.01)
                check(f"S8.9 {key.strip()} p", byl[key]["p_permutation"], p, tol=0.002)
        # power: 80% at n=50 for the observed failure-rate contrast
        pw = st["power"].get("observed 1/5 vs 0/5 (p=0.20 vs 0.00)", {})
        if pw:
            check_int("S7.4 power at n=5 is ~0.01", int(pw.get("5", 1) <= 0.03), 1)
            check_int("S7.4 power reaches 80% by n=50", int(pw.get("50", 0) >= 0.8), 1)

    # ---- section 10.5: throughput ----------------------------------------
    # The three measured speedups and the compounded figure the section quotes.
    # Internal arithmetic only -- the timings themselves are measurements, but
    # a transcription error in the product is still a transcription error.
    check("S10.5 compounded speedup", 1.22 * 1.15 * 3.0, 4.2, tol=0.05)

    # ---- section 10: artefact count --------------------------------------
    # The reported count excludes the in-progress L = 4 depth-robustness arms: they
    # are incomplete, and no reported statistic uses them.
    used = [d for d in glob.glob(os.path.join(RUNS, "*"))
            if "_L4_" not in d and os.path.exists(os.path.join(d, "metrics.json"))]
    check_int("S10 run directories (excluding in-progress L=4)", len(used), 257)

    # ---- SECTION 6: the symmetric control --------------------------------
    # Both arms from the authors' unmodified code under one protocol. These are
    # the numbers Section VI of the NQComp manuscript quotes; they are re-derived
    # here from runs/ so a transcription slip cannot survive a verify run.
    # The exact tests come from stats_analysis.py, not reimplemented.
    import json as _json
    from stats_analysis import permutation_test as _perm, cliffs_delta as _cd

    def _sym(arm):
        return [_json.load(open(f"{RUNS}/sym4_{arm}_s{i}/replication.json"))["err_u_pct"]
                for i in range(10)]

    try:
        _q, _c = _sym("qcpinn"), _sym("model2")
    except (FileNotFoundError, OSError):
        _q = _c = None

    if _q and _c:
        check_int("SYM symmetric control: quantum runs", len(_q), 10)
        check_int("SYM symmetric control: classical runs", len(_c), 10)
        check("SYM quantum median", float(np.median(_q)), 1.73, tol=5e-3)
        check("SYM classical median", float(np.median(_c)), 4.30, tol=5e-3)
        check_int("SYM quantum failures (>50%)", sum(1 for v in _q if v > 50), 0)
        check_int("SYM classical failures (>50%)", sum(1 for v in _c if v > 50), 3)
        _d, _p, _n = _perm(_q, _c, stat="median")
        check_int("SYM permutation splits C(20,10)", int(_n), 184756)
        check("SYM permutation p (all seeds)", float(_p), 0.0068, tol=5e-5)
        check("SYM Cliff delta (all seeds)", float(_cd(_q, _c)[0]), -0.64, tol=5e-3)
        # sensitivity: the seven seeds where the classical arm trains at all
        _keep = [i for i in range(10) if _c[i] <= 50]
        _qs, _cs = [_q[i] for i in _keep], [_c[i] for i in _keep]
        check_int("SYM non-failing seeds", len(_keep), 7)
        check("SYM quantum median (non-failing)", float(np.median(_qs)), 1.77, tol=5e-3)
        check("SYM classical median (non-failing)", float(np.median(_cs)), 3.43, tol=5e-3)
        _d2, _p2, _ = _perm(_qs, _cs, stat="median")
        check("SYM permutation p (non-failing)", float(_p2), 0.0583, tol=5e-5)
        # our own implementation of the SAME 2,751-parameter classical network
        _pm = [_json.load(open(f"{RUNS}/pm_model2_ours_s{i}/metrics.json"))["rel_l2"] * 100
               for i in range(8)]
        check("SYM our classical implementation median", float(np.median(_pm)), 0.70,
              tol=5e-3)
        check_int("SYM our classical implementation failures", sum(1 for v in _pm if v > 50), 0)

    # ---- CITATIONS: every marker resolves, every entry is used ------------
    # The manuscript shipped through eight revisions with no reference list at all
    # and every source named only descriptively. Now that the list exists, the
    # failure mode shifts to drift -- a [N] marker with no entry, an entry nothing
    # cites, or a gap in the numbering after an edit. All three are mechanical.
    if os.path.exists("PAPER.md"):
        import re as _re
        ptxt = open("PAPER.md", encoding="utf-8").read()
        if "## References" in ptxt:
            body, refs = ptxt.split("## References", 1)
            refs = refs.split("## Appendix A", 1)[0]
            cited = {int(m) for m in _re.findall(r"\[(\d+)\]", body)}
            listed = {int(m) for m in _re.findall(r"^\[(\d+)\]", refs, _re.M)}
            check_int("CITE every [n] marker has a reference entry",
                      len(cited - listed), 0)
            check_int("CITE every reference entry is cited in the text",
                      len(listed - cited), 0)
            check_int("CITE numbering is contiguous from 1",
                      int(listed == set(range(1, len(listed) + 1))), 1)
            check_int("CITE reference list is non-empty", int(len(listed) > 0), 1)
            if cited - listed:
                print(f"    unresolved markers: {sorted(cited - listed)}")
            if listed - cited:
                print(f"    uncited entries: {sorted(listed - cited)}")

    # ---- GENERATED TABLES: rendered from runs/, not typed ------------------
    # The deepest fix for stale numbers. render_tables.py builds each data table
    # from the artefacts in the exact markdown PAPER.md uses; if what is in the
    # paper differs, the paper is stale by construction rather than by opinion.
    # This caught 8.3's inversion table and Appendices B.1 and B.2 still carrying
    # n = 5 numbers a full revision after the arms went to n = 8.
    try:
        import render_tables as rt
    except ImportError:
        rt = None
    if rt is not None and os.path.exists("PAPER.md"):
        paper_txt = open("PAPER.md", encoding="utf-8").read()
        for tname in rt.TABLES:
            loc = rt.locate(paper_txt, tname)
            if loc is None:
                check_int(f"TABLE {tname} found in PAPER.md", 0, 1)
                continue
            rendered = rt.TABLES[tname][2]()
            check_int(f"TABLE {tname} matches the artefacts",
                      int(loc[3] == rendered), 1)

    # ---- FRESHNESS: figures must be newer than the data they plot ---------
    # The seed extension changed every arm; the figures were not regenerated and
    # kept plotting n = 5 for a full revision. Nothing in this script noticed,
    # because it only ever compared PAPER.md against runs/. A figure older than
    # the newest run it could be plotting is stale by construction.
    newest_run = 0.0
    for d in glob.glob(os.path.join(RUNS, "*")):
        p = os.path.join(d, "metrics.json")
        if "_L4_" not in d and os.path.exists(p):
            newest_run = max(newest_run, os.path.getmtime(p))
    DATA_FIGURES = ["fig1_trajectories", "fig2_init_gain", "fig3_interaction",
                    "fig4_budget_curve", "fig5_cross_pde"]
    for stem in DATA_FIGURES:
        for d in ("figures", os.path.join("paper_latex", "figures")):
            f = os.path.join(d, stem + ".pdf")
            if os.path.exists(f):
                check_int(f"FRESH {d}/{stem}.pdf newer than newest run",
                          int(os.path.getmtime(f) >= newest_run), 1)

    # ---- CAPTIONS: they live in build.py, not in PAPER.md -----------------
    # fig 2's caption still said "2/5 to 0/5" after the extension made it 3/8
    # to 0/8, and no check looked at it, because captions are not in PAPER.md.
    cap = os.path.join("paper_latex", "build.py")
    if os.path.exists(cap):
        src = open(cap, encoding="utf-8").read()
        nf05 = nfail("ctrl_classical_g0.5_s*")
        n05 = len(load_cfg("ctrl_classical_g0.5_s*")[0])
        nf20 = nfail("ctrl_classical_g2.0_s*")
        n20 = len(load_cfg("ctrl_classical_g2.0_s*")[0])
        check_int(f"CAPTION fig2 states the current failure counts "
                  f"({nf05}/{n05} to {nf20}/{n20})",
                  int(f"{nf05}/{n05} to {nf20}/{n20}" in src), 1)
        sdr = sd("ctrl_classical_g0.5_s*") / sd("ctrl_classical_g2.0_s*")
        check_int(f"CAPTION fig2 states the current sd ratio ({sdr:.1f})",
                  int(f"{sdr:.1f}" in src), 1)
        check_int("CAPTION no figure caption still says 'five seeds'",
                  src.count("five seeds"), 0)

    # ---- CROSS-REFERENCE: sections must agree with each other -------------
    # 8.4 claimed its converged column "reproduces the independent runs of 8.3
    # exactly", quoting 2.08 / 5.43 / 2.55 -- true per-seed, but 8.3's medians
    # had moved to 2.21 / 3.73 / 2.85 with the extension, so the sentence was
    # false as written. Every check here passed while the paper contradicted
    # itself, because claims were only ever tested against disk, never against
    # one another. This tests the assertion 8.4 actually makes.
    if os.path.exists("PAPER.md"):
        paper84 = open("PAPER.md", encoding="utf-8").read()
        for lbl, bc, ind in [("classical", "bc_classical_s[0-4]", "conv_classical_s[0-4]"),
                             ("ring", "bc_ring_L2_s[0-4]", "conv_ring_L2_s[0-4]"),
                             ("strong", "bc_strong_L2_s[0-4]", "init_strong_L2_s[0-4]")]:
            v = med(bc)
            if v is None:
                continue
            check(f"XREF 8.4 {lbl} == 8.3 {lbl} over the shared seeds", v, med(ind),
                  tol=1e-12)
            check_int(f"XREF 8.4 quotes {lbl} as {v*100:.2f}%",
                      int(f"{v*100:.2f}%" in paper84), 1)
        # and the claim is explicitly scoped, so a future edit cannot silently
        # restore the unqualified version
        check_int("XREF 8.4 scopes its consistency claim to the shared seeds",
                  int("over the five seeds the two studies share" in paper84), 1)

    # ---- Revision 3 prose guard ------------------------------------------
    # The six passages patch_paper.py rewrote. Numbers are checked above; this
    # checks that the sentences carrying them are still the Revision 3 ones, so
    # a reverted edit or a restored backup fails here rather than silently
    # reintroducing 2/252 as the operative floor or n = 50 as the global
    # requirement.
    mods = []
    for rev, modname in (("R3", "patch_paper"), ("R4", "patch_paper_rev4"),
                         ("R5", "patch_paper_rev5"), ("R6", "patch_paper_rev6"),
                         ("R7", "patch_paper_rev7"), ("R8", "patch_paper_rev8"),
                         ("R9", "patch_paper_rev9"),
                         ("R10", "patch_paper_rev10"),
                         ("R11", "patch_paper_rev11"),
                         ("R12", "patch_paper_rev12")):
        try:
            mods.append((rev, __import__(modname)))
        except ImportError:
            pass
    if mods and os.path.exists("PAPER.md"):
        paper = open("PAPER.md", encoding="utf-8").read()
        # A later revision may consume an earlier revision's replacement text as part
        # of its own anchor -- not necessarily as the whole of it, so this tests
        # substring containment, not equality. Superseded passages are gone by design
        # and asserting their presence would fail on correct input.
        for i, (rev, mod) in enumerate(mods):
            later_olds = [o for _r, m in mods[i + 1:] for _n, o, _x in m.EDITS]
            for name, old, new in mod.EDITS:
                # Overlap runs both ways: a later edit may swallow this passage whole,
                # or may rewrite a fragment inside it. Either way the exact Revision-N
                # text is legitimately gone.
                if any(new in o or o in new for o in later_olds):
                    continue
                check_int(f"{rev} present: {name}", paper.count(new), 1)
                # Some edits INSERT around their anchor, so the anchor survives inside
                # the replacement by design. Asserting its absence is only meaningful
                # for edits that genuinely retire their old text.
                if old not in new:
                    check_int(f"{rev} retired: {name}", paper.count(old), 0)

    # ---- report ----------------------------------------------------------
    bad = [c for c in _checks if not c[3]]
    print(f"{'claim':52}{'paper':>12}{'disk':>12}  status")
    print("-" * 88)
    for label, got, want, ok, _u in _checks:
        gs = f"{got:.6g}" if isinstance(got, float) else str(got)
        ws = f"{want:.6g}" if isinstance(want, float) else str(want)
        print(f"{label:52}{ws:>12}{gs:>12}  {'ok' if ok else 'MISMATCH'}")

    print("-" * 88)
    print(f"{len(_checks) - len(bad)}/{len(_checks)} claims verified")
    if bad:
        print(f"\n{len(bad)} MISMATCH(ES) -- PAPER.md disagrees with the artefacts:")
        for label, got, want, _ok, _u in bad:
            print(f"  {label}: paper says {want}, disk says {got}")
        return 1
    print("\nEvery quantitative claim in PAPER.md matches the artefacts on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
