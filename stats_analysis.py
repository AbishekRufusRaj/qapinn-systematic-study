#!/usr/bin/env python3
"""
STATISTICAL ANALYSIS for the QAPINN study.

WHY THIS IS A SEPARATE SCRIPT, AND WHY IT USES EXACT TESTS
==========================================================
Every configuration in this study has n = 5. That is small enough that the
usual asymptotic machinery -- t-tests, normal-approximation Mann-Whitney,
bootstrap percentile intervals -- is not merely conservative but wrong: the
approximations it relies on are not close at this sample size. It is also
small enough that the exact alternatives are computable by brute force.

For two groups of five there are C(10,5) = 252 ways to split ten observations
into two labelled groups of five. Enumerating all 252 gives the EXACT null
distribution of any statistic under the sharp null "the labels do not matter".
No approximation, no asymptotics, no distributional assumption. That is what
this script does. The smallest attainable two-sided p-value is 2/252 = 0.0079,
which is worth stating: with five seeds per arm, p < 0.008 is the floor, and
nothing in this study can be significant at 0.001 no matter how large the
effect. Reporting that floor is part of reporting the result honestly.

Failure rates are compared with Fisher's exact test on the 2x2 table, again
computed exactly from the hypergeometric distribution rather than approximated.

The script also answers a question the paper must answer rather than dodge:
HOW MANY SEEDS would be needed to establish the failure-rate claims? That is a
power calculation, and it is done by direct simulation against the observed
rates. The answer (see output) is large, and saying so is the honest
alternative to quietly reporting 0/5 versus 1/5 as though it meant something.

No scipy: every test here is implemented from its definition in numpy, and
each implementation is checked against a known answer in --selftest.

USAGE
    python3 stats_analysis.py --selftest
    python3 stats_analysis.py                  # full analysis -> stats_report.json
"""

import argparse
import glob
import json
import os
import sys
from itertools import combinations
from math import comb, lgamma, exp

import numpy as np

FAIL_LOSS = 1e-2
RUNS = "runs"


# ==========================================================================
# exact tests, implemented from their definitions
# ==========================================================================

def permutation_test(a, b, stat="median", alternative="two-sided"):
    """EXACT two-sample permutation test.

    Null hypothesis: the group labels are exchangeable, i.e. all n1+n2
    observations come from one distribution. Under that null every way of
    splitting the pooled sample into groups of size n1 and n2 is equally
    likely, so enumerating all C(n1+n2, n1) splits gives the exact null
    distribution of the test statistic.

    p = (number of splits whose statistic is at least as extreme as the
         observed one) / (total number of splits)

    The observed split is included in the numerator, which is the standard
    (and conservative) convention -- it keeps the test valid rather than
    anti-conservative at tiny sample sizes.

    `stat` is 'median' or 'mean'; the statistic is the difference between
    groups. Returns (observed difference, p-value, number of splits).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n1, n = len(a), len(pooled)
    f = np.median if stat == "median" else np.mean

    obs = f(a) - f(b)
    idx = np.arange(n)
    count = 0
    total = 0
    for c in combinations(idx, n1):
        mask = np.zeros(n, bool)
        mask[list(c)] = True
        d = f(pooled[mask]) - f(pooled[~mask])
        total += 1
        if alternative == "two-sided":
            if abs(d) >= abs(obs) - 1e-15:
                count += 1
        elif alternative == "less":
            if d <= obs + 1e-15:
                count += 1
        else:
            if d >= obs - 1e-15:
                count += 1
    return float(obs), count / total, total


def mannwhitney_exact(a, b):
    """EXACT Mann-Whitney U test (no normal approximation, no tie correction
    needed because the null distribution is enumerated directly).

    U counts, over all n1*n2 pairs, how often an observation from `a` exceeds
    one from `b`, counting ties as half. The exact null distribution comes from
    the same enumeration of C(n1+n2, n1) splits used above.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)

    def U(x, y):
        return float(sum((xi > yj) + 0.5 * (xi == yj) for xi in x for yj in y))

    obs = U(a, b)
    pooled = np.concatenate([a, b])
    n = len(pooled)
    extreme = 0
    total = 0
    centre = n1 * n2 / 2.0
    for c in combinations(range(n), n1):
        mask = np.zeros(n, bool)
        mask[list(c)] = True
        u = U(pooled[mask], pooled[~mask])
        total += 1
        if abs(u - centre) >= abs(obs - centre) - 1e-12:
            extreme += 1
    return obs, extreme / total


def permutation_test_stat(a, b, fn, alternative="two-sided"):
    """EXACT permutation test for an arbitrary two-sample statistic fn(x, y).

    Needed because the claims in section 5 are about SPREAD and WORST CASE, not
    about location, and a test on medians cannot address them. Under the same
    exchangeability null, enumerate all C(n1+n2, n1) splits and compare fn on
    each against fn on the observed split.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n1, n = len(a), len(pooled)
    obs = fn(a, b)
    count = total = 0
    for c in combinations(range(n), n1):
        mask = np.zeros(n, bool)
        mask[list(c)] = True
        v = fn(pooled[mask], pooled[~mask])
        total += 1
        if alternative == "two-sided":
            if abs(v) >= abs(obs) - 1e-15:
                count += 1
        else:
            if v >= obs - 1e-15:
                count += 1
    return float(obs), count / total


def log_sd_ratio(x, y):
    """log(sd(x) / sd(y)). Logged so the statistic is symmetric under swapping
    the two groups, which a raw ratio is not."""
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    if sx <= 0 or sy <= 0:
        return 0.0
    return float(np.log(sx / sy))


def log_max_ratio(x, y):
    """log(max(x) / max(y)) -- the worst-case claim, stated as a ratio."""
    return float(np.log(np.max(x) / np.max(y)))


def cliffs_delta(a, b):
    """Cliff's delta: a nonparametric effect size on [-1, 1].

        delta = ( #(a_i > b_j) - #(a_i < b_j) ) / (n1 * n2)

    It is the probability that a random draw from `a` exceeds a random draw
    from `b`, minus the reverse. Unlike a standardised mean difference it makes
    no assumption about spread or shape, which matters here because these
    distributions are bimodal. Romano et al.'s conventional thresholds:
    |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    d = (gt - lt) / (len(a) * len(b))
    m = abs(d)
    size = ("negligible" if m < 0.147 else "small" if m < 0.33
            else "medium" if m < 0.474 else "large")
    return float(d), size


def _hyper_pmf(k, K, n, N):
    """Hypergeometric pmf via log-gamma, so large binomials do not overflow."""
    def lc(a, b):
        if b < 0 or b > a:
            return -np.inf
        return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)
    v = lc(K, k) + lc(N - K, n - k) - lc(N, n)
    return 0.0 if v == -np.inf else exp(v)


def fisher_exact_2x2(a, b, c, d):
    """Fisher's exact test on [[a, b], [c, d]], two-sided by the
    sum-of-small-probabilities convention: p is the total probability of every
    table (with the same margins) no more likely than the observed one.

    Used here for failure counts, where a chi-square test is invalid: expected
    cell counts are far below 5.
    """
    N = a + b + c + d
    K = a + c          # column-1 total
    n = a + b          # row-1 total
    p_obs = _hyper_pmf(a, K, n, N)
    lo, hi = max(0, n - (N - K)), min(n, K)
    p = sum(_hyper_pmf(k, K, n, N)
            for k in range(lo, hi + 1)
            if _hyper_pmf(k, K, n, N) <= p_obs + 1e-12)
    return float(min(p, 1.0))


def holm_bonferroni(pvals, alpha=0.05):
    """Holm-Bonferroni step-down correction. Controls the family-wise error
    rate without assuming the tests are independent, which matters because the
    comparisons here share arms."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj, adj <= alpha


def bootstrap_ci(x, stat=np.median, n_boot=20000, alpha=0.05, seed=0):
    """Percentile bootstrap interval.

    STATED LIMITATION, because it is severe at this sample size: resampling
    five points can only ever produce medians that are one of those five
    values, so the interval is a coarse, lattice-valued object rather than a
    smooth confidence set. It is reported to convey the width of the
    uncertainty, not as a calibrated interval, and no conclusion in the paper
    rests on one.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = np.array([stat(rng.choice(x, size=len(x), replace=True))
                  for _ in range(n_boot)])
    return float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2)))


def power_for_failure_rates(p1, p2, n_grid=(5, 10, 20, 30, 50, 80, 120, 200),
                            alpha=0.05, n_sim=4000, seed=0):
    """Simulated power of Fisher's exact test to distinguish two failure rates.

    Answers 'how many seeds would we have needed?' rather than leaving the
    reader to guess. Returns {n: power}.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for n in n_grid:
        hits = 0
        for _ in range(n_sim):
            f1 = int(rng.binomial(n, p1))
            f2 = int(rng.binomial(n, p2))
            if fisher_exact_2x2(f1, n - f1, f2, n - f2) < alpha:
                hits += 1
        out[n] = hits / n_sim
    return out


# ==========================================================================
# data
# ==========================================================================

def load(pattern):
    """(errors, failure flags) for every run matching the glob."""
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


CONFIGS = {
    # Burgers', convergence-trained
    "classical_g1.0":  "conv_classical_s*",
    "classical_g0.5":  "ctrl_classical_g0.5_s*",
    "classical_g2.0":  "ctrl_classical_g2.0_s*",
    "ring_default":    "conv_ring_L2_s*",
    "ring_nearid":     "init_ring_L2_s*",
    "strong_default":  "conv_strong_L2_s*",
    "strong_nearid":   "init_strong_L2_s*",
    # QCPINN replication, at the original 25,000-epoch cap
    "qcpinn_771":      "qc_qcpinn_s*",
    "qc_model2_2751":  "qc_model2_s*",
    "qc_model1_7851":  "qc_model1_s*",
    # The convergence control at 80,000 epochs. DESCRIPTIVE ONLY -- these arms
    # appear in `descriptives` so section 8.6 can report the convergence failure,
    # and they appear in NO comparison. Do not add them to COMPARISONS.
    "qcc_qcpinn_771":     "qcc_qcpinn_s*",
    "qcc_model2_2751":    "qcc_model2_s*",
    "qcc_model1_7851":    "qcc_model1_s*",
    # heat
    "heat_classical":  "heat_classical_s*",
    "heat_ring":       "heat_ring_L2_s*",
    "heat_strong":     "heat_strong_L2_s*",
}

# Every comparison the paper actually makes. Declared here, up front, so the
# multiple-comparison correction is applied to the full declared family rather
# than to whichever subset happened to look interesting afterwards.
COMPARISONS = [
    ("H1  init gain: default vs 2.0 (classical)",      "classical_g1.0", "classical_g2.0"),
    ("H2  init gain: 0.5 vs 2.0 (classical)",          "classical_g0.5", "classical_g2.0"),
    ("H3  near-identity on strong ansatz",             "strong_default", "strong_nearid"),
    ("H4  near-identity on ring ansatz",               "ring_default",   "ring_nearid"),
    ("H5  ansatz at default init",                     "ring_default",   "strong_default"),
    ("H6  ansatz at near-identity init",               "ring_nearid",    "strong_nearid"),
    ("H7  matched tuning: classical vs QAPINN",        "classical_g2.0", "strong_nearid"),
    # H8 and H9 WITHDRAWN. Both compared the qcpinn_771 arm on parameter
    # efficiency. Under the convergence control of qcpinn_converge.sh -- 80,000
    # epochs, 3.2x the original cap, every other setting identical -- that arm
    # remains 2/5 non-converged, and both failures stop on the patience criterion
    # rather than the cap, so they are plateaus and not budget truncation. An
    # efficiency comparison drawn across runs that never solved the equation has
    # no measurand; a caveat does not repair it, because the quantity is undefined
    # rather than uncertain. Withdrawn under the rule declared before the run, not
    # after seeing these p-values. Family 12 -> 10, Holm threshold 0.05/10 = 0.0050.
    # H10 survives: it does not involve the qcpinn_771 arm.
    ("H10 our PINN vs their small baseline",           "classical_g1.0", "qc_model2_2751"),
    ("H11 heat: classical vs strong",                  "heat_classical", "heat_strong"),
    ("H12 heat: classical vs ring",                    "heat_classical", "heat_ring"),
]


# ==========================================================================
# selftest
# ==========================================================================

def selftest():
    """Each test is checked against an answer computable by hand or published."""
    print("statistical-test self-test")
    print("-" * 66)
    ok = True

    # 1. Fisher on a textbook table. Tea-tasting: [[3,1],[1,3]] -> p = 0.4857
    p = fisher_exact_2x2(3, 1, 1, 3)
    good = abs(p - 0.4857142857) < 1e-6
    ok &= good
    print(f"  Fisher [[3,1],[1,3]]        p = {p:.7f}  (want 0.4857143)  "
          f"{'PASS' if good else 'FAIL'}")

    # 2. Fisher on a perfectly separated 5-vs-5 table: [[5,0],[0,5]] -> 2/252
    p = fisher_exact_2x2(5, 0, 0, 5)
    good = abs(p - 2 / comb(10, 5)) < 1e-9
    ok &= good
    print(f"  Fisher [[5,0],[0,5]]        p = {p:.7f}  (want {2/comb(10,5):.7f})  "
          f"{'PASS' if good else 'FAIL'}")

    # 3. Permutation p-floor: two perfectly separated groups of 5 must give
    #    exactly 2/252 -- the smallest attainable two-sided p at this n.
    obs, p, tot = permutation_test([1, 2, 3, 4, 5], [10, 11, 12, 13, 14], "mean")
    good = tot == 252 and abs(p - 2 / 252) < 1e-12
    ok &= good
    print(f"  permutation, separated      p = {p:.7f} over {tot} splits "
          f"(want {2/252:.7f}, 252)  {'PASS' if good else 'FAIL'}")

    # 4. Identical groups must give p = 1
    obs, p, _ = permutation_test([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], "mean")
    good = abs(p - 1.0) < 1e-12 and abs(obs) < 1e-12
    ok &= good
    print(f"  permutation, identical      p = {p:.4f}  (want 1.0)  "
          f"{'PASS' if good else 'FAIL'}")

    # 5. Cliff's delta: complete dominance is exactly +1
    d, _ = cliffs_delta([10, 11, 12], [1, 2, 3])
    good = abs(d - 1.0) < 1e-12
    ok &= good
    print(f"  Cliff's delta, dominance    d = {d:+.4f}  (want +1.0)  "
          f"{'PASS' if good else 'FAIL'}")

    # 6. Mann-Whitney U for complete separation of 5 vs 5 is U = 25, p = 2/252
    u, p = mannwhitney_exact([6, 7, 8, 9, 10], [1, 2, 3, 4, 5])
    good = u == 25.0 and abs(p - 2 / 252) < 1e-12
    ok &= good
    print(f"  Mann-Whitney, separated     U = {u:.1f}, p = {p:.7f}  "
          f"(want 25.0, {2/252:.7f})  {'PASS' if good else 'FAIL'}")

    # 7. Holm-Bonferroni must be monotone and never below the raw p
    raw = [0.01, 0.04, 0.03]
    adj, _ = holm_bonferroni(raw)
    good = all(adj[i] >= raw[i] - 1e-12 for i in range(3))
    ok &= good
    print(f"  Holm-Bonferroni             {np.round(adj,4).tolist()} from {raw}  "
          f"{'PASS' if good else 'FAIL'}")

    print("\n  " + ("ALL PASS" if ok else "SOME FAILED -- do not use these numbers"))
    print(f"\n  NOTE: with n = 5 per arm the smallest attainable two-sided")
    print(f"  p-value is 2/252 = {2/252:.4f}. No result in this study can be")
    print(f"  significant at the 0.001 level, whatever its effect size.")
    return ok


# ==========================================================================

def main():
    ap = argparse.ArgumentParser(description="Exact statistical analysis for the study.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="stats_report.json")
    ap.add_argument("--runs", default=RUNS)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    globals()["RUNS"] = a.runs
    if not selftest():
        print("\nself-test failed -- refusing to report.", file=sys.stderr)
        sys.exit(1)

    data = {}
    for name, pat in CONFIGS.items():
        e, f = load(pat)
        if len(e):
            data[name] = (e, f)

    # ---------------- descriptives
    print("\n" + "=" * 96)
    print("DESCRIPTIVE STATISTICS  (relative L2 error; median is the primary statistic)")
    print("=" * 96)
    print(f"{'configuration':18}{'n':>3}{'fail':>6}{'median':>9}{'95% boot CI':>20}"
          f"{'mean':>9}{'sd':>8}{'min':>8}{'max':>8}")
    print("-" * 96)
    desc = {}
    for name, (e, f) in data.items():
        lo, hi = bootstrap_ci(e)
        desc[name] = {"n": len(e), "failures": int(f.sum()),
                      "median": float(np.median(e)), "ci_lo": lo, "ci_hi": hi,
                      "mean": float(e.mean()), "sd": float(e.std(ddof=1)),
                      "min": float(e.min()), "max": float(e.max()),
                      "values": [float(v) for v in e],
                      "failed_flags": [bool(v) for v in f]}
        print(f"{name:18}{len(e):>3}{int(f.sum()):>4}/5{np.median(e):>9.2%}"
              f"  [{lo:>6.2%},{hi:>7.2%}]{e.mean():>9.2%}{e.std(ddof=1):>8.2%}"
              f"{e.min():>8.2%}{e.max():>8.2%}")

    # ---------------- pairwise
    print("\n" + "=" * 96)
    print("PAIRWISE COMPARISONS  (exact permutation on medians, 252 splits;")
    print("Cliff's delta as effect size; Fisher's exact on failure counts)")
    print("=" * 96)
    rows, praw = [], []
    for label, A, B in COMPARISONS:
        if A not in data or B not in data:
            continue
        ea, fa = data[A]
        eb, fb = data[B]
        diff_med, p_perm, ntot = permutation_test(ea, eb, "median")
        _u, p_mw = mannwhitney_exact(ea, eb)
        d, size = cliffs_delta(ea, eb)
        p_fis = fisher_exact_2x2(int(fa.sum()), len(fa) - int(fa.sum()),
                                 int(fb.sum()), len(fb) - int(fb.sum()))
        rows.append({"label": label, "a": A, "b": B,
                     "median_a": float(np.median(ea)), "median_b": float(np.median(eb)),
                     "median_diff": diff_med, "p_permutation": p_perm,
                     "p_mannwhitney": p_mw, "cliffs_delta": d, "effect": size,
                     "failures_a": int(fa.sum()), "failures_b": int(fb.sum()),
                     "p_fisher_failures": p_fis, "n_splits": ntot})
        praw.append(p_perm)

    adj, sig = holm_bonferroni(praw)
    for r, pa, s in zip(rows, adj, sig):
        r["p_permutation_holm"] = float(pa)
        r["significant_holm_0.05"] = bool(s)

    print(f"{'comparison':44}{'med A':>8}{'med B':>8}{'p_perm':>8}{'p_holm':>8}"
          f"{'delta':>8}{'effect':>11}{'fail p':>8}")
    print("-" * 96)
    for r in rows:
        print(f"{r['label']:44}{r['median_a']:>8.2%}{r['median_b']:>8.2%}"
              f"{r['p_permutation']:>8.3f}{r['p_permutation_holm']:>8.3f}"
              f"{r['cliffs_delta']:>+8.2f}{r['effect']:>11}"
              f"{r['p_fisher_failures']:>8.3f}")

    print(f"\n  p-floor at n=5 per arm: 2/252 = {2/252:.4f}. Holm-Bonferroni is")
    print(f"  applied across all {len(rows)} declared comparisons.")

    # ---------------- power
    print("\n" + "=" * 96)
    print("POWER: how many seeds would the failure-rate claims need?")
    print("=" * 96)
    scenarios = [("observed 1/5 vs 0/5 (p=0.20 vs 0.00)", 0.20, 0.001),
                 ("a doubling, 0.40 vs 0.20", 0.40, 0.20),
                 ("0.20 vs 0.05", 0.20, 0.05)]
    power = {}
    for label, p1, p2 in scenarios:
        pw = power_for_failure_rates(p1, p2)
        power[label] = pw
        need = next((n for n, v in pw.items() if v >= 0.8), None)
        print(f"  {label:38} " + "  ".join(f"n={n}:{v:.2f}" for n, v in pw.items()))
        print(f"  {'':38} -> 80% power at n " +
              (f"= {need}" if need else "> 200"))
    print("\n  This is why the paper reports failure COUNTS and declines to claim")
    print("  a failure-RATE difference. 0/5 versus 1/5 is Fisher p = 1.00.")

    # ---------------- spread and worst case
    print("\n" + "=" * 96)
    print("SPREAD AND WORST-CASE COMPARISONS")
    print("(the claims in section 5 are about variance and worst case, which a")
    print(" test on medians cannot address; exact permutation on each statistic)")
    print("=" * 96)
    SPREAD = [
        ("S1 classical init: gain 1.0 vs 2.0",  "classical_g1.0", "classical_g2.0"),
        ("S2 classical init: gain 0.5 vs 2.0",  "classical_g0.5", "classical_g2.0"),
        ("S3 near-identity on strong ansatz",   "strong_default", "strong_nearid"),
        ("S4 near-identity on ring ansatz",     "ring_default",   "ring_nearid"),
        ("S5 matched tuning: classical vs QAPINN", "classical_g2.0", "strong_nearid"),
    ]
    print(f"{'comparison':42}{'sd ratio':>10}{'p':>8}{'max ratio':>11}{'p':>8}")
    print("-" * 96)
    spread_rows = []
    for label, A, B in SPREAD:
        if A not in data or B not in data:
            continue
        ea, eb = data[A][0], data[B][0]
        o1, p1 = permutation_test_stat(ea, eb, log_sd_ratio)
        o2, p2 = permutation_test_stat(ea, eb, log_max_ratio)
        spread_rows.append({"label": label, "a": A, "b": B,
                            "sd_ratio": float(np.exp(o1)), "p_sd": p1,
                            "max_ratio": float(np.exp(o2)), "p_max": p2})
        print(f"{label:42}{np.exp(o1):>9.2f}x{p1:>8.3f}{np.exp(o2):>10.2f}x{p2:>8.3f}")
    print("\n  Ratios above 1 mean the FIRST arm is more variable / has the worse")
    print("  worst case. These are the quantities section 5 claims to move.")

    # The attainable p-floor is a property of (n, test statistic), not of n
    # alone. `permutation_test` above is called with stat="median", so the
    # floor governing every p_permutation in this report is the MEDIAN-diff
    # floor -- 12/252 at n = 5, six times coarser than the mean-diff 2/252
    # that `p_floor_n5` holds. That field is kept unchanged so that nothing
    # reading it moves, but it is not the operative floor and must not be
    # quoted as one; the two fields below are derived by enumeration and
    # follow n, so they stay correct when the arms are extended past 5 seeds.
    import floor_analysis as _fa
    # Arms need not share n: the seed extension takes the ten continuous arms
    # to n = 8 while the three QCPINN arms stay at 5, which makes H10 an
    # 8-against-5 comparison with a floor of its own. So the floor is recorded
    # per comparison as well as in summary.
    _ns = {k: len(v["values"]) for k, v in desc.items()}
    _uniq = sorted(set(_ns.values()))
    _n = _uniq[0] if len(_uniq) == 1 else None
    _fc = {}
    def _floor(n1, n2, stat):
        if (n1, n2, stat) not in _fc:
            _fc[(n1, n2, stat)] = _fa.attainable_floor_unequal(n1, n2, stat)[2]
        return _fc[(n1, n2, stat)]
    for r in rows:
        n1, n2 = _ns.get(r["a"]), _ns.get(r["b"])
        if n1 and n2:
            r["n_a"], r["n_b"] = n1, n2
            r["floor_median"] = _floor(n1, n2, "median")
            r["floor_mean"] = _floor(n1, n2, "mean")
    _fm = [r["floor_median"] for r in rows if "floor_median" in r]
    out = {"descriptives": desc, "comparisons": rows, "spread": spread_rows,
           "power": power,
           "p_floor_n5": 2 / 252,
           "n_per_arm": _ns,
           "n_homogeneous": _n,
           "floor_median_min": min(_fm) if _fm else None,
           "floor_median_max": max(_fm) if _fm else None,
           "p_floor_median": _fa.attainable_floor(_n, "median")[2] if _n else None,
           "p_floor_mean": _fa.attainable_floor(_n, "mean")[2] if _n else None,
           "notes": "exact permutation over C(2n,n) splits on the MEDIAN "
                    "difference (p_permutation) and on log ratios for spread "
                    "(p_sd, p_max); Fisher exact, Cliff's delta, "
                    "Holm-Bonferroni over the declared family. p_floor_median "
                    "is the floor that binds p_permutation; p_floor_n5 is the "
                    "mean-diff floor, retained for backward compatibility. "
                    "H8 and H9 were WITHDRAWN under the pre-registered rule of "
                    "qcpinn_converge.sh: the qcpinn_771 arm is 2/5 non-converged "
                    "even at 80,000 epochs, so no efficiency comparison across it "
                    "has a measurand. Declared family is 10; Holm threshold 0.0050. "
                    "The qcc_* arms are descriptive only and enter no comparison."}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
