#!/usr/bin/env python3
"""
ANALYSIS OF THE SYMMETRIC CONTROL (sym4), for Section VI.

WHAT THIS ANSWERS
=================
Reviewers 4 and 5 both objected that the headline comparison in the submitted
paper was confounded: the quantum arm (QCPINN) and the classical arm (model2)
were trained under different protocols, so a difference in final error could
not be attributed to the architecture. The symmetric control removes that
confound by running both arms under one identical protocol -- same optimiser,
same convergence criterion, same L-BFGS refinement, same evaluation points,
same seeds 0-9.

This script reads the 20 resulting runs and applies the SAME exact tests the
rest of the study uses, by importing them from stats_analysis.py rather than
reimplementing them. At n = 10 per arm the exact permutation null has
C(20,10) = 184,756 splits, so the smallest attainable two-sided p-value is
2/184756 = 1.08e-5 -- the n = 5 p-floor that constrains the main study does
not bind here.

A NOTE ON WHICH NUMBER IS THE RESULT
====================================
Each run yields two figures: error after the Adam phase, and error after the
L-BFGS refinement ("final"). The FINAL number is the result, because the
protocol being controlled for includes the refinement. An earlier, superseded
version of this control (protocol2) crashed during refinement on every quantum
run, and its surviving after-Adam numbers were reported in the interim as
"quantum 4.33% vs classical 8.09%, p = 0.1136". That interim result is NOT the
result and is superseded here: it compared a quantum arm that never got its
refinement against a classical arm measured the same way, and the refinement
turns out to matter by a factor of two to three on both arms.

THE THREE DEAD CLASSICAL SEEDS
==============================
Seeds 5, 6 and 9 of the classical arm finish at ~100% error with
err_f ~= 99.99998%, meaning the network output is identically ~zero: these are
dead-on-arrival initialisations, not slow convergence. They fail identically
under BOTH protocols (established earlier: rep_model2 gives 100.31/100.00/100.09
against sym4's 100.16/101.29/100.28), so they are a property of the
initialisation, not of the training budget. They are therefore reported, not
discarded -- the failure rate IS part of the finding. Both the all-seeds
comparison and a sensitivity analysis excluding them are printed below, because
a reader is entitled to know the result does not hinge on how failures are
handled.

USAGE
    python3 sym4_analysis.py            # -> sym4_report.json
"""

import glob
import json
import os
import sys

import numpy as np

from stats_analysis import (
    cliffs_delta,
    fisher_exact_2x2,
    mannwhitney_exact,
    permutation_test,
)

RUNS = "runs"
FAIL_PCT = 50.0  # a run above this is a failure, not a bad fit
SEEDS = list(range(10))


def load_arm(arm):
    """Read the 10 replication.json files for one arm, ordered by seed."""
    out = []
    for s in SEEDS:
        p = os.path.join(RUNS, f"sym4_{arm}_s{s}", "replication.json")
        if not os.path.exists(p):
            sys.exit(f"MISSING: {p} -- run symmetric_protocol4.sh first")
        with open(p) as fh:
            j = json.load(fh)
        out.append(
            dict(
                seed=s,
                final=float(j["err_u_pct"]),
                after_adam=float(j["err_u_pct_after_adam"]),
                err_f=float(j["err_f_pct"]),
                stop=j["stop_reason"],
                epochs=int(j["adam_epochs_used"]),
            )
        )
    return out


def describe(xs):
    a = np.asarray(xs, float)
    return dict(
        n=len(a),
        median=float(np.median(a)),
        iqr=[float(np.percentile(a, 25)), float(np.percentile(a, 75))],
        min=float(a.min()),
        max=float(a.max()),
    )


def compare(q, c, label):
    """Exact tests on one pair of vectors. Returns a dict, prints a block."""
    diff, p, nsplits = permutation_test(q, c, stat="median")
    d, interp = cliffs_delta(q, c)
    try:
        u_p = mannwhitney_exact(q, c)
        u_p = float(u_p[-1]) if isinstance(u_p, (tuple, list)) else float(u_p)
    except Exception:
        u_p = None
    res = dict(
        label=label,
        quantum=describe(q),
        classical=describe(c),
        median_diff=float(diff),
        perm_p=float(p),
        perm_splits=int(nsplits),
        cliffs_delta=float(d),
        cliffs_interp=interp,
        mannwhitney_p=u_p,
    )
    print(f"\n--- {label} ---")
    print(f"  quantum   n={res['quantum']['n']:2d}  median {res['quantum']['median']:7.3f}%"
          f"  IQR [{res['quantum']['iqr'][0]:.3f}, {res['quantum']['iqr'][1]:.3f}]"
          f"  range [{res['quantum']['min']:.3f}, {res['quantum']['max']:.3f}]")
    print(f"  classical n={res['classical']['n']:2d}  median {res['classical']['median']:7.3f}%"
          f"  IQR [{res['classical']['iqr'][0]:.3f}, {res['classical']['iqr'][1]:.3f}]"
          f"  range [{res['classical']['min']:.3f}, {res['classical']['max']:.3f}]")
    print(f"  median difference (q - c) = {diff:+.3f} pp")
    print(f"  exact permutation p = {p:.6g}   ({nsplits} splits, floor {2/nsplits:.3g})")
    print(f"  Cliff's delta = {d:+.4f}  ({interp})")
    if u_p is not None:
        print(f"  exact Mann-Whitney p = {u_p:.6g}")
    return res


def main():
    q = load_arm("qcpinn")
    c = load_arm("model2")

    print("=" * 74)
    print("SYMMETRIC CONTROL (sym4): both arms, one identical protocol, seeds 0-9")
    print("=" * 74)
    print(f"\n{'seed':>4}  {'QCPINN final':>13} {'adam':>7}   {'model2 final':>13} {'adam':>7}")
    for qi, ci in zip(q, c):
        print(f"{qi['seed']:>4}  {qi['final']:12.3f}% {qi['after_adam']:6.2f}%   "
              f"{ci['final']:12.3f}% {ci['after_adam']:6.2f}%")

    qf = [r["final"] for r in q]
    cf = [r["final"] for r in c]

    # failure accounting
    q_fail = [r["seed"] for r in q if r["final"] > FAIL_PCT]
    c_fail = [r["seed"] for r in c if r["final"] > FAIL_PCT]
    print(f"\nfailures (> {FAIL_PCT:g}% error):")
    print(f"  quantum   {len(q_fail)}/10  {q_fail}")
    print(f"  classical {len(c_fail)}/10  {c_fail}")
    for s in c_fail:
        r = next(x for x in c if x["seed"] == s)
        print(f"    seed {s}: err_u {r['final']:.2f}%, err_f {r['err_f']:.5f}%"
              f"  -> output identically ~zero (dead init, stopped {r['stop']}"
              f" at {r['epochs']})")
    fisher_p = fisher_exact_2x2(len(q_fail), 10 - len(q_fail),
                                len(c_fail), 10 - len(c_fail))
    fisher_p = float(fisher_p[-1]) if isinstance(fisher_p, (tuple, list)) else float(fisher_p)
    print(f"  Fisher exact on failure counts: p = {fisher_p:.6g}")

    # primary comparison: every seed, as run
    primary = compare(qf, cf, "PRIMARY: all 10 seeds, final error (after refinement)")

    # sensitivity: drop the three dead classical inits
    keep = [s for s in SEEDS if s not in c_fail]
    qs = [r["final"] for r in q if r["seed"] in keep]
    cs = [r["final"] for r in c if r["seed"] in keep]
    sens = compare(qs, cs,
                   f"SENSITIVITY: excluding dead classical seeds {c_fail}"
                   f" (n={len(keep)} per arm, paired seeds)")

    # the superseded interim view, for the record
    interim = compare([r["after_adam"] for r in q],
                      [r["after_adam"] for r in c],
                      "FOR THE RECORD: after Adam only (supersedes the"
                      " protocol2 interim result)")

    report = dict(
        runs=dict(qcpinn=q, model2=c),
        failures=dict(quantum=q_fail, classical=c_fail, fisher_p=fisher_p),
        primary=primary,
        sensitivity=sens,
        interim_after_adam=interim,
        note=("Primary result is final error after L-BFGS refinement, all 10"
              " seeds, no exclusions. The after-Adam comparison is reported"
              " only to supersede the protocol2 interim figure."),
    )
    with open("sym4_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote sym4_report.json")


if __name__ == "__main__":
    main()
