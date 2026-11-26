"""
Verdict on the parameter-efficiency claim.

The claim under test, from arXiv:2503.16678: "comparable accuracy to
classical PINNs while requiring up to 89% fewer trainable parameters".

There are two ways to get that claim wrong, and this script is built to
avoid both.

FIRST: "comparable" needs a definition that cannot be chosen after seeing
the data. Here it is fixed in advance -- the difference in mean error must
be smaller than the combined seed spread of the two models. If the gap is
inside the noise, the models are comparable; if it is outside, they are
not, whichever direction it points. Reporting a percentage gap without
this test is what allows a single lucky seed to become a headline, and
this project already produced one of those (a 4.38% run that looked like a
5.6x win and evaporated on repetition).

SECOND: comparable TO WHAT. The paper's abstract compares against
Model-1, the 7,851-parameter baseline. Model-2 has 2,751 parameters and,
in the paper's own tables, beats the quantum model on the cavity problem
and on Helmholtz. Efficiency measured against an over-parameterised
baseline is not efficiency. So both comparisons are printed, and the one
against Model-2 is labelled as the honest one.

Efficiency is also reported as accuracy-per-parameter, since that is what
the claim is actually about, and a model can win on it while losing on raw
accuracy -- which is precisely the interesting case.
"""

import glob
import json
import os
import statistics as st

FAIL_LOSS = 1e-2

GROUPS = {
    "QCPINN (771 par)":   "runs/qc_qcpinn_s*",
    "Model-2 (2751 par)": "runs/qc_model2_s*",
    "Model-1 (7851 par)": "runs/qc_model1_s*",
}


def load(run_dir):
    mp, tp = os.path.join(run_dir, "metrics.json"), os.path.join(run_dir, "meta.json")
    if not (os.path.exists(mp) and os.path.exists(tp)):
        return None
    m, t = json.load(open(mp)), json.load(open(tp))
    return {"err": m["rel_l2"] * 100,
            "shock": m["rel_l2_shock_region"] * 100,
            "smooth": m["rel_l2_smooth_region"] * 100,
            "loss": t["final_total_loss"],
            "params": t["param_counts"]["total"],
            "mins": (t.get("total_seconds") or 0) / 60,
            "failed": t["final_total_loss"] > FAIL_LOSS}


def summarize(pattern):
    rows = [r for r in (load(d) for d in sorted(glob.glob(pattern))) if r]
    if not rows:
        return None
    ok = [r["err"] for r in rows if not r["failed"]]
    errs = [r["err"] for r in rows]
    return {"n": len(rows), "nfail": sum(r["failed"] for r in rows),
            "params": rows[0]["params"],
            "median": st.median(errs),
            "mean": st.mean(ok) if ok else float("nan"),
            "std": st.stdev(ok) if len(ok) > 1 else 0.0,
            "best": min(errs), "worst": max(errs),
            "shock": st.mean(r["shock"] for r in rows),
            "smooth": st.mean(r["smooth"] for r in rows),
            "mins": st.mean(r["mins"] for r in rows)}


def compare(name_a, a, name_b, b):
    """Is A comparable to B? Pre-registered rule: gap < combined spread."""
    gap = a["mean"] - b["mean"]
    spread = a["std"] + b["std"]
    print(f"\n  {name_a} vs {name_b}")
    print(f"    {a['mean']:.2f}% +/- {a['std']:.2f}%   vs   {b['mean']:.2f}% +/- {b['std']:.2f}%")
    print(f"    gap {gap:+.2f}%, combined spread {spread:.2f}%")
    if abs(gap) <= spread:
        ratio = b["params"] / a["params"]
        print(f"    -> COMPARABLE. {name_a} matches accuracy with "
              f"{ratio:.1f}x fewer parameters")
        print(f"       ({100 * (1 - a['params'] / b['params']):.0f}% parameter reduction). "
              f"The efficiency claim HOLDS here.")
        return True
    direction = "WORSE" if gap > 0 else "BETTER"
    print(f"    -> NOT comparable: {name_a} is {direction} by more than the spread.")
    return False


def main():
    print(f"non-convergence threshold: final loss > {FAIL_LOSS:.0e}\n")
    print(f"{'model':22s} {'par':>6} {'n':>2} {'fail':>5} {'median':>8} {'mean(ok)':>9} "
          f"{'std':>7} {'best':>7} {'worst':>7} {'shock':>7} {'smooth':>7} {'mins':>6}")
    print("-" * 112)

    stats = {}
    for label, pattern in GROUPS.items():
        s = summarize(pattern)
        if not s:
            print(f"{label:22s}  (no runs found)")
            continue
        stats[label] = s
        print(f"{label:22s} {s['params']:>6} {s['n']:>2} {s['nfail']:>2}/{s['n']:<2} "
              f"{s['median']:7.2f}% {s['mean']:8.2f}% {s['std']:6.2f}% "
              f"{s['best']:6.2f}% {s['worst']:6.2f}% {s['shock']:6.2f}% "
              f"{s['smooth']:6.2f}% {s['mins']:6.1f}")

    q = stats.get("QCPINN (771 par)")
    m2 = stats.get("Model-2 (2751 par)")
    m1 = stats.get("Model-1 (7851 par)")
    if not q or not (m1 or m2):
        print("\nNeed the QCPINN and at least one baseline to reach a verdict.")
        return

    print("\n" + "=" * 112)
    print("THE PARAMETER-EFFICIENCY CLAIM")
    print('  "comparable accuracy ... up to 89% fewer trainable parameters"')
    print("  Rule fixed in advance: comparable means the gap in mean error is")
    print("  smaller than the two models' combined seed spread.")

    holds_2 = compare("QCPINN", q, "Model-2", m2) if m2 else None
    holds_1 = compare("QCPINN", q, "Model-1", m1) if m1 else None

    print("\n" + "-" * 112)
    print("EFFICIENCY (accuracy per parameter -- lower error x fewer params is better)")
    for label, s in stats.items():
        print(f"    {label:22s} {s['mean']:6.2f}% at {s['params']:>5} params  "
              f"-> {s['mean'] * s['params'] / 1000:8.2f}  (error x kparams)")

    print("\n" + "-" * 112)
    print("VERDICT")
    if holds_2:
        print("  The efficiency claim survives seed repetition against the HONEST")
        print("  baseline. That is a confirmation of the published result with the")
        print("  statistical control the original lacked -- report it as such.")
    elif holds_1 and not holds_2:
        print("  The claim holds only against Model-1, the over-parameterised")
        print("  baseline. Against Model-2 -- same accuracy, 2,751 params -- it does")
        print("  not. The paper's own tables already show Model-2 winning on two of")
        print("  three problems; this quantifies that with seeds.")
    else:
        print("  The efficiency claim does not survive seed repetition on Burgers'.")
        print("  Before reporting that: check the failure counts above. If the QCPINN")
        print("  failed runs that the baselines did not, this is a trainability")
        print("  result, not a capacity one -- and near-identity init is the known")
        print("  mitigation (see init_scale_test.sh).")

    if q["nfail"]:
        print(f"\n  NOTE: {q['nfail']}/{q['n']} QCPINN runs were non-converged. With a")
        print("  15-parameter quantum layer this is worth checking against the")
        print("  barren-plateau reading before drawing a capacity conclusion.")


if __name__ == "__main__":
    main()
