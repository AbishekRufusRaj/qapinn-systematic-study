"""
Verdict on the initialisation confound.

THE QUESTION, STATED SO IT CANNOT BE FUDGED AFTER THE FACT
----------------------------------------------------------
The QAPINN's seed spread collapsed from 14.11% to 0.55% under near-identity
initialisation. The paper wants to claim the quantum model is more reliable
than the classical baseline. That claim is only sound if the classical
baseline's spread is NOT similarly reducible.

So the test is: does the classical baseline's standard deviation respond to
initialisation tuning at all?

The decision rule is fixed in advance -- the best tuned gain must reduce the
spread by more than 2x for the confound to count as real. A 10% wobble
across three points with n=5 each is noise, and picking whichever gain
happened to look tightest would be exactly the kind of post-hoc selection
this project exists to criticise.
"""

import glob
import json
import os
import statistics as st

FAIL_LOSS = 1e-2
REDUCTION_THRESHOLD = 2.0      # fixed before looking at the numbers

GROUPS = {
    "classical, default (gain 1.0)": "runs/conv_classical_s*",
    "classical, gain 0.5":           "runs/ctrl_classical_g0.5_s*",
    "classical, gain 2.0":           "runs/ctrl_classical_g2.0_s*",
    "QAPINN strong, default init":   "runs/conv_strong_L2_s*",
    "QAPINN strong, near-identity":  "runs/init_strong_L2_s*",
}


def load(d):
    mp, tp = os.path.join(d, "metrics.json"), os.path.join(d, "meta.json")
    if not (os.path.exists(mp) and os.path.exists(tp)):
        return None
    m, t = json.load(open(mp)), json.load(open(tp))
    return m["rel_l2"] * 100, t["final_total_loss"] > FAIL_LOSS


def summarize(pattern):
    rows = [r for r in (load(d) for d in sorted(glob.glob(pattern))) if r]
    if not rows:
        return None
    errs = [e for e, _ in rows]
    ok = [e for e, f in rows if not f]
    return {"n": len(rows), "nfail": sum(f for _, f in rows),
            "mean": st.mean(ok) if ok else float("nan"),
            "std": st.stdev(ok) if len(ok) > 1 else 0.0,
            "median": st.median(errs), "best": min(errs), "worst": max(errs)}


def main():
    print(f"decision rule fixed in advance: the confound is real only if a tuned")
    print(f"gain reduces the classical spread by more than {REDUCTION_THRESHOLD:.0f}x\n")
    print(f"{'configuration':32s} {'n':>2} {'fail':>5} {'mean':>8} {'std':>8} "
          f"{'median':>8} {'best':>7} {'worst':>7}")
    print("-" * 90)

    stats = {}
    for label, pattern in GROUPS.items():
        s = summarize(pattern)
        if not s:
            print(f"{label:32s}  (not run)")
            continue
        stats[label] = s
        print(f"{label:32s} {s['n']:>2} {s['nfail']:>2}/{s['n']:<2} {s['mean']:7.2f}% "
              f"{s['std']:7.2f}% {s['median']:7.2f}% {s['best']:6.2f}% {s['worst']:6.2f}%")

    base = stats.get("classical, default (gain 1.0)")
    tuned = {k: v for k, v in stats.items() if k.startswith("classical, gain")}
    if not base or not tuned:
        print("\nNeed the default classical runs and at least one tuned gain.")
        return

    best_label = min(tuned, key=lambda k: tuned[k]["std"])
    best = tuned[best_label]
    factor = base["std"] / best["std"] if best["std"] > 0 else float("inf")

    print("\n" + "=" * 90)
    print("DOES CLASSICAL VARIANCE RESPOND TO INITIALISATION TUNING?")
    print(f"    default      std {base['std']:.2f}%")
    print(f"    best tuned   std {best['std']:.2f}%   ({best_label})")
    print(f"    -> reduction {factor:.2f}x")

    q_def = stats.get("QAPINN strong, default init")
    q_ni = stats.get("QAPINN strong, near-identity")
    if q_def and q_ni and q_ni["std"] > 0:
        qf = q_def["std"] / q_ni["std"]
        print(f"\n    for comparison, the QAPINN's own reduction was {qf:.1f}x "
              f"({q_def['std']:.2f}% -> {q_ni['std']:.2f}%)")

    print()
    if factor > REDUCTION_THRESHOLD:
        print("  CONFOUND IS REAL. Classical variance is reducible by init tuning too.")
        print("  The reliability comparison must be made against the TUNED classical")
        print("  baseline, not the default one. Rewrite that claim before submitting;")
        print("  finding it yourself is much better than a reviewer finding it.")
    else:
        print("  CONFOUND DOES NOT HOLD UP. Classical spread is not meaningfully")
        print("  reducible by init tuning, so the QAPINN's collapse is specific to")
        print("  the quantum layer. That is a STRONGER claim than the current one --")
        print("  state it as 'the classical baseline was given a matched")
        print("  initialisation sweep and did not respond', and report this table.")

    print("\n  Caveat for the paper: rescaling a classical init is not the identical")
    print("  intervention as near-identity initialisation of a variational circuit.")
    print("  It is the closest honest analogue. Say so; do not imply equivalence.")


if __name__ == "__main__":
    main()
