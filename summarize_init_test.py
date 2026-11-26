"""
Summarise the near-identity initialisation test against the default-init
convergence runs it is meant to improve on.

WHY THIS REPORTS MEDIANS AND FAILURE COUNTS, NOT JUST MEANS
-----------------------------------------------------------
The convergence sweep showed the error distribution is bimodal, not spread:
runs either break through their plateau (final loss 1e-4 to 6e-3) or never
do (2e-1). A mean over a bimodal sample describes neither mode. One
non-converged run in five moved the strong-ansatz mean from 2.41% to 8.70%
and its standard deviation to 14.11%, which reads as "high variance" when
what actually happened is "four good runs and one that never started".

So this reports three things separately, because they are three different
claims about the intervention:

  failure rate  -- how often the model never leaves the plateau. This is
                   what near-identity init is theoretically supposed to fix
                   (Grant et al. 2019), so it is the primary endpoint.
  median        -- typical accuracy, robust to the failures by construction.
  mean ex-fail  -- accuracy conditional on training succeeding at all. This
                   is what tells you whether the init also helps runs that
                   were already fine, which is a separate question from
                   whether it prevents failures.

The non-convergence threshold is fixed in advance at final loss > 1e-2, a
rule chosen because exactly one run in the fifteen-run convergence sweep
trips it and the next-worst is 33x below it. Setting it after seeing these
results would make the failure counts meaningless.
"""

import glob
import json
import os
import statistics as st

FAIL_LOSS = 1e-2      # pre-registered non-convergence threshold

GROUPS = {
    "strong, default init":   "runs/conv_strong_L2_s*",
    "strong, near-identity":  "runs/init_strong_L2_s*",
    "ring, default init":     "runs/conv_ring_L2_s*",
    "classical baseline":     "runs/conv_classical_s*",
}


def load(run_dir):
    mpath = os.path.join(run_dir, "metrics.json")
    tpath = os.path.join(run_dir, "meta.json")
    if not (os.path.exists(mpath) and os.path.exists(tpath)):
        return None
    m, t = json.load(open(mpath)), json.load(open(tpath))
    return {
        "err": m["rel_l2"],
        "shock": m["rel_l2_shock_region"],
        "smooth": m["rel_l2_smooth_region"],
        "loss": t["final_total_loss"],
        "epochs": t.get("adam_epochs_used"),
        "seconds": t.get("total_seconds"),
        "failed": t["final_total_loss"] > FAIL_LOSS,
    }


def main():
    print(f"non-convergence threshold: final loss > {FAIL_LOSS:.0e}\n")
    print(f"{'configuration':24s} {'n':>2} {'fail':>5} {'median':>8} "
          f"{'mean(ok)':>9} {'std(ok)':>8} {'best':>7} {'worst':>7} {'mins':>6}")
    print("-" * 92)

    stats = {}
    for label, pattern in GROUPS.items():
        rows = [r for r in (load(d) for d in sorted(glob.glob(pattern))) if r]
        if not rows:
            print(f"{label:24s}  0   (no runs found)")
            continue

        errs = [r["err"] * 100 for r in rows]
        ok = [r["err"] * 100 for r in rows if not r["failed"]]
        nfail = sum(r["failed"] for r in rows)
        mins = st.mean(r["seconds"] for r in rows if r["seconds"]) / 60

        mean_ok = st.mean(ok) if ok else float("nan")
        std_ok = st.stdev(ok) if len(ok) > 1 else 0.0
        stats[label] = {"rows": rows, "ok": ok, "nfail": nfail, "n": len(rows),
                        "median": st.median(errs), "mean_ok": mean_ok,
                        "std_ok": std_ok}

        print(f"{label:24s} {len(rows):>2} {nfail:>2}/{len(rows):<2} "
              f"{st.median(errs):7.2f}% {mean_ok:8.2f}% {std_ok:7.2f}% "
              f"{min(errs):6.2f}% {max(errs):6.2f}% {mins:6.1f}")

    a = stats.get("strong, default init")
    b = stats.get("strong, near-identity")
    if not (a and b):
        print("\nNeed both strong-ansatz groups on disk to draw a conclusion.")
        return

    print("\n" + "=" * 92)
    print("PRIMARY ENDPOINT -- does near-identity init prevent non-convergence?")
    print(f"    default init  : {a['nfail']}/{a['n']} runs never left the plateau")
    print(f"    near-identity : {b['nfail']}/{b['n']} runs never left the plateau")
    if b["nfail"] < a["nfail"]:
        print("    -> Fewer failures. Consistent with the barren-plateau reading.")
        print("       With n=5 this is suggestive, not significant: 0/5 vs 1/5 is")
        print("       not a p-value. Say 'consistent with', and if the claim is")
        print("       load-bearing for the paper, run more seeds.")
    elif b["nfail"] == a["nfail"] == 0:
        print("    -> Neither group failed, so this test cannot separate them.")
        print("       The failure was rare enough that 5 seeds may simply have")
        print("       missed it. More seeds, or report the accuracy effect only.")
    else:
        print("    -> No reduction in failures. The barren-plateau reading of the")
        print("       original failure is not supported; do not put it in the paper.")

    print("\nSECONDARY -- does it also help runs that were already converging?")
    gap = a["mean_ok"] - b["mean_ok"]
    spread = a["std_ok"] + b["std_ok"]
    print(f"    default init  : {a['mean_ok']:.2f}% +/- {a['std_ok']:.2f}%  (converged runs only)")
    print(f"    near-identity : {b['mean_ok']:.2f}% +/- {b['std_ok']:.2f}%")
    print(f"    -> gap {gap:+.2f}%, combined spread {spread:.2f}%")
    if abs(gap) > spread:
        direction = "better" if gap > 0 else "WORSE"
        print(f"       Gap exceeds the spread: near-identity is {direction} on")
        print("       accuracy, not only on failure rate.")
    else:
        print("       Gap is inside the spread. No accuracy effect on runs that")
        print("       already converged -- whatever this init does, it does it by")
        print("       preventing failures, not by improving good runs.")

    c = stats.get("classical baseline")
    if c:
        print("\nAGAINST THE BASELINE (converged runs only)")
        gap = c["mean_ok"] - b["mean_ok"]
        spread = c["std_ok"] + b["std_ok"]
        print(f"    near-identity QAPINN : {b['mean_ok']:.2f}% +/- {b['std_ok']:.2f}%")
        print(f"    classical PINN       : {c['mean_ok']:.2f}% +/- {c['std_ok']:.2f}%")
        print(f"    -> gap {gap:+.2f}%, combined spread {spread:.2f}%  "
              f"({'SUPPORTED' if gap > spread else 'NOT supported'})")


if __name__ == "__main__":
    main()
