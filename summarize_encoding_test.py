"""
Report the ansatz x encoding factorial.

Four cells, five seeds each. Reporting the grid rather than a single best
configuration means the two factors can be separated: how much comes from
changing the ansatz, how much from changing the encoding, and whether the
two interact. A single "best config" number cannot answer any of those.

Cells:
    ring   + angle          runs/conv_ring_L2_s*      (from the convergence sweep)
    strong + angle          runs/conv_strong_L2_s*    (from the convergence sweep)
    ring   + angle_scaled   runs/enc_ring_L2_s*       (this experiment)
    strong + angle_scaled   runs/enc_strong_L2_s*     (this experiment)

Needs only the standard library and numpy, so it runs anywhere.
"""

import glob
import json
import os
import statistics as st

CELLS = {
    ("ring", "angle"):          "runs/conv_ring_L2_s*",
    ("strong", "angle"):        "runs/conv_strong_L2_s*",
    ("ring", "angle_scaled"):   "runs/enc_ring_L2_s*",
    ("strong", "angle_scaled"): "runs/enc_strong_L2_s*",
}
REFERENCE = ("classical baseline", "runs/conv_classical_s*")


def load(pattern):
    out = []
    for d in sorted(glob.glob(pattern)):
        path = os.path.join(d, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                out.append(json.load(f))
    return out


def stats(ms):
    if not ms:
        return None
    errs = [m["rel_l2"] for m in ms]
    return {
        "n": len(errs),
        "mean": st.mean(errs),
        "std": st.stdev(errs) if len(errs) > 1 else 0.0,
        "best": min(errs),
        "shock": st.mean(m["rel_l2_shock_region"] for m in ms),
    }


def main():
    grid = {k: stats(load(p)) for k, p in CELLS.items()}

    print(f"{'ansatz':8s} {'encoding':14s} {'n':>2}  {'mean':>7} {'std':>7} {'best':>7} {'shock':>8}")
    print("-" * 60)
    for (ans, enc), s in grid.items():
        if s is None:
            print(f"{ans:8s} {enc:14s}  0   (not run yet)")
        else:
            print(f"{ans:8s} {enc:14s} {s['n']:>2}  {s['mean']:>6.2%} {s['std']:>6.2%} "
                  f"{s['best']:>6.2%} {s['shock']:>7.2%}")

    ref = stats(load(REFERENCE[1]))
    if ref:
        print(f"{'--':8s} {REFERENCE[0]:14s} {ref['n']:>2}  {ref['mean']:>6.2%} {ref['std']:>6.2%} "
              f"{ref['best']:>6.2%} {ref['shock']:>7.2%}")

    complete = {k: v for k, v in grid.items() if v}
    if len(complete) < 4:
        print("\nAll four cells are needed before the factorial effects mean anything.")
        return

    def m(a, e):
        return grid[(a, e)]["mean"]

    ansatz_effect = ((m("ring", "angle") + m("ring", "angle_scaled")) -
                     (m("strong", "angle") + m("strong", "angle_scaled"))) / 2
    encoding_effect = ((m("ring", "angle") + m("strong", "angle")) -
                       (m("ring", "angle_scaled") + m("strong", "angle_scaled"))) / 2
    interaction = ((m("ring", "angle") - m("ring", "angle_scaled")) -
                   (m("strong", "angle") - m("strong", "angle_scaled"))) / 2

    spread = st.mean(v["std"] for v in grid.values())

    print("\nMain effects (positive = the change reduces error)")
    print(f"  switching ring -> strong ansatz      {ansatz_effect:+.2%}")
    print(f"  switching angle -> angle_scaled      {encoding_effect:+.2%}")
    print(f"  interaction between the two          {interaction:+.2%}")
    print(f"  mean within-cell spread (std)        {spread:.2%}")
    print()
    for name, val in [("ansatz", ansatz_effect), ("encoding", encoding_effect)]:
        if abs(val) > spread:
            print(f"  {name}: effect exceeds the noise floor -- reportable")
        else:
            print(f"  {name}: effect is within seed noise -- report as a trend only")


if __name__ == "__main__":
    main()
