"""
post_review_analysis.py -- the two post-review additions, analysed OUTSIDE the
declared comparison family.

READ THIS BEFORE USING THE OUTPUT
---------------------------------
Section 7 declares a family of comparisons and Holm-corrects it. The arms analysed
here were added AFTER results were seen, in response to review. Adding them to that
family would be exactly the practice this paper criticises -- correction applied to
what looked interesting rather than to what was planned.

So nothing here is a member of the family, nothing here is Holm-corrected against it,
and no p-value here may be reported as though it were. These are DESCRIPTIVE: effect
sizes, spreads, failure counts, and exact permutation p-values reported for what they
are -- uncorrected, on arms chosen after the fact.

Two parts:

  --part init   The literature-standard initialisation axis. ClassicalPINN inherits
                PyTorch's nn.Linear default (kaiming_uniform_, a=sqrt(5)), which is
                NOT the Xavier/Glorot initialisation the PINN literature specifies.
                gain sqrt(3)=1.7321 maps the default onto Xavier for the 7 hidden
                20x20 layers that hold 2641 of 3021 parameters; 2.8868 maps it onto
                Xavier x tanh gain (5/3). The question this answers is whether the
                study's "default" arm is a strawman.

  --part depth  Whether the architecture x initialisation interaction survives a
                change of re-uploading depth. The study runs at L = 2, which was
                selected on a SINGLE-SEED sweep (sweep_results.csv).
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np

RUNS = "runs"
FAIL_LOSS = 1e-2

# gain -> what it means. The two added arms are marked.
GAIN_MEANING = {
    "0.5":    "half the PyTorch default",
    "1.0":    "PyTorch nn.Linear default (kaiming_uniform_, a=sqrt(5))",
    "1.7321": "** Xavier/Glorot -- the PINN literature standard **  [ADDED]",
    "2.0":    "twice the PyTorch default",
    "2.8868": "** Xavier x tanh gain (5/3) **                       [ADDED]",
}


def load(pattern):
    e, f, ep = [], [], []
    for d in sorted(glob.glob(os.path.join(RUNS, pattern))):
        mp, tp = os.path.join(d, "meta.json"), os.path.join(d, "metrics.json")
        if not (os.path.exists(mp) and os.path.exists(tp)):
            continue
        m, t = json.load(open(mp)), json.load(open(tp))
        e.append(t["rel_l2"])
        f.append(float(m.get("final_total_loss", np.inf)) > FAIL_LOSS)
        ep.append(m.get("adam_epochs_used", np.nan))
    return np.array(e), np.array(f, bool), np.array(ep, float)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def perm_p(a, b, stat="median"):
    """Exact two-sided permutation p on the difference in medians -- the same
    convention as stats_analysis.permutation_test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n1, n = len(a), len(pooled)
    f = np.median if stat == "median" else np.mean
    obs = f(a) - f(b)
    c = t = 0
    for cc in itertools.combinations(range(n), n1):
        m = np.zeros(n, bool)
        m[list(cc)] = True
        v = f(pooled[m]) - f(pooled[~m])
        t += 1
        c += abs(v) >= abs(obs) - 1e-15
    return c / t, t


def describe(label, e, f, ep, extra=""):
    if not len(e):
        print(f"{label:46}{'-- not run --':>34}")
        return
    print(f"{label:46}{len(e):>4}{np.median(e)*100:>9.2f}%{e.mean()*100:>9.2f}%"
          f"{e.std(ddof=1)*100:>8.2f}%{e.max()*100:>9.2f}%{int(f.sum()):>6}/{len(e)}"
          f"{np.nanmean(ep):>10.0f}  {extra}")


HDR = (f"{'arm':46}{'n':>4}{'median':>10}{'mean':>10}{'sd':>9}{'worst':>10}"
       f"{'fail':>8}{'epochs':>10}")


def part_init():
    print("\n" + "=" * 118)
    print(" THE INITIALISATION AXIS, INCLUDING THE LITERATURE STANDARD")
    print(" Burgers', classical PINN. NOT members of the declared family -- see header.")
    print("=" * 118 + "\n")
    print(HDR)
    print("-" * 118)
    data = {}
    for g in ("0.5", "1.0", "1.7321", "2.0", "2.8868"):
        pat = "conv_classical_s*" if g == "1.0" else f"ctrl_classical_g{g}_s*"
        e, f, ep = load(pat)
        data[g] = (e, f, ep)
        describe(f"gain {g:<7} {GAIN_MEANING[g]}", e, f, ep)

    ref_g = "1.0"
    lit_g = "1.7321"
    print("\n" + "-" * 118)
    print(" Contrasts against the PyTorch default (gain 1.0) -- the study's 'default' arm")
    print("-" * 118)
    print(f"{'contrast':46}{'delta':>9}{'p_perm':>10}{'splits':>9}  reading")
    ref = data[ref_g][0]
    for g in ("0.5", "1.7321", "2.0", "2.8868"):
        e = data[g][0]
        if not len(e) or not len(ref):
            continue
        d = cliffs_delta(e, ref)
        p, tot = perm_p(e, ref)
        mag = ("negligible" if abs(d) < 0.147 else "small" if abs(d) < 0.33
               else "medium" if abs(d) < 0.474 else "large")
        sign = "worse than default" if d > 0 else "better than default"
        print(f"gain {g:<7} vs gain 1.0{'':22}{d:>9.2f}{p:>10.4f}{tot:>9}  {mag}, {sign}")

    if len(data[lit_g][0]) and len(ref):
        print("\n" + "=" * 118)
        print(" THE QUESTION THIS PART EXISTS TO ANSWER")
        print("=" * 118)
        el, fl = data[lit_g][0], data[lit_g][1]
        er, fr = ref, data[ref_g][1]
        print(f"  PyTorch default (gain 1.0) : median {np.median(er)*100:.2f}%  "
              f"sd {er.std(ddof=1)*100:.2f}%  worst {er.max()*100:.2f}%  "
              f"failures {int(fr.sum())}/{len(er)}")
        print(f"  Xavier   (gain 1.7321)     : median {np.median(el)*100:.2f}%  "
              f"sd {el.std(ddof=1)*100:.2f}%  worst {el.max()*100:.2f}%  "
              f"failures {int(fl.sum())}/{len(el)}")
        better = np.median(el) < np.median(er)
        print()
        if better:
            print("  The literature-standard initialisation BEATS the study's 'default' arm.")
            print("  The 13.2x result is therefore measured against a baseline the field")
            print("  does not use, and section 8.2 must say so. The defensible claim becomes:")
            print("  PyTorch's default is materially worse for PINNs than the Xavier")
            print("  initialisation the literature specifies, and no comparison reports which")
            print("  was used -- which is a sharper and more actionable claim than the")
            print("  current one, not a weaker one.")
        else:
            print("  The literature-standard initialisation does NOT beat the study's")
            print("  'default' arm. The strawman objection is answered empirically: the")
            print("  default is not an unusually bad starting point. Report this axis in")
            print("  full so a reader can see that rather than take it on trust.")


def part_depth():
    print("\n" + "=" * 118)
    print(" DEPTH ROBUSTNESS OF THE ARCHITECTURE x INITIALISATION INTERACTION")
    print(" Strong ansatz. L = 2 is the study's depth, chosen on a SINGLE-SEED sweep.")
    print("=" * 118 + "\n")
    print(HDR)
    print("-" * 118)
    cells = {}
    for depth in (2, 4):
        for init, pre in (("default", "conv"), ("near-identity", "init")):
            e, f, ep = load(f"{pre}_strong_L{depth}_s*")
            cells[(depth, init)] = e
            describe(f"strong  L={depth}  {init}", e, f, ep)

    print("\n" + "-" * 118)
    print(" The near-identity effect, measured separately at each depth")
    print("-" * 118)
    print(f"{'depth':>8}{'delta':>9}{'p_perm':>10}{'splits':>9}   direction")
    signs = {}
    for depth in (2, 4):
        a, b = cells[(depth, "near-identity")], cells[(depth, "default")]
        if not len(a) or not len(b):
            print(f"{depth:>8}{'-- incomplete --':>28}")
            continue
        d = cliffs_delta(a, b)
        p, tot = perm_p(a, b)
        signs[depth] = np.sign(d)
        print(f"{depth:>8}{d:>9.2f}{p:>10.4f}{tot:>9}   "
              f"near-identity {'worse' if d > 0 else 'better'} than default")

    if len(signs) == 2:
        print("\n" + "=" * 118)
        print(" THE QUESTION THIS PART EXISTS TO ANSWER")
        print("=" * 118)
        if signs[2] == signs[4]:
            print("  The near-identity effect has the SAME SIGN at L = 2 and L = 4.")
            print("  The interaction is not an artifact of the depth the single-seed sweep")
            print("  happened to pick. Critique C6 is answered on the merits: disclose the")
            print("  sweep was single-seed, and cite this arm as the robustness check.")
        else:
            print("  The near-identity effect CHANGES SIGN between L = 2 and L = 4.")
            print("  The interaction IS depth-specific. Every claim built on it must be")
            print("  restated as conditional on L = 2, and the single-seed origin of that")
            print("  choice becomes a material limitation rather than a disclosure. This is")
            print("  the outcome worth having run for; report it plainly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["init", "depth", "both"], default="both")
    a = ap.parse_args()
    print(__doc__)
    if a.part in ("init", "both"):
        part_init()
    if a.part in ("depth", "both"):
        part_depth()
    print()


if __name__ == "__main__":
    main()
