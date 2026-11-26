"""
factorial_analysis.py -- does the title's claim survive being tested as a contrast?

THE PROBLEM THIS ADDRESSES
--------------------------
The paper is titled "Initialisation dominates architecture". Nothing in it tests that
as a comparison. The initialisation effects (H1, H2) and the architecture effects
(H5, H6) are separate two-sample tests on separate arms, and "dominates" is asserted by
placing their effect sizes next to each other. A referee is entitled to point out that
comparing two effects requires a contrast, not a juxtaposition.

There is a second problem, and it is the larger one. The headline initialisation number
-- 13.2x worst-case movement -- comes from the CLASSICAL model (8.2), while the
architecture numbers come from the QUANTUM arms (8.3). Those are different experiments
on different models. A claim that one factor dominates the other, drawn across two
experiments, is not a measured comparison at all.

WHAT THIS SCRIPT DOES
---------------------
The study already contains a complete balanced 2 x 2 at L = 2:

                      default init        near-identity init
        ring          conv_ring_L2_s*     init_ring_L2_s*
        strong        conv_strong_L2_s*   init_strong_L2_s*

so the within-design question CAN be answered: within the quantum arms, is the
initialisation main effect larger than the ansatz main effect? That is a contrast, it is
measurable, and it is the strongest honest form of the title's claim.

Analysis is on log10(rel_l2). Errors here are multiplicative and right-skewed -- the arms
span 2% to 34% -- so a difference of logs is the scale on which "moves error by a factor
of k" is a constant, which is the quantity the paper actually claims.

Exact enumeration is infeasible for a four-cell design, so main effects and the
interaction use Monte-Carlo permutation with the MC standard error reported. This is
stated because the rest of the paper uses exact tests and the difference must not be
glossed. Main effects use RESTRICTED permutation (labels of one factor shuffled within
levels of the other), which is the correct null for a main effect in a balanced design.

NOT A MEMBER OF THE DECLARED FAMILY. A post-review contrast, reported with effect sizes.
"""

import glob
import json
import os

import numpy as np

RUNS = "runs"
NDRAW = 200_000
RNG = np.random.default_rng(20260920)

CELLS = {
    ("ring", "default"):       "conv_ring_L2_s*",
    ("ring", "near-identity"): "init_ring_L2_s*",
    ("strong", "default"):     "conv_strong_L2_s*",
    ("strong", "near-identity"): "init_strong_L2_s*",
}


def load(pattern):
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, pattern))):
        p = os.path.join(d, "metrics.json")
        if os.path.exists(p):
            out.append(json.load(open(p))["rel_l2"])
    return np.log10(np.array(out)) if out else np.array([])


def med(x):
    return float(np.median(x))


def main():
    print(__doc__)
    cells = {k: load(v) for k, v in CELLS.items()}
    ns = {k: len(v) for k, v in cells.items()}
    if min(ns.values()) == 0:
        print("  some cells are empty -- run the arms first\n")
        return
    if len(set(ns.values())) != 1:
        print(f"  NOTE: cells are unbalanced {ns}. Restricted permutation assumes")
        print("  balance; results below are provisional until every cell is at n = 8.\n")

    print("=" * 96)
    print(" THE 2 x 2, median log10(rel L2)   [in brackets: median rel L2 as a percentage]")
    print("=" * 96)
    print(f"\n{'':10}{'default':>22}{'near-identity':>22}{'init effect':>18}")
    for a in ("ring", "strong"):
        d, n = cells[(a, "default")], cells[(a, "near-identity")]
        row = f"{a:10}"
        for c in (d, n):
            row += f"{med(c):>12.3f} [{10**med(c)*100:>5.2f}%]"
        row += f"{med(n) - med(d):>18.3f}"
        print(row)
    ring_eff = med(cells[("ring", "near-identity")]) - med(cells[("ring", "default")])
    strong_eff = med(cells[("strong", "near-identity")]) - med(cells[("strong", "default")])
    print(f"\n  n per cell: {ns}")

    # ---- the three statistics -------------------------------------------
    def main_effect(factor):
        """median over the two cells at level 2 minus median over the two at level 1."""
        if factor == "ansatz":
            hi = np.concatenate([cells[("strong", i)] for i in ("default", "near-identity")])
            lo = np.concatenate([cells[("ring", i)] for i in ("default", "near-identity")])
        else:
            hi = np.concatenate([cells[(a, "near-identity")] for a in ("ring", "strong")])
            lo = np.concatenate([cells[(a, "default")] for a in ("ring", "strong")])
        return med(hi) - med(lo)

    obs_A = main_effect("ansatz")
    obs_B = main_effect("init")
    obs_I = strong_eff - ring_eff

    # restricted permutation for the main effects
    def perm_main(factor):
        cnt = 0
        if factor == "ansatz":
            blocks = [(cells[("ring", i)], cells[("strong", i)]) for i in ("default", "near-identity")]
        else:
            blocks = [(cells[(a, "default")], cells[(a, "near-identity")]) for a in ("ring", "strong")]
        for _ in range(NDRAW):
            his, los = [], []
            for lo_c, hi_c in blocks:
                pool = np.concatenate([lo_c, hi_c])
                idx = RNG.permutation(len(pool))
                los.append(pool[idx[:len(lo_c)]])
                his.append(pool[idx[len(lo_c):]])
            v = med(np.concatenate(his)) - med(np.concatenate(los))
            cnt += abs(v) >= abs(obs_A if factor == "ansatz" else obs_B) - 1e-15
        return cnt / NDRAW

    pooled = np.concatenate(list(cells.values()))
    sizes = [len(cells[k]) for k in CELLS]

    def perm_inter():
        cnt = 0
        for _ in range(NDRAW):
            idx = RNG.permutation(len(pooled))
            parts, o = [], 0
            for s in sizes:
                parts.append(pooled[idx[o:o + s]]); o += s
            rd, rn, sd_, sn = parts
            v = (med(sn) - med(sd_)) - (med(rn) - med(rd))
            cnt += abs(v) >= abs(obs_I) - 1e-15
        return cnt / NDRAW

    pA, pB, pI = perm_main("ansatz"), perm_main("init"), perm_inter()
    mcse = lambda p: (p * (1 - p) / NDRAW) ** 0.5

    print("\n" + "=" * 96)
    print(" MAIN EFFECTS AND INTERACTION   (log10 units; 10^x is the factor on error)")
    print("=" * 96)
    print(f"\n{'term':34}{'effect':>10}{'factor':>10}{'p_MC':>10}{'MCSE':>9}  null")
    print("-" * 96)
    print(f"{'ansatz (strong - ring)':34}{obs_A:>10.3f}{10**obs_A:>9.2f}x{pA:>10.4f}"
          f"{mcse(pA):>9.4f}  restricted")
    print(f"{'init (near-id - default)':34}{obs_B:>10.3f}{10**obs_B:>9.2f}x{pB:>10.4f}"
          f"{mcse(pB):>9.4f}  restricted")
    print(f"{'ansatz x init interaction':34}{obs_I:>10.3f}{10**obs_I:>9.2f}x{pI:>10.4f}"
          f"{mcse(pI):>9.4f}  global")

    # ---- the title's claim, as a contrast --------------------------------
    print("\n" + "=" * 96)
    print(" DOES INITIALISATION DOMINATE ARCHITECTURE, WITHIN THIS DESIGN?")
    print("=" * 96)
    diff = abs(obs_B) - abs(obs_A)
    boot = []
    for _ in range(20000):
        rs = {k: RNG.choice(v, len(v), replace=True) for k, v in cells.items()}
        hiA = np.concatenate([rs[("strong", i)] for i in ("default", "near-identity")])
        loA = np.concatenate([rs[("ring", i)] for i in ("default", "near-identity")])
        hiB = np.concatenate([rs[(a, "near-identity")] for a in ("ring", "strong")])
        loB = np.concatenate([rs[(a, "default")] for a in ("ring", "strong")])
        boot.append(abs(med(hiB) - med(loB)) - abs(med(hiA) - med(loA)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  |init effect| - |ansatz effect| = {diff:+.3f} log10 units")
    print(f"  bootstrap 95% CI               = [{lo:+.3f}, {hi:+.3f}]  (20,000 resamples)")
    print()
    if lo > 0:
        print("  The CI excludes zero from above: within the quantum arms, initialisation")
        print("  has the larger main effect. The title is supported AS A WITHIN-DESIGN")
        print("  CLAIM about these arms -- state it that way, and cite this contrast.")
    elif hi < 0:
        print("  The CI excludes zero from BELOW: the ansatz has the larger main effect.")
        print("  The title is not supported and must change.")
    else:
        print("  The CI SPANS ZERO. The two main effects are not distinguishable at this")
        print("  sample size, so 'initialisation dominates architecture' is not supported")
        print("  as a measured contrast. The defensible claim is the one section 1 already")
        print("  makes: initialisation produces effects OF THE SAME MAGNITUDE as the")
        print("  architectural differences being reported -- which is enough to establish")
        print("  that leaving it uncontrolled invalidates the comparison, and does not")
        print("  require dominance. Retitle accordingly.")

    print("\n" + "-" * 96)
    print(" AND THE LARGER ISSUE, WHICH THIS SCRIPT CANNOT FIX")
    print("-" * 96)
    print("  The 13.2x figure in the abstract is the CLASSICAL model's init response")
    print("  (section 8.2); the ansatz effects above are QUANTUM arms (section 8.3).")
    print("  Comparing them is a cross-experiment juxtaposition, not a contrast, and no")
    print("  reanalysis of existing runs can make it one. Either the title's claim is")
    print("  scoped to the within-design contrast computed above, or it is scoped to")
    print("  'both models respond to initialisation at magnitudes comparable to the")
    print("  architectural differences reported in this literature' -- which is what the")
    print("  evidence actually supports.\n")


if __name__ == "__main__":
    main()
