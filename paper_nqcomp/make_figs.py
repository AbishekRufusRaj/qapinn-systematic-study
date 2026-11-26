"""
make_figs.py -- the two figures for the NQComp submission.

Every point is one training run read from runs/. Nothing is smoothed, averaged
across arms, or redrawn by hand: the per-seed points are plotted so a reader can
see the spread that medians hide, which is the paper's own argument.

fig1  the published comparison, and what moves it
fig2  what the circuit contributes, in both architectures
"""

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = ".."
GREY, BLUE, RED = "#4d4d4d", "#1f4e79", "#a83232"


def rep(tag):
    return np.array([json.load(open(f))["err_u_pct"]
                     for f in sorted(glob.glob(f"{ROOT}/runs/{tag}_s*/replication.json"))])


def ours(tag, n=8):
    return np.array([json.load(open(f"{ROOT}/runs/{tag}_s{s}/metrics.json"))["rel_l2"] * 100
                     for s in range(n)])


def strip(ax, x, v, color, jitter=0.07):
    r = np.random.default_rng(0).uniform(-jitter, jitter, len(v))
    ax.plot(x + r, v, "o", ms=4, mfc="none", mec=color, alpha=0.85, zorder=3)
    ax.plot([x - 0.22, x + 0.22], [np.median(v)] * 2, "-", color=color, lw=2.2, zorder=4)


# ------------------------------------------------------------------ figure 1
fig, ax = plt.subplots(figsize=(7.2, 3.0))
# The last two arms are the symmetric control: both from the authors' code under
# one protocol. They are the only pair in this figure that differs by nothing
# but the model, which is why they are separated by a rule below.
arms = [
    ("QCPINN\n(quantum)", rep("rep_qcpinn"), BLUE),
    ("baseline\ncode init", rep("rep_model2"), GREY),
    ("baseline\npaper init", rep("rep_model2_paperinit"), GREY),
    ("baseline\nconvergence\nprotocol", ours("pm_model2_ours"), RED),
    ("classical PINN\ntheir protocol", ours("pm_xavier_theirs"), RED),
    ("QCPINN\nsymmetric\nprotocol", rep("sym4_qcpinn"), BLUE),
    ("baseline\nsymmetric\nprotocol", rep("sym4_model2"), GREY),
]
for i, (lab, v, c) in enumerate(arms):
    strip(ax, i, v, c)
ax.axhline(6.69, ls=":", color=BLUE, lw=1, zorder=1)
ax.text(4.45, 17, "published QCPINN\n6.69 $\\pm$ 1.51%", fontsize=6.5, color=BLUE, ha="right")
# rule marking off the symmetric control
ax.axvline(4.5, ls="-", color="#bbbbbb", lw=0.8, zorder=1)
ax.text(5.5, 165, "symmetric control\n(both arms, one protocol)", fontsize=6.5,
        color="#333333", ha="center", va="top")
ax.set_yscale("log")
ax.set_ylabel("relative $L_2$ error (%)", fontsize=8)
ax.set_xticks(range(len(arms)))
ax.set_xticklabels([a[0] for a in arms], fontsize=7)
ax.tick_params(axis="y", labelsize=7)
ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("fig1.pdf", bbox_inches="tight")
fig.savefig("fig1.png", dpi=200, bbox_inches="tight")  # proofing copy
print("fig1: " + "  ".join(f"{a[0].splitlines()[0]} n={len(a[1])} med={np.median(a[1]):.2f}"
                           for a in arms))

# ------------------------------------------------------------------ figure 2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7),
                             gridspec_kw={"width_ratios": [1.05, 1]})

left = [("ring\ncircuit", ours("conv_ring_L2"), BLUE),
        ("ring\nablated", ours("abl_ring"), GREY),
        ("strong\ncircuit", ours("conv_strong_L2"), BLUE),
        ("strong\nablated", ours("abl_strong"), GREY)]
for i, (lab, v, c) in enumerate(left):
    strip(a1, i, v, c)
a1.set_yscale("log")
a1.set_xticks(range(len(left)))
a1.set_xticklabels([l[0] for l in left], fontsize=7)
a1.set_ylabel("relative $L_2$ error (%)", fontsize=8)
a1.set_title("(a) this work's architecture, Burgers'", fontsize=8)
a1.text(0.5, 0.06, "$p = 0.79$", transform=a1.transAxes, fontsize=7, ha="center")
a1.text(0.87, 0.06, "$p = 0.62$", transform=a1.transAxes, fontsize=7, ha="center")

right = [("circuit\ntrained", rep("rep_qcpinn"), BLUE),
         ("circuit\nfrozen", rep("frz_qcpinn"), BLUE),
         ("classical\n$\\sin$", rep("ablsin_qcpinn"), GREY),
         ("classical\n$\\tanh$", rep("abl_qcpinn"), GREY)]
for i, (lab, v, c) in enumerate(right):
    strip(a2, i, v, c)
a2.set_yscale("log")
a2.set_xticks(range(len(right)))
a2.set_xticklabels([r[0] for r in right], fontsize=7)
a2.set_title("(b) the published architecture, Helmholtz", fontsize=8)

for ax in (a1, a2):
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("fig2.pdf", bbox_inches="tight")
print("fig2 left : " + "  ".join(f"{l[0].splitlines()[0]} {np.median(l[1]):.2f}" for l in left))
print("fig2 right: " + "  ".join(f"{r[0].splitlines()[0]} {np.median(r[1]):.2f}" for r in right))
