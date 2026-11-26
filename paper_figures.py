#!/usr/bin/env python3
"""
The paper's figures, built straight from runs/ -- no intermediate CSV, so a
figure can never disagree with the run directory it came from.

(make_figures.py is the OLD script. It reads sweep_results.csv and draws the
Study-1-era sweep panels. Those figures are historical -- single seed, fixed
budget -- and none of them belong in the paper. This file replaces it for
publication figures; make_figures.py is left alone so the old sweep plots
stay reproducible.)

WHAT RUNS WITHOUT TORCH
=======================
Figures 1-5 read only `loss_history.npy` and `metrics.json`, which are numpy
and JSON. They need matplotlib and nothing else, so they build anywhere.

Figures 6-7 (per-neuron XAI feature maps, layer-wise spectral content) need a
forward pass through trained models, so they need torch + pennylane and belong
to xai_feature_maps.py. Figure 8 (per-mode decomposition) needs
mode_decomposition.py to have been run first, which also needs torch. This
script reports what it could not build rather than silently emitting five
figures and letting someone think that was all of them.

HOW RUNS ARE SELECTED
=====================
Not by tag prefix alone. Tag prefixes say which STUDY a run belongs to, but the
seed-0 runs of Study 1 were produced by the original sweep and carry sweep
names (`classical_baseline`, `ansatz_strong_L2_q4`, ...). So study membership
comes from the tag and model identity comes from `meta.json` -- `entangler`,
`reupload_layers`, `init_scale`, `init_gain`, `convergence_trained`. If a run's
metadata does not match what the figure claims to plot, it is not plotted, and
the script says so. Silently plotting the wrong seed-0 run is exactly the kind
of error that produced this project's "budget flip" (handoff §6c).

COLOUR
======
Okabe-Ito, the standard colourblind-safe scientific set, restricted to three
hues that were checked rather than assumed:

    classical #0072B2   ring #D55E00   strong #009E73

Worst adjacent pair under deuteranopia is dE 11.0 (target >= 8), under normal
vision 25.8 (floor 15), and all three clear 3:1 contrast against white. Every
series also carries a distinct linestyle and marker, so identity never rests on
colour alone -- which is what makes the figures survive a greyscale print run.

USAGE
=====
    python3 paper_figures.py                 # all buildable figures -> figures/
    python3 paper_figures.py --figs 2 3      # just those
    python3 paper_figures.py --format pdf    # vector only, for LaTeX
"""

import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUNS = "runs"
OUT = "figures"

# ---- colour and style ----------------------------------------------------
# Keys are the internal model names used throughout the study.
C = {"classical": "#0072B2", "ring": "#D55E00", "strong": "#009E73",
     "other": "#A0439B"}
LS = {"classical": "-", "ring": "--", "strong": "-."}
MK = {"classical": "o", "ring": "s", "strong": "^"}
LABEL = {"classical": "Classical PINN",
         "ring": "QAPINN (ring)",
         "strong": "QAPINN (strong)"}

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e2e2e2"

# The pre-registered non-convergence threshold. A run whose FINAL TOTAL LOSS
# exceeds this is a failure, declared before the runs were made. It is a loss
# threshold, not an error threshold -- so it is read from meta, never inferred
# from rel_l2.
FAIL_LOSS = 1e-2


def style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
    })


def despine(ax, left=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    ax.xaxis.grid(False)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_meta(d):
    with open(os.path.join(d, "meta.json")) as f:
        return json.load(f)


def load_metrics(d, checkpoint=None):
    f = "metrics.json" if checkpoint is None else f"metrics_ep{checkpoint}.json"
    p = os.path.join(d, f)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_history(d):
    """loss_history.npy is a 0-d object array wrapping a dict of arrays:
    total / pde / ic / bc / phase. phase 0 = Adam, 1 = L-BFGS."""
    h = np.load(os.path.join(d, "loss_history.npy"), allow_pickle=True)
    return h.item() if h.shape == () else h


def failed(meta):
    return float(meta.get("final_total_loss", np.inf)) > FAIL_LOSS


def collect(pattern, want, need_metrics=True):
    """Run dirs matching `pattern` whose meta satisfies the predicate `want`.

    Returns a list of (dir, meta, metrics), sorted by seed. Mismatches are not
    silently dropped -- the caller reports how many it found so a missing seed
    is visible rather than invisible.
    """
    out = []
    for d in sorted(glob(os.path.join(RUNS, pattern))):
        if not os.path.isdir(d):
            continue
        try:
            m = load_meta(d)
        except (OSError, json.JSONDecodeError):
            continue
        if not want(m):
            continue
        met = load_metrics(d) if need_metrics else None
        if need_metrics and met is None:
            continue
        out.append((d, m, met))
    return sorted(out, key=lambda r: r[1].get("seed", 0))


def is_(kind, **kw):
    """Predicate factory, tolerant of the TWO meta.json schemas in runs/.

    The Study-1-era sweep runs predate pde_train.py and record
    `classical: true/false` with no `model_type`, no `convergence_trained`, and
    no `entangler` key at all when the ring ansatz was used (it was the CLI
    default). The later runs record `model_type`, `convergence_trained` and an
    explicit `entangler`. Normalising here rather than editing 40+ old
    meta.json files keeps those runs untouched and reproducible.

    is_('classical', convergence_trained=False) matches a fixed-budget
    classical run; is_('strong', init_scale=0.1) the near-identity QAPINN.
    """
    def want(m):
        mk = m.get("model_kwargs", {})

        # --- normalise the model identity across both schemas
        mt = m.get("model_type")
        if mt is None:
            mt = "classical" if m.get("classical") else "qapinn"
        if kind == "classical":
            if mt != "classical":
                return False
        else:
            if mt != "qapinn":
                return False
            # `entangler` absent means the old default, which was the ring.
            if mk.get("entangler", "ring") != kind:
                return False

        # --- normalise the queried fields
        for k, v in kw.items():
            if k == "convergence_trained":
                got = m.get("convergence_trained", False)
            elif k in m:
                got = m[k]
            else:
                got = mk.get(k)          # absent -> None, which matches
            if got != v:                 # init_scale=None on old runs is correct:
                return False             # near-identity init did not exist yet
        return True
    return want


def rel(met):
    return met["rel_l2"]


def note(msg):
    print(f"    {msg}")


# --------------------------------------------------------------------------
# Figure 1 -- plateau-then-breakthrough
# --------------------------------------------------------------------------

def fig1(fmt):
    """The figure the whole protocol argument rests on.

    Five seeds per model, fixed 8,000-epoch budget (Study 1), loss on a log
    axis. What the reader should see: a long flat plateau, then a near-vertical
    drop whose TIMING varies by seed, and a cutoff line that several runs have
    not reached their drop by. The claim "a fixed budget measures which seeds
    finished, not which model is better" is visible rather than asserted.

    Only the Adam phase is drawn (phase == 0). L-BFGS is a different optimiser
    on a different x-scale (500 steps against 8,000 epochs) and was separately
    shown to improve every run -- splicing it on would compress the part of the
    plot that carries the argument.
    """
    groups = [("classical", collect("*", is_("classical", convergence_trained=False),
                                    need_metrics=False)),
              ("ring", collect("*", is_("ring", convergence_trained=False,
                                        reupload_layers=2), need_metrics=False)),
              ("strong", collect("*", is_("strong", convergence_trained=False,
                                          reupload_layers=2), need_metrics=False))]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, (kind, runs) in zip(axes, groups):
        note(f"fig1 {kind}: {len(runs)} runs -> "
             f"{[os.path.basename(d) for d, _, _ in runs]}")
        for d, m, _ in runs:
            h = load_history(d)
            tot, ph = np.asarray(h["total"]), np.asarray(h["phase"])
            adam = tot[ph == 0]
            ax.plot(np.arange(len(adam)), adam, color=C[kind], alpha=0.75,
                    linewidth=1.0)
        ax.axvline(8000, color=MUTED, linewidth=0.9, linestyle=":")
        ax.set_xlim(-300, 8800)          # headroom, or the cutoff line sits
        ax.set_yscale("log")             # on the spine and reads as a border
        ax.set_title(LABEL[kind], color=C[kind])
        ax.set_xlabel("Adam epoch")
        despine(ax)
    axes[0].set_ylabel("total loss")
    for ax in axes:
        ax.annotate("8,000-epoch budget", xy=(8000, 0.03),
                    xycoords=("data", "axes fraction"), rotation=90,
                    ha="right", va="bottom", fontsize=6.5, color=MUTED)
    fig.suptitle("Training is plateau-then-breakthrough; breakthrough time varies by seed",
                 fontsize=9.5, y=1.04)
    save(fig, "fig1_trajectories", fmt)


# --------------------------------------------------------------------------
# Figure 2 -- classical initialisation control
# --------------------------------------------------------------------------

def fig2(fmt):
    """A scalar on the initialisation of a purely classical PINN, and what it
    does to the error distribution. Every seed is drawn as its own point --
    with n=5 a bar of the mean hides the whole result, which is about SPREAD
    and the WORST CASE, not about the mean.

    Failures (final loss > 1e-2, pre-registered) are drawn as open markers, so
    the reader can see that gain 0.5 does not merely score worse, it stops
    converging.
    """
    # The full axis, including the two literature-standard points added after
    # review. gain sqrt(3)=1.7321 is Xavier/Glorot for the seven hidden 20x20
    # layers that hold 2,641 of 3,021 parameters; 2.8868 is Xavier scaled by the
    # recommended tanh gain of 5/3. Section 8.2's table shows all five, so this
    # figure must too -- a figure plotting a subset of the table beside it is the
    # kind of quiet mismatch this study is about.
    arms = [("gain 0.5", collect("ctrl_classical_g0.5_*", is_("classical"))),
            ("gain 1.0\n(PyTorch\ndefault)", collect("conv_classical_*", is_("classical"))),
            ("1.7321\n(Xavier)", collect("ctrl_classical_g1.7321_*", is_("classical"))),
            ("gain 2.0", collect("ctrl_classical_g2.0_*", is_("classical"))),
            ("2.8868\n(Xavier\n x tanh)", collect("ctrl_classical_g2.8868_*", is_("classical")))]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    rng = np.random.default_rng(0)   # jitter only, never affects a value
    for i, (name, runs) in enumerate(arms):
        note(f"fig2 {name.splitlines()[0]}: {len(runs)} seeds")
        vals = np.array([rel(met) for _, _, met in runs])
        fails = np.array([failed(m) for _, m, _ in runs])
        x = i + rng.uniform(-0.09, 0.09, len(vals))
        ax.scatter(x[~fails], vals[~fails], s=26, color=C["classical"],
                   zorder=3, linewidths=0)
        ax.scatter(x[fails], vals[fails], s=30, facecolors="white", zorder=3,
                   edgecolors=C["classical"], linewidths=1.2)
        med = np.median(vals)
        ax.plot([i - 0.22, i + 0.22], [med, med], color=INK, linewidth=1.6, zorder=4)
        ax.annotate(f"{med:.2%}", (i + 0.26, med), fontsize=7.5, color=INK,
                    va="center")

    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms])
    ax.set_yscale("log")
    ax.set_ylabel("relative $L_2$ error")
    ax.set_xlabel("initialisation gain (classical PINN)")
    ax.set_title("Initialisation alone moves a classical PINN's\nworst case by 13$\\times$",
                 fontsize=9)
    despine(ax)
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=C["classical"],
               markersize=5, label="converged seed"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor=C["classical"], markersize=5,
               label="failed (final loss > $10^{-2}$)"),
        Line2D([], [], color=INK, linewidth=1.6, label="median"),
    ], loc="upper right")
    save(fig, "fig2_init_gain", fmt)


# --------------------------------------------------------------------------
# Figure 3 -- the ansatz x initialisation interaction
# --------------------------------------------------------------------------

def fig3(fmt):
    """The project's sharpest result -- and a demonstration of why the summary
    statistic has to be chosen before the result is read.

    Three panels, and the third one is the point.

    (a) every seed, failures as open markers. With n=5 and a bimodal outcome
        (a run breaks through or it does not), the distribution IS the result;
        any single summary number throws away what matters.

    (b) the interaction on MEDIANS. Near-identity initialisation barely moves
        the strong ansatz and badly damages the ring: the lines diverge, and
        strong is below ring at BOTH initialisations.

    (c) the same interaction on all-seed MEANS. Here the lines CROSS, which
        would license the claim "initialisation inverts the ansatz ranking".

    They disagree, and (b) is the one to believe. The crossing in (c) exists
    because strong/default contains one failed seed at 33.86%, which drags a
    five-sample mean above ring's 6.78%. On medians (2.37% vs 5.43%) and on
    converged-only means (2.41% vs 6.78%), strong is ahead at both
    initialisations and nothing inverts.

    This project already documented that mean orderings are unreliable under
    skew -- the cross-PDE "ordering flip" was one 9.29% seed -- and taught
    summarize_pde.py to refuse to call a flip unless mean and median agree.
    The same discipline has to apply to this figure's own headline, which is
    why the inversion claim is retired here rather than illustrated.

    What survives on EVERY statistic: the same intervention moves the two
    ansaetze in OPPOSITE DIRECTIONS. Strong goes 1/5 failures to 0/5, worst
    case 33.86% to 2.91%. Ring goes 0/5 to 1/5, worst 13.35% to 37.45%. That
    is the claim the gate-algebra mechanism (RX/RX commutation) explains, and
    it does not need an inverted ranking to be interesting.
    """
    cells = [("ring", "default", collect("conv_ring_L2_*", is_("ring", init_scale=None))),
             ("ring", "near-identity", collect("init_ring_L2_*", is_("ring", init_scale=0.1))),
             ("strong", "default", collect("conv_strong_L2_*", is_("strong", init_scale=None))),
             ("strong", "near-identity", collect("init_strong_L2_*", is_("strong", init_scale=0.1)))]

    data = {}
    for ansatz, init, runs in cells:
        vals = np.array([rel(met) for _, _, met in runs])
        fails = np.array([failed(m) for _, m, _ in runs])
        data[(ansatz, init)] = (vals, fails)
        note(f"fig3 {ansatz}/{init}: n={len(vals)} median {np.median(vals):.2%} "
             f"mean {vals.mean():.2%} fails {int(fails.sum())}/{len(vals)}")

    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(7.4, 3.0),
        gridspec_kw={"width_ratios": [1.5, 1, 1], "wspace": 0.45})
    inits = ["default", "near-identity"]
    rng = np.random.default_rng(1)

    # ---- (a) every seed
    ticks = []
    for i, ansatz in enumerate(["ring", "strong"]):
        for j, init in enumerate(inits):
            p = i * 2.6 + j
            ticks.append((p, init))
            vals, fails = data[(ansatz, init)]
            x = p + rng.uniform(-0.11, 0.11, len(vals))
            axA.scatter(x[~fails], vals[~fails], s=24, color=C[ansatz], zorder=3,
                        linewidths=0)
            axA.scatter(x[fails], vals[fails], s=28, facecolors="white",
                        edgecolors=C[ansatz], linewidths=1.2, zorder=3)
            med = np.median(vals)
            axA.plot([p - 0.26, p + 0.26], [med, med], color=INK, linewidth=1.6,
                     zorder=4)
    axA.set_xticks([t[0] for t in ticks])
    axA.set_xticklabels([t[1] for t in ticks], fontsize=6.8, rotation=22, ha="right")
    for i, ansatz in enumerate(["ring", "strong"]):
        axA.annotate(LABEL[ansatz], xy=(i * 2.6 + 0.5, -0.34),
                     xycoords=("data", "axes fraction"), ha="center",
                     fontsize=7.5, color=C[ansatz], annotation_clip=False)
    axA.set_yscale("log")
    axA.set_ylabel("relative $L_2$ error")
    axA.set_title("(a) every seed", fontsize=8.5)
    axA.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=4.5,
               label="converged"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor=MUTED, markersize=4.5, label="failed"),
        Line2D([], [], color=INK, linewidth=1.6, label="median"),
    ], loc="lower left", fontsize=6.5, handletextpad=0.5, borderpad=0.2)
    despine(axA)

    # ---- (b) and (c): the same interaction under two statistics
    for ax, stat, name, tag in (
            (axB, np.median, "median", "(b) median: no inversion"),
            (axC, np.mean, "all-seed mean", "(c) mean: crosses")):
        for ansatz in ["ring", "strong"]:
            ys = [float(stat(data[(ansatz, i)][0])) for i in inits]
            ax.plot([0, 1], ys, color=C[ansatz], linestyle=LS[ansatz],
                    marker=MK[ansatz], markersize=6, label=LABEL[ansatz])
            for k, y in enumerate(ys):
                # left point -> label to its left, right point -> to its right.
                # Stacking above/below collides wherever the two lines meet.
                ax.annotate(f"{y:.1%}", (k, y), textcoords="offset points",
                            xytext=(-7, 0) if k == 0 else (7, 0),
                            ha="right" if k == 0 else "left", va="center",
                            fontsize=6.8, color=INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(inits, fontsize=7, rotation=22, ha="right")
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(1.1e-2, 6e-1)
        ax.set_yscale("log")
        ax.set_title(tag, fontsize=8.5)
        ax.set_ylabel(f"{name} rel. $L_2$ error", fontsize=7.5)
        despine(ax)
    fig.legend(handles=[Line2D([], [], color=C[a], linestyle=LS[a], marker=MK[a],
                               markersize=5, label=LABEL[a])
                        for a in ("ring", "strong")],
               loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.62, -0.14))

    fig.suptitle("Same intervention, opposite sign on each ansatz -- but the "
                 "ranking inversion is one failed seed", fontsize=9, y=1.08)
    save(fig, "fig3_interaction", fmt)

# --------------------------------------------------------------------------
# Figure 4 -- budget curve
# --------------------------------------------------------------------------

def fig4(fmt):
    """Error against training budget, with every budget point taken from the
    SAME trajectory (checkpoints at 800/2000/8000, then convergence). That is
    why this figure can be read as a budget effect at all: with independent
    runs per budget, seed noise and budget would be confounded.

    This came back NEGATIVE for the budget-flip hypothesis -- the ordering is
    the same at every budget. It is in the paper because the surviving result
    matters: absolute error at 800 epochs is an order of magnitude off
    convergence, so magnitudes reported from short budgets are noise.
    """
    ckpts = [800, 2000, 8000, None]
    labels = ["800", "2,000", "8,000", "converged"]
    fig, ax = plt.subplots(figsize=(3.6, 2.9))

    for kind, pat in [("classical", "bc_classical_*"), ("ring", "bc_ring_L2_*"),
                      ("strong", "bc_strong_L2_*")]:
        runs = collect(pat, is_(kind), need_metrics=False)
        note(f"fig4 {kind}: {len(runs)} runs")
        med, lo, hi = [], [], []
        for c in ckpts:
            v = [m["rel_l2"] for d, _, _ in runs
                 if (m := load_metrics(d, c)) is not None]
            if not v:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            v = np.array(v)
            med.append(np.median(v))
            lo.append(np.percentile(v, 25))
            hi.append(np.percentile(v, 75))
        x = np.arange(len(ckpts))
        # Median with an interquartile band, not mean +- std: these
        # distributions are bimodal (break through or don't), and a symmetric
        # error bar on a bimodal sample draws a region containing no runs.
        ax.fill_between(x, lo, hi, color=C[kind], alpha=0.15, linewidth=0)
        ax.plot(x, med, color=C[kind], linestyle=LS[kind], marker=MK[kind],
                markersize=5, label=LABEL[kind])

    ax.set_xticks(np.arange(len(ckpts)))
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_xlabel("Adam budget (epochs)")
    ax.set_ylabel("relative $L_2$ error (median, IQR)")
    ax.set_title("Ordering does not flip with budget;\nshort-budget magnitudes are 10-15$\\times$ off",
                 fontsize=9)
    ax.legend(loc="lower left")
    despine(ax)
    save(fig, "fig4_budget_curve", fmt)


# --------------------------------------------------------------------------
# Figure 5 -- cross-PDE
# --------------------------------------------------------------------------

def fig5(fmt):
    """The generality check, on medians.

    Medians and not means, deliberately and for a reason that belongs in the
    caption: on Burgers' the classical mean (3.36%) is carried by a single
    9.29% seed, and an earlier version of the analysis declared a cross-PDE
    "ordering flip" that was purely that one seed. On medians the ordering is
    the same on both PDEs. The two panels have independent y-scales because
    heat errors are an order of magnitude smaller -- a shared scale would
    flatten the heat panel into a line.
    """
    panels = [
        ("Burgers' (shock)", [("classical", "conv_classical_*", is_("classical")),
                              ("ring", "conv_ring_L2_*", is_("ring", init_scale=None)),
                              ("strong", "init_strong_L2_*", is_("strong", init_scale=0.1))]),
        ("Heat (multi-mode)", [("classical", "heat_classical_*", is_("classical")),
                               ("ring", "heat_ring_L2_*", is_("ring")),
                               ("strong", "heat_strong_L2_*", is_("strong", init_scale=0.1))]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    rng = np.random.default_rng(2)
    for ax, (title, specs) in zip(axes, panels):
        for i, (kind, pat, pred) in enumerate(specs):
            runs = collect(pat, pred)
            note(f"fig5 {title} {kind}: {len(runs)} seeds")
            vals = np.array([rel(met) for _, _, met in runs])
            fails = np.array([failed(m) for _, m, _ in runs])
            x = i + rng.uniform(-0.09, 0.09, len(vals))
            ax.scatter(x[~fails], vals[~fails], s=26, color=C[kind],
                       zorder=3, linewidths=0)
            ax.scatter(x[fails], vals[fails], s=30, facecolors="white",
                       edgecolors=C[kind], linewidths=1.2, zorder=3)
            med = np.median(vals)
            ax.plot([i - 0.24, i + 0.24], [med, med], color=INK, linewidth=1.6,
                    zorder=4)
            ax.annotate(f"{med:.2%}", (i + 0.28, med), fontsize=7.5, va="center")
        ax.set_xticks(range(len(specs)))
        ax.set_xticklabels(["classical", "ring", "strong"], fontsize=8)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.5, len(specs) - 0.2)
        despine(ax)
    axes[0].set_ylabel("relative $L_2$ error")
    fig.suptitle("Median ordering is identical on both PDEs (each point is one seed)",
                 fontsize=9.5, y=1.03)
    save(fig, "fig5_cross_pde", fmt)


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Figure 6 -- per-neuron feature maps (assembled, not recomputed)
# --------------------------------------------------------------------------

XAI_RUNS = [("classical", "conv_classical_s0"),
            ("strong", "init_strong_L2_s0"),
            ("other", "qc_qcpinn_s2")]
XAI_TITLE = {"conv_classical_s0": "Classical PINN",
             "init_strong_L2_s0": "QAPINN (strong)",
             "qc_qcpinn_s2": "QCPINN"}


def _xai_metrics():
    """Load xai_metrics_<tag>.json for the three runs the paper compares.

    These are produced by xai_feature_maps.py, which needs torch. The JSON it
    leaves behind does not, so every layer-wise number in the paper can be
    replotted here without the quantum stack -- which is the point of writing
    the metrics to disk rather than only into a figure.
    """
    out = []
    for kind, tag in XAI_RUNS:
        p = os.path.join(RUNS, tag, f"xai_metrics_{tag}.json")
        if not os.path.exists(p):
            note(f"missing {p} -- run: xai_feature_maps.py runs/{tag}")
            continue
        with open(p) as f:
            out.append((kind, tag, json.load(f)))
    return out


def fig6(fmt):
    """Figure 6 is the per-neuron activation grid, which cannot be rebuilt
    without a forward pass. xai_feature_maps.py already wrote one PNG per run;
    this just reports where they are so the figure list stays complete and
    nobody goes looking for a fig6_*.png that was never meant to exist.
    """
    for _kind, tag in XAI_RUNS:
        p = os.path.join(RUNS, tag, f"feature_maps_{tag}.png")
        note(("have " if os.path.exists(p) else "MISSING ") + p)
    note("fig 6 is these three panels side by side -- assemble in LaTeX with")
    note("subfigure, or regenerate via: xai_feature_maps.py <runs> --compare")


# --------------------------------------------------------------------------
# Figure 7 -- layer-wise spectral content and utilisation
# --------------------------------------------------------------------------

def fig7(fmt):
    """How high-frequency content and neuron utilisation evolve along the
    forward pass, for the three architectures.

    The three quantities answer three different questions and are NOT put on
    one pair of axes (a dual-axis chart would make the crossing points
    meaningless):

    (a) HF energy fraction -- the share of a layer's non-DC spectral energy
        above mode 2. The classical PINN BUILDS this through depth: 3.0% at
        layer 1 to 44.9% at layer 8. High frequency is not present at the
        input; the network composes it.

    (b) highest mode present -- the accessible-spectrum question. The QAPINN's
        quantum layer was MEASURED to reach mode 4 (fourier_analysis.py) but
        the trained layer carries only mode 1. Capacity present, unused.

    (c) utilisation, PR / n_neurons -- how many units do independent work. The
        4-wire quantum layer sits at 76%, the highest of any layer in the
        study; QCPINN's 50-unit post-processor collapses to 12%.

    Together these say the quantum layer is a frequency BOTTLENECK rather than
    a frequency source: it is the most efficiently used layer in the study and
    it still hands its successors less bandwidth than the classical layer it
    replaced.

    x is position along the forward pass, not a shared semantic axis -- the
    three models have different depths, so the lines are read as trajectories,
    and the quantum layer is ringed wherever it occurs.
    """
    data = _xai_metrics()
    if not data:
        note("no xai_metrics_*.json found -- skipping fig 7")
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    specs = [("hf_energy_fraction", "high-frequency energy share", True),
             ("max_freq_present", "highest mode present", False),
             ("pr_fraction", "utilisation (PR / units)", True)]

    for ax, (key, ylab, as_pct) in zip(axes, specs):
        for kind, tag, d in data:
            names = list(d["layers"])
            ys = [d["layers"][n][key] for n in names]
            xs = np.arange(1, len(names) + 1)
            ax.plot(xs, ys, color=C[kind], linestyle=LS.get(kind, "-"),
                    marker=MK.get(kind, "D"), markersize=4.5,
                    label=XAI_TITLE[tag], linewidth=1.3)
            # ring the quantum layer wherever it sits in the stack
            for i, n in enumerate(names):
                if n == "Q-layer":
                    ax.scatter([xs[i]], [ys[i]], s=95, facecolors="none",
                               edgecolors=INK, linewidths=1.1, zorder=5)
        ax.set_xlabel("layer (forward-pass order)")
        ax.set_ylabel(ylab, fontsize=7.5)
        if as_pct:
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        despine(ax)

    axes[1].annotate("circled = quantum layer", xy=(0.5, 1.13),
                     xycoords="axes fraction", ha="center", fontsize=6.8,
                     color=MUTED, annotation_clip=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("The quantum layer is the most-utilised layer in the study "
                 "and the narrowest in bandwidth", fontsize=9, y=1.04)
    fig.tight_layout()
    save(fig, "fig7_layer_spectral", fmt)


# --------------------------------------------------------------------------
# Figure 8 -- per-mode decomposition, heat equation
# --------------------------------------------------------------------------

FIG8_CFGS = [("classical", "heat_classical"), ("ring", "heat_ring_L2"),
             ("strong", "heat_strong_L2")]


def fig8(fmt):
    """Which mode was lost, and when -- the replacement for the early/late split
    that turned out to measure IC anchoring rather than spectral difficulty
    (handoff section 6e).

    THE FIRST VERSION OF THIS FIGURE PLOTTED THE AMPLITUDES THEMSELVES and was
    useless: every model tracks every mode to within half a percent, so four
    curves lay exactly on top of each other and the figure showed nothing. The
    quantity with information in it is the RESIDUAL, so that is what is drawn.

    Each panel is one mode present in the initial condition. y is the signed
    error in that mode's amplitude,

        e_k(t) = c_k^pred(t) - c_k^exact(t),   as a fraction of a_k

    normalised by a_k, the mode's amplitude at t = 0, and NOT by c_k^exact(t).
    That denominator decays toward zero, and dividing by it manufactures huge
    late-time "errors" for a model that is predicting a small number slightly
    wrong -- the exact trap that broke the early/late metric this figure
    replaces. Lines are the median over five seeds, bands the interquartile
    range. The dotted vertical in the k=5 panel is where that mode has decayed
    to 5% of a_k; past it there is nothing left to get right, and the summary
    statistics stop averaging there.

    WHAT IT SHOWS, AND IT IS A NEGATIVE RESULT
    ------------------------------------------
    The prediction was that the QAPINN should lose mode 5 specifically: the
    angle encoding's accessible spectrum was measured to cut off at mode 4, and
    the trained quantum layer carries only mode 1. It does not happen. All
    three architectures track all three modes to within 0.4% of initial
    amplitude, and the k5/k1 error ratios (classical 3.56, ring 2.59, strong
    2.89) do not separate at n = 5 -- if anything the quantum models are less
    skewed toward mode-5 error than the classical baseline.

    That is worth stating plainly rather than burying, because it answers the
    BQP brief's "what is lost because of the quantum layer" with evidence: the
    frequency bottleneck is real AT the quantum layer and does not propagate to
    a per-mode accuracy penalty, because the seven classical layers behind it
    rebuild the bandwidth. The bottleneck is a fact about the layer, not about
    the model. It also reinforces the paper's thesis from a new direction --
    architecture is not the binding constraint here either.

    Needs mode_decomposition.py to have been run.
    """
    found = {}
    for kind, prefix in FIG8_CFGS:
        files = sorted(glob(os.path.join(RUNS, prefix + "_s*", "mode_coeffs.npz")))
        if files:
            found[kind] = files
        else:
            note(f"no mode_coeffs.npz for {prefix}_s* -- run mode_decomposition.py")
    if not found:
        note("skipping fig 8")
        return

    sample = np.load(next(iter(found.values()))[0])
    t, c_true = sample["t"], sample["c_true"]
    modes = [k for k in range(1, c_true.shape[1] + 1)
             if np.abs(c_true[:, k - 1]).max() > 0]

    # alive window: where the mode still retains >= 5% of its t=0 amplitude.
    # Mirrors LIVE_FRAC in mode_decomposition.py -- if that changes, this
    # annotation is wrong, so it is derived from the data rather than hardcoded.
    LIVE = 0.05

    fig, axes = plt.subplots(1, len(modes), figsize=(2.5 * len(modes), 2.7))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, modes):
        a_k = c_true[0, k - 1]
        ax.axhline(0, color=INK, linewidth=0.9, linestyle=(0, (4, 2)), zorder=4)
        for kind, files in found.items():
            stack = np.stack([(np.load(f)["c_pred"][:, k - 1] - c_true[:, k - 1]) / a_k
                              for f in files])
            ax.plot(t, np.median(stack, 0), color=C[kind], linestyle=LS[kind],
                    linewidth=1.3, label=LABEL[kind])
            ax.fill_between(t, np.percentile(stack, 25, axis=0),
                            np.percentile(stack, 75, axis=0),
                            color=C[kind], alpha=0.15, linewidth=0)

        alive = t[np.abs(c_true[:, k - 1]) >= LIVE * abs(a_k)]
        if len(alive) and alive[-1] < t[-1]:
            ax.axvline(alive[-1], color=MUTED, linewidth=0.9, linestyle=":")
            ax.annotate(f"decayed to {LIVE:.0%}", xy=(alive[-1], 1.02),
                        xycoords=("data", "axes fraction"), ha="center",
                        va="bottom", fontsize=6.2, color=MUTED,
                        annotation_clip=False)

        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1%}"))
        ax.set_title(f"mode $k={k}$   ($a_k$ = {a_k:.3f})", fontsize=8.5)
        ax.set_xlabel("$t$")
        despine(ax)
    axes[0].set_ylabel("amplitude error, fraction of $a_k$", fontsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("No mode is lost: every architecture tracks every mode to better "
                 "than 1% of its initial amplitude at all times", fontsize=9, y=1.04)
    fig.tight_layout()
    save(fig, "fig8_mode_decomposition", fmt)


def save(fig, name, fmt):
    os.makedirs(OUT, exist_ok=True)
    for ext in (["png", "pdf"] if fmt == "both" else [fmt]):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p)
        print(f"  wrote {p}")
    plt.close(fig)


BUILDABLE = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5,
             6: fig6, 7: fig7, 8: fig8}

# Figures 6-8 are built here from artifacts on disk, but those artifacts are
# produced by scripts that DO need torch. This says which script to run first.
UPSTREAM = {
    6: "xai_feature_maps.py runs/conv_classical_s0 runs/init_strong_L2_s0 runs/qc_qcpinn_s2",
    7: "xai_feature_maps.py runs/conv_classical_s0 runs/init_strong_L2_s0 runs/qc_qcpinn_s2",
    8: "mode_decomposition.py",
}


def main():
    p = argparse.ArgumentParser(description="Build the paper's figures from runs/.")
    p.add_argument("--figs", nargs="*", type=int, default=sorted(BUILDABLE))
    p.add_argument("--format", choices=["png", "pdf", "both"], default="both")
    p.add_argument("--runs", default=RUNS)
    a = p.parse_args()

    globals()["RUNS"] = a.runs
    if not os.path.isdir(RUNS):
        print(f"no such run directory: {RUNS}", file=sys.stderr)
        sys.exit(1)

    style()
    for n in a.figs:
        if n not in BUILDABLE:
            print(f"\nfigure {n}: unknown")
            continue
        print(f"\nfigure {n}")
        BUILDABLE[n](a.format)

    print("\nif a figure above reported missing inputs, produce them with "
          "(needs torch + pennylane):")
    for n, how in UPSTREAM.items():
        print(f"  fig {n}  ->  {how}")


if __name__ == "__main__":
    main()
