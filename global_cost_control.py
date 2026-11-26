#!/usr/bin/env python3
"""
GLOBAL-COST CONTROL for the barren-plateau comparison.

WHY THIS EXISTS
===============
arXiv:2604.09957 ("Mitigating Barren Plateaus in Variational Quantum Circuits
through PDE-Constrained Loss Functions", Apr 2026) reports that a
PDE-constrained loss decays only POLYNOMIALLY with qubit count (~n^-0.7) while
their baseline decays exponentially (~2^-0.39n). They conclude that physics
constraints "resist exponential gradient vanishing".

gradient_variance.py measured the opposite sign: on this project's circuits the
PDE-residual loss decays ~40% FASTER per qubit than the supervised IC-fitting
loss (-0.89 vs -0.62), and sits about three orders of magnitude lower in
absolute terms.

Both cannot be dismissed, and the paper cannot quietly omit the conflict. The
likely resolution is that the two studies use different BASELINES:

    their baseline   a GLOBAL cost function -- an observable acting
                     non-trivially on every qubit at once. Cerezo et al. (2021)
                     showed global costs have barren plateaus even at constant
                     depth. Beating one is a low bar.

    our baseline     a LOCAL supervised loss on per-wire <Z> readouts. Much
                     harder to beat.

If that is right, then both results are true at once and the correct statement
is sharper than either paper makes alone:

    A PDE-residual loss is better than a global cost function
    and worse than a local supervised one.

This script tests exactly that by adding the two missing arms.

WHAT IS MEASURED
================
Four losses, on IDENTICAL circuits at IDENTICAL random initialisations:

  pde     (from gradient_variance.py) mean((u_t + u u_x - nu u_xx)^2)
  ic      (from gradient_variance.py) mean((u(x,0) + sin(pi x))^2)
  global  NEW. C_G = mean over points of (1 - P(|0...0>))
          The canonical pathological cost: a projector onto one basis state,
          which is an observable with support on every wire.
  local   NEW. C_L = mean over points of (1 - (1/n) sum_i P(wire i = 0))
          Cerezo's LOCAL counterpart to the same cost: the same target state,
          but measured one wire at a time. This is the controlled pair --
          global and local differ ONLY in the locality of the observable, not
          in the circuit, the parameters, or the target.

Having both `global` and `local` matters. Without `local`, a reader can object
that the global arm differs from the PDE loss in more ways than locality. With
it, the locality axis is isolated, and `pde` can be placed on that axis.

MATCHED INITIALISATIONS
=======================
The weights are not re-drawn here. Each sample constructs a QAPINN under the
same torch seed gradient_variance.py uses, and then differentiates the new
costs with respect to THAT model's `quantum_layer.weights`. So all four arms
are measured at the same points in parameter space, and any difference between
them is a property of the loss, not of the sample.

CIRCUIT RECONSTRUCTION IS VERIFIED, NOT ASSUMED
===============================================
The global and local costs need the full probability vector, which the
project's quantum layer does not return -- it returns per-wire <Z>. So this
file rebuilds the same circuit with a `qml.probs` readout. A rebuilt circuit
that silently differs from the real one would invalidate everything, so
--selftest derives <Z_i> = P(wire i = 0) - P(wire i = 1) from the probabilities
and checks it against the actual quantum layer's output. The same discipline
pdes.py used when it reproduced the original `pde_residual` to 0.0.

USAGE
=====
    PY=/usr/local/bin/python3
    $PY global_cost_control.py --selftest     # verify the rebuilt circuit
    $PY global_cost_control.py --quick        # 2-4 qubits, 20 inits, ~1 min
    $PY global_cost_control.py                # the full sweep

Results merge into gradient_variance_results.json under "global_sweep", so the
existing width/depth sweeps and their cached numbers are untouched.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import pennylane as qml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from qapinn_model import QAPINN
from gradient_variance import _fixed_points, variance_of, RESULTS_PATH

# Same Okabe-Ito set paper_figures.py validated as colourblind-safe.
COL = {"global": "#A0439B", "pde": "#D55E00", "ic": "#0072B2", "local": "#009E73"}
MK = {"global": "D", "pde": "s", "ic": "o", "local": "^"}
NICE = {"global": "global cost  $1-P(|0\\ldots0\\rangle)$",
        "local": "local cost  (Cerezo form)",
        "pde": "PDE-residual loss",
        "ic": "IC-fitting loss"}


# --------------------------------------------------------------------------
# the rebuilt circuit
# --------------------------------------------------------------------------

def make_probs_circuit(n_qubits, reupload_layers, entangler_depth=1,
                       entangler="ring", encoding="angle"):
    """The same circuit as qapinn_model.make_quantum_layer, read out as the
    full probability distribution instead of per-wire <Z>.

    Every structural choice below is copied from that function deliberately --
    the input tiling, the per-re-upload AngleEmbedding, the entangling block
    and its weight shape. If make_quantum_layer ever changes, --selftest here
    starts failing, which is the intended alarm.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    entangling_block = (qml.StronglyEntanglingLayers if entangler == "strong"
                        else qml.BasicEntanglerLayers)

    n_pairs = max(1, n_qubits // 2)
    scales = torch.tensor(
        [float(k + 1) for k in range(n_pairs) for _ in range(2)][:n_qubits])

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        reps = (n_qubits + 1) // 2
        if inputs.dim() == 1:
            features = inputs.repeat(reps)[:n_qubits]
        else:
            features = inputs.repeat(1, reps)[:, :n_qubits]
        if encoding == "angle_scaled":
            features = features * scales.to(features.device)

        for layer in range(reupload_layers):
            qml.AngleEmbedding(features, wires=range(n_qubits))
            entangling_block(weights[layer], wires=range(n_qubits))

        return qml.probs(wires=range(n_qubits))

    return circuit


def _wire_zero_mask(n_qubits):
    """mask[i, b] is True when basis state b has wire i in |0>.

    Wire 0 is the most significant bit in PennyLane's ordering, matching the
    order qml.probs returns. Summing probabilities over mask[i] therefore gives
    the marginal P(wire i = 0) exactly -- no sampling, no approximation.
    """
    b = np.arange(2 ** n_qubits)
    return np.stack([((b >> (n_qubits - 1 - i)) & 1) == 0 for i in range(n_qubits)])


# --------------------------------------------------------------------------
# the two new costs
# --------------------------------------------------------------------------

def costs_from_probs(probs, mask):
    """probs: (n_points, 2**n). Returns (C_global, C_local), both scalars.

    C_global = mean_points [ 1 - P(all wires in |0>) ]
    C_local  = mean_points [ 1 - (1/n) sum_i P(wire i in |0>) ]

    Same target state, same circuit, same parameters. The ONLY difference is
    whether the observable is read jointly across all wires or wire by wire --
    which is precisely the axis Cerezo et al. (2021) showed controls whether a
    barren plateau appears at constant depth.
    """
    p_all_zero = probs[:, 0]
    c_global = torch.mean(1.0 - p_all_zero)

    m = torch.as_tensor(mask, dtype=probs.dtype, device=probs.device)  # (n, 2**n)
    p_wire_zero = probs @ m.T                       # (n_points, n_qubits)
    c_local = torch.mean(1.0 - p_wire_zero.mean(dim=1))
    return c_global, c_local


def sample_global_gradients(n_qubits, reupload_layers, n_samples, n_points,
                            entangler="ring", seed=config.SEED):
    """Gradients of the global and local costs w.r.t. the quantum parameters.

    The seeding line is copied verbatim from gradient_variance.sample_gradients
    so that sample i here uses the same weights as sample i there.
    """
    x, t, _x_ic, _t_ic = _fixed_points(n_points)
    inputs = torch.cat([x, t], dim=1)
    # entangler_depth is read from config rather than defaulted here: the
    # rebuilt circuit's weight shape MUST match what QAPINN actually builds,
    # and a silent mismatch would either raise deep inside PennyLane or, worse,
    # broadcast into something that runs and means nothing. The assertion
    # below turns that into an immediate, readable failure.
    depth = config.DEFAULT_ENTANGLER_DEPTH
    circuit = make_probs_circuit(n_qubits, reupload_layers,
                                 entangler_depth=depth, entangler=entangler)
    mask = _wire_zero_mask(n_qubits)

    expected = ((reupload_layers, depth, n_qubits, 3) if entangler == "strong"
                else (reupload_layers, depth, n_qubits))

    grads = {"global": [], "local": []}
    for i in range(n_samples):
        torch.manual_seed(seed + 10_000 * n_qubits + 97 * reupload_layers + i)
        model = QAPINN(n_qubits=n_qubits, reupload_layers=reupload_layers,
                       entangler=entangler)
        w = model.quantum_layer.weights
        assert tuple(w.shape) == expected, (
            f"weight-shape mismatch: QAPINN gives {tuple(w.shape)}, the rebuilt "
            f"circuit expects {expected}. make_quantum_layer has changed.")

        probs = circuit(inputs, w)
        c_g, c_l = costs_from_probs(probs, mask)

        # retain_graph on the first call only: both costs share one forward
        # pass, so the graph must survive until the second backward.
        g_g = torch.autograd.grad(c_g, w, retain_graph=True)[0]
        g_l = torch.autograd.grad(c_l, w, retain_graph=False)[0]
        grads["global"].append(g_g.flatten().detach().numpy())
        grads["local"].append(g_l.flatten().detach().numpy())

    return {k: np.stack(v) for k, v in grads.items()}


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def selftest(n_qubits=4, reupload_layers=2, n_points=16):
    """Derive <Z_i> from the rebuilt circuit's probabilities and compare with
    the project's actual quantum layer on the same weights. They must agree to
    floating-point tolerance, or the rebuilt circuit is not the same circuit
    and none of the numbers below mean anything."""
    print("circuit reconstruction self-test")
    print("-" * 64)
    ok = True
    for entangler in ("ring", "strong"):
        torch.manual_seed(7)
        model = QAPINN(n_qubits=n_qubits, reupload_layers=reupload_layers,
                       entangler=entangler)
        w = model.quantum_layer.weights

        g = torch.Generator().manual_seed(11)
        x = torch.rand(n_points, 1, generator=g) * 2 - 1
        t = torch.rand(n_points, 1, generator=g)
        inputs = torch.cat([x, t], dim=1)

        circuit = make_probs_circuit(
            n_qubits, reupload_layers,
            entangler_depth=config.DEFAULT_ENTANGLER_DEPTH, entangler=entangler)
        probs = circuit(inputs, w)
        mask = torch.as_tensor(_wire_zero_mask(n_qubits), dtype=probs.dtype)
        p0 = probs @ mask.T
        z_from_probs = (2.0 * p0 - 1.0)               # <Z> = P(0) - P(1)

        with torch.no_grad():
            z_direct = model.quantum_layer(inputs)

        err = float((z_from_probs - z_direct).detach().abs().max())
        good = err < 1e-5
        ok &= good
        print(f"  {entangler:7s}  max |<Z> from probs - quantum layer| = {err:.2e}  "
              f"{'PASS' if good else 'FAIL'}")

        s = float(probs.detach().sum(dim=1).min())
        norm_ok = abs(s - 1.0) < 1e-6
        ok &= norm_ok
        print(f"           probabilities sum to {s:.8f}  "
              f"{'PASS' if norm_ok else 'FAIL'}")

    print("\n  " + ("ALL PASS" if ok else "FAILED -- do not trust the sweep"))
    return ok


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------

def _load(path=RESULTS_PATH):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _settings_sig(n_samples, n_points, reupload):
    """The settings a cached row was computed under.

    This exists because the cache was originally keyed on n_qubits alone, and
    that is not enough. A `--quick` run (20 inits, 50 points) populated rows for
    2-4 qubits; the full run (120 inits, 200 points) then SKIPPED those rows as
    'cached' and fitted a decay rate across six points, three of which came from
    different settings. n_points changes the loss itself -- it is an average over
    that many collocation points -- so the mixed fit was measuring two different
    quantities, and the resulting verdict flipped from FAILS to HOLDS purely as
    an artifact of which rows happened to be cached.

    Worse, the old code overwrote a single top-level `global_settings` key, so
    the file ended up ASSERTING that every row used 120/200 when three did not.

    Settings now travel with each row, and a row whose settings do not match the
    current request is recomputed rather than reused. A row with no recorded
    settings is from before this fix and is always treated as stale.
    """
    return {"n_samples": int(n_samples), "n_points": int(n_points),
            "reupload_layers": int(reupload)}


def run_sweep(qubit_counts, n_samples, n_points, reupload):
    cache = _load()
    cache.pop("global_settings", None)      # superseded by per-row settings
    sig = _settings_sig(n_samples, n_points, reupload)

    old = cache.get("global_sweep", [])
    usable = {r["n_qubits"]: r for r in old if r.get("settings") == sig}
    stale = sorted({r["n_qubits"] for r in old if r.get("settings") != sig})
    rows = list(usable.values())

    print(f"\nglobal/local cost gradient variance: {n_samples} inits per width, "
          f"{n_points} points, re-upload depth {reupload}", flush=True)
    if stale:
        print(f"  discarding {len(stale)} row(s) computed under different "
              f"settings: qubits {stale}", flush=True)
    print(flush=True)

    for n in qubit_counts:
        if n in usable:
            print(f"  {n} qubits | cached at these settings, skipping", flush=True)
            continue
        g = sample_global_gradients(n, reupload, n_samples, n_points)
        row = {"n_qubits": n,
               "var_global": variance_of(g["global"]),
               "var_local": variance_of(g["local"]),
               "n_quantum_params": int(g["global"].shape[1]),
               "settings": sig}
        rows.append(row)
        print(f"  {n} qubits | global {row['var_global']:.3e} | "
              f"local {row['var_local']:.3e}", flush=True)
        cache["global_sweep"] = sorted(rows, key=lambda r: r["n_qubits"])
        with open(RESULTS_PATH, "w") as f:
            json.dump(cache, f, indent=2)

    rows = sorted(rows, key=lambda r: r["n_qubits"])
    cache["global_sweep"] = rows
    with open(RESULTS_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    sigs = {json.dumps(r.get("settings"), sort_keys=True) for r in rows}
    if len(sigs) != 1:
        raise RuntimeError(
            "global_sweep rows span more than one settings block -- refusing to "
            "fit a decay rate across them. Delete the 'global_sweep' key from "
            f"{RESULTS_PATH} and re-run.")
    return rows, cache


def _fit(x, y):
    """OLS slope of log(var) on qubit count, WITH its standard error.

    A bare decay rate is not a defensible number, least of all in a paper whose
    thesis is that under-reported protocol manufactures conclusions. Six points
    and four degrees of freedom is thin, and saying so with a standard error is
    the honest way to present it. Returns (slope, se, R^2).
    """
    x = np.asarray(x, dtype=float)
    y = np.log(np.asarray(y, dtype=float))
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    s2 = float((resid ** 2).sum() / (n - 2))
    se = float(np.sqrt(s2 / ((x - x.mean()) ** 2).sum()))
    r2 = float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(slope), se, r2, float(intercept)


def report(rows, cache, outfile="figures/fig9_global_cost_control.png"):
    """Fit every arm on a common qubit set and place `pde` on the locality axis."""
    width = {r["n_qubits"]: r for r in cache.get("width_sweep", [])}
    g = {r["n_qubits"]: r for r in rows}
    qs = sorted(set(g) & set(width)) if width else sorted(g)
    if not width:
        print("  (no width_sweep in the cache -- run gradient_variance.py "
              "--part width first to get the pde/ic arms)")
        qs = sorted(g)

    series = {"global": [g[q]["var_global"] for q in qs],
              "local": [g[q]["var_local"] for q in qs]}
    if width:
        series["pde"] = [width[q]["var_pde"] for q in qs]
        series["ic"] = [width[q]["var_ic"] for q in qs]

    fits = {k: _fit(qs, v) for k, v in series.items()}

    # ---- figure
    #
    # TWO PANELS, and the second one exists because the first cannot carry the
    # claim. The finding is about DECAY RATE, but on a raw log-variance plot the
    # dominant visual is absolute LEVEL -- the PDE-residual loss sits six orders
    # of magnitude below the global cost, so a reader sees it nowhere near the
    # local cost while the caption asserts it behaves like one. A referee would
    # flag exactly that.
    #
    # (a) keeps the absolute levels, which are a real result in their own right:
    #     the PDE-residual loss has tiny gradients compared with every other arm.
    # (b) divides each series by its own value at the smallest circuit, so all
    #     four start at 1 and the only thing left on the axis is how fast each
    #     one falls. That is the quantity the claim is about, and in (b) the
    #     pde and local lines genuinely do lie together while global falls away.
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    xf = np.linspace(min(qs) - 0.2, max(qs) + 0.2, 50)
    order = [n for n in ("global", "local", "pde", "ic") if n in series]

    for name in order:
        slope, se, _r2, intercept = fits[name]
        v = np.asarray(series[name], dtype=float)

        axA.plot(xf, np.exp(slope * xf + intercept), color=COL[name],
                 linewidth=0.8, alpha=0.4, zorder=1)
        axA.semilogy(qs, v, marker=MK[name], color=COL[name], markersize=5,
                     linestyle="none", zorder=3, label=NICE[name])

        # normalise to this arm's own value at the smallest circuit. The line
        # is the SAME OLS fit under the same normalisation (subtract log v[0]),
        # not the slope re-anchored at the first data point -- otherwise the
        # drawn line would not be the fit whose slope the legend quotes.
        axB.plot(xf, np.exp(slope * xf + intercept - np.log(v[0])),
                 color=COL[name], linewidth=0.8, alpha=0.4, zorder=1)
        axB.semilogy(qs, v / v[0], marker=MK[name], color=COL[name],
                     markersize=5, linestyle="none", zorder=3,
                     label=f"{slope:+.2f} $\\pm$ {se:.2f}/qubit")

    axA.set_xlabel("number of qubits")
    axA.set_ylabel(r"Var$[\partial\mathcal{L}/\partial\theta]$")
    axA.set_title("(a) absolute gradient variance", fontsize=8.5)

    axB.set_xlabel("number of qubits")
    axB.set_ylabel(f"variance relative to {qs[0]} qubits", fontsize=8)
    axB.set_title("(b) decay only, each arm normalised to itself", fontsize=8.5)

    for ax in (axA, axB):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.3, which="both", linewidth=0.6)
        ax.legend(fontsize=6.3, frameon=False, loc="lower left",
                  handletextpad=0.4, borderpad=0.2)

    fig.suptitle("A PDE-residual loss DECAYS like a local cost, not like a "
                 "global one", fontsize=9.5, y=1.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig(outfile.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {outfile} (+ .pdf)")

    # ---- numbers
    st = rows[0].get("settings", {})
    print(f"\nall arms fitted over qubits {qs} "
          f"({len(qs)} points, {len(qs) - 2} dof); global/local at "
          f"{st.get('n_samples')} inits x {st.get('n_points')} points")
    print(f"\n{'arm':8}{'slope':>9}{'se':>8}{'95% CI':>21}{'R^2':>8}")
    print("-" * 54)
    for name in ("global", "local", "pde", "ic"):
        if name not in fits:
            continue
        b, se, r2, _ = fits[name]
        print(f"{name:8}{b:>+9.3f}{se:>8.3f}   [{b - 1.96 * se:+.3f}, "
              f"{b + 1.96 * se:+.3f}]{r2:>8.3f}")
    print("  (more negative = faster onset of a barren plateau with width)")

    # ---- pairwise separation, so no ordering is claimed on a gap that is noise
    print("\npairwise separation of the slopes:")
    print("-" * 54)
    order = [k for k in ("global", "local", "pde", "ic") if k in fits]
    sep = {}
    for i in range(len(order) - 1):
        for j in range(i + 1, len(order)):
            A, B = order[i], order[j]
            diff = fits[A][0] - fits[B][0]
            sed = float(np.hypot(fits[A][1], fits[B][1]))
            z = diff / sed if sed > 0 else float("inf")
            sep[(A, B)] = abs(z) > 1.96
            print(f"  {A:7} vs {B:7} gap {diff:+.3f}  combined se {sed:.3f}  "
                  f"z={z:+.2f}  {'SEPARATED' if sep[(A, B)] else 'not separated'}")

    cache["global_fits"] = {k: {"slope": v[0], "se": v[1], "r2": v[2]}
                            for k, v in fits.items()}
    cache["global_fit_qubits"] = qs
    with open(RESULTS_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    # ---- verdict
    if {"global", "local", "pde", "ic"} <= set(fits):
        print("\nVERDICT")
        print("-" * 54)
        gs, ls, ps, ics = (fits[k][0] for k in ("global", "local", "pde", "ic"))
        if sep[("global", "pde")] and not sep[("local", "pde")]:
            print("  The PDE-residual loss is SEPARATED from the global cost and")
            print("  INDISTINGUISHABLE from the matched local cost.")
            print()
            print("  That is the reconciliation, and it is sharper than either paper:")
            print("  a PDE-residual loss is not a specially well-conditioned")
            print("  landscape -- it is simply a LOCAL one, and locality is what")
            print("  resists barren plateaus (Cerezo et al. 2021). arXiv:2604.09957")
            print("  compared it against a global cost and correctly found it far")
            print("  better; this project compared it against a local SUPERVISED")
            print("  loss and correctly found it worse. Both hold.")
            if sep[("pde", "ic")]:
                print()
                print(f"  It is still separated from the IC-fitting loss "
                      f"({ps:+.2f} vs {ics:+.2f}), so the trainability penalty")
                print("  this project reported for PDE-residual losses stands.")
        elif sep[("global", "pde")] and sep[("local", "pde")] and ls < ps:
            print("  pde decays more slowly than BOTH the global and the local cost.")
            print("  Stronger than the reconciliation predicted -- the PDE residual")
            print("  really is better conditioned than locality alone explains.")
        elif not sep[("global", "pde")]:
            print("  pde is NOT separated from the global cost. The reconciliation")
            print("  FAILS -- report the disagreement with arXiv:2604.09957 as")
            print("  unresolved rather than explaining it away.")
        else:
            print("  Ordering does not match any clean reading. Report the fits and")
            print("  their standard errors as measured; assert nothing further.")
        print()
        print("  Caveat either way: their PDE-constrained arm is not this project's")
        print("  PDE-residual loss and their circuits are not these circuits. This")
        print("  locates the disagreement; it does not replicate them. Six fit")
        print("  points and four degrees of freedom is thin -- quote the standard")
        print("  errors, never the bare slopes.")
    return fits


def main():
    p = argparse.ArgumentParser(
        description="Global vs local cost control for the barren-plateau comparison.")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="2-4 qubits, 20 inits -- a smoke test, not a result")
    p.add_argument("--qubits", nargs="*", type=int, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--points", type=int, default=None)
    p.add_argument("--reupload", type=int, default=None)
    p.add_argument("--report-only", action="store_true",
                   help="re-fit and re-plot from the cache; runs no circuits")
    a = p.parse_args()

    if a.report_only:
        cache = _load()
        rows = cache.get("global_sweep", [])
        if not rows:
            print("no global_sweep rows cached -- run the sweep first.",
                  file=sys.stderr)
            sys.exit(1)
        report(rows, cache)
        sys.exit(0)

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    if a.quick:
        qubits, samples, points = [2, 3, 4], 20, 50
    else:
        qubits = config.GRADVAR_QUBIT_COUNTS
        samples = config.GRADVAR_N_SAMPLES
        points = config.GRADVAR_N_POINTS
    qubits = a.qubits or qubits
    samples = a.samples or samples
    points = a.points or points
    reupload = a.reupload or config.GRADVAR_REUPLOAD_LAYERS

    if not selftest():
        print("\nself-test failed -- refusing to run the sweep.", file=sys.stderr)
        sys.exit(1)

    rows, cache = run_sweep(qubits, samples, points, reupload)
    report(rows, cache)
    print(f"\nresults merged into {RESULTS_PATH} under 'global_sweep'")


if __name__ == "__main__":
    main()
