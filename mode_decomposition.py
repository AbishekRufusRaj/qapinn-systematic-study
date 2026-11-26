#!/usr/bin/env python3
"""
Per-mode spectral decomposition for the heat equation.

WHY THIS SCRIPT EXISTS
======================
The heat runs currently report an early/late time split (`rel_l2_early_time`
vs `rel_l2_late_time`). That metric was built to answer "does a spectrally
limited model fail where the high modes live?" -- and it does NOT answer it.
It came back at ratios of 0.2-1.0, meaning models do BETTER early, which is
the opposite of the prediction. Two reasons, both artifacts of the metric
rather than facts about the models:

  1. The IC loss term directly supervises t = 0 with weight 10. Early time is
     anchored by supervision; late time is constrained only by the residual
     and accumulates error forward. The split therefore measures IC anchoring.

  2. Late-time amplitude is smaller (everything decays), so the relative-error
     DENOMINATOR shrinks and inflates the late number for free.

So the early/late column must not be cited as a spectral measure. This script
is its replacement: instead of asking "where in time is the error", it asks
the question we actually care about --

        WHICH MODE was lost, and WHEN?

THE MATH
========
Both PDEs in this study live on x in [-1, 1] with u(+-1, t) = 0. The functions

        phi_k(x) = sin(k pi x),    k = 1, 2, 3, ...

all vanish at both boundaries, so they are the natural basis here, and they
are orthogonal on this interval:

        int_{-1}^{1} sin(k pi x) sin(m pi x) dx = delta_km

Note the norm is exactly 1, not 2 or 1/2: int_{-1}^{1} sin^2(k pi x) dx
= int_{-1}^{1} (1 - cos(2 k pi x))/2 dx = 1 - 0 = 1. That means the projection
needs no normalising constant -- the coefficient IS the integral:

        c_k(t) = int_{-1}^{1} u(x, t) sin(k pi x) dx

For the exact solution the coefficients are known in closed form, because each
mode of the heat equation decays independently:

        c_k^exact(t) = a_k * exp(-alpha k^2 pi^2 t)   for k in {1, 3, 5}
        c_k^exact(t) = 0                              for every other k

That second line is what makes this measurement clean. The true solution has
LITERALLY ZERO content outside three modes, at every time. So any content a
model puts anywhere else is not approximation error in the usual sense -- it
is the model inventing structure the physics does not contain. We measure both.

TWO ERROR FAMILIES
==================
(a) TRACKING ERROR, for the modes that are present (k = 1, 3, 5):

        e_k(t) = c_k^pred(t) - c_k^exact(t)

    reported as a fraction of a_k -- the mode's amplitude at t = 0 -- and NOT
    as a fraction of c_k^exact(t). This is deliberate and is the whole lesson
    of the early/late failure: c_k^exact(t) decays toward zero, so dividing by
    it manufactures enormous "errors" at late time for a model that is simply
    predicting a small number slightly wrong. Normalising by the fixed a_k
    keeps the scale physical -- "this model misplaced 4% of mode 3's original
    amplitude" is a statement that means the same thing at every t.

(b) SPURIOUS CONTENT (leakage), for the modes that are absent (k = 2, 4, 6...):

        l_k(t) = |c_k^pred(t)|,   normalised by a_1

    The exact answer is zero everywhere, so this is a pure artifact measure.
    Even-k leakage is worth watching separately: the initial condition is a
    sum of odd modes and is odd about x = 0, and the heat operator preserves
    that symmetry, so any even-k content is a symmetry the model broke.

MODE LIFETIME
=============
Mode 5 has decayed to 0.7% of its initial amplitude by t = 1. Averaging its
error over the full time interval would mostly average over a period where
there is nothing left to get right. So each mode gets an "alive window":

        t_k = ln(1/LIVE_FRAC) / (alpha k^2 pi^2)     clipped to [0, t_max]

i.e. the time at which the mode has decayed to LIVE_FRAC of its start. With
LIVE_FRAC = 0.05 and alpha = 0.02: mode 1 and mode 3 are alive for the whole
interval, mode 5 until t ~ 0.61. Errors are averaged over each mode's own
window, and the window is printed so nobody has to guess what was averaged.

WHAT THIS TESTS
===============
fourier_analysis.py measured that the `angle` encoding's ACCESSIBLE spectrum
cuts off at mode 4. The obvious prediction is that the QAPINN should lose
mode 5 specifically, while the classical PINN should not.

Be careful with that prediction, and state the caveat in the paper: the QAPINN
has seven classical layers BEHIND the quantum layer, and those layers can
synthesise higher frequencies from what they receive. The VQC is a frequency
BOTTLENECK, not a hard low-pass filter -- the XAI maps already showed exactly
this (quantum layer at mode 1, final hidden layer at 37% high-frequency
energy). So a clean mode-5 cliff is NOT expected. What is expected, if the
bottleneck story is right, is a disproportionate degradation on mode 5
relative to mode 1 for the quantum models. The script prints that ratio
directly so the result can go either way on the record.

USAGE
=====
    PY=/usr/local/bin/python3        # torch + pennylane live here
    $PY mode_decomposition.py --selftest          # verify the projector
    $PY mode_decomposition.py                     # all heat_* runs
    $PY mode_decomposition.py --runs runs/heat_classical_s0

No training. Forward passes on existing weights only -- a couple of minutes
for all fifteen heat runs.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

# pdes.py imports torch at module level, but only USES it inside methods --
# Heat.__init__, Heat.ground_truth and the mode-amplitude maths are pure numpy.
# --selftest needs exactly those. So when torch is genuinely missing we register
# a placeholder module purely to let `import pdes` succeed, and anything that
# actually touches torch then fails loudly rather than silently. This is the
# same reasoning as validate_criterion.py being numpy-only: a tool that checks
# the maths should run in an environment that cannot train.
try:
    import torch as _torch_probe            # noqa: F401
except ModuleNotFoundError:
    import types
    sys.modules["torch"] = types.ModuleType("torch")

import pdes

# torch and pennylane are imported LAZILY, inside decompose_run, for the same
# reason validate_criterion.py is numpy-only: --selftest verifies the maths of
# the projector against closed-form answers and needs no model at all, so it
# must be runnable in an environment that has numpy and nothing else. Only
# scoring real runs needs the full stack.

# np.trapz was renamed np.trapezoid in numpy 2.0. Support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

# Modes to project onto. Goes past the three that are present so that leakage
# into absent modes is measured, and past the encoding's measured ceiling of 4
# so the ceiling prediction has somewhere to show up.
DEFAULT_KMAX = 8

# A mode counts as "alive" while it retains at least this fraction of its
# t = 0 amplitude. Errors are averaged only over each mode's alive window.
LIVE_FRAC = 0.05

# Spatial quadrature resolution. The integrand u(x,t) sin(k pi x) is smooth and
# vanishes at both endpoints, so the trapezoid rule converges fast; 801 points
# resolves k = 8 to ~1e-6, which --selftest verifies rather than assumes.
DEFAULT_NX = 801
DEFAULT_NT = 41

# Forward passes are chunked so a quantum circuit is never asked for 30k points
# in one call.
CHUNK = 4000


# --------------------------------------------------------------------------
# the projector
# --------------------------------------------------------------------------

def sine_basis(x, kmax):
    """Matrix of basis functions, shape (len(x), kmax), column k-1 = sin(k pi x)."""
    ks = np.arange(1, kmax + 1)
    return np.sin(np.pi * np.outer(x, ks))


def project(U, x, kmax):
    """Project a solution field onto the sine basis.

    U : (nt, nx) values of u on a uniform x grid, one row per time.
    returns : (nt, kmax) coefficients c_k(t).

    The integral int u(x,t) sin(k pi x) dx is evaluated with the trapezoid
    rule. Because the basis is orthonormal on [-1, 1] (norm exactly 1), the
    integral is the coefficient -- there is no 2/L factor to remember.
    """
    Phi = sine_basis(x, kmax)             # (nx, kmax)
    # np.trapz over the x axis, for every time and every mode at once:
    # integrand[t, i, k] = U[t, i] * Phi[i, k]
    integrand = U[:, :, None] * Phi[None, :, :]
    return _trapz(integrand, x=x, axis=1)          # (nt, kmax)


def exact_coefficients(pde, t, kmax):
    """c_k^exact(t) for every k -- a_k exp(-alpha k^2 pi^2 t) on the modes that
    are present in the initial condition, and exactly zero on every other."""
    C = np.zeros((len(t), kmax))
    for k, a in zip(pde.modes, pde.amps):
        if k <= kmax:
            C[:, k - 1] = a * np.exp(-pde.alpha * (k * np.pi) ** 2 * t)
    return C


def alive_until(pde, k):
    """Time at which mode k has decayed to LIVE_FRAC of its t=0 amplitude,
    clipped to the simulated interval. Modes that never decay that far inside
    the interval return t_max."""
    rate = pde.alpha * (k * np.pi) ** 2
    if rate <= 0:
        return pde.t_max
    return float(min(np.log(1.0 / LIVE_FRAC) / rate, pde.t_max))


# --------------------------------------------------------------------------
# evaluation of one run
# --------------------------------------------------------------------------

def predict_field(model, x, t):
    """u_pred on the (t, x) grid, chunked. Shape (nt, nx)."""
    import torch                                   # lazy: see top of file

    X, T = np.meshgrid(x, t)                       # both (nt, nx)
    xf = torch.tensor(X.ravel(), dtype=torch.float32).reshape(-1, 1)
    tf = torch.tensor(T.ravel(), dtype=torch.float32).reshape(-1, 1)

    out = []
    with torch.no_grad():
        for i in range(0, xf.shape[0], CHUNK):
            out.append(model(xf[i:i + CHUNK], tf[i:i + CHUNK]).numpy().ravel())
    return np.concatenate(out).reshape(X.shape)


def decompose_run(run_dir, nx=DEFAULT_NX, nt=DEFAULT_NT, kmax=DEFAULT_KMAX,
                  verbose=True):
    import torch                                   # lazy: see top of file
    from pde_train import build_model

    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)

    if meta["pde"] != "heat":
        raise ValueError(f"{run_dir}: this decomposition is heat-only "
                         f"(got pde='{meta['pde']}'). Burgers' is broadband and "
                         f"has no finite exact mode set to compare against.")

    pde = pdes.get(meta["pde"], **meta.get("pde_kwargs", {}))
    model = build_model(meta["model_type"], **meta["model_kwargs"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "weights.pt")))
    model.eval()

    x = np.linspace(pde.x_min, pde.x_max, nx)
    t = np.linspace(0.0, pde.t_max, nt)

    U_pred = predict_field(model, x, t)
    C_pred = project(U_pred, x, kmax)               # (nt, kmax)
    C_true = exact_coefficients(pde, t, kmax)

    a1 = pde.amps[0]                                # reference scale for leakage
    present = set(pde.modes)

    per_mode = {}

    # ---- (a) tracking error on the modes that are present ----------------
    for k, a_k in zip(pde.modes, pde.amps):
        if k > kmax:
            continue
        t_alive = alive_until(pde, k)
        m = t <= t_alive
        err = C_pred[:, k - 1] - C_true[:, k - 1]

        per_mode[str(k)] = {
            "present": True,
            "a_k": float(a_k),
            "alive_until_t": t_alive,
            "n_times_in_window": int(m.sum()),
            # all three normalised by a_k, NOT by c_k^exact(t) -- see docstring
            "mean_abs_err_frac_a_k": float(np.abs(err[m]).mean() / a_k),
            "max_abs_err_frac_a_k": float(np.abs(err[m]).max() / a_k),
            "err_at_t0_frac_a_k": float(abs(err[0]) / a_k),
            # signed mean says whether the model systematically under- or
            # over-shoots this mode; a bottleneck should under-shoot
            "signed_mean_err_frac_a_k": float(err[m].mean() / a_k),
            "c_pred_t0": float(C_pred[0, k - 1]),
            "c_true_t0": float(C_true[0, k - 1]),
        }

    # ---- (b) leakage into modes the solution does not contain -------------
    leak_odd, leak_even = 0.0, 0.0
    for k in range(1, kmax + 1):
        if k in present:
            continue
        mag = float(np.abs(C_pred[:, k - 1]).max() / a1)
        per_mode[str(k)] = {
            "present": False,
            "max_abs_coeff_frac_a1": mag,
            "mean_abs_coeff_frac_a1": float(np.abs(C_pred[:, k - 1]).mean() / a1),
            "parity": "even" if k % 2 == 0 else "odd",
        }
        if k % 2 == 0:
            leak_even = max(leak_even, mag)
        else:
            leak_odd = max(leak_odd, mag)

    # ---- headline summary numbers ----------------------------------------
    m1 = per_mode["1"]["mean_abs_err_frac_a_k"]
    m5 = per_mode["5"]["mean_abs_err_frac_a_k"] if "5" in per_mode else float("nan")

    out = {
        "tag": meta["tag"],
        "model_type": meta["model_type"],
        "model_kwargs": meta.get("model_kwargs", {}),
        "seed": meta["seed"],
        "grid": {"nx": nx, "nt": nt, "kmax": kmax, "live_frac": LIVE_FRAC},
        "modes_present": list(pde.modes),
        "per_mode": per_mode,
        # The ceiling test, stated as one number: how much worse is the
        # highest present mode than the lowest, for this model? A frequency
        # bottleneck should push this up relative to the classical baseline.
        "mode5_over_mode1_err": float(m5 / m1) if m1 > 0 else float("nan"),
        "max_leakage_odd_frac_a1": leak_odd,
        "max_leakage_even_frac_a1": leak_even,
    }

    with open(os.path.join(run_dir, "mode_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Coefficient trajectories, for figure 8. Small, so just save them.
    np.savez(os.path.join(run_dir, "mode_coeffs.npz"),
             t=t, c_pred=C_pred, c_true=C_true, kmax=kmax)

    if verbose:
        bits = "  ".join(
            f"k={k}:{per_mode[str(k)]['mean_abs_err_frac_a_k']:6.2%}"
            for k in pde.modes if str(k) in per_mode)
        print(f"{meta['tag']:24s} {bits}   k5/k1 {out['mode5_over_mode1_err']:5.2f}"
              f"   leak(odd/even) {leak_odd:.1%}/{leak_even:.1%}")
    return out


# --------------------------------------------------------------------------
# aggregation across seeds
# --------------------------------------------------------------------------

def config_of(tag):
    """heat_strong_L2_s3 -> heat_strong_L2. Seeds aggregate, configs don't."""
    return tag.rsplit("_s", 1)[0]


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault(config_of(r["tag"]), []).append(r)

    summary = {}
    for cfg, rs in sorted(groups.items()):
        entry = {"n_seeds": len(rs), "per_mode": {}}
        modes = rs[0]["modes_present"]
        for k in modes:
            vals = [r["per_mode"][str(k)]["mean_abs_err_frac_a_k"] for r in rs]
            signed = [r["per_mode"][str(k)]["signed_mean_err_frac_a_k"] for r in rs]
            entry["per_mode"][str(k)] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
                "worst": float(np.max(vals)),
                "signed_mean": float(np.mean(signed)),
            }
        ratios = [r["mode5_over_mode1_err"] for r in rs]
        entry["mode5_over_mode1_err"] = {
            "mean": float(np.mean(ratios)),
            "std": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
        }
        entry["max_leakage_even_frac_a1"] = {
            "mean": float(np.mean([r["max_leakage_even_frac_a1"] for r in rs])),
            "worst": float(np.max([r["max_leakage_even_frac_a1"] for r in rs])),
        }
        entry["max_leakage_odd_frac_a1"] = {
            "mean": float(np.mean([r["max_leakage_odd_frac_a1"] for r in rs])),
            "worst": float(np.max([r["max_leakage_odd_frac_a1"] for r in rs])),
        }
        summary[cfg] = entry
    return summary


def print_summary(summary):
    print("\n" + "=" * 78)
    print("PER-MODE AMPLITUDE ERROR  (mean |c_k^pred - c_k^exact| over the mode's")
    print("alive window, as a fraction of that mode's t=0 amplitude a_k)")
    print("=" * 78)

    cfgs = list(summary)
    modes = sorted(int(k) for k in summary[cfgs[0]]["per_mode"])

    head = f"{'config':<22}{'n':>3}  " + "".join(f"{'k=' + str(k):>16}" for k in modes)
    print(head)
    print("-" * len(head))
    for cfg in cfgs:
        e = summary[cfg]
        row = f"{cfg:<22}{e['n_seeds']:>3}  "
        for k in modes:
            s = e["per_mode"][str(k)]
            row += f"{s['mean']:>9.2%} ±{s['std']:<5.2%}"
        print(row)

    print(f"\n{'config':<22}{'k5/k1 ratio':>16}{'even leak':>12}{'odd leak':>12}")
    print("-" * 62)
    for cfg in cfgs:
        e = summary[cfg]
        print(f"{cfg:<22}{e['mode5_over_mode1_err']['mean']:>9.2f} "
              f"±{e['mode5_over_mode1_err']['std']:<5.2f}"
              f"{e['max_leakage_even_frac_a1']['mean']:>12.2%}"
              f"{e['max_leakage_odd_frac_a1']['mean']:>12.2%}")

    # The interpretation the paper needs, computed rather than eyeballed.
    print("\nREADING THIS TABLE")
    print("-" * 62)
    base = next((c for c in cfgs if "classical" in c), None)
    if base:
        b = summary[base]["mode5_over_mode1_err"]["mean"]
        for cfg in cfgs:
            if cfg == base:
                continue
            q = summary[cfg]["mode5_over_mode1_err"]["mean"]
            spread = (summary[cfg]["mode5_over_mode1_err"]["std"]
                      + summary[base]["mode5_over_mode1_err"]["std"])
            sep = "SEPARATED" if abs(q - b) > spread else "not separated"
            print(f"  {cfg} k5/k1 = {q:.2f} vs classical {b:.2f}  "
                  f"(gap {q - b:+.2f}, combined spread {spread:.2f}) -> {sep}")
        print("\n  A quantum model degrading disproportionately on mode 5 relative to")
        print("  mode 1 is the frequency-bottleneck prediction. 'not separated' at")
        print("  n=5 means the data does not support the prediction -- report it that")
        print("  way rather than reaching for the mean ordering alone.")


# --------------------------------------------------------------------------
# self-test: verify the projector before trusting it on model output
# --------------------------------------------------------------------------

def selftest(nx=DEFAULT_NX, nt=DEFAULT_NT, kmax=DEFAULT_KMAX):
    """The FFT in this project was once wrong for a subtle reason (spectral
    leakage from non-periodic ramps) and reported a centroid of 10.4 for a
    straight line. That is why every spectral tool here gets a test with a
    known answer before it is pointed at real data."""
    pde = pdes.get("heat")
    x = np.linspace(pde.x_min, pde.x_max, nx)
    t = np.linspace(0.0, pde.t_max, nt)
    X, T = np.meshgrid(x, t)

    print("projector self-test")
    print("-" * 62)

    # 1. a single pure mode must return 1.0 on its own index and 0 elsewhere
    U = np.sin(3 * np.pi * X)
    C = project(U, x, kmax)
    ok1 = abs(C[0, 2] - 1.0) < 1e-6 and np.abs(np.delete(C[0], 2)).max() < 1e-6
    print(f"  pure sin(3 pi x):     c_3 = {C[0, 2]:.6f} (want 1.000000), "
          f"max other |c_k| = {np.abs(np.delete(C[0], 2)).max():.2e}  "
          f"{'PASS' if ok1 else 'FAIL'}")

    # 2. the exact heat solution must return its analytic coefficients
    U = np.asarray(pde.ground_truth()(X, T), dtype=float)
    C = project(U, x, kmax)
    C_ref = exact_coefficients(pde, t, kmax)
    err = np.abs(C - C_ref).max()
    ok2 = err < 1e-5
    print(f"  exact heat solution:  max |c_k^proj - c_k^analytic| = {err:.2e}  "
          f"{'PASS' if ok2 else 'FAIL'}")

    # 3. absent modes must come back at exactly zero on the exact solution --
    #    this is the baseline against which model leakage is measured
    absent = [k for k in range(1, kmax + 1) if k not in pde.modes]
    leak = np.abs(C[:, [k - 1 for k in absent]]).max()
    ok3 = leak < 1e-5
    print(f"  absent modes {absent}: max |c_k| = {leak:.2e}  "
          f"{'PASS' if ok3 else 'FAIL'}")

    # 4. a linear ramp -- the function that broke the old FFT. It is NOT zero
    #    in this basis and should not be: sin(k pi x) genuinely has overlap
    #    with x. The point is that the coefficients are the true Fourier-sine
    #    coefficients of a ramp (2(-1)^(k+1)/(k pi)), not smeared garbage.
    C = project(X, x, kmax)
    ref = np.array([2 * (-1) ** (k + 1) / (k * np.pi) for k in range(1, kmax + 1)])
    err = np.abs(C[0] - ref).max()
    ok4 = err < 1e-4
    print(f"  linear ramp u = x:    max |c_k - analytic| = {err:.2e}  "
          f"{'PASS' if ok4 else 'FAIL'}")
    print(f"      (analytic c_k = 2(-1)^(k+1)/(k pi); the old FFT reported a")
    print(f"       centroid of 10.4 for this function -- this basis does not)")

    # 5. mode lifetimes, so the averaging windows are on the record
    print("\n  alive windows (mode retains >= "
          f"{LIVE_FRAC:.0%} of a_k):")
    for k, a in zip(pde.modes, pde.amps):
        print(f"      k={k}  a_k={a:.4f}  alive until t = {alive_until(pde, k):.3f}")

    allok = ok1 and ok2 and ok3 and ok4
    print("\n  " + ("ALL PASS" if allok else "SOME FAILED -- do not use these numbers"))
    return allok


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Per-mode spectral decomposition for heat-equation runs.")
    p.add_argument("--runs", nargs="*", default=None,
                   help="run directories (default: runs/heat_*)")
    p.add_argument("--nx", type=int, default=DEFAULT_NX)
    p.add_argument("--nt", type=int, default=DEFAULT_NT)
    p.add_argument("--kmax", type=int, default=DEFAULT_KMAX)
    p.add_argument("--out", default="mode_decomposition_summary.json")
    p.add_argument("--selftest", action="store_true",
                   help="verify the projector against known answers and exit")
    a = p.parse_args()

    if a.selftest:
        sys.exit(0 if selftest(a.nx, a.nt, a.kmax) else 1)

    run_dirs = a.runs or sorted(glob.glob("runs/heat_*"))
    if not run_dirs:
        print("no heat runs found under runs/", file=sys.stderr)
        sys.exit(1)

    if not selftest(a.nx, a.nt, a.kmax):
        print("\nprojector self-test failed -- refusing to score runs.", file=sys.stderr)
        sys.exit(1)
    print()

    results = []
    for d in run_dirs:
        if not os.path.exists(os.path.join(d, "weights.pt")):
            print(f"skip {d}: no weights.pt", file=sys.stderr)
            continue
        results.append(decompose_run(d, a.nx, a.nt, a.kmax))

    summary = aggregate(results)
    print_summary(summary)

    with open(a.out, "w") as f:
        json.dump({"config": {"nx": a.nx, "nt": a.nt, "kmax": a.kmax,
                              "live_frac": LIVE_FRAC},
                   "summary": summary,
                   "runs": results}, f, indent=2)
    print(f"\nwrote {a.out}  (+ mode_metrics.json and mode_coeffs.npz per run)")


if __name__ == "__main__":
    main()
