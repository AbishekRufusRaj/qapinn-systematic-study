"""
BARREN-PLATEAU / GRADIENT-VARIANCE ANALYSIS (Weeks 11-12 milestone).

This targets the project's stated research gap head-on. From the
proposal: "barren-plateau theory is well established for
classification-style losses, but rarely tested specifically under
PDE-residual losses -- the PINN setting is different (no labels, a
physics-derived loss landscape)."

So this script measures the same thing twice, under two different losses,
on identical circuits:

  A) the PDE-RESIDUAL loss  -- the physics loss: mean((u_t + u*u_x - nu*u_xx)^2).
     Depends on SECOND DERIVATIVES of the circuit output. No labels.
  B) the IC-fitting loss    -- mean((u(x,0) - (-sin(pi x)))^2). An ordinary
     supervised data-fitting loss on the circuit output itself.

If barren-plateau behaviour differs between (A) and (B) on the same
circuits, that is a direct, novel, quantitative result for the gap the
proposal claims -- and it is measurable without training anything, which
is why it is worth running early.

METHOD (standard barren-plateau protocol)
--------------------------------------------
For each circuit size: draw many independent random initializations,
compute the gradient of the loss with respect to the quantum parameters
at each, and take the VARIANCE across initializations. A barren plateau
shows up as that variance shrinking exponentially with qubit count --
a straight downward line on a log-scale plot. Flat or slowly-decaying
variance means the landscape is still navigable.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from qapinn_model import QAPINN, pde_residual


def _fixed_points(n_points, seed=1234):
    """One fixed set of interior points, shared across every measurement so
    variance reflects the circuit, not the sampling."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n_points, 1, generator=g) * (config.X_MAX - config.X_MIN) + config.X_MIN
    t = torch.rand(n_points, 1, generator=g) * config.T_MAX
    x_ic = torch.rand(n_points, 1, generator=g) * (config.X_MAX - config.X_MIN) + config.X_MIN
    t_ic = torch.zeros(n_points, 1)
    return x, t, x_ic, t_ic


def sample_gradients(n_qubits, reupload_layers, n_samples, n_points, entangler="ring", seed=config.SEED):
    """Return {loss_name: (n_samples, n_quantum_params) gradient array}.

    The standard barren-plateau protocol measures Var over random
    initializations of ONE fixed parameter's gradient. Doing that for
    every parameter position separately and then averaging those
    variances gives the same quantity with far less estimator noise, so
    that is what `variance_of()` below does with this array."""
    x, t, x_ic, t_ic = _fixed_points(n_points)
    grads = {"pde": [], "ic": []}

    for i in range(n_samples):
        torch.manual_seed(seed + 10_000 * n_qubits + 97 * reupload_layers + i)
        model = QAPINN(n_qubits=n_qubits, reupload_layers=reupload_layers, entangler=entangler)
        q_params = list(model.quantum_layer.parameters())

        # (A) PDE-residual loss -- the physics loss, uses second derivatives
        model.zero_grad()
        loss_pde = torch.mean(pde_residual(model, x, t, config.NU) ** 2)
        g_pde = torch.autograd.grad(loss_pde, q_params, retain_graph=False, allow_unused=True)
        grads["pde"].append(torch.cat([g.flatten() for g in g_pde]).detach().numpy())

        # (B) IC-fitting loss -- an ordinary supervised loss on the same circuit
        model.zero_grad()
        u_ic = model(x_ic, t_ic)
        loss_ic = torch.mean((u_ic - (-torch.sin(np.pi * x_ic))) ** 2)
        g_ic = torch.autograd.grad(loss_ic, q_params, retain_graph=False, allow_unused=True)
        grads["ic"].append(torch.cat([g.flatten() for g in g_ic]).detach().numpy())

    return {k: np.stack(v) for k, v in grads.items()}


def variance_of(grad_array):
    """Var across random initializations, computed per parameter position
    and then averaged over positions."""
    return float(np.mean(np.var(grad_array, axis=0)))


RESULTS_PATH = "gradient_variance_results.json"


def _load_cache():
    try:
        with open(RESULTS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(data):
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def run_width_sweep():
    """Gradient variance vs. number of qubits -- the classic barren-plateau axis."""
    qubit_counts = config.GRADVAR_QUBIT_COUNTS
    reupload = config.GRADVAR_REUPLOAD_LAYERS
    n_samples = config.GRADVAR_N_SAMPLES
    n_points = config.GRADVAR_N_POINTS

    cache = _load_cache()
    rows = cache.get("width_sweep", [])
    done = {r["n_qubits"] for r in rows}

    print(f"measuring gradient variance over {n_samples} random inits per circuit "
          f"(re-upload depth {reupload}, {n_points} collocation points)\n", flush=True)

    for n_qubits in qubit_counts:
        if n_qubits in done:
            print(f"  {n_qubits} qubits | cached, skipping", flush=True)
            continue
        g = sample_gradients(n_qubits, reupload, n_samples, n_points)
        var_pde = variance_of(g["pde"])
        var_ic = variance_of(g["ic"])
        rows.append({"n_qubits": n_qubits, "var_pde": var_pde, "var_ic": var_ic,
                     "mean_abs_pde": float(np.mean(np.abs(g["pde"]))),
                     "mean_abs_ic": float(np.mean(np.abs(g["ic"]))),
                     "n_quantum_params": int(g["pde"].shape[1])})
        print(f"  {n_qubits} qubits | Var[dL/dtheta] pde-residual = {var_pde:.3e} | "
              f"ic-fitting = {var_ic:.3e}", flush=True)
        cache["width_sweep"] = sorted(rows, key=lambda r: r["n_qubits"])
        cache.setdefault("settings", {}).update(
            {"n_samples": n_samples, "n_points": n_points, "reupload_layers": reupload})
        _save_cache(cache)

    rows = sorted(rows, key=lambda r: r["n_qubits"])
    qs = np.array([r["n_qubits"] for r in rows], dtype=float)
    v_pde = np.array([r["var_pde"] for r in rows])
    v_ic = np.array([r["var_ic"] for r in rows])

    # exponential-decay fit: log(var) = a*n + b  -> barren plateau if a < 0
    fit_pde = np.polyfit(qs, np.log(v_pde), 1)
    fit_ic = np.polyfit(qs, np.log(v_ic), 1)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.semilogy(qs, v_pde, "o-", color="#DD8452", markersize=8,
                label=f"PDE-residual loss (decay rate {fit_pde[0]:.2f}/qubit)")
    ax.semilogy(qs, v_ic, "s-", color="#4C72B0", markersize=8,
                label=f"IC-fitting loss (decay rate {fit_ic[0]:.2f}/qubit)")
    ax.set_xlabel("number of qubits")
    ax.set_ylabel(r"Var$[\partial \mathcal{L} / \partial \theta]$ across random inits")
    ax.set_title("Barren-plateau diagnostic:\ngradient variance vs. circuit width", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("gradient_variance.png", dpi=150)
    print("\nsaved gradient_variance.png")

    cache["fit_pde_decay_per_qubit"] = float(fit_pde[0])
    cache["fit_ic_decay_per_qubit"] = float(fit_ic[0])
    _save_cache(cache)

    print(f"\nfitted exponential decay rate (log-variance per qubit):")
    print(f"  PDE-residual loss: {fit_pde[0]:+.3f}   IC-fitting loss: {fit_ic[0]:+.3f}")
    print("  (more negative = faster onset of a barren plateau with circuit width)")
    return rows


def run_depth_sweep():
    """Gradient variance vs. re-uploading depth at fixed width -- i.e. what
    buying more frequency budget costs in trainability."""
    n_samples = config.GRADVAR_N_SAMPLES
    n_points = config.GRADVAR_N_POINTS
    fixed_qubits = 4

    cache = _load_cache()
    depth_rows = cache.get("depth_sweep", [])
    done = {r["reupload"] for r in depth_rows}

    for depth in [1, 2, 4, 6, 8]:
        if depth in done:
            print(f"  depth {depth} | cached, skipping", flush=True)
            continue
        g = sample_gradients(fixed_qubits, depth, n_samples, n_points)
        depth_rows.append({"reupload": depth, "var_pde": variance_of(g["pde"]),
                           "var_ic": variance_of(g["ic"])})
        print(f"  depth {depth} | pde {depth_rows[-1]['var_pde']:.3e} | "
              f"ic {depth_rows[-1]['var_ic']:.3e}", flush=True)
        cache["depth_sweep"] = sorted(depth_rows, key=lambda r: r["reupload"])
        _save_cache(cache)

    depth_rows = sorted(depth_rows, key=lambda r: r["reupload"])
    ds = np.array([r["reupload"] for r in depth_rows], dtype=float)
    fig2, ax2 = plt.subplots(figsize=(6.6, 4.6))
    ax2.semilogy(ds, [r["var_pde"] for r in depth_rows], "o-", color="#DD8452",
                 markersize=8, label="PDE-residual loss")
    ax2.semilogy(ds, [r["var_ic"] for r in depth_rows], "s-", color="#4C72B0",
                 markersize=8, label="IC-fitting loss")
    ax2.set_xlabel("data re-uploading depth L")
    ax2.set_ylabel(r"Var$[\partial \mathcal{L} / \partial \theta]$")
    ax2.set_title(f"Gradient variance vs. circuit depth ({fixed_qubits} qubits)\n"
                  "the trainability cost of buying more frequency budget")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    fig2.savefig("gradient_variance_vs_depth.png", dpi=150)
    print("saved gradient_variance_vs_depth.png")
    return depth_rows


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Barren-plateau / gradient-variance analysis.")
    p.add_argument("--part", choices=["width", "depth", "all"], default="all",
                   help="run one sweep or both; results cache to "
                        "gradient_variance_results.json so parts can run separately")
    args = p.parse_args()

    if args.part in ("width", "all"):
        run_width_sweep()
    if args.part in ("depth", "all"):
        run_depth_sweep()
    print(f"\nresults cached in {RESULTS_PATH}")
