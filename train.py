"""
The training engine. One function, `run_experiment`, that takes a full
configuration and produces a complete, self-describing run directory:
weights, loss history, the exact hyperparameters used, and timing.

Two-phase training, matching what the classical Burgers' baseline needed
to reach 11.2% (and what got the QAPINN from 40.7% to 16.6%):

  Phase 1, Adam       -- fast, explores the landscape, gets close.
  Phase 2, L-BFGS     -- second-order with line search, polishes the last
                         stretch to a precision Adam can't reach alone.

Skipping phase 2 is the single most common reason a PINN result looks
worse than it should. Every run in this study does both, so no
configuration is unfairly penalized by an incomplete recipe.

Usage (single run):
    python3 train.py --qubits 4 --reupload 6 --budget full --tag headline
    python3 train.py --classical --budget full --tag classical_baseline
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import config
from qapinn_model import QAPINN, ClassicalPINN, pde_residual


def sample_points(budget, generator=None):
    """Collocation points for the physics loss, plus IC/BC points.

    Collocation points are drawn in two groups: uniform across the domain,
    plus a concentrated cluster near x=0. The shock lives at x=0 and is
    where the residual is hardest to satisfy -- uniform-only sampling
    under-represents exactly the region that matters most."""
    n_uni = budget["n_collocation_uniform"]
    n_shock = budget["n_collocation_shock"]
    n_ic = budget["n_ic"]
    n_bc = budget["n_bc"]

    x_uniform = torch.rand(n_uni, 1, generator=generator) * (config.X_MAX - config.X_MIN) + config.X_MIN
    x_shock = (torch.randn(n_shock, 1, generator=generator) * 0.15).clamp(config.X_MIN, config.X_MAX)
    x_f = torch.cat([x_uniform, x_shock], dim=0)
    t_f = torch.rand(x_f.shape[0], 1, generator=generator) * config.T_MAX

    x_ic = torch.rand(n_ic, 1, generator=generator) * (config.X_MAX - config.X_MIN) + config.X_MIN
    t_ic = torch.zeros(n_ic, 1)

    t_bc = torch.rand(n_bc, 1, generator=generator) * config.T_MAX
    x_bc_left = torch.full((n_bc, 1), config.X_MIN)
    x_bc_right = torch.full((n_bc, 1), config.X_MAX)

    return x_f, t_f, x_ic, t_ic, x_bc_left, x_bc_right, t_bc


def compute_losses(model, pts, budget):
    x_f, t_f, x_ic, t_ic, x_bc_left, x_bc_right, t_bc = pts

    residual = pde_residual(model, x_f, t_f, config.NU)
    loss_pde = torch.mean(residual ** 2)

    u_ic_pred = model(x_ic, t_ic)
    u_ic_true = -torch.sin(np.pi * x_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

    u_bc_left = model(x_bc_left, t_bc)
    u_bc_right = model(x_bc_right, t_bc)
    loss_bc = torch.mean(u_bc_left ** 2) + torch.mean(u_bc_right ** 2)

    loss = loss_pde + budget["w_ic"] * loss_ic + budget["w_bc"] * loss_bc
    return loss, loss_pde, loss_ic, loss_bc


def _log(history, loss, loss_pde, loss_ic, loss_bc, phase):
    history["total"].append(loss.item())
    history["pde"].append(loss_pde.item())
    history["ic"].append(loss_ic.item())
    history["bc"].append(loss_bc.item())
    history["phase"].append(phase)


def train_adam(model, budget, history, print_every=250, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=budget["lr"])
    epochs = budget["adam_epochs"]

    for epoch in range(1, epochs + 1):
        pts = sample_points(budget)
        loss, loss_pde, loss_ic, loss_bc = compute_losses(model, pts, budget)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _log(history, loss, loss_pde, loss_ic, loss_bc, phase=0)

        if verbose and (epoch % print_every == 0 or epoch == 1):
            print(f"  [adam]  epoch {epoch:6d}/{epochs} | total {loss.item():.3e} | "
                  f"pde {loss_pde.item():.3e} | ic {loss_ic.item():.3e} | bc {loss_bc.item():.3e}",
                  flush=True)

    return model, history


def train_lbfgs(model, budget, history, print_every=50, verbose=True):
    """L-BFGS needs a FIXED batch (it runs its own line search, so
    resampling mid-optimization would make its internal curvature estimate
    meaningless). Draw one batch and refine on it."""
    steps = budget["lbfgs_steps"]
    if steps <= 0:
        return model, history

    pts = sample_points(budget)
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=steps,
        tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50,
        line_search_fn="strong_wolfe",
    )
    counter = {"n": 0}

    def closure():
        lbfgs.zero_grad()
        loss, loss_pde, loss_ic, loss_bc = compute_losses(model, pts, budget)
        loss.backward()
        counter["n"] += 1
        _log(history, loss, loss_pde, loss_ic, loss_bc, phase=1)
        if verbose and counter["n"] % print_every == 0:
            print(f"  [lbfgs] step {counter['n']:5d} | total {loss.item():.3e} | "
                  f"pde {loss_pde.item():.3e} | ic {loss_ic.item():.3e} | bc {loss_bc.item():.3e}",
                  flush=True)
        return loss

    lbfgs.step(closure)
    if verbose:
        print(f"  [lbfgs] done after {counter['n']} closure evaluations", flush=True)
    return model, history


def run_experiment(model_kwargs=None, budget=None, tag=None, out_root="runs",
                   classical=False, seed=config.SEED, verbose=True):
    """Train one configuration end to end and save everything about it."""
    model_kwargs = model_kwargs or {}
    budget = budget or config.SWEEP

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ClassicalPINN() if classical else QAPINN(**model_kwargs)
    counts = model.param_counts()

    if tag is None:
        tag = "classical" if classical else "_".join(f"{k}{v}" for k, v in sorted(model_kwargs.items()))
    run_dir = os.path.join(out_root, tag)
    os.makedirs(run_dir, exist_ok=True)

    if verbose:
        print(f"\n=== {tag} === params q/c = {counts['quantum']}/{counts['classical']}", flush=True)

    history = {"total": [], "pde": [], "ic": [], "bc": [], "phase": []}

    t0 = time.time()
    model, history = train_adam(model, budget, history, verbose=verbose)
    t_adam = time.time() - t0

    t1 = time.time()
    model, history = train_lbfgs(model, budget, history, verbose=verbose)
    t_lbfgs = time.time() - t1

    meta = {
        "tag": tag,
        "classical": classical,
        "model_kwargs": model_kwargs if not classical else model.hparams,
        "budget": budget,
        "param_counts": counts,
        "seed": seed,
        "adam_seconds": round(t_adam, 1),
        "lbfgs_seconds": round(t_lbfgs, 1),
        "total_seconds": round(t_adam + t_lbfgs, 1),
        "final_total_loss": history["total"][-1],
        "final_pde_loss": history["pde"][-1],
        "final_ic_loss": history["ic"][-1],
        "final_bc_loss": history["bc"][-1],
        "logged_steps": len(history["total"]),
    }

    torch.save(model.state_dict(), os.path.join(run_dir, "weights.pt"))
    np.save(os.path.join(run_dir, "loss_history.npy"), history, allow_pickle=True)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"  saved -> {run_dir}  ({meta['total_seconds']}s total)", flush=True)

    return model, history, meta


def _build_argparser():
    p = argparse.ArgumentParser(description="Train one QAPINN (or classical baseline) configuration.")
    p.add_argument("--qubits", type=int, default=config.DEFAULT_N_QUBITS)
    p.add_argument("--reupload", type=int, default=config.DEFAULT_REUPLOAD_LAYERS)
    p.add_argument("--entangler-depth", type=int, default=config.DEFAULT_ENTANGLER_DEPTH)
    p.add_argument("--entangler", choices=["ring", "strong"], default="ring")
    p.add_argument("--encoding", choices=["angle", "angle_scaled"], default="angle")
    p.add_argument("--budget", choices=["full", "sweep"], default="sweep")
    p.add_argument("--classical", action="store_true", help="train the classical PINN baseline instead")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-root", default="runs")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    budget = config.FULL if args.budget == "full" else config.SWEEP

    model_kwargs = {} if args.classical else dict(
        n_qubits=args.qubits, reupload_layers=args.reupload,
        entangler_depth=args.entangler_depth, entangler=args.entangler,
        encoding=args.encoding,
    )

    run_experiment(model_kwargs=model_kwargs, budget=budget, tag=args.tag,
                   out_root=args.out_root, classical=args.classical, seed=args.seed)
