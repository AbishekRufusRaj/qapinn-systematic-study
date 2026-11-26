"""
TRAINING TO CONVERGENCE (the mitigation).

WHAT THE SEED STUDY ACTUALLY FOUND
-------------------------------------
The five-seed repetition showed enormous spread: QAPINN 17.47% +/- 14.43%,
classical 8.05% +/- 9.83%. The obvious reading is that some runs fall into
bad basins and never recover. Inspecting the loss trajectories shows that
reading is wrong.

Every run follows the same shape: a long plateau around 0.3-0.4, then a
sharp breakthrough after which the loss falls by an order of magnitude.
What varies across seeds is WHEN the breakthrough happens:

    seed 0   broke through at ~3,500 epochs   ->  4.4% error
    seed 3   broke through at ~5,500 epochs   ->  6.2%
    seed 4   broke through at ~6,000 epochs   -> 10.9%
    seed 1   had not broken through by 8,000  -> 30.1%
    seed 2   had not broken through by 8,000  -> 35.7%

And critically, at the 8,000-epoch cutoff EVERY run was still descending --
the fitted slope over the final 2,000 epochs is negative for all ten runs,
classical and quantum alike. The classical baseline's apparently terrible
24.61% run had the steepest slope of the lot: its loss fell from 0.252 to
0.076 over its last 1,900 epochs. It was cut off mid-breakthrough.

So the measured "variance between architectures" was largely variance in
breakthrough time, sampled at an arbitrary cutoff. A fixed-epoch budget
does not measure model quality; it measures how many seeds happened to
break through before the clock ran out.

THE FIX
----------
Stop at a convergence criterion instead of an epoch count. Training
continues while the loss is still meaningfully improving and stops when it
genuinely flattens, with a generous cap for safety.

The criterion has to survive the plateau, which is the subtle part. During
the plateau the loss still creeps down by roughly 5% per 1,000 epochs, so a
naive early-stopping rule with short patience would terminate exactly where
these runs most need to keep going. Patience is therefore set in thousands
of epochs, and the improvement threshold is deliberately loose.

Everything here applies identically to the classical baseline, so no
improvement can come from giving the quantum model a procedure the
classical one was denied.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import config
from qapinn_model import QAPINN, ClassicalPINN
from train import compute_losses, sample_points, train_lbfgs

# convergence settings -- see the docstring for why patience is this large
DEFAULT_MAX_EPOCHS = 25000
DEFAULT_PATIENCE = 3000      # window over which improvement is measured
DEFAULT_TOL = 0.03           # need >3% relative improvement across that window
CHECK_EVERY = 500
EMA_ALPHA = 0.002            # smoothing for the noisy per-batch loss


def train_to_convergence(model, budget, history, max_epochs=DEFAULT_MAX_EPOCHS,
                         patience=DEFAULT_PATIENCE, tol=DEFAULT_TOL,
                         verbose=True, print_every=1000):
    """Adam until the loss stops improving. Returns (model, stop_info)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=budget["lr"])

    ema = None
    ema_trace = []       # (epoch, ema) sampled every CHECK_EVERY
    stop_reason = "max_epochs"
    stopped_at = max_epochs

    for epoch in range(1, max_epochs + 1):
        pts = sample_points(budget)
        loss, loss_pde, loss_ic, loss_bc = compute_losses(model, pts, budget)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        v = loss.item()
        ema = v if ema is None else (1 - EMA_ALPHA) * ema + EMA_ALPHA * v

        history["total"].append(v)
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())
        history["bc"].append(loss_bc.item())
        history["phase"].append(0)

        if epoch % CHECK_EVERY == 0:
            ema_trace.append((epoch, ema))
            # compare against the reading from `patience` epochs ago
            past = [e for e in ema_trace if e[0] <= epoch - patience]
            if past:
                then = past[-1][1]
                rel = (then - ema) / max(then, 1e-12)
                if rel < tol:
                    stop_reason = f"converged (only {rel:.1%} improvement over last {patience} epochs)"
                    stopped_at = epoch
                    if verbose:
                        print(f"  [adam]  stopping at epoch {epoch}: {stop_reason}", flush=True)
                    break

        if verbose and (epoch % print_every == 0 or epoch == 1):
            print(f"  [adam]  epoch {epoch:6d}/{max_epochs} | loss {v:.3e} | ema {ema:.3e}",
                  flush=True)

    return model, {"stop_reason": stop_reason, "adam_epochs_used": stopped_at,
                   "final_ema": ema}


def run(model_kwargs=None, budget=None, classical=False, seed=config.SEED,
        max_epochs=DEFAULT_MAX_EPOCHS, patience=DEFAULT_PATIENCE, tol=DEFAULT_TOL,
        tag=None, out_root="runs", verbose=True):
    model_kwargs = model_kwargs or {}
    budget = dict(budget or config.SWEEP)

    tag = tag or ("conv_classical" if classical else "conv_" +
                  "_".join(f"{k}{v}" for k, v in sorted(model_kwargs.items())))
    run_dir = os.path.join(out_root, tag)
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))

    model = ClassicalPINN() if classical else QAPINN(**model_kwargs)
    counts = model.param_counts()
    if verbose:
        print(f"\n=== {tag} === params q/c = {counts['quantum']}/{counts['classical']}",
              flush=True)

    history = {"total": [], "pde": [], "ic": [], "bc": [], "phase": []}

    t0 = time.time()
    model, stop_info = train_to_convergence(model, budget, history, max_epochs,
                                            patience, tol, verbose=verbose)
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
        "convergence_trained": True,
        "max_epochs": max_epochs,
        "patience": patience,
        "tol": tol,
        **stop_info,
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
        print(f"  saved -> {run_dir} | {stop_info['adam_epochs_used']} Adam epochs, "
              f"{meta['total_seconds']}s", flush=True)
    return model, history, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train one configuration to convergence.")
    p.add_argument("--qubits", type=int, default=config.DEFAULT_N_QUBITS)
    p.add_argument("--reupload", type=int, default=config.DEFAULT_REUPLOAD_LAYERS)
    p.add_argument("--entangler-depth", type=int, default=config.DEFAULT_ENTANGLER_DEPTH)
    p.add_argument("--entangler", choices=["ring", "strong"], default="ring")
    p.add_argument("--encoding", choices=["angle", "angle_scaled"], default="angle")
    p.add_argument("--init-scale", type=float, default=None,
                   help="near-identity init (try 0.1); omit for PennyLane's default")
    p.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p.add_argument("--budget", choices=["full", "sweep"], default="sweep")
    p.add_argument("--classical", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-root", default="runs")
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    budget = config.FULL if args.budget == "full" else config.SWEEP
    model_kwargs = {} if args.classical else dict(
        n_qubits=args.qubits, reupload_layers=args.reupload,
        entangler_depth=args.entangler_depth, entangler=args.entangler,
        encoding=args.encoding, init_scale=args.init_scale,
    )

    run(model_kwargs=model_kwargs, budget=budget, classical=args.classical,
        seed=args.seed, max_epochs=args.max_epochs, patience=args.patience,
        tol=args.tol, tag=args.tag, out_root=args.out_root)
