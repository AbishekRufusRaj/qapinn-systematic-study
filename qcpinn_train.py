"""
Train a QCPINN (or one of the paper's two classical baselines) using
THEIR recipe and OUR protocol.

WHAT IS BORROWED AND WHAT IS NOT
--------------------------------
Borrowed from arXiv:2503.16678, because the point is to test their claim
on their terms rather than a strawman of it:

    Adam, lr = 0.005                     (this project had been using 1e-3)
    ReduceLROnPlateau, factor 0.9, patience 1000
    gradient-norm clipping to 1.0
    the architecture itself (qcpinn_model.py)

NOT borrowed -- replaced with this project's protocol, which is the entire
contribution:

    their fixed 20,000-epoch budget  ->  our convergence criterion
    their single run                 ->  five seeds
    their no-variance reporting      ->  pre-registered failure threshold
                                         (final loss > 1e-2 = non-converged)

The fixed-budget point is not a stylistic quibble. Our own seed study
found every run still descending at an 8,000-epoch cutoff, with the
classical baseline's worst run cut off mid-breakthrough -- a fixed budget
measures which seeds happened to finish, not which model is better. A
20,000-epoch budget is longer but has the same defect, and the paper
reports no seed variance at all, so there is no way to tell from it
whether any reported gap exceeds run-to-run noise.

L-BFGS is kept from this project's recipe and applied identically to all
three models. It is worth one to two orders of magnitude on every run we
have measured, so omitting it would flatter whichever model happens to be
further from its optimum when Adam stops.

The data budget is this project's SWEEP budget for all three models rather
than the paper's batch size of 64, so these numbers stay directly
comparable to the fifteen convergence runs already on disk. All models get
the identical budget, which is what makes the comparison fair.

Usage:
    python3 qcpinn_train.py --model qcpinn --seed 0 --tag qc_qcpinn_s0
    python3 qcpinn_train.py --model model2 --seed 0 --tag qc_model2_s0
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import config
from qcpinn_model import QCPINN, model_1, model_2
from train import compute_losses, sample_points, train_lbfgs

# their optimiser settings
QC_LR = 0.005
QC_SCHED_FACTOR = 0.9
QC_SCHED_PATIENCE = 1000
QC_CLIP_NORM = 1.0

# our convergence settings (identical to robust_train.py, so QAPINN and
# QCPINN runs are stopped by the same rule)
DEFAULT_MAX_EPOCHS = 25000
DEFAULT_PATIENCE = 3000
DEFAULT_TOL = 0.03
CHECK_EVERY = 500
EMA_ALPHA = 0.002

FAIL_LOSS = 1e-2      # pre-registered non-convergence threshold


def build_model(kind, **kw):
    if kind == "qcpinn":
        return QCPINN(**kw)
    if kind == "model1":
        return model_1()
    if kind == "model2":
        return model_2()
    raise ValueError(f"unknown model kind: {kind}")


def train_to_convergence(model, budget, history, max_epochs=DEFAULT_MAX_EPOCHS,
                         patience=DEFAULT_PATIENCE, tol=DEFAULT_TOL,
                         verbose=True, print_every=1000):
    """Adam with the paper's schedule and clipping, stopped by our criterion."""
    optimizer = torch.optim.Adam(model.parameters(), lr=QC_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=QC_SCHED_FACTOR, patience=QC_SCHED_PATIENCE)

    ema = None
    ema_trace = []
    stop_reason = "max_epochs"
    stopped_at = max_epochs

    for epoch in range(1, max_epochs + 1):
        pts = sample_points(budget)
        loss, loss_pde, loss_ic, loss_bc = compute_losses(model, pts, budget)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), QC_CLIP_NORM)
        optimizer.step()

        v = loss.item()
        scheduler.step(v)
        ema = v if ema is None else (1 - EMA_ALPHA) * ema + EMA_ALPHA * v

        history["total"].append(v)
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())
        history["bc"].append(loss_bc.item())
        history["phase"].append(0)

        if epoch % CHECK_EVERY == 0:
            ema_trace.append((epoch, ema))
            past = [e for e in ema_trace if e[0] <= epoch - patience]
            if past:
                then = past[-1][1]
                rel = (then - ema) / max(then, 1e-12)
                if rel < tol:
                    stop_reason = (f"converged (only {rel:.1%} improvement "
                                   f"over last {patience} epochs)")
                    stopped_at = epoch
                    if verbose:
                        print(f"  [adam]  stopping at epoch {epoch}: {stop_reason}",
                              flush=True)
                    break

        if verbose and (epoch % print_every == 0 or epoch == 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [adam]  epoch {epoch:6d}/{max_epochs} | loss {v:.3e} | "
                  f"ema {ema:.3e} | lr {lr_now:.2e}", flush=True)

    return model, {"stop_reason": stop_reason, "adam_epochs_used": stopped_at,
                   "final_ema": ema,
                   "final_lr": optimizer.param_groups[0]["lr"]}


def run(kind="qcpinn", model_kwargs=None, budget=None, seed=config.SEED,
        max_epochs=DEFAULT_MAX_EPOCHS, patience=DEFAULT_PATIENCE,
        tol=DEFAULT_TOL, tag=None, out_root="runs", verbose=True):
    model_kwargs = model_kwargs or {}
    budget = dict(budget or config.SWEEP)

    tag = tag or f"qc_{kind}_s{seed}"
    run_dir = os.path.join(out_root, tag)
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))

    model = build_model(kind, **model_kwargs)
    counts = model.param_counts()
    if verbose:
        print(f"\n=== {tag} === params q/c = {counts['quantum']}/{counts['classical']} "
              f"(total {counts['total']})", flush=True)

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
        # `model_type` is what evaluate.py dispatches on. `classical` is kept
        # so older tooling that reads it still behaves sensibly.
        "model_type": kind,
        "classical": kind in ("model1", "model2"),
        "model_kwargs": model.hparams,
        "budget": budget,
        "param_counts": counts,
        "seed": seed,
        "convergence_trained": True,
        "recipe": {"lr": QC_LR, "sched_factor": QC_SCHED_FACTOR,
                   "sched_patience": QC_SCHED_PATIENCE, "clip_norm": QC_CLIP_NORM},
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
        "converged": history["total"][-1] <= FAIL_LOSS,
        "logged_steps": len(history["total"]),
    }

    torch.save(model.state_dict(), os.path.join(run_dir, "weights.pt"))
    np.save(os.path.join(run_dir, "loss_history.npy"), history, allow_pickle=True)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        flag = "" if meta["converged"] else "   *** NON-CONVERGED ***"
        print(f"  saved -> {run_dir} | {stop_info['adam_epochs_used']} Adam epochs, "
              f"{meta['total_seconds']}s{flag}", flush=True)
    return model, history, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train a QCPINN or one of the paper's classical baselines.")
    p.add_argument("--model", choices=["qcpinn", "model1", "model2"], default="qcpinn")
    p.add_argument("--layers", type=int, default=1, help="L, variational layers (qcpinn only)")
    p.add_argument("--reupload", action="store_true",
                   help="re-encode before every layer; off matches the paper")
    p.add_argument("--trainable-entangler", action="store_true",
                   help="learn the CRX angles too; off matches the paper's param count")
    p.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p.add_argument("--budget", choices=["full", "sweep"], default="sweep")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-root", default="runs")
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    budget = config.FULL if args.budget == "full" else config.SWEEP
    model_kwargs = {} if args.model != "qcpinn" else dict(
        n_layers=args.layers, reupload=args.reupload,
        trainable_entangler=args.trainable_entangler)

    run(kind=args.model, model_kwargs=model_kwargs, budget=budget, seed=args.seed,
        max_epochs=args.max_epochs, patience=args.patience, tol=args.tol,
        tag=args.tag, out_root=args.out_root)
