"""
PDE-agnostic training and evaluation.

This is the multi-PDE path. It deliberately does NOT modify train.py,
robust_train.py, evaluate.py or qapinn_model.py: those files are what the
forty runs already on disk were produced with, and editing them would put
every existing result one refactor away from being unreproducible. This
module reimplements the same pipeline through the `pdes.py` interface and
writes the identical run-directory format, so old and new runs sit side by
side and are read by the same tooling.

Verified equivalence: `pdes.Burgers().residual` reproduces the original
`qapinn_model.pde_residual` to 0.0 absolute difference, so a Burgers' run
launched from here is comparable to one launched from robust_train.py.

TRAINING RECIPE
---------------
Two recipes, selected by model unless overridden:

  plain   lr 1e-3, no scheduler, no clipping        (robust_train.py)
  qcpinn  lr 5e-3, ReduceLROnPlateau 0.9/1000,
          gradient-norm clipping to 1.0             (arXiv:2503.16678)

Auto-selection matters: a QCPINN trained with the plain recipe is not the
architecture that paper evaluated, and a QAPINN trained with theirs is not
comparable to the runs already on disk. Both mistakes are easy to make by
hand and neither is visible in the results.

Stopping is this project's convergence criterion in both cases, with the
same pre-registered non-convergence threshold, because that is the part
the paper is actually about.

Usage:
    python3 pde_train.py --pde heat --model classical --seed 0
    python3 pde_train.py --pde heat --model qapinn --entangler strong \\
                         --reupload 2 --init-scale 0.1 --seed 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import config
import pdes

FAIL_LOSS = 1e-2
DEFAULT_MAX_EPOCHS = 25000
DEFAULT_PATIENCE = 3000
DEFAULT_TOL = 0.03
CHECK_EVERY = 500
EMA_ALPHA = 0.002

RECIPES = {
    "plain":  {"lr": 1e-3, "sched": None, "clip": None},
    "qcpinn": {"lr": 5e-3, "sched": (0.9, 1000), "clip": 1.0},
}


# ---------------------------------------------------------------- models

def build_model(kind, **kw):
    if kind == "classical":
        from qapinn_model import ClassicalPINN
        return ClassicalPINN(**kw)
    if kind == "qapinn":
        from qapinn_model import QAPINN
        return QAPINN(**kw)
    if kind in ("qcpinn", "model1", "model2"):
        from qcpinn_model import QCPINN, model_1, model_2
        return {"qcpinn": lambda: QCPINN(**kw),
                "model1": model_1, "model2": model_2}[kind]()
    raise ValueError(f"unknown model kind: {kind}")


def default_recipe(kind):
    return "qcpinn" if kind in ("qcpinn", "model1", "model2") else "plain"


# ---------------------------------------------------------------- data

def sample_points(pde, budget, generator=None):
    x_f, t_f = pde.sample_collocation(budget["n_collocation_uniform"],
                                      budget["n_collocation_shock"], generator)
    n_ic, n_bc = budget["n_ic"], budget["n_bc"]

    x_ic = torch.rand(n_ic, 1, generator=generator) * (pde.x_max - pde.x_min) + pde.x_min
    t_ic = torch.zeros(n_ic, 1)

    t_bc = torch.rand(n_bc, 1, generator=generator) * pde.t_max
    x_l = torch.full((n_bc, 1), pde.x_min)
    x_r = torch.full((n_bc, 1), pde.x_max)
    return x_f, t_f, x_ic, t_ic, x_l, x_r, t_bc


def compute_losses(model, pde, pts, budget):
    x_f, t_f, x_ic, t_ic, x_l, x_r, t_bc = pts

    loss_pde = torch.mean(pde.residual(model, x_f, t_f) ** 2)
    loss_ic = torch.mean((model(x_ic, t_ic) - pde.ic(x_ic)) ** 2)
    loss_bc = (torch.mean((model(x_l, t_bc) - pde.bc_value(x_l, t_bc)) ** 2)
               + torch.mean((model(x_r, t_bc) - pde.bc_value(x_r, t_bc)) ** 2))

    loss = loss_pde + budget["w_ic"] * loss_ic + budget["w_bc"] * loss_bc
    return loss, loss_pde, loss_ic, loss_bc


# ---------------------------------------------------------------- training

def train_to_convergence(model, pde, budget, history, recipe,
                         max_epochs=DEFAULT_MAX_EPOCHS, patience=DEFAULT_PATIENCE,
                         tol=DEFAULT_TOL, verbose=True, print_every=1000,
                         checkpoints=(), on_checkpoint=None):
    """`checkpoints` are epoch numbers at which to snapshot the model.

    THE BUDGET CURVE COMES FROM ONE RUN, NOT FOUR
    ---------------------------------------------
    The obvious way to measure "how do conclusions change with training
    budget" is to train separately at 800, 2000, 8000 and convergence --
    four times the compute, and four different random trajectories whose
    differences confound the budget effect with seed noise.

    Snapshotting a single run instead gives all four budget points from
    ONE trajectory: same seed, same data, same initialisation, so the only
    thing that differs between the points IS the budget. That is both 4x
    cheaper and a strictly cleaner comparison.

    The snapshots also feed budget-dependent XAI: run xai_feature_maps.py
    with --checkpoint 800 against the same directory to see what the
    network's features looked like before it converged."""
    r = RECIPES[recipe]
    opt = torch.optim.Adam(model.parameters(), lr=r["lr"])
    sched = None
    if r["sched"]:
        factor, pat = r["sched"]
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=factor, patience=pat)

    ema, trace = None, []
    stop_reason, stopped_at = "max_epochs", max_epochs

    for epoch in range(1, max_epochs + 1):
        loss, l_pde, l_ic, l_bc = compute_losses(model, pde, sample_points(pde, budget), budget)

        opt.zero_grad()
        loss.backward()
        if r["clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), r["clip"])
        opt.step()

        v = loss.item()
        if sched:
            sched.step(v)
        ema = v if ema is None else (1 - EMA_ALPHA) * ema + EMA_ALPHA * v

        history["total"].append(v)
        history["pde"].append(l_pde.item())
        history["ic"].append(l_ic.item())
        history["bc"].append(l_bc.item())
        history["phase"].append(0)

        if epoch in checkpoints and on_checkpoint is not None:
            on_checkpoint(epoch, model)
            if verbose:
                print(f"  [ckpt]  snapshot at epoch {epoch} (loss {v:.3e})", flush=True)

        if epoch % CHECK_EVERY == 0:
            trace.append((epoch, ema))
            past = [e for e in trace if e[0] <= epoch - patience]
            if past:
                rel = (past[-1][1] - ema) / max(past[-1][1], 1e-12)
                if rel < tol:
                    stop_reason = f"converged (only {rel:.1%} over last {patience} epochs)"
                    stopped_at = epoch
                    if verbose:
                        print(f"  [adam]  stopping at {epoch}: {stop_reason}", flush=True)
                    break

        if verbose and (epoch % print_every == 0 or epoch == 1):
            print(f"  [adam]  epoch {epoch:6d}/{max_epochs} | loss {v:.3e} | ema {ema:.3e} "
                  f"| lr {opt.param_groups[0]['lr']:.2e}", flush=True)

    return model, {"stop_reason": stop_reason, "adam_epochs_used": stopped_at,
                   "final_ema": ema}


def train_lbfgs(model, pde, budget, history, verbose=True, print_every=100):
    steps = budget["lbfgs_steps"]
    if steps <= 0:
        return model, history
    pts = sample_points(pde, budget)          # fixed batch: L-BFGS line-searches
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=steps,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=50, line_search_fn="strong_wolfe")
    n = {"i": 0}

    def closure():
        opt.zero_grad()
        loss, l_pde, l_ic, l_bc = compute_losses(model, pde, pts, budget)
        loss.backward()
        n["i"] += 1
        history["total"].append(loss.item())
        history["pde"].append(l_pde.item())
        history["ic"].append(l_ic.item())
        history["bc"].append(l_bc.item())
        history["phase"].append(1)
        if verbose and n["i"] % print_every == 0:
            print(f"  [lbfgs] step {n['i']:5d} | total {loss.item():.3e}", flush=True)
        return loss

    opt.step(closure)
    if verbose:
        print(f"  [lbfgs] done after {n['i']} evaluations", flush=True)
    return model, history


# ---------------------------------------------------------------- evaluation

def evaluate_run(run_dir, nx=200, nt=200, verbose=True, checkpoint=None):
    """checkpoint: an epoch number, to score weights_ep<N>.pt and write
    metrics_ep<N>.json instead of the final weights and metrics.json."""
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    pde = pdes.get(meta["pde"], **meta.get("pde_kwargs", {}))

    wfile = "weights.pt" if checkpoint is None else f"weights_ep{checkpoint}.pt"
    mfile = "metrics.json" if checkpoint is None else f"metrics_ep{checkpoint}.json"

    model = build_model(meta["model_type"], **meta["model_kwargs"])
    model.load_state_dict(torch.load(os.path.join(run_dir, wfile)))
    model.eval()

    x = np.linspace(pde.x_min, pde.x_max, nx)
    t = np.linspace(0.0, pde.t_max, nt)
    X, T = np.meshgrid(x, t)
    with torch.no_grad():
        xf = torch.tensor(X.ravel(), dtype=torch.float32).reshape(-1, 1)
        tf = torch.tensor(T.ravel(), dtype=torch.float32).reshape(-1, 1)
        U = model(xf, tf).numpy().ravel().reshape(X.shape)

    U_true = np.asarray(pde.ground_truth()(X, T), dtype=float)
    err = U - U_true

    out = dict(meta)
    out["checkpoint_epoch"] = checkpoint
    out["rel_l2"] = float(np.linalg.norm(err) / np.linalg.norm(U_true))
    out.update(pde.decompose(X, T, err, U_true))

    with open(os.path.join(run_dir, mfile), "w") as f:
        json.dump(out, f, indent=2)

    if verbose:
        label = meta["tag"] + ("" if checkpoint is None else f" @ep{checkpoint}")
        print(f"{label:30s} rel L2 {out['rel_l2']:7.2%} | "
              f"hard/easy {out['hard_easy_ratio']:6.1f} | "
              f"params {meta['param_counts']['quantum']}q/{meta['param_counts']['classical']}c")
    return out


# ---------------------------------------------------------------- driver

def run(pde_name="burgers", pde_kwargs=None, kind="classical", model_kwargs=None,
        budget=None, recipe=None, seed=config.SEED, max_epochs=DEFAULT_MAX_EPOCHS,
        patience=DEFAULT_PATIENCE, tol=DEFAULT_TOL, tag=None, out_root="runs",
        checkpoints=(), verbose=True):
    pde_kwargs = pde_kwargs or {}
    model_kwargs = model_kwargs or {}
    budget = dict(budget or config.SWEEP)
    recipe = recipe or default_recipe(kind)

    pde = pdes.get(pde_name, **pde_kwargs)
    tag = tag or f"{pde_name}_{kind}_s{seed}"
    run_dir = os.path.join(out_root, tag)
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))

    model = build_model(kind, **model_kwargs)
    counts = model.param_counts()
    if verbose:
        print(f"\n=== {tag} === pde={pde_name} recipe={recipe} "
              f"params q/c={counts['quantum']}/{counts['classical']}", flush=True)

    history = {"total": [], "pde": [], "ic": [], "bc": [], "phase": []}

    saved_ckpts = []

    def snapshot(epoch, m):
        torch.save(m.state_dict(), os.path.join(run_dir, f"weights_ep{epoch}.pt"))
        saved_ckpts.append(epoch)

    t0 = time.time()
    model, stop = train_to_convergence(model, pde, budget, history, recipe,
                                       max_epochs, patience, tol, verbose,
                                       checkpoints=set(checkpoints),
                                       on_checkpoint=snapshot)
    t_adam = time.time() - t0
    t1 = time.time()
    model, history = train_lbfgs(model, pde, budget, history, verbose)
    t_lbfgs = time.time() - t1

    meta = {
        "tag": tag, "pde": pde_name, "pde_kwargs": pde_kwargs,
        "model_type": kind, "model_kwargs": getattr(model, "hparams", {}),
        "classical": counts["quantum"] == 0,
        "recipe": recipe, "budget": budget, "param_counts": counts, "seed": seed,
        "convergence_trained": True, "max_epochs": max_epochs,
        "patience": patience, "tol": tol, **stop,
        "adam_seconds": round(t_adam, 1), "lbfgs_seconds": round(t_lbfgs, 1),
        "total_seconds": round(t_adam + t_lbfgs, 1),
        "final_total_loss": history["total"][-1],
        "final_pde_loss": history["pde"][-1],
        "final_ic_loss": history["ic"][-1],
        "final_bc_loss": history["bc"][-1],
        "converged": history["total"][-1] <= FAIL_LOSS,
        "checkpoints": saved_ckpts,
        "logged_steps": len(history["total"]),
    }

    torch.save(model.state_dict(), os.path.join(run_dir, "weights.pt"))
    np.save(os.path.join(run_dir, "loss_history.npy"), history, allow_pickle=True)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        flag = "" if meta["converged"] else "   *** NON-CONVERGED ***"
        print(f"  saved -> {run_dir} | {stop['adam_epochs_used']} epochs, "
              f"{meta['total_seconds']}s{flag}", flush=True)

    for ep in saved_ckpts:
        evaluate_run(run_dir, verbose=verbose, checkpoint=ep)
    evaluate_run(run_dir, verbose=verbose)
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train one model on one PDE, to convergence.")
    p.add_argument("--pde", choices=list(pdes.REGISTRY), default="burgers")
    p.add_argument("--model", choices=["classical", "qapinn", "qcpinn", "model1", "model2"],
                   default="classical")
    p.add_argument("--recipe", choices=list(RECIPES), default=None,
                   help="default: qcpinn recipe for QCPINN-family models, plain otherwise")
    # QAPINN knobs
    p.add_argument("--qubits", type=int, default=config.DEFAULT_N_QUBITS)
    p.add_argument("--reupload", type=int, default=config.DEFAULT_REUPLOAD_LAYERS)
    p.add_argument("--entangler", choices=["ring", "strong"], default="ring")
    p.add_argument("--encoding", choices=["angle", "angle_scaled"], default="angle")
    p.add_argument("--init-scale", type=float, default=None)
    p.add_argument("--init-gain", type=float, default=None,
                   help="classical baseline only: rescale default init "
                        "(the matched control for the QAPINN's --init-scale)")
    # QCPINN knobs
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--reupload-qc", action="store_true")
    # shared
    p.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p.add_argument("--checkpoints", default="",
                   help="comma-separated epochs to snapshot, e.g. 800,2000,8000 "
                        "-- gives the whole budget curve from one trajectory")
    p.add_argument("--budget", choices=["full", "sweep"], default="sweep")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-root", default="runs")
    p.add_argument("--seed", type=int, default=config.SEED)
    a = p.parse_args()

    if a.model == "qapinn":
        mk = dict(n_qubits=a.qubits, reupload_layers=a.reupload,
                  entangler=a.entangler, encoding=a.encoding, init_scale=a.init_scale)
    elif a.model == "qcpinn":
        mk = dict(n_layers=a.layers, reupload=a.reupload_qc)
    elif a.model == "classical":
        mk = dict(init_gain=a.init_gain)
    else:
        mk = {}

    run(pde_name=a.pde, kind=a.model, model_kwargs=mk, recipe=a.recipe,
        budget=config.FULL if a.budget == "full" else config.SWEEP,
        seed=a.seed, max_epochs=a.max_epochs, patience=a.patience, tol=a.tol,
        tag=a.tag, out_root=a.out_root,
        checkpoints=[int(c) for c in a.checkpoints.split(",") if c.strip()])
