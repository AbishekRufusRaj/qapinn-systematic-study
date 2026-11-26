"""
HYPERPARAMETER SENSITIVITY (Reviewer 2, point 3).

R2 asked whether the reported gains are sensitive to optimiser choice,
learning-rate scheduling, batch size and training duration. Training duration is
already answered by the convergence-criterion study; this script answers the
other three.

It is protocol4_qcpinn.py with exactly three knobs added -- --optimizer, --lr and
--batch -- and NOTHING else changed. protocol4 is left untouched on purpose: the
Section VI results are reproducible from it, and editing it would invalidate
them. The defaults here reproduce protocol4 exactly, so the sym4 runs at seeds
0-2 serve as this sweep's baseline arm and are not re-run.

The authors' solvers build their own Adam + ReduceLROnPlateau in __init__. We do
not modify external/QCPINN; we replace model.optimizer and model.scheduler after
construction, keeping the authors' scheduler settings (factor 0.9, patience
1000) so that only the optimiser family changes.

WHAT VARIES, AND WHY EACH IS THE FAIR CHOICE
  sgd       SGD + momentum 0.9, the standard non-adaptive comparator
  rmsprop   RMSprop, adaptive but without Adam's bias correction
  lr0.001   Adam at one fifth the authors' rate
  lr0.01    Adam at twice the authors' rate
  batch256  the authors' 64-point minibatch widened 4x

Three seeds per cell per arm. That is small, and the paper says so: the claim
this supports is "the ranking does not move", not a precise effect size.

USAGE
    python3 protocol5_hyper.py --model qcpinn --seed 0 --tag hyp_qcpinn_sgd_s0 \
        --optimizer sgd
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "external", "QCPINN")

EMA_BETA = 0.002
TOL = 0.03
PATIENCE = 3000
CHECK_EVERY = 500
CAP = 25000
LBFGS_STEPS = 500
REFINE_RES = 1000          # this study's protocol: 1,400 points per epoch
REFINE_BC = 100            # 1,000 interior + 4 x 100 boundary = 1,400
CHUNK = 500                # PennyLane 0.29 cachetools limit workaround


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qcpinn", "model2"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--lbfgs", type=int, default=LBFGS_STEPS)
    ap.add_argument("--refine-res", type=int, default=REFINE_RES)
    ap.add_argument("--refine-bc", type=int, default=REFINE_BC)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--resume", action="store_true",
                    help="load after_adam.pt and run the refinement phase only")
    ap.add_argument("--check-against", type=float, default=None,
                    help="assert the after-Adam error equals this (percent)")
    a = ap.parse_args()

    out_dir = os.path.join(ROOT, "runs", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "after_adam.pt")
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    from src.utils.logger import Logging
    from src.nn.pde import helmholtz_operator
    from src.data.helmholtz_dataset import u, f, generate_training_dataset
    from src.nn.DVPDESolver import DVPDESolver
    from src.nn.ClassicalSolver import ClassicalSolver

    DEVICE = torch.device("cpu")
    args = {
        "batch_size": a.batch, "epochs": a.cap, "lr": a.lr, "seed": a.seed,
        "print_every": 500, "log_path": os.path.join(out_dir, "authors_log"),
        "input_dim": 2, "output_dim": 1, "num_qubits": 5, "hidden_dim": 50,
        "num_quantum_layers": 1, "classic_network": [2, 50, 1],
        "q_ansatz": "cascade", "mode": "hybrid", "activation": "tanh",
        "shots": None, "problem": "helmholtz",
        "solver": "DV" if a.model == "qcpinn" else "Classical",
        "device": DEVICE, "method": "None", "cutoff_dim": 20,
        "class": "CVNeuralNetwork1", "encoding": "angle",
    }

    A1, A2, LAMBDA, npts = 1, 4, 1.0, 100
    t_ax = torch.linspace(-1, 1, npts).unsqueeze(1)
    x_ax = torch.linspace(-1, 1, npts).unsqueeze(1)
    T, X = torch.meshgrid(t_ax.squeeze(), x_ax.squeeze())
    X_star = torch.hstack((T.flatten().unsqueeze(1), X.flatten().unsqueeze(1)))
    u_star, f_star = u(X_star, A1, A2), f(X_star, A1, A2, LAMBDA)

    logger = Logging(args["log_path"])
    model = (ClassicalSolver if args["solver"] == "Classical" else DVPDESolver)(
        args, logger, X_star, DEVICE)
    # Replace the authors' Adam only when asked. The scheduler is rebuilt on the
    # new optimiser with the authors' own settings, so the optimiser family is
    # the single thing that differs.
    if a.optimizer != "adam":
        _tr = filter(lambda p: p.requires_grad, model.parameters())
        if a.optimizer == "sgd":
            model.optimizer = torch.optim.SGD(_tr, lr=a.lr, momentum=0.9)
        else:
            model.optimizer = torch.optim.RMSprop(_tr, lr=a.lr)
        model.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model.optimizer, mode="min", factor=0.9, patience=1000)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== {a.tag} === {args['solver']} params={n_params} chunk={a.chunk}", flush=True)

    bcs_sampler, res_sampler = generate_training_dataset(model.device)

    def batch(n_res=None, n_bc=None):
        n_res = n_res or model.batch_size
        n_bc = n_bc or model.batch_size
        bcs = [s.sample(n_bc) for s in bcs_sampler]
        Xr, fr = res_sampler.sample(n_res)
        return bcs, Xr, fr

    def residual_loss(Xr, fr):
        """MSE of the authors' Helmholtz residual, chunked for the 0.29 cache.
        Chunks are equal-sized, so the mean of chunk means is the true mean."""
        if Xr.shape[0] <= a.chunk:
            _, r = helmholtz_operator(model, Xr[:, 0:1], Xr[:, 1:2])
            return model.loss_fn(r, fr)
        parts = []
        for i in range(0, Xr.shape[0], a.chunk):
            c, g = Xr[i:i + a.chunk], fr[i:i + a.chunk]
            _, r = helmholtz_operator(model, c[:, 0:1], c[:, 1:2])
            parts.append(model.loss_fn(r, g) * c.shape[0])
        return sum(parts) / Xr.shape[0]

    def boundary_loss(bcs):
        # protocol2's exact expression whenever no edge needs chunking, so the
        # Adam phase reproduces bit for bit
        if all(Xb.shape[0] <= a.chunk for Xb, _ in bcs):
            return sum(model.loss_fn(model.forward(Xb), ub) for Xb, ub in bcs)
        tot = 0.0
        for Xb, ub in bcs:
            if Xb.shape[0] <= a.chunk:
                tot = tot + model.loss_fn(model.forward(Xb), ub)
            else:
                parts = [model.loss_fn(model.forward(Xb[i:i + a.chunk]),
                                       ub[i:i + a.chunk]) * Xb[i:i + a.chunk].shape[0]
                         for i in range(0, Xb.shape[0], a.chunk)]
                tot = tot + sum(parts) / Xb.shape[0]
        return tot

    def objective(pts):
        # ORDER IS LOAD-BEARING: the boundary term is built before the residual
        # term, as in protocol2. backward() accumulates gradients in
        # graph-construction order, so swapping these changes the run.
        bcs, Xr, fr = pts
        loss_bc = boundary_loss(bcs)
        loss_r = residual_loss(Xr, fr)
        return loss_r + 10.0 * loss_bc

    def evaluate():
        ups, fps = [], []
        for i in range(0, X_star.shape[0], 1000):
            c = X_star[i:i + 1000]
            up, fp = helmholtz_operator(model, c[:, 0:1], c[:, 1:2])
            ups.append(up.detach()); fps.append(fp.detach())
        u_pred, f_pred = torch.cat(ups), torch.cat(fps)
        return ((torch.norm(u_pred - u_star, 2) / torch.norm(u_star, 2) * 100).item(),
                (torch.norm(f_pred - f_star, 2) / torch.norm(f_star, 2) * 100).item())

    # ---------------------------------------------------------- Adam phase
    if a.resume and os.path.exists(ckpt):
        st = torch.load(ckpt)
        model.load_state_dict(st["model"])
        stop_reason, used, t_adam = st["stop_reason"], st["adam_epochs_used"], st["adam_seconds"]
        err_u_adam, err_f_adam = st["err_u_pct_after_adam"], st["err_f_pct_after_adam"]
        print(f"  [adam] resumed from checkpoint: err_u = {err_u_adam:.2f}% "
              f"({stop_reason} at {used})", flush=True)
    else:
        t0 = time.time()
        ema, trace, stop_reason, used = None, [], "cap", a.cap
        for it in range(1, a.cap + 1):
            model.optimizer.zero_grad()
            loss = objective(batch())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            model.optimizer.step()
            v = loss.item()
            model.scheduler.step(v)
            ema = v if ema is None else (1 - EMA_BETA) * ema + EMA_BETA * v
            if it % CHECK_EVERY == 0:
                trace.append((it, ema))
                print(f"  [adam] {it}/{a.cap} loss {v:.3e} ema {ema:.3e}", flush=True)
                if it > PATIENCE:
                    past = [e for i, e in trace if i <= it - PATIENCE]
                    if past and (past[-1] - ema) / past[-1] < TOL:
                        stop_reason, used = "converged", it
                        break
        t_adam = time.time() - t0
        err_u_adam, err_f_adam = evaluate()
        torch.save({"model": model.state_dict(), "stop_reason": stop_reason,
                    "adam_epochs_used": used, "adam_seconds": round(t_adam, 1),
                    "err_u_pct_after_adam": err_u_adam,
                    "err_f_pct_after_adam": err_f_adam}, ckpt)
        print(f"  [adam] done: err_u = {err_u_adam:.2f}% ({stop_reason} at {used}); "
              f"checkpoint saved", flush=True)

    if a.check_against is not None:
        d = abs(err_u_adam - a.check_against)
        assert d < 0.01, (f"after-Adam error {err_u_adam:.4f}% does not reproduce "
                          f"{a.check_against}% (differs by {d:.4f})")
        print(f"  [check] after-Adam error reproduces {a.check_against}%", flush=True)

    # ---------------------------------------------------------- L-BFGS phase
    t1 = time.time()
    n_eval = 0
    if a.lbfgs > 0:
        pts = batch(a.refine_res, a.refine_bc)
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=a.lbfgs,
                                tolerance_grad=1e-9, tolerance_change=1e-12,
                                history_size=50, line_search_fn="strong_wolfe")
        n = {"i": 0}

        def closure():
            opt.zero_grad()
            l = objective(pts)
            l.backward()
            n["i"] += 1
            return l
        opt.step(closure)
        n_eval = n["i"]
        print(f"  [lbfgs] {n_eval} evaluations on {a.refine_res} interior "
              f"+ 4 x {a.refine_bc} boundary, chunk {a.chunk}", flush=True)
    t_lbfgs = time.time() - t1

    err_u, err_f = evaluate()

    json.dump({"tag": a.tag, "model": a.model, "seed": a.seed,
               "protocol": "convergence+lbfgs(1400-point refinement, chunked, protocol2 summation order)",
               "stop_reason": stop_reason, "adam_epochs_used": used,
               "lbfgs_steps": a.lbfgs, "lbfgs_evaluations": n_eval,
               "refine_interior_points": a.refine_res,
               "refine_boundary_points_per_edge": a.refine_bc,
               "refine_total_points": a.refine_res + 4 * a.refine_bc,
               "chunk": a.chunk, "params": n_params,
               "optimizer": a.optimizer, "lr": a.lr, "batch_size": a.batch,
               "err_u_pct_after_adam": err_u_adam, "err_f_pct_after_adam": err_f_adam,
               "err_u_pct": err_u, "err_f_pct": err_f,
               "adam_seconds": t_adam, "lbfgs_seconds": round(t_lbfgs, 1),
               "resumed": bool(a.resume and os.path.exists(ckpt)),
               "source": "github.com/afrah/QCPINN @ 0dc467be, models and loss unmodified"},
              open(os.path.join(out_dir, "replication.json"), "w"), indent=2)
    print(f"  FINAL err_u = {err_u:.2f}%  (after Adam {err_u_adam:.2f}%; "
          f"{stop_reason} at {used}; {(t_adam + t_lbfgs) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
