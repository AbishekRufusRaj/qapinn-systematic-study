"""
protocol3_qcpinn.py -- symmetric protocol control, third and corrected attempt.

History, so the artefacts are readable later:

  protocol_qcpinn.py   gave the L-BFGS refinement the authors' TRAINING
                       mini-batch of 64 points. 500 strong-Wolfe iterations
                       against 64 residual points overfits: the quantum arm
                       degraded to a 14.22% median and three classical runs
                       left an early plateau for 319%, 1882% and nan.

  protocol2_qcpinn.py  enlarged the refinement batch to 4,000 residual points,
                       which fixed the overfitting (classical: 8.09% -> 4.16%
                       median) but exceeded the cachetools value limit in
                       PennyLane 0.29. All ten quantum runs completed Adam,
                       evaluated correctly, and then died in the refinement
                       phase. Its docstring also claimed the batch "matched"
                       this study's protocol at ~4,600 points; that was wrong.
                       This study's convergence runs use config.SWEEP, which is
                       1,400 points per epoch, not 4,600 (that is config.FULL,
                       used elsewhere).

This version:

  1. refines on 1,400 points -- 1,000 interior and 100 per boundary edge --
     which is the scale this study's own protocol actually uses, and is 22x the
     authors' mini-batch rather than 62x;

  2. evaluates every quantum circuit call in chunks of 500 points, the same
     workaround replicate_qcpinn.py uses for the PennyLane 0.29 cache. Chunk
     means are averaged with equal chunk sizes, so the chunked loss equals the
     unchunked one;

  3. saves the model state after Adam to runs/<tag>/after_adam.pt and accepts
     --resume to reload it and run refinement alone, so a failure in the
     refinement phase can never again cost the two hours of Adam before it;

  4. records the relative L2 error after Adam and again after refinement.

The Adam phase draws from the samplers in exactly the order protocol2 used, so
its trajectory is bit-identical for a given seed. The after-Adam errors this
script reports must therefore reproduce the ones in logs_extend/sym2_qcpinn_s*.log;
--check-against asserts that, which verifies the salvage and this script at once.

Nothing in the authors' code is modified: their models, samplers, loss
(residual + 10 x boundary), Adam at 5e-3, ReduceLROnPlateau, gradient clipping
at 1.0 and relative-L2 computation are used as released.
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
        "batch_size": 64, "epochs": a.cap, "lr": 0.005, "seed": a.seed,
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
        bcs, Xr, fr = pts
        return residual_loss(Xr, fr) + 10.0 * boundary_loss(bcs)

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
               "protocol": "convergence+lbfgs(1400-point refinement, chunked)",
               "stop_reason": stop_reason, "adam_epochs_used": used,
               "lbfgs_steps": a.lbfgs, "lbfgs_evaluations": n_eval,
               "refine_interior_points": a.refine_res,
               "refine_boundary_points_per_edge": a.refine_bc,
               "refine_total_points": a.refine_res + 4 * a.refine_bc,
               "chunk": a.chunk, "params": n_params,
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
