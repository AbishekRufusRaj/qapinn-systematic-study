"""
protocol_qcpinn.py -- the symmetric protocol control the reviewers asked for.

The paper compares the published fixed-budget protocol against a
convergence-criterion protocol, but so far only the CLASSICAL arm has been run
under both. Reviewers 4 and 5 both identified that asymmetry. This script
removes it by running BOTH arms under BOTH protocols inside the authors' own
code, so the only thing that differs between the two protocols is the stopping
rule and the refinement phase.

What is kept from the authors' implementation (external/QCPINN, MIT):
    the models (DVPDESolver for QCPINN, ClassicalSolver for the baseline),
    the Helmholtz problem and its samplers, their mini-batch size of 64,
    their loss (residual + 10 x boundary), their Adam learning rate 5e-3,
    their ReduceLROnPlateau scheduler, their gradient clipping at 1.0,
    their relative-L2 error computation.

What this script changes, and only this:
    stopping   fixed 20,000 epochs  ->  EMA convergence criterion
               (beta = 0.002, tau = 3% over a patience of 3,000 epochs,
               checked every 500, hard cap 25,000) -- identical to the
               criterion used everywhere else in this study
    refinement none                 ->  500 L-BFGS steps on a fixed batch,
                                        strong-Wolfe line search

Both are applied identically to the quantum and the classical arm. Their own
code path is untouched: this file imports their modules and re-implements the
training LOOP around their objective, because their loop hard-codes the epoch
count.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qcpinn", "model2"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--lbfgs", type=int, default=LBFGS_STEPS)
    a = ap.parse_args()

    out_dir = os.path.join(ROOT, "runs", a.tag)
    os.makedirs(out_dir, exist_ok=True)
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

    logger = Logging(args["log_path"])
    model = (ClassicalSolver if args["solver"] == "Classical" else DVPDESolver)(
        args, logger, X_star, DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== {a.tag} === {args['solver']} params={n_params}", flush=True)

    bcs_sampler, res_sampler = generate_training_dataset(model.device)

    def batch():
        """One draw of the authors' five mini-batches."""
        bcs = [s.sample(model.batch_size) for s in bcs_sampler]
        Xr, fr = res_sampler.sample(model.batch_size)
        return bcs, Xr, fr

    def objective(pts):
        """The authors' DV loss: residual + 10 x boundary."""
        bcs, Xr, fr = pts
        loss_bc = sum(model.loss_fn(model.forward(Xb), ub) for Xb, ub in bcs)
        _, r_pred = helmholtz_operator(model, Xr[:, 0:1], Xr[:, 1:2])
        loss_r = model.loss_fn(r_pred, fr)
        return loss_r + 10.0 * loss_bc

    # ---------------------------------------------------------- Adam phase
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

    # ---------------------------------------------------------- L-BFGS phase
    t1 = time.time()
    if a.lbfgs > 0:
        pts = batch()                     # fixed batch: the line search needs it
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
        print(f"  [lbfgs] {n['i']} evaluations", flush=True)
    t_lbfgs = time.time() - t1

    # ---------------------------------------------------------- evaluation
    ups, fps = [], []
    for i in range(0, X_star.shape[0], 1000):
        c = X_star[i:i + 1000]
        up, fp = helmholtz_operator(model, c[:, 0:1], c[:, 1:2])
        ups.append(up.detach()); fps.append(fp.detach())
    u_pred, f_pred = torch.cat(ups), torch.cat(fps)
    u_star, f_star = u(X_star, A1, A2), f(X_star, A1, A2, LAMBDA)
    err_u = (torch.norm(u_pred - u_star, 2) / torch.norm(u_star, 2) * 100).item()
    err_f = (torch.norm(f_pred - f_star, 2) / torch.norm(f_star, 2) * 100).item()

    json.dump({"tag": a.tag, "model": a.model, "seed": a.seed,
               "protocol": "convergence+lbfgs", "stop_reason": stop_reason,
               "adam_epochs_used": used, "lbfgs_steps": a.lbfgs,
               "params": n_params, "err_u_pct": err_u, "err_f_pct": err_f,
               "adam_seconds": round(t_adam, 1), "lbfgs_seconds": round(t_lbfgs, 1),
               "source": "github.com/afrah/QCPINN @ 0dc467be, models and loss unmodified"},
              open(os.path.join(out_dir, "replication.json"), "w"), indent=2)
    print(f"  err_u = {err_u:.2f}%  ({stop_reason} at {used}; "
          f"{(t_adam + t_lbfgs) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
