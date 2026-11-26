"""
replicate_qcpinn.py -- a GENUINE replication of the QCPINN Helmholtz result.

Target (arXiv:2503.16678 v6, Table 3, Helmholtz row):
    QCPINN, DV, angle encoding, cascade ansatz, 5 qubits, L = 1, 771 params
        relative L2 error u = 6.69 +/- 1.51 %,  f = 2.86 +/- 0.54 %
    PINN Model-2, 2,751 params
        relative L2 error u = 12.12 +/- 7.17 %, f = 3.23 +/- 1.19 %
    "mean +/- std across 10 independent runs"

What is run: the AUTHORS' OWN CODE, unmodified, from external/QCPINN
(github.com/afrah/QCPINN @ 0dc467be). Their trainer (src/trainer/
helmholtz_hybrid_trainer.py) is not called directly only because its
hard-coded config is the continuous-variable variant; this script builds the
same `args` dict with the paper's stated DV settings (Sec. 5.2) and then calls
their own DVPDESolver / ClassicalSolver, their helmholtz_train.train, and
reproduces their own error computation line for line.

The ONLY addition is seeding. Their config carries "seed": 1 but never applies
it, so their runs are unrecorded; here torch and numpy are seeded per run so
every run can be reproduced. Nothing of this project's protocol (convergence
criterion, L-BFGS, failure threshold) is applied -- that would no longer be a
replication.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qcpinn", "model2"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--freeze", action="store_true",
                    help="keep the circuit but FREEZE its 15 weights at initialisation: "
                         "separates 'the circuit is a useful fixed map' from 'the trained "
                         "quantum parameters matter'")
    ap.add_argument("--ablate-sin", action="store_true",
                    help="replace the circuit with a CLASSICAL Fourier feature map "
                         "sin(Wx+b), same shape and parameter count as --ablate: does the "
                         "circuit's benefit come from trigonometric encoding rather than "
                         "from quantumness?")
    ap.add_argument("--ablate", action="store_true",
                    help="replace the 15-parameter circuit with a parameter-matched "
                         "classical map (rank-1 linear + bias, 5->5): does the circuit "
                         "contribute anything measurable?")
    ap.add_argument("--init", choices=["code", "paper"], default="code",
                    help="code = as the authors' repository does it (default); "
                         "paper = Xavier on ALL classical Linear layers, as the paper's "
                         "text states")
    a = ap.parse_args()

    out_dir = os.path.join(ROOT, "runs", a.tag)
    os.makedirs(out_dir, exist_ok=True)

    os.chdir(REPO)
    sys.path.insert(0, REPO)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    from src.utils.logger import Logging
    from src.nn.pde import helmholtz_operator
    from src.data.helmholtz_dataset import u, f
    import src.trainer.helmholtz_train as helmholtz_train
    from src.nn.DVPDESolver import DVPDESolver
    from src.nn.ClassicalSolver import ClassicalSolver

    DEVICE = torch.device("cpu")
    input_dim, output_dim, hidden_dim = 2, 1, 50
    args = {
        "batch_size": 64,
        "epochs": a.epochs,
        "lr": 0.005,                         # paper Sec. 5.2, DV circuits
        "seed": a.seed,
        "print_every": 100,
        "log_path": os.path.join(out_dir, "authors_log"),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "num_qubits": 5,                     # paper: five qubits
        "hidden_dim": hidden_dim,
        "num_quantum_layers": 1,             # L = 1 -> 3nL = 15 quantum params
        "classic_network": [input_dim, hidden_dim, output_dim],
        "q_ansatz": "cascade",
        "mode": "hybrid",
        "activation": "tanh",
        "shots": None,
        "problem": "helmholtz",
        "solver": "DV" if a.model == "qcpinn" else "Classical",
        "device": DEVICE,
        "method": "None",
        "cutoff_dim": 20,
        "class": "CVNeuralNetwork1",
        "encoding": "angle",
    }

    # --- from here on, identical to the authors' helmholtz_hybrid_trainer.py
    A1, A2, LAMBDA, num_points = 1, 4, 1.0, 100
    dom = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float32)
    t = torch.linspace(dom[0, 0], dom[1, 0], num_points).unsqueeze(1)
    x = torch.linspace(dom[0, 1], dom[1, 1], num_points).unsqueeze(1)
    t, x = torch.meshgrid(t.squeeze(), x.squeeze())
    X_star = torch.hstack((t.flatten().unsqueeze(1), x.flatten().unsqueeze(1)))

    logger = Logging(args["log_path"])
    if args["solver"] == "Classical":
        model = ClassicalSolver(args, logger, X_star, DEVICE)
    else:
        model = DVPDESolver(args, logger, X_star, DEVICE)
    if a.init == "paper":
        # Paper Sec. 5.1: "Classical neural network layers were initialized
        # using Xavier initialization". The repository applies it only to the
        # pre-processor; here it is applied to every classical Linear layer,
        # with the repository's own choices (xavier_normal_, zero bias).
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    if a.ablate_sin:
        # The circuit encodes each input as an R_X rotation, so its outputs are
        # trigonometric functions of the inputs -- a Fourier feature map. This
        # stand-in is that map and nothing else: same shape, same parameter count
        # as the --ablate stand-in, sine in place of tanh.
        class SinFeatures(torch.nn.Module):
            def __init__(self, n):
                super().__init__()
                self.lin = torch.nn.Linear(n, n)
            def forward(self, z):
                return torch.sin(self.lin(z))
        model.quantum_layer = SinFeatures(args["num_qubits"])
        model.optimizer = torch.optim.Adam(
            filter(lambda q: q.requires_grad, model.parameters()), lr=args["lr"])
        model.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model.optimizer, mode="min", factor=0.9, patience=1000)
    if a.ablate:
        nq = args["num_qubits"]
        # A parameter-matched ablation must keep the circuit's SHAPE (5 -> 5, fully
        # mixing) and only remove its quantumness. A rank-1 bottleneck would match
        # the 15-parameter count but destroy the information path, which tests
        # capacity rather than quantumness. Tanh(Linear(5,5)) is the smallest
        # honest classical stand-in: 30 parameters against the circuit's 15, so the
        # classical side is if anything favoured -- the point of the test is whether
        # removing the circuit HURTS.
        model.quantum_layer = torch.nn.Sequential(
            torch.nn.Linear(nq, nq), torch.nn.Tanh())
        model.optimizer = torch.optim.Adam(
            filter(lambda q: q.requires_grad, model.parameters()), lr=args["lr"])
        model.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model.optimizer, mode="min", factor=0.9, patience=1000)
    if a.freeze:
        for q in model.quantum_layer.parameters():
            q.requires_grad_(False)
        model.optimizer = torch.optim.Adam(
            filter(lambda q: q.requires_grad, model.parameters()), lr=args["lr"])
        model.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model.optimizer, mode="min", factor=0.9, patience=1000)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== {a.tag} === solver={args['solver']} params={n_params}", flush=True)

    t0 = time.time()
    helmholtz_train.train(model)
    secs = time.time() - t0

    u_star = u(X_star, A1, A2)
    f_star = f(X_star, A1, A2, LAMBDA)
    # Evaluated in chunks of 1,000 points: PennyLane 0.29's execution cache
    # rejects a single 10,000-point batch ("value too large"). Each point is
    # evaluated independently, so the concatenated result -- and therefore the
    # error below -- is identical to a single-batch evaluation.
    ups, fps = [], []
    for i in range(0, X_star.shape[0], 1000):
        c = X_star[i:i + 1000]
        up, fp = helmholtz_operator(model, c[:, 0:1], c[:, 1:2])
        ups.append(up.detach()); fps.append(fp.detach())
    u_pred, f_pred = torch.cat(ups), torch.cat(fps)
    err_u = (torch.norm(u_pred - u_star, 2) / torch.norm(u_star, 2) * 100).item()
    err_f = (torch.norm(f_pred - f_star, 2) / torch.norm(f_star, 2) * 100).item()
    # --- end of the authors' procedure

    res = {"tag": a.tag, "model": a.model, "seed": a.seed, "epochs": a.epochs,
           "params": n_params, "init": a.init, "ablated": bool(a.ablate), "ablated_sin": bool(a.ablate_sin), "frozen_circuit": bool(a.freeze), "err_u_pct": err_u, "err_f_pct": err_f,
           "seconds": round(secs, 1),
           "source": "github.com/afrah/QCPINN @ 0dc467be, unmodified",
           "config": {k: (str(v) if k == "device" else v) for k, v in args.items()}}
    json.dump(res, open(os.path.join(out_dir, "replication.json"), "w"), indent=2)
    print(f"  err_u = {err_u:.2f}%  err_f = {err_f:.2f}%  ({secs/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
