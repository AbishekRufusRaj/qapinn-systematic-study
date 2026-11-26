"""
helmholtz_ours.py -- this project's models and protocol on the 2D Helmholtz
problem used by QCPINN (arXiv:2503.16678).

The problem, taken from the authors' code (external/QCPINN/src/data/
helmholtz_dataset.py and src/nn/pde.py):

    u_x1x1 + u_x2x2 + lambda u = f   on [-1, 1]^2,  lambda = 1
    u*(x1, x2) = sin(pi x1) sin(4 pi x2)      (a1 = 1, a2 = 4)
    f = (1 - 17 pi^2) u*,  u = 0 on all four edges

This closes the "two-dimensional PDE" gap in the future-work list: two
spatial dimensions, an exact solution, and the oscillation sits in x2.

How it plugs in: pde_train.py assumes (x, t) with an initial condition at
t = 0. It is imported, NOT edited (it produced the runs already on disk).
The second coordinate is passed through the model's `t` input, and the three
functions that encode the time assumption -- point sampling, the loss, and
evaluation -- are replaced for this PDE only. Everything else is pde_train.py
unchanged: models, recipes, the convergence criterion, L-BFGS, the
pre-registered threshold, the run-directory format.

Loss, matching the structure of both this project and the authors' code:
    L = mean(residual^2) + 10 * [bottom+top edges] + 10 * [left+right edges]
The "ic" and "bc" history slots hold the x2-edges and x1-edges respectively.

The residual is NOT rescaled, so the loss has the same scale as the authors'
(|f| up to ~167). The pre-registered 1e-2 loss threshold was calibrated on the
1D problems and is not meaningful at this scale; runs are judged on relative
L2 error, reported below as `rel_l2`, as the authors do.
"""

import argparse
import json
import os

import numpy as np
import torch

import config
import pdes
import pde_train as pt

A1, A2, LAM = 1.0, 4.0, 1.0


class Helmholtz(pdes.PDE):
    name = "helmholtz"
    x_min, x_max = -1.0, 1.0
    t_min, t_max = -1.0, 1.0          # "t" carries x2

    def residual(self, model, x, t):
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        u = model(x, t)
        ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
        utt = torch.autograd.grad(ut, t, torch.ones_like(ut), create_graph=True)[0]
        return uxx + utt + LAM * u - self.forcing(x, t)

    @staticmethod
    def exact_t(x, t):
        return torch.sin(A1 * np.pi * x) * torch.sin(A2 * np.pi * t)

    def forcing(self, x, t):
        return (LAM - (A1 * np.pi) ** 2 - (A2 * np.pi) ** 2) * self.exact_t(x, t)

    def ic(self, x):
        return torch.zeros_like(x)

    def ground_truth(self):
        return lambda X, T: np.sin(A1 * np.pi * X) * np.sin(A2 * np.pi * T)

    def sample_collocation(self, n_uniform, n_focus, generator=None):
        n = n_uniform + n_focus
        x = torch.rand(n, 1, generator=generator) * 2 - 1
        t = torch.rand(n, 1, generator=generator) * 2 - 1
        return x, t

    def decompose(self, X, T, err, U_true):
        band = (np.abs(X) > 0.9) | (np.abs(T) > 0.9)
        rb = float(np.linalg.norm(err[band]) / np.linalg.norm(U_true[band]))
        ri = float(np.linalg.norm(err[~band]) / np.linalg.norm(U_true[~band]))
        return {"rel_l2_boundary_band": rb, "rel_l2_interior": ri,
                "hard_easy_ratio": rb / ri if ri > 0 else float("nan"),
                "region_definition": "boundary band |x1|>0.9 or |x2|>0.9"}


pdes.REGISTRY["helmholtz"] = Helmholtz
_orig_sample, _orig_losses, _orig_eval = pt.sample_points, pt.compute_losses, pt.evaluate_run


def sample_points(pde, budget, generator=None):
    if not isinstance(pde, Helmholtz):
        return _orig_sample(pde, budget, generator)
    x_f, t_f = pde.sample_collocation(budget["n_collocation_uniform"],
                                      budget["n_collocation_shock"], generator)
    n = budget["n_bc"]
    r = lambda: torch.rand(n, 1, generator=generator) * 2 - 1
    lo, hi = torch.full((n, 1), -1.0), torch.full((n, 1), 1.0)
    edges_t = (r(), lo, r(), hi)       # bottom (x2=-1), top (x2=+1)
    edges_x = (lo, r(), hi, r())       # left (x1=-1), right (x1=+1)
    return x_f, t_f, edges_t, edges_x


def compute_losses(model, pde, pts, budget):
    if not isinstance(pde, Helmholtz):
        return _orig_losses(model, pde, pts, budget)
    x_f, t_f, (xb, tb, xt, tt), (xl, tl, xr, tr) = pts
    loss_pde = torch.mean(pde.residual(model, x_f, t_f) ** 2)
    loss_ic = torch.mean(model(xb, tb) ** 2) + torch.mean(model(xt, tt) ** 2)
    loss_bc = torch.mean(model(xl, tl) ** 2) + torch.mean(model(xr, tr) ** 2)
    loss = loss_pde + budget["w_ic"] * loss_ic + budget["w_bc"] * loss_bc
    return loss, loss_pde, loss_ic, loss_bc


def evaluate_run(run_dir, nx=200, nt=200, verbose=True, checkpoint=None):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    if meta["pde"] != "helmholtz":
        return _orig_eval(run_dir, nx, nt, verbose, checkpoint)
    pde = Helmholtz()
    model = pt.build_model(meta["model_type"], **meta["model_kwargs"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "weights.pt")))
    model.eval()
    X, T = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, nt))
    with torch.no_grad():
        U = model(torch.tensor(X.ravel(), dtype=torch.float32).reshape(-1, 1),
                  torch.tensor(T.ravel(), dtype=torch.float32).reshape(-1, 1)
                  ).numpy().reshape(X.shape)
    U_true = pde.ground_truth()(X, T)
    err = U - U_true
    out = dict(meta, rel_l2=float(np.linalg.norm(err) / np.linalg.norm(U_true)),
               checkpoint_epoch=None)
    out.update(pde.decompose(X, T, err, U_true))
    json.dump(out, open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
    if verbose:
        print(f"{meta['tag']:30s} rel L2 {out['rel_l2']:7.2%}", flush=True)
    return out


pt.sample_points, pt.compute_losses, pt.evaluate_run = sample_points, compute_losses, evaluate_run


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["classical", "qapinn", "model2"], required=True,
                   help="model2 is the QCPINN paper's 2,751-parameter classical baseline, "
                        "run here under THIS project's protocol so the comparison with the "
                        "published numbers is protocol-matched")
    p.add_argument("--their-recipe", action="store_true",
                   help="train as the QCPINN paper does instead: fixed 20,000 epochs, "
                        "Adam at 5e-3 with their scheduler, and no L-BFGS phase")
    p.add_argument("--entangler", choices=["ring", "strong"], default="ring")
    p.add_argument("--init-scale", type=float, default=None,
                   help="quantum near-identity: 0.1 (as every QAPINN arm in the study)")
    p.add_argument("--init-gain", type=float, default=None,
                   help="classical init gain: 1.7321 = Xavier-equivalent")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--max-epochs", type=int, default=25000)
    a = p.parse_args()

    if a.model == "qapinn":
        mk = dict(n_qubits=4, reupload_layers=2, entangler=a.entangler,
                  encoding="angle", init_scale=a.init_scale)
    elif a.model == "classical":
        mk = dict(init_gain=a.init_gain)
    else:
        mk = {}
    budget = dict(config.SWEEP)
    kw = {}
    if a.their_recipe:
        budget["lbfgs_steps"] = 0          # their protocol has no L-BFGS phase
        kw["recipe"] = "qcpinn"            # lr 5e-3, ReduceLROnPlateau 0.9/1000
        kw["patience"] = 10 ** 9           # no convergence criterion: run the full budget
        a.max_epochs = 20000
    pt.run(pde_name="helmholtz", kind=a.model, model_kwargs=mk, budget=budget,
           seed=a.seed, max_epochs=a.max_epochs, tag=a.tag, **kw)
