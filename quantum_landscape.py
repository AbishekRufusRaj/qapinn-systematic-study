"""
quantum_landscape.py -- why the ansatz x initialisation interaction happens,
measured on the circuits themselves. No training: every number here comes from
freshly initialised models, so it is a property of the ARCHITECTURE AND ITS
INITIALISATION, not of any particular run.

Chapter 4 argues the mechanism algebraically: the ring's trainable R_X is the
same gate type as the encoding, so R_X(w)R_X(f) = R_X(f+w) and its weights are
additive offsets that can never produce a Y or Z rotation. Appendix B shows the
consequence for accessible frequency at L >= 4. What has never been measured is
the consequence for the OPTIMISATION LANDSCAPE at L = 2, which is the depth every
statistical arm actually used -- and where today's anatomy shows both ansaetze
reach the same frequency ceiling and the same entanglement. This script measures
that landscape.

Four measurements per arm (ring/strong x default/near-identity):

  1. GRADIENT MAGNITUDE AND VARIANCE of the PDE loss with respect to the quantum
     weights, over independent initialisations -- the standard barren-plateau
     quantity, here under the PDE-residual loss and per ansatz/init combination
     rather than per qubit count (which gradient_variance.py already did).

  2. QUANTUM-VS-CLASSICAL GRADIENT SHARE: the same draw, comparing the gradient
     norm on the 8 or 24 circuit weights against the ~2,600 classical weights.
     If the circuit receives a negligible share of the learning signal, the
     "quantum" model is a classical network with a fixed front end.

  3. EFFECTIVE PARAMETER RANK of the circuit output: the participation ratio of
     the singular values of d(circuit output)/d(weights), probed with random
     functionals. P parameters whose effect on the output is degenerate behave
     as fewer than P knobs. This is the direct landscape reading of the gate
     algebra: additive offsets should show a lower effective rank than full
     rotations.

  4. EXPRESSIBILITY and ENTANGLING CAPABILITY (Sim, Johnson & Aspuru-Guzik 2019):
     KL divergence of the circuit's output-state fidelity distribution from the
     Haar distribution (lower = more expressible), and mean Meyer-Wallach Q over
     random weight draws. Reported for L = 1, 2, 4 so the L = 2 result sits in a
     depth trend.

Outputs quantum_landscape.json and a table on stdout.
"""

import json

import numpy as np
import pennylane as qml
import torch

import config
import pdes
import pde_train as pt
from quantum_anatomy import meyer_wallach, state_fn

ARMS = [("ring", None, "default"), ("ring", 0.1, "near-identity"),
        ("strong", None, "default"), ("strong", 0.1, "near-identity")]
N, L_MAIN = 4, 2
N_DRAWS = 60          # independent initialisations per arm
N_POINTS = 400        # collocation subset for the gradient measurement
N_FUNCTIONALS = 64    # random output functionals for the rank probe
N_PAIRS = 1500        # weight-pairs for the expressibility histogram


def make(entangler, init_scale, L=L_MAIN):
    return pt.build_model("qapinn", n_qubits=N, reupload_layers=L,
                          entangler=entangler, encoding="angle", init_scale=init_scale)


def sample_weights(entangler, init_scale, L, gen):
    shape = (L, 1, N, 3) if entangler == "strong" else (L, 1, N)
    if init_scale is None:
        return torch.rand(shape, generator=gen, dtype=torch.float64) * 2 * np.pi
    return torch.randn(shape, generator=gen, dtype=torch.float64) * init_scale


# ------------------------------------------------------- 1, 2: gradients

def gradient_stats(entangler, init_scale, burgers, pts):
    gq, gc, per_param = [], [], []
    for d in range(N_DRAWS):
        torch.manual_seed(10_000 + d)
        m = make(entangler, init_scale)
        loss, *_ = pt.compute_losses(m, burgers, pts, config.SWEEP)
        loss.backward()
        w = m.quantum_layer.weights
        gq.append(float(w.grad.norm()))
        per_param.append(w.grad.detach().flatten().numpy().copy())
        gc.append(float(torch.sqrt(sum((p.grad ** 2).sum()
                                       for p in m.classical_net.parameters()))))
    per_param = np.array(per_param)
    return {"grad_norm_quantum_median": float(np.median(gq)),
            "grad_norm_classical_median": float(np.median(gc)),
            "quantum_share_median": float(np.median(np.array(gq) /
                                                    (np.array(gq) + np.array(gc)))),
            "per_param_variance_median": float(np.median(per_param.var(axis=0))),
            "per_param_variance_min": float(per_param.var(axis=0).min()),
            "n_params": int(per_param.shape[1])}


# ------------------------------------------------------- 3: effective rank

def effective_rank(entangler, init_scale, xt, seed=0):
    torch.manual_seed(seed)
    m = make(entangler, init_scale)
    w = m.quantum_layer.weights
    q = m.quantum_layer(xt)                       # (points, 4)
    g = torch.Generator().manual_seed(seed)
    rows = []
    for r in range(N_FUNCTIONALS):
        c = torch.randn(q.shape, generator=g)
        s = (q * c).sum()
        gr = torch.autograd.grad(s, w, retain_graph=(r < N_FUNCTIONALS - 1))[0]
        rows.append(gr.flatten().numpy().copy())
    M = np.array(rows)
    sv = np.linalg.svd(M, compute_uv=False)
    pr = sv.sum() ** 2 / (sv ** 2).sum() if sv.sum() > 0 else float("nan")
    return {"effective_rank": float(pr), "n_params": int(M.shape[1]),
            "rank_fraction": float(pr / M.shape[1])}


# ------------------------------------------------------- 4: Sim et al. metrics

def haar_kl_and_entanglement(entangler, init_scale, L, seed=0):
    f = state_fn(entangler, L)
    gen = torch.Generator().manual_seed(seed)
    x, t = torch.tensor(0.3, dtype=torch.float64), torch.tensor(0.5, dtype=torch.float64)
    fid, mw = [], []
    for _ in range(N_PAIRS):
        a = f(x, t, sample_weights(entangler, init_scale, L, gen)).detach().numpy()
        b = f(x, t, sample_weights(entangler, init_scale, L, gen)).detach().numpy()
        fid.append(abs(np.vdot(a, b)) ** 2)
        mw.append(meyer_wallach(a))
    bins = np.linspace(0, 1, 76)
    p, _ = np.histogram(fid, bins=bins, density=False)
    p = p / p.sum()
    dim = 2 ** N
    centres = (bins[:-1] + bins[1:]) / 2
    haar = (dim - 1) * (1 - centres) ** (dim - 2)
    haar = haar / haar.sum()
    mask = p > 0
    kl = float(np.sum(p[mask] * np.log(p[mask] / haar[mask])))
    return {"expressibility_kl": kl, "entangling_capability": float(np.mean(mw))}


def main():
    burgers = pdes.get("burgers")
    torch.manual_seed(0)
    pts = pt.sample_points(burgers, dict(config.SWEEP, n_collocation_uniform=N_POINTS,
                                         n_collocation_shock=100, n_ic=100, n_bc=100))
    X, T = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(0, 1, 12))
    xt = torch.tensor(np.stack([X.ravel(), T.ravel()], 1), dtype=torch.float32)

    out = {"arms": {}, "depth_trend": {}}
    print(f"\nGRADIENT LANDSCAPE AT INITIALISATION  ({N_DRAWS} independent draws, "
          f"PDE loss, L = {L_MAIN})\n")
    print(f"{'arm':22}{'|grad| quantum':>16}{'|grad| classical':>18}"
          f"{'q share':>9}{'min per-param var':>19}{'eff. rank':>12}")
    print("-" * 96)
    for ent, sc, label in ARMS:
        g = gradient_stats(ent, sc, burgers, pts)
        r = effective_rank(ent, sc, xt)
        out["arms"][f"{ent}_{label}"] = {**g, **r}
        print(f"{ent + ', ' + label:22}{g['grad_norm_quantum_median']:16.3e}"
              f"{g['grad_norm_classical_median']:18.3e}{g['quantum_share_median']:9.3f}"
              f"{g['per_param_variance_min']:19.3e}"
              f"{r['effective_rank']:8.2f}/{r['n_params']:<3}", flush=True)

    print(f"\nSIM ET AL. ANSATZ METRICS  ({N_PAIRS} weight pairs; "
          f"expressibility KL: lower = closer to Haar)\n")
    print(f"{'arm':22}{'L':>3}{'expressibility KL':>20}{'entangling capability':>24}")
    print("-" * 70)
    for ent, sc, label in ARMS:
        for L in (1, 2, 4):
            m = haar_kl_and_entanglement(ent, sc, L)
            out["depth_trend"][f"{ent}_{label}_L{L}"] = m
            print(f"{ent + ', ' + label:22}{L:3}{m['expressibility_kl']:20.3f}"
                  f"{m['entangling_capability']:24.3f}", flush=True)

    json.dump(out, open("quantum_landscape.json", "w"), indent=2)
    print("\nwrote quantum_landscape.json")


if __name__ == "__main__":
    main()
