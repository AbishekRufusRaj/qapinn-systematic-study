"""
quantum_scaling.py -- do the mechanism measurements hold across circuit sizes?

quantum_landscape.py measures the gradient landscape and the Sim et al. ansatz
metrics at the four qubits every statistical arm of this study uses. The obvious
objection is that four qubits is one point. This script repeats the measurements
at n = 2, 4, 6, 8, which is the range the comparable literature works in
(Ukwatta Hewage et al. use n in {4,6,8}; dos Santos et al. 3-4; QCPINN 5). No
training: these are properties of the circuit and its initialisation.

Reported per (n, ansatz, initialisation):
  quantum gradient share  fraction of the total gradient norm landing on the
                          circuit rather than on the classical stack
  effective rank          participation ratio of the singular values of
                          d(circuit output)/d(weights), against parameter count
  expressibility KL       distance of the output-state fidelity distribution
                          from Haar (lower = more expressible)
  entangling capability   mean Meyer-Wallach Q over random weight draws

The barren-plateau prediction is that gradient variance falls exponentially in n
for default initialisation; the expressibility prediction is that KL falls with
n and with depth. Both are checked here under a PDE-residual loss.
"""

import json

import numpy as np
import pennylane as qml
import torch

import config
import pdes
import pde_train as pt

def meyer_wallach(psi, n):
    """Global entanglement, generalised to any n (quantum_anatomy's copy is
    hard-wired to the four qubits used by the statistical arms)."""
    psi = psi.reshape([2] * n)
    purity = 0.0
    for k in range(n):
        m = np.moveaxis(psi, k, 0).reshape(2, -1)
        rho = m @ m.conj().T
        purity += np.real(np.trace(rho @ rho))
    return 2 * (1 - purity / n)


QUBITS = (2, 4, 6, 8)
ARMS = [("ring", None, "default"), ("ring", 0.1, "near-identity"),
        ("strong", None, "default"), ("strong", 0.1, "near-identity")]
L = 2
N_DRAWS = 30
N_PAIRS = 400
N_FUNCTIONALS = 80


def state_fn(n, entangler):
    dev = qml.device("default.qubit", wires=n)
    block = qml.StronglyEntanglingLayers if entangler == "strong" else qml.BasicEntanglerLayers

    @qml.qnode(dev, interface="torch")
    def f(x, t, w):
        feats = torch.stack([x, t] * ((n + 1) // 2))[:n]
        for l in range(L):
            qml.AngleEmbedding(feats, wires=range(n))
            block(w[l], wires=range(n))
        return qml.state()
    return f


def sample_weights(n, entangler, init_scale, gen):
    shape = (L, 1, n, 3) if entangler == "strong" else (L, 1, n)
    if init_scale is None:
        return torch.rand(shape, generator=gen, dtype=torch.float64) * 2 * np.pi
    return torch.randn(shape, generator=gen, dtype=torch.float64) * init_scale


def main():
    burgers = pdes.get("burgers")
    torch.manual_seed(0)
    pts = pt.sample_points(burgers, dict(config.SWEEP, n_collocation_uniform=200,
                                         n_collocation_shock=60, n_ic=60, n_bc=60))
    X, T = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(0, 1, 10))
    xt = torch.tensor(np.stack([X.ravel(), T.ravel()], 1), dtype=torch.float32)

    rows = []
    print(f"{'n':>3}  {'arm':22}{'q grad share':>14}{'grad variance':>15}"
          f"{'eff rank':>12}{'expr KL':>10}{'ent cap':>9}")
    print("-" * 88)
    for n in QUBITS:
        for ent, sc, label in ARMS:
            gq, gc, per = [], [], []
            for d in range(N_DRAWS):
                torch.manual_seed(5000 + d)
                m = pt.build_model("qapinn", n_qubits=n, reupload_layers=L,
                                   entangler=ent, encoding="angle", init_scale=sc)
                loss, *_ = pt.compute_losses(m, burgers, pts, config.SWEEP)
                loss.backward()
                w = m.quantum_layer.weights
                gq.append(float(w.grad.norm()))
                per.append(w.grad.detach().flatten().numpy().copy())
                gc.append(float(torch.sqrt(sum((p.grad ** 2).sum()
                                               for p in m.classical_net.parameters()))))
            per = np.array(per)
            share = float(np.median(np.array(gq) / (np.array(gq) + np.array(gc))))
            gvar = float(np.median(per.var(axis=0)))

            torch.manual_seed(0)
            m = pt.build_model("qapinn", n_qubits=n, reupload_layers=L, entangler=ent,
                               encoding="angle", init_scale=sc)
            w = m.quantum_layer.weights
            q = m.quantum_layer(xt)
            g = torch.Generator().manual_seed(0)
            M = []
            for r in range(N_FUNCTIONALS):
                s = (q * torch.randn(q.shape, generator=g)).sum()
                M.append(torch.autograd.grad(s, w, retain_graph=(r < N_FUNCTIONALS - 1))[0]
                         .flatten().numpy().copy())
            sv = np.linalg.svd(np.array(M), compute_uv=False)
            rank = float(sv.sum() ** 2 / (sv ** 2).sum())
            n_par = int(np.array(M).shape[1])

            f = state_fn(n, ent)
            gen = torch.Generator().manual_seed(0)
            x0 = torch.tensor(0.3, dtype=torch.float64)
            t0 = torch.tensor(0.5, dtype=torch.float64)
            fid, mw = [], []
            for _ in range(N_PAIRS):
                a = f(x0, t0, sample_weights(n, ent, sc, gen)).detach().numpy()
                b = f(x0, t0, sample_weights(n, ent, sc, gen)).detach().numpy()
                fid.append(abs(np.vdot(a, b)) ** 2)
                mw.append(meyer_wallach(a, n))
            bins = np.linspace(0, 1, 76)
            p, _ = np.histogram(fid, bins=bins)
            p = p / p.sum()
            dim = 2 ** n
            c = (bins[:-1] + bins[1:]) / 2
            haar = (dim - 1) * (1 - c) ** (dim - 2)
            haar = haar / haar.sum()
            # At n = 8 the Haar density (1-F)^(2^n - 2) underflows to zero in the
            # bins where near-identity circuits place all their mass, which sends
            # the KL to infinity. Clipping the reference density at the smallest
            # representable positive value turns those entries into a finite LOWER
            # BOUND on the divergence, flagged by kl_is_bound.
            haar_c = np.clip(haar, 1e-300, None)
            msk = p > 0
            kl = float(np.sum(p[msk] * np.log(p[msk] / haar_c[msk])))
            kl_bound = bool(np.any(haar[msk] <= 0))

            rows.append(dict(n=n, entangler=ent, init=label, grad_share=share,
                             grad_variance=gvar, eff_rank=rank, n_params=n_par,
                             expressibility_kl=kl, kl_is_bound=kl_bound,
                             entangling_capability=float(np.mean(mw))))
            print(f"{n:>3}  {ent + ', ' + label:22}{share:14.4f}{gvar:15.3e}"
                  f"{rank:8.2f}/{n_par:<3}{('>' if kl_bound else '') + format(kl, '.2f'):>10}{np.mean(mw):9.3f}", flush=True)

    json.dump(rows, open("quantum_scaling.json", "w"), indent=2)
    print("\nwrote quantum_scaling.json")


if __name__ == "__main__":
    main()
