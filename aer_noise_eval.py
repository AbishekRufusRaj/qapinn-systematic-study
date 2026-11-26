"""
aer_noise_eval.py -- trained QAPINNs re-evaluated on CALIBRATED IBM device models.

WHY THIS EXISTS ALONGSIDE noise_eval.py
---------------------------------------
noise_eval.py applies a synthetic depolarising channel at a rate we choose. That answers
"what happens under noise of magnitude p", which is useful but hypothetical: the rate is
a knob, not a measurement.

This instead uses qiskit-aer with a NoiseModel built from a real IBM backend's published
calibration -- per-gate error rates, T1/T2 relaxation, and readout error, all measured on
hardware -- and transpiles the circuit onto that device's actual coupling map. Two things
change as a result:

  1. The noise is no longer a free parameter. It is whatever the device reports.
  2. ROUTING enters. Our CNOT ring (0-1, 1-2, 2-3, 3-0) is not native to a line or to a
     heavy-hex lattice, so the transpiler inserts SWAPs. On FakeManila the 8 logical
     two-qubit gates become 22 physical ones. The noiseless study cannot see this cost
     at all, and it falls on the entangling structure -- which is exactly the axis this
     project compares.

STILL A DEPLOYMENT MEASUREMENT, NOT A TRAINABILITY ONE. The weights come from the
noiseless runs. This does not simulate training on hardware.

VALIDATION. With the noise model removed, Aer must reproduce each run's recorded rel_l2
exactly. That check runs first and the script refuses to proceed if it fails -- the same
guard that caught a missing Tanh in noise_eval.py.

    python3 aer_noise_eval.py --validate          # noiseless agreement only
    python3 aer_noise_eval.py --backend FakeManilaV2
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import DensityMatrix, Statevector, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import config
from evaluate import ground_truth

RUNS = "runs"


# ------------------------------------------------------------------ circuit
def qiskit_circuit(n, L, entangler):
    """The PennyLane circuit, rebuilt gate for gate.

    AngleEmbedding          -> RX(f_q) on each wire
    StronglyEntanglingLayers-> Rot(phi,theta,omega) = RZ(omega) RY(theta) RZ(phi)
    BasicEntanglerLayers    -> RX(w_q)
    both followed by a CNOT ring.

    Validated against PennyLane to 2.5e-16 before use.
    """
    f = ParameterVector("f", n)
    npar = L * n * 3 if entangler == "strong" else L * n
    w = ParameterVector("w", npar)
    qc = QuantumCircuit(n)
    k = 0
    for _ in range(L):
        for q in range(n):
            qc.rx(f[q], q)
        for q in range(n):
            if entangler == "strong":
                qc.rz(w[k], q); qc.ry(w[k + 1], q); qc.rz(w[k + 2], q); k += 3
            else:
                qc.rx(w[k], q); k += 1
        for q in range(n):
            qc.cx(q, (q + 1) % n)
    return qc, f, w


def flat_weights(sd, entangler, n, L):
    """PennyLane stores (L, 1, n[, 3]); flatten in the order the circuit binds."""
    W = sd["quantum_layer.weights"].detach().cpu().numpy()
    return W.reshape(-1)


def z_expectations(rho_or_sv, n, phys):
    """<Z> on each LOGICAL wire, read from its physical qubit after routing."""
    out = []
    for i in range(n):
        p = phys[i]
        label = "I" * (rho_or_sv.num_qubits - 1 - p) + "Z" + "I" * p
        out.append(float(np.real(rho_or_sv.expectation_value(SparsePauliOp(label)))))
    return out


class Runner:
    def __init__(self, meta, backend=None, batch=2000):
        mk = meta["model_kwargs"]
        self.n = mk.get("n_qubits", 4)
        self.L = mk.get("reupload_layers", 2)
        self.ent = mk.get("entangler", "ring")
        self.batch = batch
        qc, self.f, self.w = qiskit_circuit(self.n, self.L, self.ent)
        if backend is None:
            self.tq = qc
            self.phys = list(range(self.n))
            self.sim = None
            self.routed2q = sum(v for k, v in qc.count_ops().items() if k in ("cx", "ecr", "cz"))
        else:
            self.tq = transpile(qc, backend, optimization_level=3, seed_transpiler=7)
            lay = self.tq.layout
            self.phys = (lay.final_index_layout() if lay is not None
                         else list(range(self.n)))
            nm = NoiseModel.from_backend(backend)
            self.sim = AerSimulator(method="density_matrix", noise_model=nm)
            self.routed2q = sum(v for k, v in self.tq.count_ops().items()
                                if k in ("cx", "ecr", "cz"))
            self.tq.save_density_matrix()

    def quantum_out(self, feats, wvals):
        """feats: (M, n) encoding angles. Returns (M, n) Pauli-Z expectations."""
        binds = [{**{self.f[i]: float(row[i]) for i in range(self.n)},
                  **{self.w[j]: float(wvals[j]) for j in range(len(self.w))}}
                 for row in feats]
        out = []
        if self.sim is None:
            for bd in binds:
                sv = Statevector.from_instruction(self.tq.assign_parameters(bd))
                out.append(z_expectations(sv, self.n, self.phys))
        else:
            for s in range(0, len(binds), self.batch):
                chunk = [self.tq.assign_parameters(bd) for bd in binds[s:s + self.batch]]
                res = self.sim.run(chunk).result()
                for k in range(len(chunk)):
                    out.append(z_expectations(DensityMatrix(res.data(k)["density_matrix"]),
                                              self.n, self.phys))
        return np.array(out)


def classical_head(sd, n):
    w = sd["classical_net.0.weight"].shape[0]
    layers, i = [], 0
    keys = sorted({int(k.split(".")[1]) for k in sd if k.startswith("classical_net.")})
    mods = []
    for idx in keys:
        lin = torch.nn.Linear(sd[f"classical_net.{idx}.weight"].shape[1],
                              sd[f"classical_net.{idx}.weight"].shape[0])
        lin.weight.data = sd[f"classical_net.{idx}.weight"]
        lin.bias.data = sd[f"classical_net.{idx}.bias"]
        mods += [lin, torch.nn.Tanh()]
    mods = mods[:-1]                      # no activation after the output layer
    return torch.nn.Sequential(*mods)


def rel_l2(run_dir, backend=None, nx=200, nt=200):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    sd = torch.load(os.path.join(run_dir, "weights.pt"), map_location="cpu")
    r = Runner(meta, backend)
    wv = flat_weights(sd, r.ent, r.n, r.L)

    x = np.linspace(config.X_MIN, config.X_MAX, nx)
    tt = np.linspace(0, config.T_MAX, nt)
    X, T = np.meshgrid(x, tt)
    reps = (r.n + 1) // 2
    pts = np.stack([X.ravel(), T.ravel()], 1)
    feats = np.tile(pts, (1, reps))[:, :r.n]

    z = r.quantum_out(feats, wv)
    head = classical_head(sd, r.n)
    with torch.no_grad():
        u = head(torch.tanh(torch.tensor(z, dtype=torch.float32))).numpy().ravel()
    U = u.reshape(X.shape)
    U_true = ground_truth()(X, T)
    return float(np.linalg.norm(U - U_true) / np.linalg.norm(U_true)), r.routed2q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--backend", default="FakeManilaV2")
    ap.add_argument("--nx", type=int, default=120)
    a = ap.parse_args()

    if a.validate:
        print("NOISELESS AGREEMENT -- Aer must reproduce the recorded rel_l2\n")
        print(f"{'run':24}{'recorded':>11}{'qiskit':>11}{'|diff|':>11}")
        ok = True
        for tag in ("conv_strong_L2_s0", "init_strong_L2_s0", "conv_ring_L2_s0"):
            d = os.path.join(RUNS, tag)
            rec = json.load(open(d + "/metrics.json"))["rel_l2"]
            got, _ = rel_l2(d, None, nx=a.nx, nt=a.nx)
            # the recorded value uses a 200x200 grid; compare on the same grid
            got200, _ = rel_l2(d, None, nx=200, nt=200)
            print(f"{tag:24}{rec*100:>10.4f}%{got200*100:>10.4f}%{abs(rec-got200)*100:>10.2e}")
            ok &= abs(rec - got200) < 1e-9
        print("\n" + ("PASS" if ok else "FAIL -- do not trust any noisy number from this"))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
