"""
noise_eval.py -- does the initialisation effect survive hardware noise?

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
---------------------------------------
Section 9.6 records that every quantum result uses exact statevector simulation, so
conclusions are about the idealised circuit. The obvious objection is that any
initialisation advantage might evaporate under hardware noise. This measures that --
partially, and the partiality is the point.

Training under noise is NOT what this does, and cannot be afforded: `default.mixed`
propagates a density matrix and benchmarks ~30x slower than `default.qubit` on this
circuit (~37x with channels attached). The L = 2 quantum arms take ~30 min noiseless, so
a single noisy training run is ~18 hours and the 2 x 2 at eight seeds is ~300 hours.

So this takes the CONVERGED weights from the noiseless runs and re-evaluates them on a
noisy device. That is a deployment question -- "the model was trained on a simulator;
what happens when it runs on hardware?" -- not a trainability question. It cannot tell
you whether the initialisation advantage would still appear if training itself were
noisy. Say so wherever the numbers are used.

Two things it does settle, both of which section 9.6 currently only speculates about:

  1. whether the ordering between arms survives noise, or inverts
  2. the noise level at which the arms become indistinguishable

NOISE MODEL. Depolarising channel on every wire after each entangling block, plus a
bit-flip channel before measurement standing in for readout error at 10x the gate rate,
which is the usual order of magnitude on superconducting hardware. `default.mixed` is
exact, so there is NO shot noise here -- a separate axis that would need finite shots and
would add sampling variance on top of what is measured. This is decoherence and readout
error only.

    python3 noise_eval.py --quick          # one arm, one seed, sanity check
    python3 noise_eval.py                  # the full 2x2 sweep -> noise_sweep.json
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pennylane as qml
import torch

import config
from evaluate import ground_truth

RUNS = "runs"
LEVELS = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2]
READOUT_MULT = 10.0          # readout error is ~10x the per-gate rate

ARMS = [
    ("strong, default",       "conv_strong_L2_s*"),
    ("strong, near-identity", "init_strong_L2_s*"),
    ("ring, default",         "conv_ring_L2_s*"),
    ("ring, near-identity",   "init_ring_L2_s*"),
]


def noisy_layer(n_qubits, reupload_layers, entangler_depth, entangler, p):
    """The same circuit as qapinn_model.build_qlayer, on a density-matrix device
    with channels attached. Kept as a separate implementation rather than a flag
    on the original because the original is read by running trainers and must not
    be touched mid-experiment."""
    dev = qml.device("default.mixed" if p > 0 else "default.qubit", wires=n_qubits)
    if entangler == "strong":
        shapes = {"weights": (reupload_layers, entangler_depth, n_qubits, 3)}
        block = qml.StronglyEntanglingLayers
    else:
        shapes = {"weights": (reupload_layers, entangler_depth, n_qubits)}
        block = qml.BasicEntanglerLayers

    @qml.qnode(dev, interface="torch", diff_method=None)
    def circuit(inputs, weights):
        reps = (n_qubits + 1) // 2
        f = (inputs.repeat(reps)[:n_qubits] if inputs.dim() == 1
             else inputs.repeat(1, reps)[:, :n_qubits])
        for l in range(reupload_layers):
            qml.AngleEmbedding(f, wires=range(n_qubits))
            block(weights[l], wires=range(n_qubits))
            if p > 0:
                for w in range(n_qubits):
                    qml.DepolarizingChannel(p, wires=w)
        if p > 0:
            for w in range(n_qubits):
                qml.BitFlip(min(0.5, p * READOUT_MULT), wires=w)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, shapes)


class NoisyQAPINN(torch.nn.Module):
    """Reconstructed to match QAPINN's module layout exactly so a state_dict from a
    noiseless run loads without remapping."""

    def __init__(self, meta, p):
        super().__init__()
        mk = meta["model_kwargs"]
        n = mk.get("n_qubits", config.DEFAULT_N_QUBITS)
        # attribute names must match QAPINN's exactly or the state_dict will not load
        self.quantum_layer = noisy_layer(n, mk.get("reupload_layers", 2),
                                         mk.get("entangler_depth", 1),
                                         mk.get("entangler", "ring"), p)
        # QAPINN passes the Pauli-Z expectations through a Tanh BEFORE the
        # classical stack (qapinn_model.QAPINN.forward). Omitting it silently
        # changes the function: the p = 0 check came out at 4.22% against a
        # recorded 1.25% until this was added.
        self.quantum_activation = torch.nn.Tanh()
        w = mk.get("classical_width") or config.DEFAULT_CLASSICAL_WIDTH
        d = mk.get("classical_layers") or config.DEFAULT_CLASSICAL_LAYERS
        layers = [torch.nn.Linear(n, w), torch.nn.Tanh()]
        for _ in range(d - 1):
            layers += [torch.nn.Linear(w, w), torch.nn.Tanh()]
        layers += [torch.nn.Linear(w, 1)]
        self.classical_net = torch.nn.Sequential(*layers)

    def forward(self, x, t):
        z = self.quantum_layer(torch.cat([x, t], dim=1))
        z = self.quantum_activation(z)
        return self.classical_net(z)


def rel_l2_under_noise(run_dir, p, nx=200, nt=200, batch=512):
    """Identical grid and metric to evaluate.predict_grid / evaluate.metrics, so the
    p = 0 column must reproduce the run's recorded rel_l2 exactly. That equality is
    the sanity check on this whole module."""
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    model = NoisyQAPINN(meta, p)
    model.load_state_dict(torch.load(os.path.join(run_dir, "weights.pt"),
                                     map_location="cpu"))
    model.eval()

    x = np.linspace(config.X_MIN, config.X_MAX, nx)
    tt = np.linspace(0, config.T_MAX, nt)
    X, T = np.meshgrid(x, tt)
    xf = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1)
    tf = torch.tensor(T.flatten(), dtype=torch.float32).reshape(-1, 1)

    out = []
    with torch.no_grad():
        for i in range(0, len(xf), batch):
            out.append(model(xf[i:i + batch], tf[i:i + batch]).numpy().ravel())
    U = np.concatenate(out).reshape(X.shape)
    U_true = ground_truth()(X, T)
    return float(np.linalg.norm(U - U_true) / np.linalg.norm(U_true))


def seeds_of(pattern):
    return sorted(glob.glob(os.path.join(RUNS, pattern)),
                  key=lambda d: int(re.search(r"_s(\d+)$", d).group(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default="noise_sweep.json")
    a = ap.parse_args()

    arms = ARMS[:1] if a.quick else ARMS
    levels = [0.0, 1e-2] if a.quick else LEVELS
    nseed = 1 if a.quick else a.seeds

    result = {"levels": levels, "readout_multiplier": READOUT_MULT, "arms": {}}
    for label, pat in arms:
        dirs = seeds_of(pat)[:nseed]
        result["arms"][label] = {}
        for p in levels:
            vals, t0 = [], time.time()
            for d in dirs:
                vals.append(rel_l2_under_noise(d, p))
            result["arms"][label][f"{p:g}"] = vals
            print(f"  {label:24} p={p:<8g} median={np.median(vals) * 100:6.2f}%  "
                  f"n={len(vals)}  {time.time() - t0:5.1f}s")
        print()

    json.dump(result, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
