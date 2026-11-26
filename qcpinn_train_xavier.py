"""
qcpinn_train_xavier.py -- the QCPINN replication, re-run with the AUTHORS'
initialisation.

Why this exists
---------------
The qc_* / qcc_* replication runs used PyTorch's framework default for every
layer. The QCPINN paper (arXiv:2503.16678, all versions checked through v6)
states "Classical neural network layers were initialized using Xavier
initialization", and the authors' public code (github.com/afrah/QCPINN,
commit 0dc467be, 15 Dec 2025) does the following, which is reproduced here
exactly rather than inferred from the sentence:

  QCPINN (DVPDESolver + DVQuantumLayer, cascade ansatz)
    pre-processor Linear layers  -> xavier_normal_ weights, zero biases
    post-processor               -> NOT re-initialised (framework default)
    quantum weights              -> xavier_normal_ on the (L, n_qubits*3) view
  Model-2 (ClassicalSolver:  pre = Linear(2,50);  post = Linear, Tanh, Linear)
    first Linear                 -> xavier_normal_, zero bias; rest default
  Model-1 (ClassicalSolver2: pre = Linear, Tanh, Linear; hidden; post)
    first TWO Linear layers      -> xavier_normal_, zero bias; rest default

Everything else -- architecture, optimiser recipe, convergence criterion,
pre-registered failure threshold, seeds -- is IDENTICAL to qcc_* (the 80,000
epoch control), so the only difference between qcx_* and qcc_* is the
initialisation. The authors' code sets "seed": 1 in its config but never calls
manual_seed, so their runs are unseeded; ours keep seeds 0-4.

qcpinn_train.py is imported, not edited: the existing runs were produced by it.
"""

import argparse
import json
import os

import torch
import torch.nn as nn

import config
import qcpinn_train as qt
from qcpinn_model import QCPINN, ClassicalModelQC

_orig_build = qt.build_model


def _xavier(lin):
    nn.init.xavier_normal_(lin.weight)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)


def authors_init(model):
    if isinstance(model, QCPINN):
        for m in model.pre:
            if isinstance(m, nn.Linear):
                _xavier(m)
        L, n = model.hparams["n_layers"], model.hparams["n_qubits"]
        with torch.no_grad():
            rot = model.quantum_layer.rot            # TorchLayer weight, (L, n, 3)
            nn.init.xavier_normal_(rot.view(L, n * 3))
    elif isinstance(model, ClassicalModelQC):
        lins = [m for m in model.net if isinstance(m, nn.Linear)]
        n_pre = 2 if model.hparams["hidden_blocks"] == 3 else 1   # Model-1 : Model-2
        for m in lins[:n_pre]:
            _xavier(m)
    else:
        raise TypeError(type(model))
    return model


def build_model_xavier(kind, **kw):
    return authors_init(_orig_build(kind, **kw))


qt.build_model = build_model_xavier


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qcpinn", "model1", "model2"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--max-epochs", type=int, default=80000)
    p.add_argument("--out-root", default="runs")
    a = p.parse_args()

    model_kwargs = {} if a.model != "qcpinn" else dict(n_layers=1)
    qt.run(kind=a.model, model_kwargs=model_kwargs, budget=config.SWEEP, seed=a.seed,
           max_epochs=a.max_epochs, tag=a.tag, out_root=a.out_root)

    mp = os.path.join(a.out_root, a.tag, "meta.json")
    meta = json.load(open(mp))
    meta["init"] = "authors_xavier (github.com/afrah/QCPINN @ 0dc467be)"
    json.dump(meta, open(mp, "w"), indent=2)
