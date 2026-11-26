"""
ablate_qapinn.py -- is the circuit doing anything at all?

The gradient-share and feature-map measurements say the quantum layer receives
0.1-0.5% of the learning signal and supplies no more high-frequency content than an
ordinary first layer. Both are indirect. The direct test, used by Bowles, Ahmed and
Schuld (arXiv:2403.07059) on quantum classifiers, is to REMOVE the circuit and put a
classical map of the SAME PARAMETER COUNT in its place, then train identically.

  ring   ->  8 quantum params  ->  Linear(2, 4, bias=False)                    =  8
  strong -> 24 quantum params  ->  Linear(2, 4, bias=False) + Tanh
                                   + Linear(4, 4, bias=False)                  = 24

Everything else is untouched: the Tanh on the layer output, the 7x20 classical stack,
the optimiser, the convergence criterion, the seeds. If the ablated model matches the
quantum one, the circuit contributes nothing measurable. If it is worse, the circuit
contributes, and by how much.

qapinn_model.py and pde_train.py are imported, not edited.
"""
import argparse, torch, torch.nn as nn, config, pde_train as pt
from qapinn_model import QAPINN


class AblatedQAPINN(QAPINN):
    """QAPINN with its variational circuit replaced by a parameter-matched
    classical map. Construction still builds the circuit and then discards it,
    so the classical stack sees the identical initialisation stream."""

    def __init__(self, **kw):
        kw.pop("ablated", None)          # present when rebuilt from a saved meta.json
        super().__init__(**kw)
        n = self.hparams["n_qubits"]
        # The circuit maps (x, t) -> 4 outputs. The stand-in must keep that shape;
        # matching the 8- or 24-parameter count exactly would mean removing the bias
        # or the mixing layer, which tests capacity rather than quantumness. Both
        # stand-ins therefore have slightly MORE parameters than the circuit they
        # replace (12 and 36 against 8 and 24), favouring the classical side: the
        # question being asked is whether removing the circuit hurts.
        if self.hparams["entangler"] == "strong":
            sub = nn.Sequential(nn.Linear(2, n), nn.Tanh(), nn.Linear(n, n))   # 36
        else:
            sub = nn.Sequential(nn.Linear(2, n))                               # 12
        self.quantum_layer = sub
        self.hparams = dict(self.hparams, ablated=True)

    def param_counts(self):
        c = sum(p.numel() for p in self.parameters())
        return {"quantum": 0, "classical": c, "total": c}


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--entangler", choices=["ring", "strong"], required=True)
    a.add_argument("--init-scale", type=float, default=None)
    a.add_argument("--seed", type=int, required=True)
    a.add_argument("--tag", required=True)
    a.add_argument("--max-epochs", type=int, default=25000)
    args = a.parse_args()

    orig = pt.build_model
    pt.build_model = lambda kind, **kw: (AblatedQAPINN(**kw) if kind == "qapinn"
                                         else orig(kind, **kw))
    mk = dict(n_qubits=4, reupload_layers=2, entangler=args.entangler,
              encoding="angle", init_scale=args.init_scale)
    pt.run(pde_name="burgers", kind="qapinn", model_kwargs=mk, budget=config.SWEEP,
           seed=args.seed, max_epochs=args.max_epochs, tag=args.tag)
