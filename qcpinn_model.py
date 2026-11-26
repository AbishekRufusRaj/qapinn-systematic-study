"""
The QCPINN architecture (arXiv:2503.16678), reimplemented here so its
parameter-efficiency claim can be tested under this project's protocol.

WHY THIS FILE EXISTS
--------------------
The QAPINN in qapinn_model.py feeds (x, t) straight into the circuit and
puts a full 8-layer, width-20 classical stack BEHIND it. That is roughly
2,600 classical parameters carrying a 24-parameter quantum layer. Under
that arrangement the quantum layer cannot possibly account for the model's
capacity, and the parameter-efficiency claim the literature actually makes
is structurally invisible: you cannot show a model is efficient if you
have bolted a large classical network to it.

QCPINN is built the other way round, and the difference is the whole point:

    (x,t) -> classical PRE-processor -> quantum circuit -> classical POST -> u
             2 layers, 50 wide          5 qubits          symmetric, thin

Two structural consequences follow, and both matter for this project.

1. THE PRE-PROCESSOR IS LEARNABLE, SO THE ENCODING IS LEARNABLE.
   Our Fourier analysis measured the accessible frequency spectrum of a
   circuit fed RAW inputs, where the spectrum is fixed by the encoding
   gates and the re-upload count. Here a trainable map sits in front of
   the encoding, so the network can learn to rescale its inputs -- which
   is exactly what our `angle_scaled` encoding did by hand, except
   learned. That reframes the frequency-ceiling result rather than
   contradicting it: the ceiling is still real, but the model can move
   its inputs around underneath it.

2. THERE IS NO RE-UPLOADING. The input is encoded once, then L variational
   layers follow. On the Schuld/Sweke/Meyer picture that is the minimum
   possible frequency budget. Whether the learnable pre-processor
   substitutes for re-uploading is an open question this file makes
   testable -- `reupload=True` turns it on, off by default to match the
   paper.

RECONSTRUCTION NOTES (what the paper states vs what was inferred)
-----------------------------------------------------------------
Stated: 5 qubits; angle encoding; cascade = ring topology of CRX gates;
PauliZ expectation on every wire; pre-processor of 2 layers x 50 neurons
with Tanh; post-processor symmetric; total 771 parameters for
angle-cascade against Classical Model-1 at 7,851.

Not stated: the number of variational layers L, and whether the CRX ring
angles are trainable. Both were pinned by matching the reported parameter
counts, which the defaults below reproduce EXACTLY:

    pre   Linear(2,50) + Linear(50,5)   = 150 + 255 = 405
    post  Linear(5,50) + Linear(50,1)   = 300 +  51 = 351
    quantum  5 qubits x Rot(3 angles) x L=1        =  15
                                          total    = 771   (paper: 771)

The two classical baselines fall out of the same reading and also land
exactly on the paper's numbers, which is the strongest evidence available
that this reconstruction is right:

    Model-1  Linear(2,50) + 3 x Linear(50,50) + Linear(50,1) = 7,851
    Model-2  Linear(2,50) +     Linear(50,50) + Linear(50,1) = 2,751

If a future reading of the paper changes these, change them here -- but
check the parameter counts still match before trusting any comparison.
"""

import pennylane as qml
import torch
import torch.nn as nn

import config

QCPINN_QUBITS = 5
QCPINN_HIDDEN = 50


def make_cascade_layer(n_qubits: int = QCPINN_QUBITS,
                       n_layers: int = 1,
                       reupload: bool = False,
                       trainable_entangler: bool = False,
                       crx_angle: float = torch.pi) -> qml.qnn.TorchLayer:
    """The angle-cascade circuit.

    n_qubits            -- 5 in the paper
    n_layers            -- L; 1 reproduces the reported 771-parameter total
    reupload            -- re-encode the input before every variational
                           layer. False matches the paper. True is the
                           experiment: does re-uploading still help once a
                           learnable pre-processor is in front of it?
    trainable_entangler -- learn the CRX angles too (+n_qubits params per
                           layer). False matches the paper's count.
    crx_angle           -- fixed CRX rotation when the entangler is not
                           trainable. pi makes CRX a controlled-X.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    def _body(inputs, rot, crx):
        # The pre-processor already emits exactly n_qubits features, so
        # unlike the QAPINN there is no tiling of (x, t) across wires --
        # every wire gets its own learned feature.
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        for layer in range(n_layers):
            if reupload and layer > 0:
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

            # General single-qubit rotation. Rot is Z-Y-Z, so it does NOT
            # merge into the RY encoding the way a bare RY would -- the
            # same-axis commutation trap that cost this project a working
            # frequency budget earlier on.
            for w in range(n_qubits):
                qml.Rot(rot[layer, w, 0], rot[layer, w, 1], rot[layer, w, 2], wires=w)

            # cascade: CRX around a ring
            for w in range(n_qubits):
                angle = crx[layer, w] if crx is not None else crx_angle
                qml.CRX(angle, wires=[w, (w + 1) % n_qubits])

        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    # TorchLayer requires a shape for EVERY non-input argument, so the two
    # variants need genuinely different signatures -- a defaulted `crx=None`
    # is rejected rather than treated as absent.
    if trainable_entangler:
        weight_shapes = {"rot": (n_layers, n_qubits, 3), "crx": (n_layers, n_qubits)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, rot, crx):
            return _body(inputs, rot, crx)
    else:
        weight_shapes = {"rot": (n_layers, n_qubits, 3)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, rot):
            return _body(inputs, rot, None)

    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QCPINN(nn.Module):
    """Classical pre-processor -> variational circuit -> classical post-processor."""

    def __init__(self,
                 n_qubits: int = QCPINN_QUBITS,
                 n_layers: int = 1,
                 hidden: int = QCPINN_HIDDEN,
                 reupload: bool = False,
                 trainable_entangler: bool = False):
        super().__init__()
        self.hparams = dict(n_qubits=n_qubits, n_layers=n_layers, hidden=hidden,
                            reupload=reupload, trainable_entangler=trainable_entangler)

        self.pre = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, n_qubits), nn.Tanh(),
        )
        self.quantum_layer = make_cascade_layer(
            n_qubits, n_layers, reupload, trainable_entangler)
        self.post = nn.Sequential(
            nn.Linear(n_qubits, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = self.pre(torch.cat([x, t], dim=1))
        q = self.quantum_layer(z)
        return self.post(q)

    def param_counts(self):
        q = sum(p.numel() for p in self.quantum_layer.parameters())
        c = (sum(p.numel() for p in self.pre.parameters())
             + sum(p.numel() for p in self.post.parameters()))
        return {"quantum": q, "classical": c, "total": q + c}


class ClassicalModelQC(nn.Module):
    """The paper's classical baselines, which the efficiency claim is
    measured against. `hidden_blocks` is the only difference between them:

        3 -> Model-1, 7,851 params  (pre + 2 hidden + post)
        1 -> Model-2, 2,751 params  (pre + post only)

    Model-2 is the honest comparison. It is the one that, in the paper's
    own tables, beats the quantum model on the cavity problem and on
    Helmholtz -- a fact the abstract does not dwell on.
    """

    def __init__(self, hidden: int = QCPINN_HIDDEN, hidden_blocks: int = 3):
        super().__init__()
        self.hparams = dict(hidden=hidden, hidden_blocks=hidden_blocks)
        # hidden_blocks counts the hidden x hidden layers, NOT the total
        # depth: 3 -> 7,851 (Model-1) and 1 -> 2,751 (Model-2), which are
        # the paper's exact figures.
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(hidden_blocks):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))

    def param_counts(self):
        c = sum(p.numel() for p in self.net.parameters())
        return {"quantum": 0, "classical": c, "total": c}


def model_1(**kw):
    return ClassicalModelQC(hidden_blocks=3, **kw)


def model_2(**kw):
    return ClassicalModelQC(hidden_blocks=1, **kw)


if __name__ == "__main__":
    from qapinn_model import pde_residual

    torch.manual_seed(config.SEED)
    print("Reconstruction check -- these must match the paper or the")
    print("comparison is meaningless.\n")

    expected = {"QCPINN angle-cascade": 771,
                "Classical Model-1": 7851,
                "Classical Model-2": 2751}

    builds = [
        ("QCPINN angle-cascade", QCPINN()),
        ("Classical Model-1", model_1()),
        ("Classical Model-2", model_2()),
    ]

    for name, m in builds:
        c = m.param_counts()
        want = expected[name]
        ok = "MATCH" if c["total"] == want else f"MISMATCH (paper says {want})"
        x = torch.rand(8, 1) * 2 - 1
        t = torch.rand(8, 1)
        r = pde_residual(m, x, t)
        grad_ok = tuple(r.shape) == (8, 1)
        print(f"  {name:24s} q/c {c['quantum']:>3}/{c['classical']:<5} "
              f"total {c['total']:>5}  {ok:32s} residual_ok={grad_ok}")

    print("\nVariants available for the follow-up experiments:")
    for name, m in [("+ re-uploading (L=2)", QCPINN(n_layers=2, reupload=True)),
                    ("+ trainable CRX (L=2)", QCPINN(n_layers=2, trainable_entangler=True))]:
        c = m.param_counts()
        print(f"  {name:24s} q/c {c['quantum']:>3}/{c['classical']:<5} total {c['total']:>5}")
