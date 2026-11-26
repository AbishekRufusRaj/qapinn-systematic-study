"""
The QAPINN, fully parameterized so every architectural choice this
project investigates is a knob rather than a hardcoded constant.

THE FOUR INVESTIGATION AXES, AS CODE
---------------------------------------
The project proposal names four axes. Here is exactly where each one
lives in this file:

  1. DATA-ENCODING METHOD  -> `encoding` argument ("angle" or "angle_scaled")
     and `reupload_layers` (how many times each input is re-encoded).
  2. ENTANGLEMENT STRUCTURE -> `entangler` argument ("ring" via
     BasicEntanglerLayers, or "strong" via StronglyEntanglingLayers).
  3. CIRCUIT DEPTH -> `reupload_layers` x `entangler_depth`.
  4. FREQUENCY EXPRESSIVITY -> not a knob, it's the CONSEQUENCE of the
     above, and what fourier_analysis.py measures.

WHY RE-UPLOADING IS THE FREQUENCY LEVER
------------------------------------------
Schuld/Sweke/Meyer (2021): a VQC's output, as a function of an encoded
input, is a truncated Fourier series, and the accessible frequency
spectrum is set by the eigenvalues of the encoding generators. For angle
encoding with Pauli rotations, encoding the same input L times gives
access to integer frequencies up to L. So `reupload_layers` is, quite
literally, "how many harmonics this circuit is allowed to use" -- which
is why it is the single most important knob in this whole study, and why
fourier_analysis.py measures it directly instead of taking it on faith.
"""

import pennylane as qml
import torch
import torch.nn as nn

import config


def make_quantum_layer(n_qubits: int = config.DEFAULT_N_QUBITS,
                       reupload_layers: int = config.DEFAULT_REUPLOAD_LAYERS,
                       entangler_depth: int = config.DEFAULT_ENTANGLER_DEPTH,
                       entangler: str = "ring",
                       encoding: str = "angle",
                       init_scale: float = None) -> qml.qnn.TorchLayer:
    """Build the variational quantum circuit as a drop-in torch layer.

    n_qubits         -- width of the circuit (inputs are tiled across wires)
    reupload_layers  -- how many times (x, t) are re-encoded == frequency budget
    entangler_depth  -- entangling sub-layers per re-upload block
    entangler        -- "ring" (BasicEntanglerLayers, RX + CNOT ring) or
                        "strong" (StronglyEntanglingLayers, full Rot + CNOT)
    encoding         -- "angle" (x -> angle x on every assigned wire) or
                        "angle_scaled" (x -> angle k*x on the k-th wire pair,
                        a cheap way to reach higher frequencies without
                        more re-upload layers -- an encoding-scheme variant
                        worth comparing in the Weeks 7-8 sweep)
    init_scale       -- None uses PennyLane's default weight initialisation,
                        uniform on [0, 2*pi). A small float instead draws
                        weights from N(0, init_scale), starting the circuit
                        NEAR IDENTITY. This is a standard barren-plateau
                        mitigation (Grant et al. 2019): a circuit that begins
                        close to the identity has a better-conditioned
                        landscape than one initialised uniformly at random,
                        which is exactly the failure mode the seed study
                        measured -- some initialisations land in basins Adam
                        never escapes.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    if entangler == "strong":
        weight_shapes = {"weights": (reupload_layers, entangler_depth, n_qubits, 3)}
        entangling_block = qml.StronglyEntanglingLayers
    else:
        weight_shapes = {"weights": (reupload_layers, entangler_depth, n_qubits)}
        entangling_block = qml.BasicEntanglerLayers

    # scaling factors used by the "angle_scaled" encoding: wire pair k gets
    # its input multiplied by (k+1), so different wires see different
    # frequencies of the same variable
    n_pairs = max(1, n_qubits // 2)
    scales = torch.tensor([float(k + 1) for k in range(n_pairs) for _ in range(2)][:n_qubits])

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # inputs is (..., 2) = (x, t); tile across the available wires
        reps = (n_qubits + 1) // 2
        if inputs.dim() == 1:
            features = inputs.repeat(reps)[:n_qubits]
        else:
            features = inputs.repeat(1, reps)[:, :n_qubits]

        if encoding == "angle_scaled":
            features = features * scales.to(features.device)

        for layer in range(reupload_layers):
            qml.AngleEmbedding(features, wires=range(n_qubits))
            entangling_block(weights[layer], wires=range(n_qubits))

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    if init_scale is None:
        return qml.qnn.TorchLayer(circuit, weight_shapes)

    def _small_angle(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=init_scale)

    return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=_small_angle)


class QAPINN(nn.Module):
    """(x, t) -> u(x, t), with a configurable VQC as the first hidden layer."""

    def __init__(self,
                 n_qubits: int = config.DEFAULT_N_QUBITS,
                 reupload_layers: int = config.DEFAULT_REUPLOAD_LAYERS,
                 entangler_depth: int = config.DEFAULT_ENTANGLER_DEPTH,
                 entangler: str = "ring",
                 encoding: str = "angle",
                 init_scale: float = None,
                 classical_width: int = config.DEFAULT_CLASSICAL_WIDTH,
                 classical_layers: int = config.DEFAULT_CLASSICAL_LAYERS):
        super().__init__()
        self.hparams = dict(n_qubits=n_qubits, reupload_layers=reupload_layers,
                            entangler_depth=entangler_depth, entangler=entangler,
                            encoding=encoding, init_scale=init_scale,
                            classical_width=classical_width,
                            classical_layers=classical_layers)

        self.quantum_layer = make_quantum_layer(
            n_qubits, reupload_layers, entangler_depth, entangler, encoding, init_scale)
        self.quantum_activation = nn.Tanh()

        layers = [nn.Linear(n_qubits, classical_width), nn.Tanh()]
        for _ in range(classical_layers - 1):
            layers += [nn.Linear(classical_width, classical_width), nn.Tanh()]
        layers += [nn.Linear(classical_width, 1)]
        self.classical_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t], dim=1)
        q_out = self.quantum_layer(xt)
        q_out = self.quantum_activation(q_out)
        return self.classical_net(q_out)

    def param_counts(self):
        q = sum(p.numel() for p in self.quantum_layer.parameters())
        c = sum(p.numel() for p in self.classical_net.parameters())
        return {"quantum": q, "classical": c, "total": q + c}


class ClassicalPINN(nn.Module):
    """The classical baseline, same interface -- included here so the sweep
    can train it under the IDENTICAL data budget and training recipe as
    every QAPINN configuration. Comparing against a number from a
    differently-configured run is the most common way this kind of study
    goes wrong; this removes that failure mode."""

    def __init__(self, hidden_width: int = 20, hidden_layers: int = 8,
                 init_gain: float = None):
        """init_gain rescales PyTorch's default weight initialisation.

        WHY THIS EXISTS -- IT CLOSES A CONFOUND, NOT A PERFORMANCE GAP
        --------------------------------------------------------------
        The QAPINN was given near-identity initialisation (`init_scale`),
        which cut its seed spread from 14.11% to 0.55% and eliminated its
        one non-converged run. The classical baseline was never given any
        equivalent treatment, so "the quantum model is more reliable" is
        currently confounded with "the quantum model got an intervention
        the baseline did not."

        This is the matched control. `init_gain` multiplies the default
        initialisation, so a sweep over gains asks the only question that
        matters: does the classical baseline's variance respond to init
        tuning at all?

          it does     -> the confound is real and must be reported; the
                         reliability claim needs the matched comparison
          it doesn't  -> the QAPINN's variance reduction is specific to
                         the quantum layer, which is a stronger result
                         than the one currently claimed

        Note this is NOT the same intervention as the QAPINN's. Near-identity
        means something specific for a variational circuit (start close to
        the identity operation) with no exact classical analogue; scaling
        the default init is the closest honest equivalent. Say so in the
        paper rather than implying the two treatments are identical.

        None leaves initialisation untouched, so every existing run
        reproduces bit-for-bit."""
        super().__init__()
        self.hparams = dict(hidden_width=hidden_width, hidden_layers=hidden_layers,
                            init_gain=init_gain)
        layers = [nn.Linear(2, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers += [nn.Linear(hidden_width, 1)]
        self.net = nn.Sequential(*layers)

        if init_gain is not None:
            with torch.no_grad():
                for m in self.net:
                    if isinstance(m, nn.Linear):
                        m.weight.mul_(init_gain)
                        m.bias.mul_(init_gain)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))

    def param_counts(self):
        c = sum(p.numel() for p in self.net.parameters())
        return {"quantum": 0, "classical": c, "total": c}


def pde_residual(model, x: torch.Tensor, t: torch.Tensor, nu: float = config.NU) -> torch.Tensor:
    """r(x,t) = u_t + u*u_x - nu*u_xx, by automatic differentiation.
    Works identically for QAPINN and ClassicalPINN -- the whole point is
    that the physics loss doesn't care what's inside the network."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)

    u = model(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    return u_t + u * u_x - nu * u_xx


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    for kwargs in [dict(n_qubits=2, reupload_layers=1),
                   dict(n_qubits=4, reupload_layers=6),
                   dict(n_qubits=4, reupload_layers=4, entangler="strong"),
                   dict(n_qubits=4, reupload_layers=4, encoding="angle_scaled")]:
        m = QAPINN(**kwargs)
        counts = m.param_counts()
        x = torch.rand(8, 1) * 2 - 1
        t = torch.rand(8, 1)
        r = pde_residual(m, x, t)
        print(f"{str(kwargs):70s} params(q/c)={counts['quantum']}/{counts['classical']}  "
              f"residual_ok={tuple(r.shape) == (8, 1)}")

    c = ClassicalPINN()
    r = pde_residual(c, torch.rand(8, 1) * 2 - 1, torch.rand(8, 1))
    print(f"{'ClassicalPINN baseline':70s} params(q/c)=0/{c.param_counts()['classical']}  "
          f"residual_ok={tuple(r.shape) == (8, 1)}")
