"""
PER-NEURON FEATURE MAPS -- the BQP challenge's headline deliverable.

WHAT THE BRIEF ASKS FOR
-----------------------
Deliverable 2: "Use Explainable AI (XAI) techniques/metrics to demonstrate
how the learning happens in each of the networks."

Fig 3 of the brief shows exactly the figure it wants: a grid of per-neuron
activation maps over the (x, t) domain, for a network trained WITHOUT a
quantum layer and the same network WITH its first hidden layer replaced by
a VQC. Nothing in this project has produced that figure until now.

WHY THIS IS ALMOST FREE
-----------------------
A feature map is a forward pass. Every model this study has trained is
already on disk as weights.pt, so this reads them and renders. Minutes,
not hours -- against four studies that each cost between one and nine hours
of training.

WHY IT SHIPS METRICS, NOT JUST PICTURES
---------------------------------------
The brief also asks for "explainability/interpretability metrics", and a
wall of coloured squares is not a measurement. Three numbers per layer,
each answering a question a reviewer will actually ask:

  participation ratio  PR = (sum s_i)^2 / sum(s_i^2), over per-neuron
      activation variances s_i. This is the EFFECTIVE number of neurons
      doing work. A 20-neuron layer with PR = 2.1 is a 20-neuron layer
      pretending to be a 2-neuron layer. Reported as a fraction of layer
      width so classical and quantum layers of different sizes compare.

  dominant spatial frequency  per neuron, the argmax of |FFT| of its
      activation along x at fixed t, excluding DC. This is where the
      spectral story becomes per-neuron rather than whole-model: it says
      WHICH units carry the high-frequency content the shock needs.

  saturation  fraction of grid points where |activation| > 0.95. Tanh
      units pinned at +/-1 encode no local gradient, so a saturated layer
      is passing a step function, not a feature.

Everything is computed from the same activations that make the figure, so
the numbers and the picture can never disagree.

Usage:
    python3 xai_feature_maps.py runs/conv_classical_s0 runs/init_strong_L2_s0
    python3 xai_feature_maps.py --compare runs/conv_classical_s0 runs/init_strong_L2_s0
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from evaluate import load_run

# a coarse grid is plenty for a feature map and keeps quantum forward
# passes cheap -- 80x80 = 6400 points rather than the 40000 evaluate.py uses
GRID = 80
SATURATION = 0.95


def _grid(nx=GRID, nt=GRID):
    x = np.linspace(config.X_MIN, config.X_MAX, nx)
    t = np.linspace(0.0, config.T_MAX, nt)
    X, T = np.meshgrid(x, t)
    xf = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1)
    tf = torch.tensor(T.flatten(), dtype=torch.float32).reshape(-1, 1)
    return X, T, xf, tf


def capture(model, nx=GRID, nt=GRID):
    """Run the grid through the model, capturing every layer's output.

    Hooks are attached to the quantum layer (when present) and to each
    activation in the classical stack, so what is captured is the
    POST-activation signal each downstream layer actually sees -- which is
    what the brief's figure shows.
    """
    X, T, xf, tf = _grid(nx, nt)
    grabbed, handles = {}, []

    def hook(name):
        def fn(_module, _inp, out):
            grabbed[name] = out.detach().cpu().numpy()
        return fn

    # the quantum layer, whatever the architecture calls it
    if hasattr(model, "quantum_layer"):
        handles.append(model.quantum_layer.register_forward_hook(hook("Q-layer")))

    # every Tanh in every Sequential, labelled by the block it sits in
    for block_name in ("pre", "classical_net", "post", "net"):
        block = getattr(model, block_name, None)
        if not isinstance(block, torch.nn.Sequential):
            continue
        idx = 0
        for layer in block:
            if isinstance(layer, torch.nn.Tanh):
                idx += 1
                label = f"{block_name} L{idx}" if block_name != "net" else f"Layer {idx}"
                handles.append(layer.register_forward_hook(hook(label)))

    with torch.no_grad():
        model(xf, tf)
    for h in handles:
        h.remove()

    # (n_points, n_neurons) -> (n_neurons, nt, nx)
    return X, T, {k: v.T.reshape(-1, nt, nx) for k, v in grabbed.items()}


N_MODES = 12               # sin(k pi x) modes resolved, k = 1..N_MODES


def sine_spectrum(maps, n_modes=N_MODES):
    """Mean |amplitude| of modes sin(k*pi*x), k = 1..n_modes, over time slices.

    WHY NOT AN FFT
    --------------
    A plain rfft along x is wrong here and quietly so. Activations are not
    periodic on [-1, 1]: most are monotone ramps. The FFT's implicit
    periodic extension therefore has a jump at the wrap-around, and that
    jump leaks energy into every frequency bin. Measured directly, a PURE
    LINEAR RAMP -- a function with no oscillation whatsoever -- reports a
    spectral centroid of 10.4. Any "high-frequency content" read off that
    number is an artifact of the transform, not a property of the network,
    and it would have gone straight into the paper's expressivity claim.

    The fix is to use the basis this problem actually lives in. Both PDEs
    have homogeneous Dirichlet conditions, so their solutions are sine
    series in sin(k*pi*x) -- the same basis the heat equation's modes and
    the Schuld frequency-ceiling argument are stated in. Each time slice is
    first detrended by subtracting the straight line through its endpoints,
    which makes it vanish at both boundaries so the sine basis is complete
    for it, and the remainder is projected onto that basis.

    Result: a ramp returns ~0 across all modes, and sin(3*pi*x) returns a
    clean peak at k = 3. Mode indices are then directly comparable to the
    heat equation's k = 1, 3, 5 and to the predicted frequency ceiling.
    """
    nt, nx = maps.shape
    x = np.linspace(-1.0, 1.0, nx)

    # detrend each time slice: subtract the line through its endpoints
    a = maps[:, :1]
    b = maps[:, -1:]
    ramp = a + (b - a) * ((x - x[0]) / (x[-1] - x[0]))[None, :]
    resid = maps - ramp

    basis = np.sin(np.outer(np.arange(1, n_modes + 1), np.pi * x))   # (n_modes, nx)
    # trapezoid projection, normalised so a unit-amplitude mode returns ~1
    coef = (resid @ basis.T) * (2.0 / nx)
    return np.abs(coef).mean(axis=0)


def layer_metrics(maps):
    """PR, saturation and per-neuron dominant frequency for one layer."""
    n = maps.shape[0]
    var = maps.reshape(n, -1).var(axis=1)

    if var.sum() <= 0:
        pr = 0.0
    else:
        pr = float(var.sum() ** 2 / np.sum(var ** 2))

    sat = float(np.mean(np.abs(maps) > SATURATION))

    # ---- spectral content, per neuron -------------------------------
    #
    # An earlier version reported only argmax of each neuron's spectrum --
    # its DOMINANT mode. That is not the same as the highest mode PRESENT,
    # and reading it as such would have supported the false claim "this
    # layer carries no high frequencies" for a layer whose mode-4 energy
    # simply sits below its mode-1 energy. Since the whole expressivity
    # argument in this project is about which frequencies a layer can
    # represent, three measures are reported instead:
    #
    #   dominant      argmax, kept for continuity (which mode leads)
    #   centroid      energy-weighted mean frequency: sum(f*|F|)/sum(|F|).
    #                 Moves whenever high-frequency energy appears at all,
    #                 even while a low mode still dominates.
    #   top_present   highest frequency holding at least HF_PRESENT of the
    #                 neuron's peak non-DC energy -- "what is actually
    #                 there", which is the quantity the Fourier-ceiling
    #                 argument is about.
    #
    # hf_energy_fraction aggregates the layer: the share of non-DC spectral
    # energy above HF_CUTOFF. One number for "how much of this layer's
    # output is high-frequency content".
    HF_PRESENT = 0.10      # counts as present at 10% of the neuron's peak
    HF_CUTOFF = 2          # "high frequency" means mode > 2

    dominant, centroid, top_present = [], [], []
    hf_energy = lf_energy = 0.0

    for i in range(n):
        spec = sine_spectrum(maps[i])                  # index k-1 holds mode k
        k = np.arange(1, len(spec) + 1)
        total = spec.sum()

        dominant.append(int(k[np.argmax(spec)]) if total > 0 else 0)
        centroid.append(float((k * spec).sum() / total) if total > 0 else 0.0)

        peak = spec.max()
        present = np.nonzero(spec >= HF_PRESENT * peak)[0] if peak > 0 else []
        top_present.append(int(k[present[-1]]) if len(present) else 0)

        hf_energy += float(spec[k > HF_CUTOFF].sum())
        lf_energy += float(total)

    return {"n_neurons": n,
            "participation_ratio": round(pr, 3),
            "pr_fraction": round(pr / n, 3),
            "saturation": round(sat, 4),
            "dominant_freqs": dominant,
            "max_dominant_freq": int(max(dominant)) if dominant else 0,
            "n_unique_freqs": int(len(set(dominant))),
            "spectral_centroid": round(float(np.mean(centroid)), 3),
            "max_freq_present": int(max(top_present)) if top_present else 0,
            "mean_freq_present": round(float(np.mean(top_present)), 2),
            "hf_energy_fraction": round(hf_energy / lf_energy, 4) if lf_energy > 0 else 0.0}


def figure(X, T, acts, title, path, max_neurons=24):
    """Grid of per-neuron heatmaps, one panel-block per layer."""
    layers = list(acts.items())
    if not layers:
        return
    ncols = 8
    rows_per = [int(np.ceil(min(len(m), max_neurons) / ncols)) for _, m in layers]
    total_rows = sum(rows_per) + len(layers)          # +1 title row per layer

    # Title rows get a fraction of the height of a map row. Without this the
    # gridspec gives every caption as much vertical space as a whole row of
    # heatmaps, which leaves a large gap under the suptitle.
    heights = []
    for nrow in rows_per:
        heights.append(0.22)                          # the layer's caption
        heights.extend([1.0] * nrow)                  # its heatmap rows

    fig = plt.figure(figsize=(ncols * 1.5, sum(heights) * 1.3))
    gs = fig.add_gridspec(total_rows, ncols, height_ratios=heights,
                          hspace=0.35, wspace=0.12)

    r = 0
    for (name, maps), nrow in zip(layers, rows_per):
        ax = fig.add_subplot(gs[r, :])
        m = layer_metrics(maps)
        ax.axis("off")
        ax.text(0.0, 0.1,
                f"{name}   —   {m['n_neurons']} units, "
                f"PR {m['participation_ratio']:.2f} ({m['pr_fraction']:.0%} utilised), "
                f"centroid {m['spectral_centroid']:.2f}, "
                f"max mode present {m['max_freq_present']}, "
                f"HF energy {m['hf_energy_fraction']:.1%}, "
                f"{m['saturation']:.0%} saturated",
                fontsize=10, fontweight="bold")
        r += 1

        lim = float(np.abs(maps).max()) or 1.0
        for k in range(min(len(maps), max_neurons)):
            ax = fig.add_subplot(gs[r + k // ncols, k % ncols])
            ax.pcolormesh(X, T, maps[k], cmap="RdBu_r", vmin=-lim, vmax=lim,
                          shading="auto", rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{k}", fontsize=6, pad=1.5)
        r += nrow

    fig.suptitle(title, fontsize=13, y=0.997)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def process(run_dir, outdir=None, checkpoint=None):
    """checkpoint: an epoch number, to read weights_ep<N>.pt instead of the
    final weights. This is what makes interpretability measurable AS A
    FUNCTION OF TRAINING BUDGET -- if a layer's utilisation and spectral
    content at 800 epochs differ from what it converges to, then XAI
    conclusions drawn at short budgets are artifacts in exactly the way
    accuracy comparisons at short budgets are."""
    weights = "weights.pt" if checkpoint is None else f"weights_ep{checkpoint}.pt"
    model, _history, meta = load_run(run_dir, weights)
    if checkpoint is not None:
        meta = dict(meta, tag=f"{meta['tag']}_ep{checkpoint}")
    model.eval()
    X, T, acts = capture(model)

    tag = meta["tag"]
    outdir = outdir or run_dir
    os.makedirs(outdir, exist_ok=True)

    counts = meta["param_counts"]
    kind = "quantum-assisted" if counts["quantum"] else "classical"
    figure(X, T, acts,
           f"{tag}  ({kind}, {counts['quantum']}q / {counts['classical']}c params)",
           os.path.join(outdir, f"feature_maps_{tag}.png"))

    metrics = {name: layer_metrics(m) for name, m in acts.items()}
    out = {"tag": tag, "param_counts": counts, "layers": metrics}
    with open(os.path.join(outdir, f"xai_metrics_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{tag}   ({kind})")
    print(f"  {'layer':16s} {'units':>6} {'PR':>7} {'used':>6} {'domin':>6} "
          f"{'centroid':>9} {'maxpres':>8} {'HFenergy':>9} {'sat':>6}")
    for name, m in metrics.items():
        print(f"  {name:16s} {m['n_neurons']:>6} {m['participation_ratio']:>7.2f} "
              f"{m['pr_fraction']:>5.0%} {m['max_dominant_freq']:>6} "
              f"{m['spectral_centroid']:>9.2f} {m['max_freq_present']:>8} "
              f"{m['hf_energy_fraction']:>8.1%} {m['saturation']:>5.0%}")
    return out


def compare(runs, path="xai_comparison.png"):
    """The brief's Fig 3: utilisation and frequency content, side by side.

    This is the figure that answers 'what does the quantum layer change',
    because it puts the same two measurements on the same axes for a
    classical network and a quantum-assisted one.
    """
    results = [process(r) for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    width = 0.8 / max(1, len(results))

    for j, res in enumerate(results):
        names = list(res["layers"])
        pos = np.arange(len(names)) + j * width
        axes[0].bar(pos, [res["layers"][n]["pr_fraction"] for n in names],
                    width, label=res["tag"])
        axes[1].bar(pos, [res["layers"][n]["hf_energy_fraction"] for n in names],
                    width, label=res["tag"])
        for ax in axes:
            ax.set_xticks(np.arange(len(names)) + 0.4 - width / 2)
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)

    axes[0].set_ylabel("effective utilisation (PR / width)")
    axes[0].set_title("How much of each layer does work")
    axes[1].set_ylabel(f"high-frequency energy fraction (modes > 2)")
    axes[1].set_title("How much high-frequency content each layer carries")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Per-neuron XAI feature maps (BQP deliverable 2).")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--compare", action="store_true",
                   help="also emit the side-by-side comparison figure")
    p.add_argument("--out", default="xai_comparison.png")
    p.add_argument("--checkpoint", type=int, default=None,
                   help="epoch of a mid-training snapshot (weights_ep<N>.pt)")
    args = p.parse_args()

    if args.compare:
        compare(args.run_dirs, args.out)
    else:
        for d in args.run_dirs:
            process(d, checkpoint=args.checkpoint)
