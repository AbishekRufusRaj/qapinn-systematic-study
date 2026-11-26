"""
EMPIRICAL VALIDATION OF THE PROJECT'S THEORETICAL ANCHOR.

This is Objective 3 of the proposal ("develop and empirically test a
Fourier-spectrum-based theory predicting the frequency expressivity of
each quantum-layer configuration"), and it needs no training at all --
the accessible frequency spectrum is a property of the CIRCUIT, not of
the trained weights.

THE CLAIM BEING TESTED
-------------------------
Schuld, Sweke & Meyer (2021): a variational quantum circuit's output, as
a function of an encoded input x, is a truncated Fourier series

    f(x) = sum_{w in Omega} c_w e^{i w x}

where the accessible frequency set Omega is determined entirely by the
data-encoding gates -- specifically by the eigenvalues of their
generators. For angle encoding with Pauli rotations, encoding x once
gives access to frequencies {-1, 0, 1}. Encoding it L times (data
re-uploading) gives access to integer frequencies up to L.

Concretely: a circuit that re-uploads x 6 times can build shapes out of
harmonics up to the 6th. One that uploads it once can only use the
fundamental -- which is why it can produce a single smooth hump and
nothing sharper, and why it failed at the Burgers' shock.

HOW THIS SCRIPT TESTS IT
---------------------------
1. Build the quantum layer at a given re-upload depth (random weights --
   the accessible spectrum doesn't depend on their values).
2. Sweep x across one full period of the encoding angle, [-pi, pi],
   holding t fixed. Record the circuit's output.
3. Take the FFT of that output. The magnitude spectrum shows exactly
   which frequencies the circuit actually has access to.
4. Repeat across re-upload depths and check whether the cutoff frequency
   tracks the depth, as the theorem says it must.

If the measured cutoff scales with re-upload depth, the theory is
validated on this project's own circuits -- and the Burgers'-shock
result stops being an anecdote and becomes a predicted consequence.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from qapinn_model import make_quantum_layer

AMPLITUDE_THRESHOLD = 1e-6   # what counts as "this frequency is present"
N_DRAWS = 12                  # random weight draws averaged per configuration


def predicted_ceiling(n_qubits, reupload_layers):
    """Frequencies the theorem says this circuit can access, for x.

    Inputs are tiled across the wires as [x, t, x, t, ...], so x is
    encoded on ceil(n_qubits/2) wires, and each re-upload layer applies
    every one of those encodings again. Total applications of x's
    generator = n_x_wires * reupload_layers, and for Pauli-rotation angle
    encoding the accessible integer frequencies run up to exactly that.

    Note this is NOT simply 'L' -- a detail worth stating explicitly when
    presenting, because it means qubit count and re-upload depth BOTH
    feed the frequency budget when inputs are tiled this way."""
    n_x_wires = (n_qubits + 1) // 2
    return n_x_wires * reupload_layers


def measure_spectrum(n_qubits, reupload_layers, entangler="ring", encoding="angle",
                     n_samples=config.FOURIER_N_SAMPLES, t_fixed=0.5, seed=config.SEED,
                     n_draws=N_DRAWS):
    """Return (freqs, mean_magnitude) for the quantum layer's x-dependence.

    Averaged over several random weight draws: the ACCESSIBLE spectrum is
    a property of the circuit, but any single random draw can accidentally
    put a near-zero coefficient on a frequency that is in fact available.
    Averaging |FFT| over draws removes those accidental gaps and shows
    what the architecture can actually reach."""
    x = torch.linspace(-np.pi, np.pi, n_samples + 1)[:-1].reshape(-1, 1)
    t = torch.full_like(x, t_fixed)
    xt = torch.cat([x, t], dim=1)

    accumulated = None
    for draw in range(n_draws):
        torch.manual_seed(seed + draw)
        layer = make_quantum_layer(n_qubits=n_qubits, reupload_layers=reupload_layers,
                                   entangler=entangler, encoding=encoding)
        with torch.no_grad():
            out = layer(xt).numpy()                       # (n_samples, n_qubits)
        spectra = np.abs(np.fft.rfft(out, axis=0)) / n_samples
        mag = spectra.mean(axis=1)                        # average across qubits
        accumulated = mag if accumulated is None else accumulated + mag

    mean_mag = accumulated / n_draws
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / n_samples)  # integer frequency indices
    return freqs, mean_mag


def cutoff_frequency(freqs, mag, threshold=AMPLITUDE_THRESHOLD):
    """Highest frequency whose amplitude clears the threshold."""
    significant = np.where(mag > threshold)[0]
    return float(freqs[significant[-1]]) if len(significant) else 0.0


def main():
    depths = config.FOURIER_REUPLOAD_LAYERS
    n_qubits = config.FOURIER_N_QUBITS
    max_plot_freq = predicted_ceiling(n_qubits, max(depths)) + 4

    results = []
    fig, axes = plt.subplots(1, len(depths), figsize=(3.3 * len(depths), 3.6), sharey=True)
    if len(depths) == 1:
        axes = [axes]

    for ax, depth in zip(axes, depths):
        freqs, mag = measure_spectrum(n_qubits, depth)
        cut = cutoff_frequency(freqs, mag)
        pred = predicted_ceiling(n_qubits, depth)
        results.append((depth, cut, pred))

        keep = freqs <= max_plot_freq
        ax.bar(freqs[keep], mag[keep], width=0.6, color="#DD8452", edgecolor="black", linewidth=0.4)
        ax.axvline(pred + 0.5, color="crimson", linestyle="--", linewidth=1.4)
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 1)
        ax.set_xlabel("frequency")
        ax.set_title(f"L = {depth}\npredicted {pred}, measured {cut:.0f}", fontsize=10)

    axes[0].set_ylabel("amplitude (log)")
    fig.suptitle("Quantum layer frequency spectrum vs. data re-uploading depth L\n"
                 "(dashed line = ceiling predicted by Schuld/Sweke/Meyer for this encoding)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("fourier_spectra.png", dpi=150)
    print("saved fourier_spectra.png")

    # summary: predicted vs measured cutoff
    depths_arr = np.array([r[0] for r in results], dtype=float)
    cuts_arr = np.array([r[1] for r in results], dtype=float)
    preds_arr = np.array([r[2] for r in results], dtype=float)

    fig2, ax2 = plt.subplots(figsize=(5.8, 4.4))
    ax2.plot(depths_arr, preds_arr, "k--", marker="s", markersize=6,
             label="theory ceiling (n_x_wires x L)")
    ax2.plot(depths_arr, cuts_arr, "o-", color="#DD8452", markersize=8,
             label="measured cutoff (FFT of circuit output)")
    ax2.set_xlabel("data re-uploading depth L")
    ax2.set_ylabel("highest accessible frequency")
    ax2.set_title(f"Frequency expressivity scales with re-uploading depth\n({n_qubits} qubits)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("fourier_cutoff_vs_depth.png", dpi=150)
    print("saved fourier_cutoff_vs_depth.png")

    # encoding-scheme comparison at fixed depth: does angle_scaled buy
    # extra frequencies without extra re-uploading?
    fig3, ax3 = plt.subplots(figsize=(6.5, 4.2))
    for encoding, color in [("angle", "#DD8452"), ("angle_scaled", "#4C72B0")]:
        freqs, mag = measure_spectrum(n_qubits, reupload_layers=2, encoding=encoding)
        keep = freqs <= max_plot_freq
        ax3.plot(freqs[keep], mag[keep], "o-", color=color, markersize=4,
                 label=f"{encoding} (cutoff {cutoff_frequency(freqs, mag):.0f})")
    ax3.set_yscale("log")
    ax3.set_ylim(1e-9, 1)
    ax3.set_xlabel("frequency")
    ax3.set_ylabel("amplitude (log)")
    ax3.set_title("Encoding scheme changes the spectrum at fixed depth (L=2)")
    ax3.legend()
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig("fourier_encoding_comparison.png", dpi=150)
    print("saved fourier_encoding_comparison.png")

    # ------------------------------------------------------------------
    # ENTANGLER/ANSATZ COMPARISON -- the axis that turned out to matter
    # more than expected.
    #
    # BasicEntanglerLayers uses trainable RX rotations. AngleEmbedding
    # also encodes with RX by default. Rotations about the SAME axis
    # COMMUTE: RX(x) RX(w) = RX(x + w). So on each wire the encoding and
    # the very gate meant to process it partially collapse into one
    # rotation, and only the CNOT ring separates successive re-uploads.
    # The theoretical frequency ceiling still rises with depth, but much
    # of that spectrum is left unpopulated.
    #
    # StronglyEntanglingLayers uses general Rot(a,b,c) = RZ RY RZ gates,
    # which do NOT commute with the RX encoding -- so successive
    # re-uploads genuinely compound instead of partially cancelling.
    # ------------------------------------------------------------------
    fig4, axes4 = plt.subplots(1, len(depths), figsize=(3.3 * len(depths), 3.6), sharey=True)
    if len(depths) == 1:
        axes4 = [axes4]
    entangler_rows = []
    for ax, depth in zip(axes4, depths):
        pred = predicted_ceiling(n_qubits, depth)
        for entangler, color, offset in [("ring", "#DD8452", -0.2), ("strong", "#4C72B0", 0.2)]:
            freqs, mag = measure_spectrum(n_qubits, depth, entangler=entangler)
            cut = cutoff_frequency(freqs, mag)
            entangler_rows.append((depth, entangler, pred, cut))
            keep = freqs <= max_plot_freq
            ax.bar(freqs[keep] + offset, mag[keep], width=0.38, color=color,
                   edgecolor="black", linewidth=0.3,
                   label=f"{entangler} (cutoff {cut:.0f})")
        ax.axvline(pred + 0.5, color="crimson", linestyle="--", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 1)
        ax.set_xlabel("frequency")
        ax.set_title(f"L = {depth} (ceiling {pred})", fontsize=10)
        ax.legend(fontsize=7)
    axes4[0].set_ylabel("amplitude (log)")
    fig4.suptitle("Ansatz choice decides how much of the available spectrum is actually populated\n"
                  "(RX ansatz partially commutes with RX encoding; general Rot gates do not)",
                  fontsize=11)
    fig4.tight_layout()
    fig4.savefig("fourier_entangler_comparison.png", dpi=150)
    print("saved fourier_entangler_comparison.png")

    with open("fourier_entangler_results.csv", "w") as f:
        f.write("reupload_depth,entangler,predicted_ceiling,measured_cutoff\n")
        for d, e, p, c in entangler_rows:
            f.write(f"{int(d)},{e},{int(p)},{int(c)}\n")

    print("\nansatz comparison (measured cutoff vs. theoretical ceiling)")
    for d, e, p, c in entangler_rows:
        print(f"  L = {int(d):2d}  {e:6s}  ceiling {int(p):3d}  measured {int(c):3d}")

    lines = ["reupload_depth,predicted_ceiling,measured_cutoff"]
    lines += [f"{int(d)},{int(p)},{int(c)}" for d, c, p in results]
    with open("fourier_results.csv", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\nre-upload depth -> frequency cutoff")
    for d, c, p in results:
        verdict = "MATCHES theory" if abs(c - p) < 1.01 else f"below ceiling by {p - c:.0f}"
        print(f"  L = {int(d):2d}  ->  predicted {int(p):3d}, measured {int(c):3d}   {verdict}")


if __name__ == "__main__":
    main()
