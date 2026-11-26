"""
Turns sweep_results.csv into the figures you actually present.

Run after run_sweep.py. Produces:

  sweep_error_vs_reupload.png   -- does more frequency budget reduce error?
  sweep_error_vs_qubits.png     -- does more width reduce error?
  sweep_ansatz_comparison.png   -- does the ansatz choice matter, as the
                                   Fourier analysis predicted it should?
  sweep_spectral_bias.png       -- shock-region vs smooth-region error, the
                                   quantitative version of "spectral bias"
  sweep_summary_table.md        -- everything, as a markdown table you can
                                   paste straight into a report or slide
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "sweep_results.csv"
QUANTUM_COLOR = "#DD8452"
CLASSICAL_COLOR = "#4C72B0"
STRONG_COLOR = "#55A868"


def load_rows(path=CSV_PATH):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run `python3 run_sweep.py` first.")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ["rel_l2", "rel_l2_shock_region", "rel_l2_smooth_region",
                  "spectral_bias_ratio", "final_pde_loss", "total_seconds"]:
            r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
        for k in ["n_qubits", "reupload_layers", "quantum_params", "total_params"]:
            r[k] = int(r[k]) if r[k] not in ("", None) else None
        r["classical"] = str(r["classical"]).lower() == "true"
    return rows


def baseline_error(rows):
    for r in rows:
        if r["classical"]:
            return r["rel_l2"]
    return None


def _axis_plot(rows, axis, xkey, xlabel, title, outfile, filt=None):
    sel = [r for r in rows if r["axis"] == axis and (filt is None or filt(r))]
    if not sel:
        print(f"  (no rows for axis '{axis}', skipping {outfile})")
        return
    sel.sort(key=lambda r: r[xkey])
    xs = [r[xkey] for r in sel]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(xs, [r["rel_l2"] * 100 for r in sel], "o-", color=QUANTUM_COLOR,
            markersize=8, label="QAPINN, overall")
    ax.plot(xs, [r["rel_l2_shock_region"] * 100 for r in sel], "^--", color="#C44E52",
            markersize=7, alpha=0.85, label="QAPINN, shock region")
    ax.plot(xs, [r["rel_l2_smooth_region"] * 100 for r in sel], "v--", color="#8172B3",
            markersize=7, alpha=0.85, label="QAPINN, smooth region")

    b = baseline_error(rows)
    if b is not None:
        ax.axhline(b * 100, color=CLASSICAL_COLOR, linestyle=":", linewidth=2,
                   label=f"classical PINN ({b:.1%})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("relative L2 error (%)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  saved {outfile}")


def ansatz_figure(rows, outfile="sweep_ansatz_comparison.png"):
    ring = {r["reupload_layers"]: r for r in rows
            if r["axis"] == "reupload" and r["entangler"] == "ring"}
    strong = {r["reupload_layers"]: r for r in rows
              if r["axis"] == "ansatz" and r["entangler"] == "strong"}
    shared = sorted(set(ring) & set(strong))
    if not shared:
        print(f"  (no matched ring/strong depths, skipping {outfile})")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    width = 0.36
    idx = range(len(shared))
    ax.bar([i - width / 2 for i in idx], [ring[L]["rel_l2"] * 100 for L in shared],
           width, color=QUANTUM_COLOR, edgecolor="black", linewidth=0.5,
           label="ring ansatz (RX -- commutes with encoding)")
    ax.bar([i + width / 2 for i in idx], [strong[L]["rel_l2"] * 100 for L in shared],
           width, color=STRONG_COLOR, edgecolor="black", linewidth=0.5,
           label="strongly-entangling ansatz (general Rot)")

    b = baseline_error(rows)
    if b is not None:
        ax.axhline(b * 100, color=CLASSICAL_COLOR, linestyle=":", linewidth=2,
                   label=f"classical PINN ({b:.1%})")

    ax.set_xticks(list(idx))
    ax.set_xticklabels([f"L = {L}" for L in shared])
    ax.set_ylabel("relative L2 error (%)")
    ax.set_title("Does ansatz choice matter?\n"
                 "(Fourier analysis predicted the ring ansatz wastes frequency budget)",
                 fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  saved {outfile}")


def spectral_bias_figure(rows, outfile="sweep_spectral_bias.png"):
    sel = [r for r in rows if not r["classical"] and r["axis"] == "reupload"]
    if not sel:
        print(f"  (no reupload rows, skipping {outfile})")
        return
    sel.sort(key=lambda r: r["reupload_layers"])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot([r["reupload_layers"] for r in sel], [r["spectral_bias_ratio"] for r in sel],
            "o-", color=QUANTUM_COLOR, markersize=8, label="QAPINN")
    b = [r for r in rows if r["classical"]]
    if b:
        ax.axhline(b[0]["spectral_bias_ratio"], color=CLASSICAL_COLOR, linestyle=":",
                   linewidth=2, label=f"classical PINN ({b[0]['spectral_bias_ratio']:.1f})")
    ax.set_xlabel("data re-uploading depth L (= frequency budget)")
    ax.set_ylabel("shock-region error / smooth-region error")
    ax.set_title("Spectral bias, quantified\n"
                 "higher = error concentrated at the sharp feature", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  saved {outfile}")


def summary_table(rows, outfile="sweep_summary_table.md"):
    rows = sorted(rows, key=lambda r: (r["axis"], r["rel_l2"]))
    lines = ["| config | qubits | L | ansatz | q params | total params | rel L2 | shock | smooth | bias ratio | mins |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['n_qubits'] or '-'} | {r['reupload_layers'] or '-'} | "
            f"{r['entangler'] or '-'} | {r['quantum_params']} | {r['total_params']} | "
            f"{r['rel_l2']:.2%} | {r['rel_l2_shock_region']:.2%} | "
            f"{r['rel_l2_smooth_region']:.2%} | {r['spectral_bias_ratio']:.1f} | "
            f"{r['total_seconds'] / 60:.1f} |")
    with open(outfile, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {outfile}")
    print("\n".join(lines))


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} configurations from {CSV_PATH}\n")

    _axis_plot(rows, "reupload", "reupload_layers",
               "data re-uploading depth L (= frequency budget)",
               "Does more frequency budget fix the shock?", "sweep_error_vs_reupload.png")
    _axis_plot(rows, "qubits", "n_qubits", "number of qubits",
               "Does more circuit width help?", "sweep_error_vs_qubits.png")
    ansatz_figure(rows)
    spectral_bias_figure(rows)
    summary_table(rows)


if __name__ == "__main__":
    main()
