"""
Evaluation for any run produced by train.py.

Beyond the usual relative L2 error, this computes an ERROR DECOMPOSITION
that turns the project's central claim into a number instead of a
picture.

WHY THE DECOMPOSITION MATTERS
--------------------------------
Every result in this project so far has been described as "the error is
concentrated at the shock". That was read off a heatmap by eye. Splitting
the domain into a narrow shock band around x=0 and everything else, and
reporting the relative L2 error separately in each, makes spectral bias
a measurable quantity: a spectrally-limited model should show a large
shock-region error and a small smooth-region error, and the RATIO between
them should shrink as frequency budget increases.

That ratio is the cleanest single number for "how badly does this
configuration suffer from spectral bias", and it is what the sweep
compares across configurations.
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
from ground_truth import BurgersGroundTruth
from qapinn_model import QAPINN, ClassicalPINN

_GT = None


def ground_truth():
    global _GT
    if _GT is None:
        _GT = BurgersGroundTruth()
    return _GT


def _build(meta):
    """Reconstruct the model a run was trained with.

    Runs written before qcpinn_train.py existed have no `model_type` key,
    so the original QAPINN/ClassicalPINN branch is kept as the fallback and
    every existing run directory evaluates exactly as it did before."""
    kind = meta.get("model_type")
    if kind is None:
        return (ClassicalPINN(**meta["model_kwargs"]) if meta["classical"]
                else QAPINN(**meta["model_kwargs"]))

    # pde_train.py names the two original models explicitly, where the
    # older runs identified them only by the `classical` boolean.
    if kind == "classical":
        return ClassicalPINN(**meta["model_kwargs"])
    if kind == "qapinn":
        return QAPINN(**meta["model_kwargs"])

    # imported lazily so runs that predate the QCPINN work do not depend on it
    from qcpinn_model import QCPINN, model_1, model_2
    if kind == "qcpinn":
        return QCPINN(**meta["model_kwargs"])
    if kind == "model1":
        return model_1()
    if kind == "model2":
        return model_2()
    raise ValueError(f"unknown model_type in {meta.get('tag')}: {kind}")


def load_run(run_dir, weights="weights.pt"):
    """`weights` selects which snapshot to load -- "weights.pt" for the
    final model, or "weights_ep8000.pt" for a mid-training checkpoint."""
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    model = _build(meta)
    model.load_state_dict(torch.load(os.path.join(run_dir, weights)))
    model.eval()
    history = np.load(os.path.join(run_dir, "loss_history.npy"), allow_pickle=True).item()
    return model, history, meta


def predict_grid(model, nx=200, nt=200):
    x = np.linspace(config.X_MIN, config.X_MAX, nx)
    t = np.linspace(0, config.T_MAX, nt)
    X, T = np.meshgrid(x, t)
    with torch.no_grad():
        xf = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1)
        tf = torch.tensor(T.flatten(), dtype=torch.float32).reshape(-1, 1)
        U = model(xf, tf).numpy().flatten().reshape(X.shape)
    return X, T, U


def metrics(run_dir, half_width=config.SHOCK_HALF_WIDTH):
    model, history, meta = load_run(run_dir)
    X, T, U_pred = predict_grid(model)
    U_true = ground_truth()(X, T)

    err = U_pred - U_true
    rel_l2 = float(np.linalg.norm(err) / np.linalg.norm(U_true))

    shock_mask = np.abs(X) <= half_width
    smooth_mask = ~shock_mask

    rel_shock = float(np.linalg.norm(err[shock_mask]) / np.linalg.norm(U_true[shock_mask]))
    rel_smooth = float(np.linalg.norm(err[smooth_mask]) / np.linalg.norm(U_true[smooth_mask]))

    out = dict(meta)
    out.update({
        "rel_l2": rel_l2,
        "rel_l2_shock_region": rel_shock,
        "rel_l2_smooth_region": rel_smooth,
        "spectral_bias_ratio": rel_shock / rel_smooth if rel_smooth > 0 else float("nan"),
        "shock_half_width": half_width,
    })

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out, (X, T, U_pred, U_true, history)


def make_plots(run_dir, data):
    X, T, U_pred, U_true, history = data
    x = X[0]

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.5), sharey=True)
    for ax, t_val in zip(axes, [0.25, 0.5, 0.75, 0.99]):
        idx = int(np.argmin(np.abs(T[:, 0] - t_val)))
        ax.plot(x, U_true[idx], "b-", linewidth=2, label="Numerical (BDF)")
        ax.plot(x, U_pred[idx], "g--", linewidth=2, label="model")
        ax.set_title(f"t = {t_val}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("u(x,t)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "prediction_vs_truth.png"), dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for key in ["total", "pde", "ic", "bc"]:
        ax2.semilogy(history[key], label=key, alpha=0.85)
    if "phase" in history:
        phase = np.array(history["phase"])
        switch = np.argmax(phase == 1) if (phase == 1).any() else None
        if switch:
            ax2.axvline(switch, color="crimson", linestyle="--", linewidth=1.2)
            ax2.text(switch, ax2.get_ylim()[1], " L-BFGS", color="crimson",
                     va="top", fontsize=8)
    ax2.set_xlabel("step")
    ax2.set_ylabel("loss (log)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    im = ax3.pcolormesh(X, T, np.abs(U_pred - U_true), shading="auto", cmap="viridis")
    fig3.colorbar(im, ax=ax3, label="|u_pred - u_true|")
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    fig3.tight_layout()
    fig3.savefig(os.path.join(run_dir, "error_heatmap.png"), dpi=150)
    plt.close(fig3)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate one or more run directories.")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    for run_dir in args.run_dirs:
        m, data = metrics(run_dir)
        if not args.no_plots:
            make_plots(run_dir, data)
        print(f"{m['tag']:32s} rel L2 {m['rel_l2']:7.2%} | shock {m['rel_l2_shock_region']:7.2%} | "
              f"smooth {m['rel_l2_smooth_region']:7.2%} | bias ratio {m['spectral_bias_ratio']:5.1f} | "
              f"q/c params {m['param_counts']['quantum']}/{m['param_counts']['classical']}")
