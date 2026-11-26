"""
quantum_anatomy.py -- what the trained quantum layer actually does.

Runs on the 32 trained QAPINNs of the Burgers 2x2 (ring/strong x default/
near-identity x 8 seeds). No training; forward passes and statevectors only.

The quantum layer is the FIRST layer: (x, t) -> 4-qubit circuit -> tanh(<Z_i>)
-> classical stack. Everything the classical stack knows about (x, t) passes
through these four numbers, so four questions characterise it:

  1. EFFECTIVE WIDTH     How many of the 4 qubit outputs carry independent
                         information? Participation ratio of the eigenvalues
                         of the 4x4 covariance of tanh(<Z>) over the domain:
                         1 = one direction of variation, 4 = all four used.
  2. SPECTRUM USED       The circuit is an exact trigonometric polynomial in x
                         with integer frequencies 0..4 (x is encoded on 2 wires
                         x 2 re-uploads). Coefficients are computed exactly by
                         sampling one full period of x at fixed t. Reported:
                         share of non-DC power at frequency >= 2 (beyond what a
                         single encoding of x can reach) and the highest
                         frequency holding >= 1% of non-DC power.
  3. ENTANGLEMENT        Meyer-Wallach Q of the circuit's state, averaged over
                         the (x, t) domain. 0 = product state (no entanglement
                         used), 1 = maximal.
  4. WEIGHT MOVEMENT     Initial weights are rebuilt exactly from the seed
                         (robust_train seeds torch immediately before building
                         the model). Reported: RMS circular distance travelled,
                         and RMS distance of the trained weights from zero (the
                         identity) -- this tests the conjecture in the noise
                         section that a near-identity circuit stays near the
                         identity after training.

Output: quantum_anatomy.json, and a table on stdout.
"""

import json
import os

import numpy as np
import pennylane as qml
import torch
from scipy.stats import spearmanr

from evaluate import load_run
from qapinn_model import QAPINN

ARMS = [("conv_ring_L2", "ring", "default"), ("init_ring_L2", "ring", "near-identity"),
        ("conv_strong_L2", "strong", "default"), ("init_strong_L2", "strong", "near-identity")]
SEEDS = range(8)
N = 4


def state_fn(entangler, L):
    dev = qml.device("default.qubit", wires=N)
    block = qml.StronglyEntanglingLayers if entangler == "strong" else qml.BasicEntanglerLayers

    @qml.qnode(dev, interface="torch")
    def f(x, t, w):
        feats = torch.stack([x, t, x, t])
        for l in range(L):
            qml.AngleEmbedding(feats, wires=range(N))
            block(w[l], wires=range(N))
        return qml.state()
    return f


def meyer_wallach(psi):
    psi = psi.reshape([2] * N)
    purity = 0.0
    for k in range(N):
        m = np.moveaxis(psi, k, 0).reshape(2, -1)
        rho = m @ m.conj().T
        purity += np.real(np.trace(rho @ rho))
    return 2 * (1 - purity / N)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def analyse(tag, entangler):
    model, _, meta = load_run(f"runs/{tag}")
    rel = json.load(open(f"runs/{tag}/metrics.json"))["rel_l2"]
    mk = meta["model_kwargs"]
    L = mk["reupload_layers"]
    w_tr = model.quantum_layer.weights.detach().clone()

    torch.manual_seed(meta["seed"])
    w_0 = QAPINN(**mk).quantum_layer.weights.detach().clone()

    # 1. effective width on the training domain
    X, T = np.meshgrid(np.linspace(-1, 1, 80), np.linspace(0, 1, 80))
    xt = torch.tensor(np.stack([X.ravel(), T.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        q = torch.tanh(model.quantum_layer(xt)).numpy()
    ev = np.clip(np.linalg.eigvalsh(np.cov(q.T)), 0, None)
    pr = ev.sum() ** 2 / (ev ** 2).sum()

    # 2. exact x-spectrum: one full period in x (2*pi), 64 samples, 11 t values
    xs = torch.linspace(0, 2 * np.pi, 65)[:-1]
    P = np.zeros(33)
    for tv in np.linspace(0, 1, 11):
        inp = torch.stack([xs, torch.full_like(xs, float(tv))], 1)
        with torch.no_grad():
            z = model.quantum_layer(inp).numpy()          # raw <Z>, the trig polynomial
        P += (np.abs(np.fft.rfft(z, axis=0)) ** 2).sum(1)
    nondc = P[1:].sum()
    share_hi = P[2:].sum() / nondc if nondc > 0 else float("nan")
    k_eff = max([k for k in range(1, 33) if P[k] >= 0.01 * nondc] or [0])

    # 3. Meyer-Wallach, trained and at initialisation, over 100 domain points
    f = state_fn(entangler, L)
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(-1, 1, 100), rng.uniform(0, 1, 100)], 1)
    mw = {}
    for name, w in (("trained", w_tr), ("init", w_0)):
        vals = [meyer_wallach(f(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64),
                                w.double()).detach().numpy()) for a, b in pts]
        mw[name] = float(np.mean(vals))

    # sanity: my rebuilt circuit reproduces the model's own <Z> outputs
    psi = f(torch.tensor(pts[0, 0], dtype=torch.float64), torch.tensor(pts[0, 1], dtype=torch.float64),
            w_tr.double()).detach().numpy().reshape([2] * N)
    probs = np.abs(psi) ** 2
    z_mine = np.array([probs.take(0, axis=k).sum() - probs.take(1, axis=k).sum() for k in range(N)])
    with torch.no_grad():
        z_model = model.quantum_layer(torch.tensor(pts[:1], dtype=torch.float32)).numpy()[0]
    check = float(np.abs(z_mine - z_model).max())

    # 4. movement and distance from identity
    move = float(np.sqrt(np.mean(wrap((w_tr - w_0).numpy()) ** 2)))
    from_id = float(np.sqrt(np.mean(wrap(w_tr.numpy()) ** 2)))
    from_id0 = float(np.sqrt(np.mean(wrap(w_0.numpy()) ** 2)))

    return dict(tag=tag, rel_l2=rel, eff_width=float(pr), share_freq_ge2=float(share_hi),
                k_eff=k_eff, power_by_freq=(P[:6] / nondc).tolist() if nondc > 0 else None,
                mw_trained=mw["trained"], mw_init=mw["init"],
                rms_move=move, rms_from_identity=from_id, rms_from_identity_init=from_id0,
                rebuild_check_maxabs=check)


def main():
    rows = []
    for arm, ent, init in ARMS:
        for s in SEEDS:
            r = analyse(f"{arm}_s{s}", ent)
            r.update(arm=arm, entangler=ent, init=init, seed=s)
            rows.append(r)
            print(f"  {r['tag']:22} err {r['rel_l2']:6.2%}  width {r['eff_width']:.2f}  "
                  f"f>=2 {r['share_freq_ge2']:.2f}  kmax {r['k_eff']}  MW {r['mw_init']:.2f}->{r['mw_trained']:.2f}  "
                  f"move {r['rms_move']:.2f}  |w| {r['rms_from_identity_init']:.2f}->{r['rms_from_identity']:.2f}  "
                  f"chk {r['rebuild_check_maxabs']:.1e}", flush=True)

    print("\nPER ARM (median over 8 seeds)")
    keys = ["rel_l2", "eff_width", "share_freq_ge2", "k_eff", "mw_init", "mw_trained",
            "rms_move", "rms_from_identity_init", "rms_from_identity"]
    summary = {}
    for arm, ent, init in ARMS:
        a = [r for r in rows if r["arm"] == arm]
        summary[arm] = {k: float(np.median([r[k] for r in a])) for k in keys}
        s = summary[arm]
        print(f"  {ent:6} {init:13} err {s['rel_l2']:6.2%} | width {s['eff_width']:.2f}/4 | "
              f"f>=2 share {s['share_freq_ge2']:.2f} | kmax {s['k_eff']:.0f}/4 | "
              f"MW {s['mw_init']:.2f}->{s['mw_trained']:.2f} | move {s['rms_move']:.2f} rad | "
              f"|w| {s['rms_from_identity_init']:.2f}->{s['rms_from_identity']:.2f} rad")

    print("\nDOES ANY OF IT PREDICT ERROR? Spearman rho over all 32 runs (exploratory)")
    corr = {}
    for k in ["eff_width", "share_freq_ge2", "mw_trained", "rms_move"]:
        rho, p = spearmanr([r[k] for r in rows], [r["rel_l2"] for r in rows])
        corr[k] = {"rho": float(rho), "p": float(p)}
        print(f"  {k:16} rho {rho:+.2f}  p {p:.3f}")

    json.dump({"rows": rows, "summary": summary, "spearman_vs_error": corr,
               "max_rebuild_error": max(r["rebuild_check_maxabs"] for r in rows)},
              open("quantum_anatomy.json", "w"), indent=2)
    print(f"\nrebuild check (my circuit vs model's own <Z>): max abs diff "
          f"{max(r['rebuild_check_maxabs'] for r in rows):.1e}\nwrote quantum_anatomy.json")


if __name__ == "__main__":
    main()
