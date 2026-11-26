"""Replay the convergence criterion against already-recorded loss histories.

The criterion must do two things: keep slow-breakthrough runs alive, and
stop runs that have genuinely flattened. Both are checkable offline
against the trajectories the seed study already produced -- no retraining.
"""
import glob, json, os, sys
import numpy as np

# Constants mirrored from robust_train.py so this validator needs only numpy
# and can run anywhere, including environments without torch installed.
# Keep these in sync if the trainer's settings change.
CHECK_EVERY = 500
DEFAULT_PATIENCE = 3000
DEFAULT_TOL = 0.03
EMA_ALPHA = 0.002


def replay(adam_losses, patience=DEFAULT_PATIENCE, tol=DEFAULT_TOL):
    ema, trace = None, []
    for epoch, v in enumerate(adam_losses, start=1):
        ema = v if ema is None else (1 - EMA_ALPHA) * ema + EMA_ALPHA * v
        if epoch % CHECK_EVERY == 0:
            trace.append((epoch, ema))
            past = [e for e in trace if e[0] <= epoch - patience]
            if past:
                then = past[-1][1]
                if (then - ema) / max(then, 1e-12) < tol:
                    return epoch, "STOP"
    return len(adam_losses), "still running at end of recorded data"


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "runs"
    groups = [("QAPINN strong L=2", [f"{root}/ansatz_strong_L2_q4"] + sorted(glob.glob(f"{root}/seed_strong_L2_s*"))),
              ("Classical PINN",    [f"{root}/classical_baseline"] + sorted(glob.glob(f"{root}/seed_classical_s*")))]

    for label, runs in groups:
        print(f"--- {label} ---")
        print(f"{'run':24s} {'final err':>10} {'would stop at':>14}  verdict")
        for d in runs:
            if not os.path.exists(os.path.join(d, "loss_history.npy")):
                continue
            h = np.load(os.path.join(d, "loss_history.npy"), allow_pickle=True).item()
            tot, ph = np.array(h["total"]), np.array(h["phase"])
            adam = tot[ph == 0]
            m = json.load(open(os.path.join(d, "metrics.json")))
            stop_at, status = replay(adam)
            if status == "STOP":
                verdict = "would have stopped early -- CHECK THIS"
            else:
                verdict = "kept running to the end -> would train longer  OK"
            print(f"{os.path.basename(d):24s} {m['rel_l2']:>9.1%} {stop_at:>14}  {verdict}")
        print()
