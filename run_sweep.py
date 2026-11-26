"""
THE SYSTEMATIC SWEEP (Weeks 7-10 milestone).

Everything before this point in the project produced individual data
points: one circuit, one number. That is enough to notice an effect, not
enough to characterize it. This script trains a whole grid of
configurations under an IDENTICAL data budget and training recipe, so
the differences between them are attributable to the architecture and
nothing else.

WHAT IS SWEPT
----------------
  Axis 1 -- re-uploading depth L at fixed qubit count.
            Tests the frequency hypothesis: fourier_analysis.py shows the
            accessible spectrum grows with L; this checks whether the
            Burgers' shock error actually falls as a result.

  Axis 2 -- qubit count at fixed re-uploading depth.
            Separates "more Hilbert space" from "more frequencies", and
            exposes the trainability cost that gradient_variance.py
            predicts should come with extra width.

  Axis 3 -- ansatz structure (ring vs strongly-entangling) at a few depths.
            fourier_analysis.py found the RX-based ring ansatz leaves much
            of its own frequency ceiling unpopulated because those
            rotations commute with the RX encoding. If that matters, the
            strongly-entangling runs should beat the ring runs at equal
            depth. This is the sweep entry most likely to produce a
            genuinely new, useful design rule.

  Baseline -- the classical PINN, trained under the exact same budget, so
            every QAPINN number has an honest reference point.

Each configuration writes its own run directory, then gets evaluated, and
everything lands in sweep_results.csv. The sweep is RESUMABLE: a config
whose run directory already has metrics.json is skipped, so you can stop
and restart without losing work.

Usage:
    python3 run_sweep.py                  # everything, in a sensible order
    python3 run_sweep.py --axis reupload  # just one axis
    python3 run_sweep.py --list           # show the plan and time estimate
"""

import argparse
import csv
import json
import os
import time

import config
import evaluate
from train import run_experiment


def build_plan():
    """Return the list of (tag, kwargs, classical) configurations to run,
    ordered cheapest-first so partial results are still useful."""
    plan = []

    # baseline first -- it's fast and everything else is compared to it
    plan.append(("classical_baseline", {}, True, "baseline"))

    # Axis 1: re-uploading depth at fixed width
    for L in config.SWEEP_REUPLOAD:
        plan.append((f"reupload_L{L}_q{config.SWEEP_REUPLOAD_FIXED_QUBITS}",
                     dict(n_qubits=config.SWEEP_REUPLOAD_FIXED_QUBITS, reupload_layers=L),
                     False, "reupload"))

    # Axis 2: qubit count at fixed depth
    for q in config.SWEEP_QUBITS:
        plan.append((f"qubits_q{q}_L{config.SWEEP_QUBITS_FIXED_REUPLOAD}",
                     dict(n_qubits=q, reupload_layers=config.SWEEP_QUBITS_FIXED_REUPLOAD),
                     False, "qubits"))

    # Axis 3: ansatz structure at a few depths
    for L in [2, 4, 6]:
        plan.append((f"ansatz_strong_L{L}_q4",
                     dict(n_qubits=4, reupload_layers=L, entangler="strong"),
                     False, "ansatz"))

    # de-duplicate by the ACTUAL configuration, not the tag -- the two
    # axes cross at one shared point (4 qubits, L=4) and training it twice
    # under the same seed would just burn compute for an identical result
    seen, unique = set(), []
    for tag, kwargs, classical, axis in plan:
        key = ("classical",) if classical else tuple(sorted(kwargs.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append((tag, kwargs, classical, axis))
    return unique


def already_done(run_dir):
    return os.path.exists(os.path.join(run_dir, "metrics.json"))


def write_csv(rows, path="sweep_results.csv"):
    if not rows:
        return
    fields = ["tag", "axis", "classical", "n_qubits", "reupload_layers", "entangler",
              "quantum_params", "classical_params", "total_params",
              "rel_l2", "rel_l2_shock_region", "rel_l2_smooth_region",
              "spectral_bias_ratio", "final_pde_loss", "total_seconds"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nwrote {path} ({len(rows)} configurations)")


def collect_rows(plan, out_root):
    rows = []
    for tag, kwargs, classical, axis in plan:
        mpath = os.path.join(out_root, tag, "metrics.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            m = json.load(f)
        rows.append({
            "tag": tag, "axis": axis, "classical": classical,
            "n_qubits": kwargs.get("n_qubits", ""),
            "reupload_layers": kwargs.get("reupload_layers", ""),
            "entangler": kwargs.get("entangler", "ring" if not classical else ""),
            "quantum_params": m["param_counts"]["quantum"],
            "classical_params": m["param_counts"]["classical"],
            "total_params": m["param_counts"]["total"],
            "rel_l2": round(m["rel_l2"], 6),
            "rel_l2_shock_region": round(m["rel_l2_shock_region"], 6),
            "rel_l2_smooth_region": round(m["rel_l2_smooth_region"], 6),
            "spectral_bias_ratio": round(m["spectral_bias_ratio"], 4),
            "final_pde_loss": m["final_pde_loss"],
            "total_seconds": m["total_seconds"],
        })
    return rows


def main():
    p = argparse.ArgumentParser(description="Run the systematic QAPINN sweep.")
    p.add_argument("--axis", choices=["baseline", "reupload", "qubits", "ansatz", "all"], default="all")
    p.add_argument("--budget", choices=["sweep", "full"], default="sweep")
    p.add_argument("--out-root", default="runs")
    p.add_argument("--list", action="store_true", help="print the plan and exit")
    p.add_argument("--quiet-training", action="store_true", help="suppress per-epoch training logs")
    args = p.parse_args()

    budget = config.FULL if args.budget == "full" else config.SWEEP
    plan = build_plan()
    if args.axis != "all":
        plan = [item for item in plan if item[3] == args.axis]

    if args.list:
        print(f"{len(plan)} configurations planned (budget: {args.budget})\n")
        for tag, kwargs, classical, axis in plan:
            print(f"  [{axis:9s}] {tag:28s} {'classical baseline' if classical else kwargs}")
        print("\nRun `python3 run_sweep.py` to execute. Already-completed configurations "
              "are skipped automatically, so this is safe to stop and restart.")
        return

    print(f"sweep: {len(plan)} configurations, budget={args.budget} "
          f"({budget['adam_epochs']} Adam epochs + {budget['lbfgs_steps']} L-BFGS steps each)\n")

    t_start = time.time()
    for i, (tag, kwargs, classical, axis) in enumerate(plan, 1):
        run_dir = os.path.join(args.out_root, tag)
        if already_done(run_dir):
            print(f"[{i}/{len(plan)}] {tag}: already complete, skipping", flush=True)
            continue

        print(f"[{i}/{len(plan)}] {tag} ...", flush=True)
        run_experiment(model_kwargs=kwargs, budget=budget, tag=tag,
                       out_root=args.out_root, classical=classical,
                       verbose=not args.quiet_training)
        m, data = evaluate.metrics(run_dir)
        evaluate.make_plots(run_dir, data)
        print(f"    -> rel L2 {m['rel_l2']:.2%} | shock {m['rel_l2_shock_region']:.2%} | "
              f"smooth {m['rel_l2_smooth_region']:.2%} | bias ratio {m['spectral_bias_ratio']:.1f}",
              flush=True)

        write_csv(collect_rows(plan, args.out_root))

    rows = collect_rows(plan, args.out_root)
    write_csv(rows)
    print(f"\ntotal sweep wall time: {(time.time() - t_start) / 60:.1f} min")
    print("next: python3 make_figures.py")


if __name__ == "__main__":
    main()
