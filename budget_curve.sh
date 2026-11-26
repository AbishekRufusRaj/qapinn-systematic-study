#!/usr/bin/env bash
# THE TITLE EXPERIMENT.
#
# Everything else in this project supports one claim: that design
# conclusions about quantum-assisted PINNs depend on how long you train,
# and that published comparisons made at short fixed budgets are therefore
# measuring the budget rather than the architecture. This is the experiment
# that measures that directly.
#
# HOW IT WORKS, AND WHY IT IS CHEAP
# ---------------------------------
# The naive design trains separately at 800, 2000, 8000 and convergence:
# four times the compute, and four different random trajectories whose
# differences confound the budget effect with seed noise.
#
# Instead each run is SNAPSHOTTED at those epoch counts. One trajectory
# yields all four budget points -- same seed, same data, same
# initialisation -- so the only thing differing between points is the
# budget. Four times cheaper and a strictly cleaner comparison.
#
# 800 is not an arbitrary choice: it is the Adam budget used by a public
# submission to this same BQP challenge whose design-axis conclusions this
# tests, and the point at which those conclusions were drawn.
#
# WHY THESE THREE CONFIGURATIONS
# ------------------------------
# The ranking of ring against strong is the contested one, and the existing
# data already hints it is budget-dependent:
#
#     at 8,000 epochs   classical 8.05%  <  strong 17.47%  <  ring 24.27%
#     at convergence    classical 3.36%  <  ring   6.78%   <  strong  8.70%
#
# Strong and ring swap. But those came from two separate studies with
# different seeds and enormous spreads (11-14%), so the swap may be noise.
# This experiment settles it: same trajectory, both budgets, five seeds.
#
# PARALLELISM
# -----------
# Quantum runs are effectively single-threaded, so the three configurations
# run concurrently and each is pinned to one thread to stop them fighting
# over cores. Expect roughly a third of the serial wall clock -- about
# 3 hours rather than 9.
#
# Resumable; completed runs are skipped.
set -u

PY="${PY:-python3}"
if ! "$PY" -c "import pennylane, torch" >/dev/null 2>&1; then
  echo "'$PY' cannot import pennylane and torch."
  echo "Re-run as:  PY=/usr/local/bin/python3 bash budget_curve.sh"
  exit 1
fi

if pgrep -f "robust_train.py|qcpinn_train.py|pde_train.py" > /dev/null 2>&1; then
  echo "A training process is already running. Override with FORCE=1."
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

CKPTS="${CKPTS:-800,2000,8000}"
PDE="${PDE:-burgers}"
echo "pde=$PDE  checkpoints=$CKPTS  (plus convergence)"
echo

# one thread per process: these are small circuits, and thread coordination
# costs more than it saves once three processes are competing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_one () {
  local tag="$1"; shift
  if [ -f "runs/$tag/metrics.json" ]; then echo "[skip] $tag"; return; fi
  echo "[run ] $tag"
  "$PY" pde_train.py --pde "$PDE" --tag "$tag" --checkpoints "$CKPTS" "$@" \
    > "runs_${tag}.log" 2>&1 || echo "[FAIL] $tag (see runs_${tag}.log)"
}

for s in 0 1 2 3 4; do
  echo "=================== seed $s ==================="
  # three configurations concurrently, then wait for all of them
  run_one "bc_classical_s$s"  --model classical --seed "$s" &
  run_one "bc_ring_L2_s$s"    --model qapinn --qubits 4 --reupload 2 \
                              --entangler ring --seed "$s" &
  run_one "bc_strong_L2_s$s"  --model qapinn --qubits 4 --reupload 2 \
                              --entangler strong --init-scale 0.1 --seed "$s" &
  wait
  echo "  seed $s complete"
done

echo
"$PY" summarize_budget_curve.py
