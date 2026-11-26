#!/usr/bin/env bash
# The second PDE. Same three configurations as the Burgers' convergence
# study (conv_seed_check.sh) and the same protocol, so the two PDEs are
# compared at identical design points and the only thing that changed is
# the equation.
#
# WHAT THIS BUYS
# --------------
# "Single PDE" is the first objection a reviewer raises. But the value is
# not simply "we did two" -- it is the CONTRAST. Burgers' is broadband with
# its hard content concentrated in space at the shock. This heat problem is
# smooth with its hard content concentrated in TIME: the initial condition
# is a sum of modes k = 1, 3, 5 decaying as exp(-alpha k^2 pi^2 t), so
# modes 3 and 5 exist early and are gone by t = 0.5.
#
#     mode amplitude   t=0.0    t=0.2    t=1.0
#         k=1          0.887    0.852    0.728
#         k=3          0.443    0.311    0.075
#         k=5          0.222    0.083    0.002
#
# A spectrally limited model therefore fails EARLY on heat and at the SHOCK
# on Burgers'. The error decomposition reports early-vs-late for heat and
# shock-vs-smooth for Burgers', both exposed as `hard_easy_ratio` so one
# column compares across equations.
#
# A design rule that holds on one and fails on the other is a far stronger
# result than either alone -- it is the difference between "we studied
# Burgers'" and "we established where these rules apply".
#
# Heat also has an EXACT solution (each mode decays independently), so
# unlike Burgers' there is no numerical reference and no solver error in
# the ground truth. Worth having in a paper about measurement error.
#
# COST
# ----
# Heat converges in far fewer epochs than Burgers' -- a classical PINN is
# already at 2.9% after 1,500 -- so the convergence criterion stops these
# runs early and the sweep is considerably cheaper than its Burgers'
# counterpart. Resumable; completed runs are skipped.
set -u

PY="${PY:-python3}"
if ! "$PY" -c "import pennylane, torch" >/dev/null 2>&1; then
  echo "'$PY' cannot import pennylane and torch."
  echo "Re-run as:  PY=/usr/local/bin/python3 bash heat_sweep.sh"
  exit 1
fi

if pgrep -f "robust_train.py|qcpinn_train.py|pde_train.py" > /dev/null 2>&1; then
  echo "A training process is already running."
  echo "Override with: FORCE=1 ..."
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

echo "=== PDE definition check (mode decay must separate, or the problem"
echo "    discriminates between nothing) ==="
"$PY" pdes.py || exit 1
echo

run_one () {
  local tag="$1"; shift
  if [ -f "runs/$tag/metrics.json" ]; then echo "[skip] $tag"; return; fi
  echo "[run ] $tag"
  "$PY" pde_train.py --pde heat --tag "$tag" "$@" || { echo "[FAIL] $tag"; return; }
}

for s in 0 1 2 3 4; do
  echo "=================== seed $s ==================="
  run_one "heat_classical_s$s"    --model classical --seed "$s"
  run_one "heat_ring_L2_s$s"      --model qapinn --qubits 4 --reupload 2 \
                                  --entangler ring --seed "$s"
  run_one "heat_strong_L2_s$s"    --model qapinn --qubits 4 --reupload 2 \
                                  --entangler strong --init-scale 0.1 --seed "$s"
done

echo
echo "Done. Summarise with:"
echo "  $PY summarize_pde.py --pde heat"
