#!/usr/bin/env bash
# Seed study, redone with convergence-based training instead of a fixed
# 8000-epoch cutoff.
#
# The fixed-budget study measured un-converged models: replaying the
# convergence criterion over its recorded histories shows all ten runs
# were still improving when they were stopped. This re-runs the same
# three configurations, 5 seeds each, training each until its loss
# genuinely flattens (cap 25000 epochs).
#
# Resumable -- completed runs are skipped. Expect ~6 hours.
set -u

run_one () {
  local tag="$1"; shift
  if [ -f "runs/$tag/metrics.json" ]; then echo "[skip] $tag"; return; fi
  echo "[run ] $tag"
  python3 robust_train.py --tag "$tag" "$@" || { echo "[FAIL] $tag"; return; }
  python3 evaluate.py "runs/$tag" --no-plots
}

for s in 0 1 2 3 4; do
  echo "=================== seed $s ==================="
  run_one "conv_strong_L2_s$s" --qubits 4 --reupload 2 --entangler strong --seed "$s"
  run_one "conv_ring_L2_s$s"   --qubits 4 --reupload 2 --entangler ring   --seed "$s"
  run_one "conv_classical_s$s" --classical                                --seed "$s"
done

echo
python3 summarize_seeds.py --conv
