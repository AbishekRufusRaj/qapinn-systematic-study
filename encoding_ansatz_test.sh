#!/usr/bin/env bash
# 2x2 factorial: ansatz (ring | strong) x encoding (angle | angle_scaled).
#
# The convergence sweep already supplies half the grid:
#     conv_ring_L2_s*      ring   + angle
#     conv_strong_L2_s*    strong + angle
# This script adds the two cells nobody has ever trained:
#     enc_ring_L2_s*       ring   + angle_scaled
#     enc_strong_L2_s*     strong + angle_scaled
#
# Running both new cells rather than only the predicted-best one is
# deliberate. With all four, the ansatz effect and the encoding effect can
# be separated from each other and from their interaction. With only the
# best cell you would see a number and not know which change produced it.
#
# Why angle_scaled is worth the compute: at identical depth and qubit
# count, fourier_analysis.py measured plain angle cutting off at frequency
# 4 and angle_scaled reaching 6 -- 50% more frequency ceiling for no extra
# qubits, depth, or runtime.
#
# Everything here uses the existing modules unchanged and writes only to
# new run tags, so it cannot disturb a sweep already in progress.
# Resumable; skips completed runs. Roughly 4-5 hours for all ten.
set -u

if pgrep -f "robust_train.py" > /dev/null 2>&1; then
  echo "A training process is still running (robust_train.py)."
  echo "Let the convergence sweep finish first, then re-run this script."
  echo "Override with: FORCE=1 ./encoding_ansatz_test.sh"
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

run_one () {
  local tag="$1"; shift
  if [ -f "runs/$tag/metrics.json" ]; then echo "[skip] $tag"; return; fi
  echo "[run ] $tag"
  python3 robust_train.py --tag "$tag" "$@" || { echo "[FAIL] $tag"; return; }
  python3 evaluate.py "runs/$tag" --no-plots
}

for s in 0 1 2 3 4; do
  echo "=================== seed $s ==================="
  run_one "enc_strong_L2_s$s" --qubits 4 --reupload 2 --entangler strong \
                              --encoding angle_scaled --seed "$s"
  run_one "enc_ring_L2_s$s"   --qubits 4 --reupload 2 --entangler ring   \
                              --encoding angle_scaled --seed "$s"
done

echo
python3 summarize_encoding_test.py
