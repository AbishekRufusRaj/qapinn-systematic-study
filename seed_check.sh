#!/usr/bin/env bash
# Seed-variance check on the three configurations that carry the argument:
#   strong L=2  -- the headline result (4.38% on seed 0)
#   ring   L=2  -- the matched comparison that tests the ansatz design rule
#   classical   -- the reference everything is measured against
#
# Seed 0 already exists from the sweep, so this adds seeds 1-4 for n=5 each.
# Resumable: a config whose run dir already has metrics.json is skipped.
# Total ~100 min. Safe to Ctrl-C and re-run.
set -u

run_one () {  # $1 = tag, rest = train.py args
  local tag="$1"; shift
  if [ -f "runs/$tag/metrics.json" ]; then
    echo "[skip] $tag (already done)"
    return
  fi
  echo "[run ] $tag"
  python3 train.py --tag "$tag" "$@" || { echo "[FAIL] $tag"; return; }
  python3 evaluate.py "runs/$tag" --no-plots
}

for s in 1 2 3 4; do
  echo "=================== seed $s ==================="
  run_one "seed_strong_L2_s$s" --qubits 4 --reupload 2 --entangler strong --seed "$s"
  run_one "seed_ring_L2_s$s"   --qubits 4 --reupload 2 --entangler ring   --seed "$s"
  run_one "seed_classical_s$s" --classical                                --seed "$s"
done

echo
python3 summarize_seeds.py
