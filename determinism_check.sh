#!/usr/bin/env bash
# determinism_check.sh -- is a run reproducible from its seed alone?
#
# sym3_model2_s0 and sym2_model2_s0 share a seed and a code path for the Adam
# phase, agree exactly at epoch 500, and end at 5.43% and 10.00%. Either the
# refactor between the two scripts changed the arithmetic, or these runs are not
# reproducible from the seed at all. This distinguishes the two, and tests
# whether single-threading restores determinism.
#
#   rep2a / rep2b   same script, same seed, 2 threads   -- differ => nondeterministic
#   rep1a / rep1b   same script, same seed, 1 thread    -- differ => still nondeterministic
#
# Classical arm only: ~2 minutes a run. No --check-against, so nothing asserts.
set -u
PY="${PY:-external/venv_qcpinn/bin/python}"
run() {  # tag, threads
  OMP_NUM_THREADS=$2 MKL_NUM_THREADS=$2 \
    "$PY" protocol3_qcpinn.py --model model2 --seed 0 --tag "$1" \
      > "logs_extend/$1.log" 2>&1
  echo "  $1 (threads=$2): $(grep -o 'FINAL err_u = .*' logs_extend/$1.log | tail -1)"
}
mkdir -p logs_extend
rm -rf runs/det_rep2a runs/det_rep2b runs/det_rep1a runs/det_rep1b
run det_rep2a 2 & run det_rep2b 2 & wait
run det_rep1a 1 & run det_rep1b 1 & wait
echo
echo "=== after-Adam error, 4 replicates of one seed ==="
for t in det_rep2a det_rep2b det_rep1a det_rep1b; do
  "$PY" -c "
import json;d=json.load(open('runs/$t/replication.json'))
print(f'  $t  after_adam={d[\"err_u_pct_after_adam\"]:.6f}%  final={d[\"err_u_pct\"]:.6f}%  stop={d[\"stop_reason\"]}@{d[\"adam_epochs_used\"]}')"
done
echo
echo "reference: sym2_model2_s0 after_adam=10.001266%, sym3_model2_s0 after_adam=5.425300%"
