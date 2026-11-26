#!/usr/bin/env bash
# swap_run.sh -- the 2^3 swap: quantum init x classical init x point stream,
# good seed 5 against bad seed 4, ring ansatz with near-identity initialisation.
set -u
PY="${PY:-/usr/local/bin/python3}"; JOBS="${JOBS:-4}"
mkdir -p logs_extend; J=./.swap_jobs.txt; : > $J
for q in 5 4; do for c in 5 4; do for p in 5 4; do
  echo "swap_q${q}c${c}p${p}|--q $q --c $c --p $p" >> $J
done; done; done
export PY
tr '\n' '\0' < $J | xargs -0 -P "$JOBS" -I {} bash -c '
  tag="${1%%|*}"; args="${1#*|}"
  [ -f "runs/$tag/metrics.json" ] && { echo "[skip] $tag"; exit 0; }
  t0=$(date +%s)
  "$PY" swap_train.py --tag "$tag" $args > "logs_extend/$tag.log" 2>&1
  echo "[$([ -f runs/$tag/metrics.json ] && echo done || echo FAIL)] $tag  $(( ($(date +%s)-t0)/60 ))m  $(grep -o "rel L2 *[0-9.]*%" logs_extend/$tag.log | tail -1)"
' _ {}
rm -f $J; echo "SWAP finished"
