#!/usr/bin/env bash
# helmholtz_ours.sh -- this project's six arms on 2D Helmholtz, 8 seeds each.
# Arms fixed BEFORE any Helmholtz result exists:
#   hh_classical        classical PINN, framework default init
#   hh_classical_g1.7321  classical PINN, Xavier-equivalent gain
#   hh_ring / hh_ring_ni     ring ansatz, default / near-identity (scale 0.1)
#   hh_strong / hh_strong_ni strong ansatz, default / near-identity
# Resumable: finished runs are skipped. Classical arms first (fast).
set -u
PY="${PY:-/usr/local/bin/python3}"; JOBS="${JOBS:-4}"; SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
mkdir -p logs_extend; J=./.hh_jobs.txt; : > $J
for s in $SEEDS; do
  echo "hh_classical_s$s|--model classical --seed $s" >> $J
  echo "hh_classical_g1.7321_s$s|--model classical --init-gain 1.7321 --seed $s" >> $J
done
for s in $SEEDS; do for e in ring strong; do
  echo "hh_${e}_s$s|--model qapinn --entangler $e --seed $s" >> $J
  echo "hh_${e}_ni_s$s|--model qapinn --entangler $e --init-scale 0.1 --seed $s" >> $J
done; done
export PY
tr '\n' '\0' < $J | xargs -0 -P "$JOBS" -I {} bash -c '
  tag="${1%%|*}"; args="${1#*|}"
  [ -f "runs/$tag/metrics.json" ] && { echo "[skip] $tag"; exit 0; }
  t0=$(date +%s)
  "$PY" helmholtz_ours.py --tag "$tag" $args > "logs_extend/$tag.log" 2>&1
  echo "[$([ -f runs/$tag/metrics.json ] && echo done || echo FAIL)] $tag  $(( ($(date +%s)-t0)/60 ))m  $(grep -o "rel L2 *[0-9.]*%" logs_extend/$tag.log | tail -1)"
' _ {}
rm -f $J; echo "HH finished"
