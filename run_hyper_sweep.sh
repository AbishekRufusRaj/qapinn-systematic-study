#!/usr/bin/env bash
# run_hyper_sweep.sh -- hyperparameter sensitivity, Reviewer 2 point 3.
#
# R2 asked whether the reported gains are sensitive to optimiser choice,
# learning-rate scheduling and batch size. Training duration is already answered
# by the convergence-criterion study, and seed count by the ten-seed arms.
#
# Runs protocol5_hyper.py, which is protocol4_qcpinn.py plus --optimizer, --lr
# and --batch and nothing else. Validated: at its defaults it reproduces
# protocol4's model2 seed 0 after-Adam error of 10.00% exactly, so the sym4 runs
# at seeds 0-2 are this sweep's baseline cell and are NOT re-run here.
#
# MUST run under the authors' interpreter. external/venv_qcpinn is Python 3.9.6
# with torch 2.0.1 and PennyLane 0.29; the system Python (3.13.5, torch 2.10)
# gives 6.44% where the venv gives 10.00% on the same seed and the same code.
# The numerics are interpreter-dependent, so PY is not optional.
#
# Classical runs first (about a minute each) so a broken config surfaces in
# minutes rather than hours. Quantum is config-major, so an interrupted run
# leaves complete three-seed cells rather than fragments of every cell.
#
# Writes runs/hyp_<arm>_<config>_s<seed>. Resumable: a finished run is skipped.
set -u
PY="${PY:-external/venv_qcpinn/bin/python}"; JOBS="${JOBS:-4}"
SEEDS="${SEEDS:-0 1 2}"
CONFIGS="${CONFIGS:-sgd rmsprop lr0.001 lr0.01 batch256}"

flags() {
  case "$1" in
    sgd)      echo "--optimizer sgd";;
    rmsprop)  echo "--optimizer rmsprop";;
    lr0.001)  echo "--lr 0.001";;
    lr0.01)   echo "--lr 0.01";;
    batch256) echo "--batch 256";;
    *) echo "UNKNOWN"; return 1;;
  esac
}

mkdir -p logs_extend; J=./.hyp_jobs.txt; : > $J
# classical first: cheap, and validates every config before the long arm starts
for m in model2 qcpinn; do
  for c in $CONFIGS; do
    f=$(flags "$c") || { echo "bad config: $c"; exit 1; }
    for s in $SEEDS; do
      echo "hyp_${m}_${c}_s$s|--model $m --seed $s $f" >> $J
    done
  done
done
total=$(wc -l < $J | tr -d ' ')
echo "queued $total runs ($(echo $CONFIGS | wc -w | tr -d ' ') configs x $(echo $SEEDS | wc -w | tr -d ' ') seeds x 2 arms), $JOBS at a time"
export PY
tr '\n' '\0' < $J | xargs -0 -P "$JOBS" -I {} bash -c '
  tag="${1%%|*}"; args="${1#*|}"
  [ -f "runs/$tag/replication.json" ] && { echo "[skip] $tag"; exit 0; }
  t0=$(date +%s); log="logs_extend/$tag.log"
  "$PY" protocol5_hyper.py --tag "$tag" $args > "$log" 2>&1
  rc=$?; mins=$(( ($(date +%s)-t0)/60 ))
  if [ $rc -eq 0 ] && [ -f "runs/$tag/replication.json" ]; then
    echo "[$tag] PASS  $(grep -o "FINAL err_u = .*" "$log" | tail -1)  ${mins}m"
  else
    echo "[$tag] FAIL rc=$rc after ${mins}m -- $(tail -1 "$log")"
  fi
' _ {}
rm -f $J
echo "HYPER SWEEP queue drained -- every line above must say PASS or skip"
echo "complete: $(ls runs/hyp_*/replication.json 2>/dev/null | wc -l | tr -d ' ') of $total"
