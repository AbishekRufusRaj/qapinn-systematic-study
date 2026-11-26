#!/usr/bin/env bash
# qcpinn_xavier.sh -- QCPINN replication with the authors' initialisation.
# Identical to qcpinn_converge.sh (80,000-epoch cap, same criterion, seeds 0-4)
# except the trainer, which applies the init from github.com/afrah/QCPINN.
# New tags qcx_*; qc_* and qcc_* are not touched. Resumable.
set -u
PY="${PY:-/usr/local/bin/python3}"; JOBS="${JOBS:-5}"; SEEDS="${SEEDS:-0 1 2 3 4}"
export PY
mkdir -p logs_extend
JOBFILE=./.qcx_jobs.txt; : > "$JOBFILE"
for m in qcpinn model1 model2; do for s in $SEEDS; do echo "qcx_${m}_s$s $m $s" >> "$JOBFILE"; done; done
work() {
  set -- $1; tag=$1; m=$2; s=$3
  [ -f "runs/$tag/metrics.json" ] && { echo "[skip] $tag"; return 0; }
  echo "[run ] $tag"; t0=$(date +%s)
  "$PY" qcpinn_train_xavier.py --model "$m" --seed "$s" --tag "$tag" --max-epochs 80000 > "logs_extend/$tag.log" 2>&1 \
    && "$PY" evaluate.py "runs/$tag" --no-plots >> "logs_extend/$tag.log" 2>&1
  rc=$?; echo "[$([ $rc = 0 ] && echo done || echo FAIL)] $tag after $(( ($(date +%s)-t0)/60 ))m"
}
export -f work
tr '\n' '\0' < "$JOBFILE" | xargs -0 -P "$JOBS" -I {} bash -c 'work "$1"' _ {}
rm -f "$JOBFILE"
echo "QCX exit=done"
