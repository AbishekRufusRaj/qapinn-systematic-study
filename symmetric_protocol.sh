#!/usr/bin/env bash
# symmetric_protocol.sh -- both arms of the published comparison under the
# convergence protocol, inside the authors' own code. Answers the asymmetry
# raised by reviewers 4 and 5. Ten seeds per arm; resumable.
set -u
PY="${PY:-external/venv_qcpinn/bin/python}"; JOBS="${JOBS:-4}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
mkdir -p logs_extend; J=./.sym_jobs.txt; : > $J
for s in $SEEDS; do
  echo "sym_model2_s$s|--model model2 --seed $s"  >> $J
  echo "sym_qcpinn_s$s|--model qcpinn --seed $s"  >> $J
done
export PY
tr '\n' '\0' < $J | xargs -0 -P "$JOBS" -I {} bash -c '
  tag="${1%%|*}"; args="${1#*|}"
  [ -f "runs/$tag/replication.json" ] && { echo "[skip] $tag"; exit 0; }
  t0=$(date +%s)
  "$PY" protocol_qcpinn.py --tag "$tag" $args > "logs_extend/$tag.log" 2>&1
  echo "[$tag] $(grep -o "err_u = [0-9.]*%" logs_extend/$tag.log | tail -1)  $(( ($(date +%s)-t0)/60 ))m"
' _ {}
rm -f $J; echo "SYMMETRIC finished"
