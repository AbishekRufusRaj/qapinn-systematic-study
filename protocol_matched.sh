#!/usr/bin/env bash
# protocol_matched.sh -- answers "the comparison is not protocol-matched", both ways.
#   pm_model2_ours      the QCPINN paper's classical baseline under THIS protocol
#   pm_classical_theirs this project's classical PINN under THEIR protocol
#   pm_xavier_theirs    the same with the Xavier-equivalent gain
# All on Helmholtz, 8 seeds each. Resumable.
set -u
PY="${PY:-/usr/local/bin/python3}"; JOBS="${JOBS:-2}"
mkdir -p logs_extend; J=./.pm_jobs.txt; : > $J
for s in 0 1 2 3 4 5 6 7; do
  echo "pm_model2_ours_s$s|--model model2 --seed $s"                                >> $J
  echo "pm_classical_theirs_s$s|--model classical --their-recipe --seed $s"         >> $J
  echo "pm_xavier_theirs_s$s|--model classical --init-gain 1.7321 --their-recipe --seed $s" >> $J
done
export PY
tr '\n' '\0' < $J | xargs -0 -P "$JOBS" -I {} bash -c '
  tag="${1%%|*}"; args="${1#*|}"
  [ -f "runs/$tag/metrics.json" ] && { echo "[skip] $tag"; exit 0; }
  "$PY" helmholtz_ours.py --tag "$tag" $args > "logs_extend/$tag.log" 2>&1
  echo "[$tag] $(grep -o "rel L2 *[0-9.]*%" logs_extend/$tag.log | tail -1)"
' _ {}
rm -f $J; echo "PM finished"
