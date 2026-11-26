#!/usr/bin/env bash
# symmetric_protocol4.sh -- symmetric protocol control, the run of record.
#
# Runs protocol4_qcpinn.py, which keeps protocol3's chunked circuit calls and
# after-Adam checkpointing but restores protocol2's loss summation order. That
# order is load-bearing: reversing it changed the classical baseline from 10.00%
# to 5.43% after Adam, through floating-point gradient accumulation alone (see
# runs/det_rep*, four replicates agreeing to six decimals).
#
# Because the Adam phase is now bit-identical to the protocol2 runs, every seed
# asserts its after-Adam error against the value protocol2 logged, so the
# salvaged quantum numbers are verified rather than assumed.
#
# Greps the FINAL line only and prints PASS/FAIL per run: protocol2's runner
# matched the after-Adam line too, and so reported ten crashed runs as finished.
#
# Both arms run, because the refinement batch is shared. Writes to runs/sym4_*.
# Resumable: a completed replication.json is skipped, and a run whose refinement
# failed can be finished with --resume off its checkpoint.
set -u
PY="${PY:-external/venv_qcpinn/bin/python}"; JOBS="${JOBS:-4}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ARMS="${ARMS:-model2 qcpinn}"

# after-Adam relative L2 error (%) recorded by the v2 runs; the Adam phase is
# unchanged, so each re-run must reproduce its own value.
expect() {
  case "$1_$2" in
    qcpinn_0) echo 2.91;;  qcpinn_1) echo 3.16;;  qcpinn_2) echo 4.36;;
    qcpinn_3) echo 4.68;;  qcpinn_4) echo 10.54;; qcpinn_5) echo 5.91;;
    qcpinn_6) echo 3.79;;  qcpinn_7) echo 4.25;;  qcpinn_8) echo 10.34;;
    qcpinn_9) echo 4.30;;
    model2_0) echo 10.00;; model2_1) echo 3.01;;  model2_2) echo 8.96;;
    model2_3) echo 7.18;;  model2_4) echo 6.73;;  model2_5) echo 101.23;;
    model2_6) echo 100.34;; model2_7) echo 4.44;; model2_8) echo 7.21;;
    model2_9) echo 100.00;;
    *) echo "";;
  esac
}

mkdir -p logs_extend; J=./.sym4_jobs.txt; : > $J
for s in $SEEDS; do
  for m in $ARMS; do
    e=$(expect "$m" "$s")
    chk=""; [ -n "$e" ] && chk="--check-against $e"
    echo "sym4_${m}_s$s|--model $m --seed $s $chk" >> $J
  done
done
export PY
tr '\n' '\0' < $J | xargs -0 -P "$JOBS" -I {} bash -c '
  tag="${1%%|*}"; args="${1#*|}"
  [ -f "runs/$tag/replication.json" ] && { echo "[skip] $tag"; exit 0; }
  t0=$(date +%s); log="logs_extend/$tag.log"
  "$PY" protocol4_qcpinn.py --tag "$tag" $args > "$log" 2>&1
  rc=$?; mins=$(( ($(date +%s)-t0)/60 ))
  if [ $rc -eq 0 ] && [ -f "runs/$tag/replication.json" ]; then
    echo "[$tag] PASS  $(grep -o "FINAL err_u = .*" "$log" | tail -1)  ${mins}m"
  else
    echo "[$tag] FAIL rc=$rc after ${mins}m -- $(tail -1 "$log")"
  fi
' _ {}
rm -f $J
echo "SYMMETRIC4 queue drained -- check every line above says PASS"
echo "complete: $(ls runs/sym4_*/replication.json 2>/dev/null | wc -l | tr -d ' ') of 20"
