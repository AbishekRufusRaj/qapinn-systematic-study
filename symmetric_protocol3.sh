#!/usr/bin/env bash
# symmetric_protocol3.sh -- symmetric protocol control, corrected run.
#
# Changes from symmetric_protocol2.sh:
#   * runs protocol3_qcpinn.py: refinement on 1,400 points (the scale this
#     study's own protocol uses), every circuit call chunked at 500 to stay
#     inside the PennyLane 0.29 cachetools limit that killed the v2 quantum arm
#   * greps the FINAL line only. v2's pattern "err_u = .*" also matched the
#     after-Adam line, so crashed runs printed a plausible-looking number and
#     the queue reported success
#   * passes --check-against per seed: each run asserts its after-Adam error
#     reproduces the value the v2 run logged, which verifies both the salvaged
#     numbers and this script
#   * reports a per-run PASS/FAIL so a crash cannot be mistaken for a result
#
# Both arms are re-run, because the refinement batch changed and the two arms
# must share it. Writes to runs/sym3_*. Resumable.
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

mkdir -p logs_extend; J=./.sym3_jobs.txt; : > $J
for s in $SEEDS; do
  for m in $ARMS; do
    e=$(expect "$m" "$s")
    chk=""; [ -n "$e" ] && chk="--check-against $e"
    echo "sym3_${m}_s$s|--model $m --seed $s $chk" >> $J
  done
done
export PY
tr '\n' '\0' < $J | xargs -0 -P "$JOBS" -I {} bash -c '
  tag="${1%%|*}"; args="${1#*|}"
  [ -f "runs/$tag/replication.json" ] && { echo "[skip] $tag"; exit 0; }
  t0=$(date +%s); log="logs_extend/$tag.log"
  "$PY" protocol3_qcpinn.py --tag "$tag" $args > "$log" 2>&1
  rc=$?; mins=$(( ($(date +%s)-t0)/60 ))
  if [ $rc -eq 0 ] && [ -f "runs/$tag/replication.json" ]; then
    echo "[$tag] PASS  $(grep -o "FINAL err_u = .*" "$log" | tail -1)  ${mins}m"
  else
    echo "[$tag] FAIL rc=$rc after ${mins}m -- $(tail -1 "$log")"
  fi
' _ {}
rm -f $J
echo "SYMMETRIC3 queue drained -- check every line above says PASS"
echo "complete: $(ls runs/sym3_*/replication.json 2>/dev/null | wc -l | tr -d ' ') of 20"
