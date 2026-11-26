#!/usr/bin/env bash
# Worker invoked once per job by seeds_extend.sh via xargs -P.
#
# Receives ONE tab-separated line:   TAG <TAB> TRAINER <TAB> ARGS
#
# It exists as a separate file because xargs -I passes the whole line as a
# single argument, and doing the field splitting inside a `bash -c` string
# would mean quoting the arguments twice.
set -u

PY="${PY:-python3}"

line="$1"
IFS=$'\t' read -r tag trainer args <<< "$line"

log="logs_extend/$tag.log"
mkdir -p logs_extend

if [ -f "runs/$tag/metrics.json" ]; then
  echo "[skip] $tag"
  exit 0
fi

echo "[run ] $tag  ($trainer)"
start=$(date +%s)

# shellcheck disable=SC2086   # word splitting of $args is intended
case "$trainer" in
  robust)
    "$PY" robust_train.py --tag "$tag" $args > "$log" 2>&1 \
      && "$PY" evaluate.py "runs/$tag" --no-plots >> "$log" 2>&1
    ;;
  pde)
    "$PY" pde_train.py --tag "$tag" $args > "$log" 2>&1
    ;;
  qcpinn)
    "$PY" qcpinn_train.py --tag "$tag" $args > "$log" 2>&1 \
      && "$PY" evaluate.py "runs/$tag" --no-plots >> "$log" 2>&1
    ;;
  *)
    echo "[FAIL] $tag -- unknown trainer '$trainer'"; exit 1
    ;;
esac

rc=$?
mins=$(( ( $(date +%s) - start ) / 60 ))

if [ $rc -ne 0 ] || [ ! -f "runs/$tag/metrics.json" ]; then
  echo "[FAIL] $tag after ${mins}m -- see $log"
  exit 1
fi

echo "[done] $tag  ${mins}m"
