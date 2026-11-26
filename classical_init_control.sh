#!/usr/bin/env bash
# THE MATCHED CONTROL. Closes the project's one live confound.
#
# WHAT IS CURRENTLY WRONG WITH THE RELIABILITY CLAIM
# --------------------------------------------------
# The QAPINN was given near-identity initialisation and its seed spread
# collapsed:
#
#     default init    2.42% mean, 14.11% std, 1/5 non-converged
#     near-identity   2.42% mean,  0.55% std, 0/5 non-converged
#
# On that basis the paper wants to say the quantum model is more reliable
# than the classical baseline (0.55% spread against 3.35%). But the
# classical baseline was never given ANY initialisation treatment. So the
# comparison is currently:
#
#     quantum model    + init tuning
#     classical model  - init tuning
#
# which measures the intervention, not the architecture. This is the same
# class of error the paper documents in other people's work, and a
# reviewer finds it in about ninety seconds.
#
# WHAT THIS RUNS
# --------------
# The classical baseline at initialisation gains 0.5 and 2.0, against the
# default (gain 1.0) runs already on disk as conv_classical_s*. Three
# points on the axis, five seeds each.
#
# EITHER OUTCOME IS PUBLISHABLE, WHICH IS WHY IT IS WORTH AN HOUR
#   variance drops    -> the confound is real. The reliability claim must be
#                        made against the tuned classical baseline, not the
#                        default one. Better to find this now.
#   variance holds    -> the classical baseline's spread is NOT reducible by
#                        init tuning, so the QAPINN's collapse is specific to
#                        the quantum layer. That is a considerably stronger
#                        claim than the one currently being made.
#
# HONEST CAVEAT, to be stated in the paper: this is not the identical
# intervention. "Near-identity" means something specific for a variational
# circuit -- start close to the identity operation -- with no exact
# classical analogue. Rescaling the default init is the closest honest
# equivalent. Do not imply the two treatments are the same thing.
#
# Classical runs take about two minutes each, so all ten finish inside an
# hour. Resumable; completed runs are skipped.
set -u

PY="${PY:-python3}"
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "'$PY' cannot import torch. Re-run as: PY=/usr/local/bin/python3 bash $0"
  exit 1
fi

if pgrep -f "robust_train.py|qcpinn_train.py|pde_train.py" > /dev/null 2>&1; then
  echo "A training process is already running. Override with FORCE=1."
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

run_one () {
  local tag="$1"; shift
  if [ -f "runs/$tag/metrics.json" ]; then echo "[skip] $tag"; return; fi
  echo "[run ] $tag"
  "$PY" pde_train.py --pde burgers --model classical --tag "$tag" "$@" \
    || echo "[FAIL] $tag"
}

for g in 0.5 2.0; do
  echo "=================== init gain $g ==================="
  for s in 0 1 2 3 4; do
    run_one "ctrl_classical_g${g}_s$s" --init-gain "$g" --seed "$s"
  done
done

echo
"$PY" summarize_init_control.py
