#!/usr/bin/env bash
# THE PARAMETER-EFFICIENCY TEST.
#
# QCPINN's claim is not that quantum is more accurate. It is:
#
#     "comparable accuracy to classical PINNs while requiring up to 89%
#      fewer trainable parameters"
#
# That claim has never been checked with seed repetition or a convergence
# criterion -- the paper reports single runs with no error bars. This
# script checks it, on Burgers', with five seeds and our stopping rule.
#
# The three models, at the paper's exact parameter counts:
#
#     QCPINN angle-cascade      771 params   (15 quantum + 756 classical)
#     Classical Model-2       2,751 params   (3.6x more)
#     Classical Model-1       7,851 params   (10.2x more)
#
# WHAT EACH OUTCOME WOULD MEAN
#   - QCPINN matches Model-2 within seed spread -> the efficiency claim
#     holds under a controlled protocol, and that is a positive result
#     worth reporting as a confirmation with the rigour the original lacked.
#   - QCPINN matches Model-1 but not Model-2 -> the claim only holds against
#     an over-parameterised baseline. Model-2 is the fair comparison and the
#     paper's own tables already show it winning on two of three problems.
#   - QCPINN loses to both -> the efficiency claim does not survive seed
#     repetition on this PDE.
#
# All three are the paper's own architectures, trained with the paper's own
# optimiser settings. The only thing changed is how they are stopped and
# how many times they are run -- so no outcome here can be dismissed as an
# unfair reimplementation.
#
# Within each seed the two classical baselines run first: they take about
# two minutes each, so every seed yields a usable comparison quickly, and
# the slow quantum run is never blocking a result you could already see.
#
# Resumable; completed runs are skipped. Roughly 6-8 hours for all fifteen
# (the QCPINN runs dominate at ~70-100 min each; the baselines are minutes).
set -u

# ---------------------------------------------------------------------
# INTERPRETER RESOLUTION
# `python3` inside a script is not always the `python3` of the shell you
# typed in: a conda/venv activation lives in the interactive shell's
# environment and does not always survive into a script, a cron job, or a
# nohup. Rather than fail three hours in, resolve it up front, verify the
# imports actually work, and -- if they do not -- go and find an
# interpreter on this machine that does.
#
# Override explicitly with:   PY=/full/path/to/python ./qcpinn_compare.sh
# ---------------------------------------------------------------------
PY="${PY:-python3}"

if ! "$PY" -c "import pennylane, torch" >/dev/null 2>&1; then
  echo "'$PY' (= $(command -v "$PY" 2>/dev/null || echo 'not found')) cannot import pennylane and torch."
  echo
  echo "Searching this machine for one that can..."
  FOUND=""
  for p in "$HOME"/*/bin/python3 "$HOME"/*/bin/python \
           "$HOME"/miniconda3/envs/*/bin/python "$HOME"/anaconda3/envs/*/bin/python \
           "$HOME"/opt/anaconda3/envs/*/bin/python "$HOME"/miniforge3/envs/*/bin/python \
           "$HOME"/.conda/envs/*/bin/python \
           /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
    [ -x "$p" ] || continue
    if "$p" -c "import pennylane, torch" >/dev/null 2>&1; then
      echo "    PY=$p ./qcpinn_compare.sh"
      FOUND="$p"
    fi
  done
  if [ -z "$FOUND" ]; then
    echo "    (none found -- activate the environment you used for the earlier runs)"
  else
    echo
    echo "Re-run with one of the lines above, or activate that environment first."
  fi
  exit 1
fi

echo "interpreter: $(command -v "$PY")"
"$PY" -c "import pennylane, torch; print('             pennylane', pennylane.__version__, '| torch', torch.__version__)"
echo

if pgrep -f "robust_train.py|qcpinn_train.py" > /dev/null 2>&1; then
  echo "A training process is already running."
  echo "Override with: FORCE=1 ./qcpinn_compare.sh"
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

echo "=== reconstruction check (must match the paper before anything runs) ==="
"$PY" qcpinn_model.py || exit 1
echo

run_one () {
  local tag="$1"; shift
  if [ -f "runs/$tag/metrics.json" ]; then echo "[skip] $tag"; return; fi
  echo "[run ] $tag"
  "$PY" qcpinn_train.py --tag "$tag" "$@" || { echo "[FAIL] $tag"; return; }
  "$PY" evaluate.py "runs/$tag" --no-plots
}

for s in 0 1 2 3 4; do
  echo "=================== seed $s ==================="
  run_one "qc_model2_s$s" --model model2 --seed "$s"
  run_one "qc_model1_s$s" --model model1 --seed "$s"
  run_one "qc_qcpinn_s$s" --model qcpinn --seed "$s"
done

echo
"$PY" summarize_qcpinn.py
