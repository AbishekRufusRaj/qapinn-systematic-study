#!/usr/bin/env bash
# qcpinn_converge.sh -- the missing control for H8-H10.
#
# THE DEFECT THIS FIXES
# ---------------------
# The QCPINN replication ran under the study's standard cap of 25,000 epochs.
# Two of five QCPINN seeds finished above the pre-registered non-convergence
# threshold of 1e-2, i.e. they never solved the PDE. The parameter-efficiency
# comparison (H8, H9) was then drawn across those runs.
#
# That comparison has no measurand. "Model A reaches a worse error than model B
# at matched parameters" means something only if both models were run until
# their error stopped improving; otherwise it reports where two non-solutions
# happened to be interrupted. A caveat in the text does not repair this,
# because the quantity being compared is undefined, not merely uncertain.
#
# WHAT CHANGES, AND WHAT DELIBERATELY DOES NOT
# --------------------------------------------
# ONLY the cap. Patience (3,000), tolerance (3%) and the EMA convergence
# criterion are identical to every other run in the study, so "converged" means
# the same thing here as everywhere else. Raising the cap to 80,000 asks
# whether these architectures converge slowly or not at all -- which is the
# question -- rather than changing the definition of convergence to get a
# friendlier answer.
#
# New tags (qcc_*) rather than overwriting qc_*. The truncated runs stay on
# disk and stay in the paper as a reproduction note: under the protocol these
# architectures were published with, they did not converge. That is itself a
# finding, and it is a different finding from the efficiency comparison.
#
# THE THREE POSSIBLE OUTCOMES, DECLARED BEFORE RUNNING
# ----------------------------------------------------
#   1. All arms converge. H8/H9 are re-run on solved models and the efficiency
#      claim is measured properly for the first time.
#   2. QCPINN converges, baselines already had. Same as 1, and the 25,000 cap
#      is shown to have been the binding constraint.
#   3. QCPINN still does not converge at 80,000. Then H8/H9 are WITHDRAWN from
#      the comparison family -- not reported with a caveat -- and replaced by a
#      statement about convergence failure, which needs no efficiency claim to
#      be interesting. The declared family drops from twelve to ten, which
#      LOOSENS the Holm threshold to 0.05/10 = 0.0050.
#
# Outcome 3 is not a failure of this control; it is the control working.
#
# COST
# ----
# QCPINN runs took 52.9 min at the 25,000 cap. Worst case here is 3.2x that,
# so budget up to ~2.8 h per run if nothing converges early; runs that DO
# converge stop on the criterion and cost no more than before. Fifteen runs,
# three at a time: overnight.
#
# Resumable. Run this BEFORE seeds_extend.sh if you only have one machine --
# it decides whether H8-H10 are still in the family, and the family size sets
# the Holm threshold that seeds_extend.sh is targeting.

set -u

SEEDS="${SEEDS:-0 1 2 3 4}"
JOBS="${JOBS:-3}"
MAXEPOCHS="${MAXEPOCHS:-80000}"
PY="${PY:-python3}"
export PY

# ---------------------------------------------------------------- preflight
if [ "${DRYRUN:-0}" != "1" ] && ! "$PY" -c "import pennylane, torch" >/dev/null 2>&1; then
  echo "'$PY' cannot import pennylane and torch."
  for c in /usr/local/bin/python3 /opt/homebrew/bin/python3; do
    [ -x "$c" ] && "$c" -c "import pennylane, torch" >/dev/null 2>&1 \
      && echo "  PY=$c bash qcpinn_converge.sh"
  done
  exit 1
fi

if pgrep -f "robust_train.py|qcpinn_train.py|pde_train.py" >/dev/null 2>&1; then
  echo "A training process is already running. Override with FORCE=1."
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

[ -f _seed_worker.sh ] || { echo "missing _seed_worker.sh"; exit 1; }

if [ "${DRYRUN:-0}" != "1" ]; then
  echo "=== reconstruction check (parameter counts must still match the paper) ==="
  "$PY" qcpinn_model.py || exit 1
fi
echo

mkdir -p logs_extend
JOBFILE="./.qcc_jobs.tsv"
: > "$JOBFILE"

for s in $SEEDS; do
  for m in qcpinn model1 model2; do
    printf '%s\t%s\t%s\n' "qcc_${m}_s$s" qcpinn \
      "--model $m --max-epochs $MAXEPOCHS --seed $s" >> "$JOBFILE"
  done
done

total=$(wc -l < "$JOBFILE" | tr -d ' ')
echo "=================================================================="
echo " cap:          $MAXEPOCHS epochs (patience and tolerance unchanged)"
echo " seeds:        $SEEDS"
echo " runs:         $total"
echo " concurrency:  $JOBS"
echo "=================================================================="
echo

if [ "${DRYRUN:-0}" = "1" ]; then
  echo "--- DRYRUN: job list, nothing launched ---"
  cat -t "$JOBFILE"
  rm -f "$JOBFILE" 2>/dev/null || true
  exit 0
fi

# BSD xargs (macOS) converts TABS TO SPACES inside the -I replacement string,
# so the worker received the whole line as $1 and split out an empty trainer.
# Feeding NUL-delimited records with -0 suppresses that blank processing and
# the tabs survive. Do not revert to a bare `xargs -I {} ... < "$JOBFILE"`:
# DRYRUN prints the job list BEFORE this point, so a dry run cannot catch it.
tr '\n' '\0' < "$JOBFILE" \
  | xargs -0 -P "$JOBS" -I {} bash _seed_worker.sh "{}"

# ---------------------------------------------------------------- verdict
echo
echo "=================================================================="
echo " convergence verdict (pre-registered threshold: final loss > 1e-2"
echo " counts as non-converged)"
echo "=================================================================="
"$PY" - "$MAXEPOCHS" <<'PY'
import json, os, sys, glob

MAXEP = int(sys.argv[1])
FAIL_LOSS = 1e-2

def read(d):
    p = os.path.join(d, "metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    mp = os.path.join(d, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    return m, meta

print(f"\n{'model':10}{'seed':>5}{'  cap':>8}{'final loss':>13}"
      f"{'rel L2':>10}{'epochs':>9}  status")
print("-" * 68)

summary = {}
for model in ("qcpinn", "model1", "model2"):
    for old_new, prefix in (("25k", "qc_"), (f"{MAXEP//1000}k", "qcc_")):
        for s in range(8):
            d = f"runs/{prefix}{model}_s{s}"
            r = read(d)
            if r is None:
                continue
            m, meta = r
            # The loss key is final_total_loss, in BOTH meta.json and
            # metrics.json -- NOT "final_loss"/"loss", and the epoch key is
            # adam_epochs_used, NOT "epochs_run". Reading the wrong key gave
            # NaN, and `NaN > FAIL_LOSS` is False, so every run printed as
            # "converged" and the verdict read 0/5 when the truth was 2/5.
            # A missing or NaN loss is therefore counted as NON-converged and
            # printed as UNKNOWN: this verdict decides a pre-registered rule,
            # so it must fail loud rather than fail permissive.
            loss = meta.get("final_total_loss", m.get("final_total_loss"))
            loss = float(loss) if loss is not None else float("nan")
            rel = m.get("rel_l2", float("nan"))
            ep = meta.get("adam_epochs_used", m.get("adam_epochs_used", "?"))
            hit_cap = isinstance(ep, int) and ep >= MAXEP - 1
            unknown = loss != loss
            bad = True if unknown else (loss > FAIL_LOSS)
            status = "UNKNOWN (no loss on disk)" if unknown else (
                "NOT CONVERGED" if bad else "converged")
            if hit_cap:
                status += " (hit cap)"
            print(f"{model:10}{s:>5}{old_new:>8}{loss:>13.5f}"
                  f"{rel*100 if rel == rel else float('nan'):>9.2f}%{str(ep):>9}  {status}")
            summary.setdefault((model, old_new), []).append(bad)

print()
for (model, cap), bads in sorted(summary.items()):
    n = len(bads); f = sum(bads)
    print(f"  {model:10} cap {cap:>4}:  {f}/{n} non-converged")

print("""
READ THIS AS:
  If qcpinn is now 0/5 non-converged, H8/H9 can be re-measured on solved
  models -- re-point CONFIGS in stats_analysis.py at the qcc_* tags.
  If it is still non-zero, WITHDRAW H8/H9 from the declared family (twelve
  comparisons become ten, and the Holm threshold loosens to 0.0050) and
  report the convergence failure instead. Do not report an efficiency
  comparison across non-solutions with a caveat attached.
""")
PY

rm -f "$JOBFILE" 2>/dev/null || true
