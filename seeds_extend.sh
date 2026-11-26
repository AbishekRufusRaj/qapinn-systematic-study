#!/usr/bin/env bash
# seeds_extend.sh -- take every continuous-comparison arm from n = 5 to n = 8.
#
# WHY n = 8 AND NOT 50
# --------------------
# floor_analysis.py computes, by enumeration, the smallest two-sided p an exact
# permutation test can return at each n:
#
#     n      difference in medians      difference in means
#     5      12/252  = 0.0476           2/252   = 0.0079
#     6      12/924  = 0.0130           2/924   = 0.0022
#     8      40/12870 = 0.0031          2/12870 = 0.0002
#
# Holm over twelve declared comparisons needs 0.05/12 = 0.0042 for the most
# significant test. The mean statistic clears that at n = 6; the median
# statistic -- which is the one stats_analysis.py actually uses -- does not
# clear it until n = 8. Going to 8 means the result cannot be attributed to
# having picked the statistic that suited us.
#
# The n = 50 figure in the paper is a POWER calculation for the binary
# failure-rate claims (Fisher's exact on 2x2 counts) and does not govern these
# comparisons at all. Conflating the two overstated the cost of a confirmatory
# study roughly tenfold.
#
# A floor is necessary, not sufficient: reaching it requires the observed
# separation to be complete. n = 8 buys the possibility of significance.
#
# WHAT THIS RUNS
# --------------
# Seeds 5, 6, 7 for the ten arms feeding H1-H7 and H11-H12, using the SAME
# trainer, the same flags and the same tag scheme as seeds 0-4, so the new
# seeds are drawn from the identical protocol. The QCPINN arms (H8-H10) are
# deliberately excluded: they need the convergence control in
# qcpinn_converge.sh first, because comparing runs that never solved the PDE
# measures nothing.
#
# PARALLELISM
# -----------
# Three concurrent processes. This is the only speedup applied to runs that
# enter the statistics, and it is applied deliberately: running two processes
# side by side does not change either one's arithmetic. Thread pinning
# (torch.set_num_threads(1), ~1.15x) and circuit-call fusion (~1.22x) DO alter
# floating-point reduction order, so mixing them into arms whose other seeds
# were run without them would introduce exactly the kind of silent confound
# this paper is about. They are measured in section 10.5 and kept out of here.
#
# Resumable: any run whose metrics.json exists is skipped, so interrupting this
# and restarting it costs nothing.
#
# Expected wall-clock at JOBS=3: about 2.5 hours.
#   classical arms   ~2 min/run   x 9 runs
#   Burgers quantum  ~32 min/run  x 12 runs
#   heat arms        ~1-10 min    x 9 runs

set -u

SEEDS="${SEEDS:-5 6 7}"
JOBS="${JOBS:-3}"
PY="${PY:-python3}"
export PY

# ---------------------------------------------------------------- preflight
if [ "${DRYRUN:-0}" != "1" ] && ! "$PY" -c "import pennylane, torch" >/dev/null 2>&1; then
  echo "'$PY' cannot import pennylane and torch."
  echo
  echo "Find the right interpreter and re-run, e.g.:"
  for c in /usr/local/bin/python3 /opt/homebrew/bin/python3 "$(command -v python3)"; do
    [ -x "$c" ] || continue
    if "$c" -c "import pennylane, torch" >/dev/null 2>&1; then
      echo "  PY=$c bash seeds_extend.sh"
    fi
  done
  exit 1
fi

if pgrep -f "robust_train.py|qcpinn_train.py|pde_train.py" >/dev/null 2>&1; then
  echo "A training process is already running. Override with FORCE=1."
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

for f in _seed_worker.sh robust_train.py pde_train.py evaluate.py; do
  [ -f "$f" ] || { echo "missing $f -- run this from the study folder"; exit 1; }
done

mkdir -p logs_extend
JOBFILE="./.extend_jobs.tsv"
: > "$JOBFILE"

emit () {  # tag  trainer  args...
  local tag="$1" trainer="$2"; shift 2
  printf '%s\t%s\t%s\n' "$tag" "$trainer" "$*" >> "$JOBFILE"
}

# ---------------------------------------------------------------- the arms
# Flags copied verbatim from the scripts that produced seeds 0-4:
#   conv_*  <- conv_seed_check.sh   (robust_train.py)
#   init_*  <- init_scale_test.sh   (robust_train.py, --init-scale 0.1)
#   ctrl_*  <- classical_init_control.sh (pde_train.py --pde burgers)
#   heat_*  <- heat_sweep.sh        (pde_train.py --pde heat)
for s in $SEEDS; do
  # --- Burgers', convergence-trained (H1, H3-H7) ---
  emit "conv_classical_s$s"  robust --classical --seed "$s"
  emit "conv_ring_L2_s$s"    robust --qubits 4 --reupload 2 --entangler ring   --seed "$s"
  emit "conv_strong_L2_s$s"  robust --qubits 4 --reupload 2 --entangler strong --seed "$s"

  # --- near-identity initialisation arms (H3, H4, H6, H7) ---
  emit "init_ring_L2_s$s"    robust --qubits 4 --reupload 2 --entangler ring \
                                    --init-scale 0.1 --seed "$s"
  emit "init_strong_L2_s$s"  robust --qubits 4 --reupload 2 --entangler strong \
                                    --init-scale 0.1 --seed "$s"

  # --- classical initialisation control (H1, H2) ---
  emit "ctrl_classical_g0.5_s$s" pde --pde burgers --model classical \
                                     --init-gain 0.5 --seed "$s"
  emit "ctrl_classical_g2.0_s$s" pde --pde burgers --model classical \
                                     --init-gain 2.0 --seed "$s"

  # --- heat (H11, H12) ---
  emit "heat_classical_s$s"  pde --pde heat --model classical --seed "$s"
  emit "heat_ring_L2_s$s"    pde --pde heat --model qapinn --qubits 4 --reupload 2 \
                                 --entangler ring --seed "$s"
  emit "heat_strong_L2_s$s"  pde --pde heat --model qapinn --qubits 4 --reupload 2 \
                                 --entangler strong --init-scale 0.1 --seed "$s"
done

total=$(wc -l < "$JOBFILE" | tr -d ' ')
todo=0
while IFS=$'\t' read -r tag _ _; do
  [ -f "runs/$tag/metrics.json" ] || todo=$((todo + 1))
done < "$JOBFILE"

echo "=================================================================="
echo " seeds:        $SEEDS"
echo " jobs total:   $total   (already complete: $((total - todo)))"
echo " to run:       $todo"
echo " concurrency:  $JOBS"
echo " logs:         logs_extend/<tag>.log"
echo "=================================================================="
echo

# Sort longest-first so the 32-minute quantum runs start before the 2-minute
# classical ones; otherwise the tail of the queue is one long job running alone.
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
sort -t$'\t' -k2,2r "$JOBFILE" \
  | tr '\n' '\0' \
  | xargs -0 -P "$JOBS" -I {} bash _seed_worker.sh "{}"

echo
echo "=================================================================="
failed=0
while IFS=$'\t' read -r tag _ _; do
  if [ ! -f "runs/$tag/metrics.json" ]; then
    echo "  MISSING: $tag"
    failed=$((failed + 1))
  fi
done < "$JOBFILE"

if [ "$failed" -gt 0 ]; then
  echo
  echo "$failed run(s) did not produce metrics.json. Check logs_extend/,"
  echo "then simply re-run this script -- completed runs are skipped."
else
  echo "  all $total runs present."
fi

rm -f "$JOBFILE" 2>/dev/null || true

echo
echo "Next:"
echo "  $PY stats_analysis.py          # re-run the twelve comparisons at n = 8"
echo "  $PY floor_analysis.py          # floors and both statistics"
