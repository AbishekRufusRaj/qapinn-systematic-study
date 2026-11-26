#!/usr/bin/env bash
# depth_ring.sh -- the RING half of the L = 4 depth experiment.
#
# WHY THIS EXISTS SEPARATELY FROM depth_robustness.sh
# ---------------------------------------------------
# depth_robustness.sh runs the STRONG ansatz at L = 4 only. That answers "is the
# near-identity effect depth-specific on the ansatz where it helps" -- a robustness
# check. It does NOT test Proposition 2, because Proposition 2 is about the DIFFERENCE
# between ring and strong, and section 5.2's own table says the two have identical
# measured spectral reach at L = 2:
#
#     L    ring cutoff   strong cutoff
#     2         4              4          <- every statistical arm in the paper
#     4         6              8          <- where reach actually diverges
#
# So the interaction observed at L = 2 cannot be a frequency-reach effect. Running the
# ring arm at L = 4 completes the 2 x 2 at BOTH equal reach and divergent reach, which
# is a direct experimental test of the mechanism rather than a robustness check.
#
#   interaction changes character between L=2 and L=4  -> Proposition 2 confirmed
#   interaction unchanged                              -> the expressivity mechanism is
#                                                         NOT what drives the finding,
#                                                         and the paper must say so
#
# The second outcome argues against the paper's own explanation, which is why it is
# worth running.
#
# WHY THIS EXISTS -- IT ANSWERS THE SHARPEST CRITIQUE OF THE PAPER
# ---------------------------------------------------------------
# `sweep_results.csv` is ONE RUN per configuration. L = 2 was chosen because it won
# that sweep:
#
#     ring    L1 0.283   L2 0.195   L4 0.306   L6 0.270   L8 0.376
#     strong             L2 0.0438  L4 0.316   L6 0.127
#
# Every statistical arm in the paper then runs at L = 2. But `conv_strong_L2_s*` over
# five seeds has median 0.0237, mean 0.0870 and worst 0.3386 -- so the single 0.0438
# that won the sweep sits inside a distribution spanning 0.02 to 0.34, and L4's single
# 0.316 could equally be a bad draw. Selecting the architecture on one seed is exactly
# the failure mode section 8.3 documents, applied to the choice the whole study rests on.
#
# This does not re-run the sweep at n = 5 (about 50 runs, and strong-L6 alone was
# 5,681 s single-seed). It answers the criticism that actually matters: does the
# near-identity initialisation effect survive a change of depth? Strong ansatz, L = 4,
# both initialisations, 8 seeds.
#
#   interaction holds at L=4  -> the finding is not depth-specific; C6 is answered
#   interaction vanishes      -> the finding IS depth-specific and must be restated
#                                as conditional on L = 2. That is a real limit on the
#                                claim and reporting it is the point of running this.
#
# NOT MEMBERS OF THE DECLARED COMPARISON FAMILY -- a post-review robustness check,
# reported separately with effect sizes. See section 7 before changing that.
#
#   PY=/usr/local/bin/python3 bash depth_robustness.sh
#   DRYRUN=1 ... to print the job list and exit
set -u

PY="${PY:-python3}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
DEPTH="${DEPTH:-4}"
ANSATZ="${ANSATZ:-ring}"
JOBS="${JOBS:-3}"

"$PY" -c "import pennylane, torch" 2>/dev/null || {
  echo "ABORT: '$PY' cannot import pennylane+torch. Set PY to the right interpreter."; exit 1; }
[ -f _seed_worker.sh ] || { echo "ABORT: missing _seed_worker.sh"; exit 1; }

JOBFILE=".depth_${ANSATZ}_jobs.tsv"
: > "$JOBFILE"
total=0
for s in $SEEDS; do
  # default init -- flags copied verbatim from conv_seed_check.sh, depth changed
  printf '%s\t%s\t%s\n' "conv_${ANSATZ}_L${DEPTH}_s$s" robust \
    "--qubits 4 --reupload $DEPTH --entangler $ANSATZ --seed $s" >> "$JOBFILE"
  # near-identity -- flags copied verbatim from init_scale_test.sh, depth changed
  printf '%s\t%s\t%s\n' "init_${ANSATZ}_L${DEPTH}_s$s" robust \
    "--qubits 4 --reupload $DEPTH --entangler $ANSATZ --init-scale 0.1 --seed $s" >> "$JOBFILE"
  total=$((total + 2))
done

echo "=================================================================="
echo " depth robustness of the architecture x initialisation interaction"
echo " depth:        L = $DEPTH   (the study's arms run at L = 2)"
echo " arms:         conv_strong_L${DEPTH} (default init), init_strong_L${DEPTH} (near-identity)"
echo " seeds:        $SEEDS"
echo " runs:         $total"
echo " concurrency:  $JOBS"
echo " estimate:     strong-L2 averaged 30 min/run; L=$DEPTH is deeper, so budget ~5-7 h"
echo "=================================================================="
echo

if [ "${DRYRUN:-0}" = "1" ]; then
  echo "--- DRYRUN: job list, nothing launched ---"
  cat -t "$JOBFILE"
  rm -f "$JOBFILE"
  exit 0
fi

# NUL-delimited records: BSD xargs turns tabs into spaces inside -I. Do not revert.
sort -t$'\t' -k1,1 "$JOBFILE" \
  | tr '\n' '\0' \
  | xargs -0 -P "$JOBS" -I {} bash _seed_worker.sh "{}"

echo
echo "=================================================================="
missing=0
while IFS=$'\t' read -r tag _ _; do
  [ -f "runs/$tag/metrics.json" ] || { echo "  MISSING: $tag"; missing=$((missing+1)); }
done < "$JOBFILE"
[ "$missing" -eq 0 ] && echo "  all $total runs produced metrics.json" \
                     || echo "  $missing run(s) missing -- re-run; completed runs are skipped"
rm -f "$JOBFILE"
echo
echo "Next:  \$PY post_review_analysis.py --part depth"
