#!/usr/bin/env bash
# init_standard.sh -- the literature-standard initialisation arms.
#
# WHY THIS EXISTS
# ---------------
# `ClassicalPINN` inherits PyTorch's nn.Linear default: kaiming_uniform_(a=sqrt(5)),
# i.e. bound = 1/sqrt(fan_in). That is NOT the initialisation the PINN literature
# specifies. Raissi et al. and essentially every PINN paper since use Xavier/Glorot
# with tanh. So the study's "gain 1.0 default" arm is the PyTorch default, not the
# field's default, and a referee is entitled to call the 13.2x result a comparison
# against a strawman.
#
# For this architecture the 7 hidden 20x20 layers hold 2641 of 3021 parameters and
# therefore dominate. For them:
#
#     torch default std = (1/sqrt(20))/sqrt(3) = 0.12910
#     Xavier std        = sqrt(2/(20+20))      = 0.22361   -> gain sqrt(3) = 1.7321
#     Xavier x tanh 5/3                        = 0.37268   -> gain         = 2.8868
#
# Both fall outside the tested set {0.5, 1.0, 2.0} -- 1.7321 sits in the untested gap
# between 1.0 and 2.0, and 2.8868 is beyond the tested range entirely. The existing
# sweep improves monotonically toward 2.0, so the optimum may well sit at or past the
# value the literature actually prescribes, unmeasured.
#
# THESE ARMS ARE NOT MEMBERS OF THE DECLARED COMPARISON FAMILY. They were added after
# results were seen, and adding them to the Holm family would be precisely the practice
# this paper criticises. They are reported as a descriptive gain axis with effect sizes,
# explicitly labelled as a post-review baseline repair. Read section 7 before changing this.
#
#   PY=/usr/local/bin/python3 bash init_standard.sh
#   DRYRUN=1 ... to print the job list and exit
set -u

PY="${PY:-python3}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"     # n = 8, matching the extended continuous arms
GAINS="${GAINS:-1.7321 2.8868}"
JOBS="${JOBS:-3}"

"$PY" -c "import torch" 2>/dev/null || {
  echo "ABORT: '$PY' cannot import torch. Set PY to the interpreter that has it."; exit 1; }
[ -f _seed_worker.sh ] || { echo "ABORT: missing _seed_worker.sh"; exit 1; }

JOBFILE=".init_std_jobs.tsv"
: > "$JOBFILE"
total=0
for g in $GAINS; do
  for s in $SEEDS; do
    printf '%s\t%s\t%s\n' "ctrl_classical_g${g}_s$s" pde \
      "--pde burgers --model classical --init-gain $g --seed $s" >> "$JOBFILE"
    total=$((total + 1))
  done
done

echo "=================================================================="
echo " literature-standard initialisation arms"
echo " gains:        $GAINS   (sqrt(3)=Xavier, 2.8868=Xavier x tanh gain)"
echo " seeds:        $SEEDS"
echo " runs:         $total"
echo " concurrency:  $JOBS"
echo "=================================================================="
echo

if [ "${DRYRUN:-0}" = "1" ]; then
  echo "--- DRYRUN: job list, nothing launched ---"
  cat -t "$JOBFILE"
  rm -f "$JOBFILE"
  exit 0
fi

# BSD xargs converts tabs to spaces inside the -I replacement string; NUL-delimited
# records with -0 suppress that. See the note in qcpinn_converge.sh -- DRYRUN prints
# the job list before this line, so a dry run cannot catch a regression here.
tr '\n' '\0' < "$JOBFILE" \
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
echo "Next:  \$PY post_review_analysis.py --part init"
