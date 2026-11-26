#!/usr/bin/env bash
# TRAINABILITY TEST: does near-identity initialisation rescue the failed run?
#
# WHAT THIS IS TESTING
# --------------------
# The convergence sweep produced one qualitative failure. conv_strong_L2_s2
# never broke through its plateau:
#
#     conv_strong_L2_s2   0.457 -> 0.379 -> 0.306 -> 0.261 -> 0.246
#     every other run     drops 2-3 orders of magnitude
#
# It finished at loss 2.15e-01 and 33.86% error, against 1.5e-04 to 6.5e-03
# and 1.2-13.4% for the other fourteen runs. L-BFGS could not rescue it
# either. That is not a bad tail of a smooth distribution, it is a different
# event: the run never left the plateau at all.
#
# The barren-plateau literature predicts exactly this, and predicts a fix.
# McClean et al. (2018) show gradient variance decaying exponentially with
# circuit width; Grant et al. (2019) show that initialising the circuit NEAR
# THE IDENTITY, rather than uniformly on [0, 2*pi), gives a better-conditioned
# landscape. `--init-scale 0.1` draws the variational weights from N(0, 0.1)
# instead of PennyLane's default uniform, which is that mitigation.
#
# So this is a falsifiable prediction, not a fishing expedition:
#   - if seed 2 now converges, the failure was CAUSALLY an initialisation
#     event, and the standard mitigation fixes it. That is the trainability
#     half of the project demonstrated end to end: diagnostic -> mechanism
#     -> fix that works.
#   - if seed 2 still fails, the failure is NOT about initialisation, and
#     the barren-plateau reading has to be dropped. That is also worth
#     knowing, and is worth knowing before it goes in a paper.
#
# SEED 2 RUNS FIRST, DELIBERATELY
# -------------------------------
# The decisive answer arrives in the first run (~30 min) rather than after
# the whole sweep. If seed 2 does not recover, kill the script -- the other
# four seeds are then measuring a hypothesis that has already failed. If it
# does recover, let it run: the remaining seeds tell you whether the fix
# lifts the whole distribution or only rescues the pathological case, and
# those are different claims.
#
# The strong ansatz is the one under test because it is the one that failed:
# 24 quantum parameters to ring's 8, which is also why the plateau theory
# points at it.
#
# Writes only to new `init_*` run tags, so nothing already on disk moves.
# Resumable; completed runs are skipped. Roughly 2.5-3 hours for all five.
set -u

if pgrep -f "robust_train.py" > /dev/null 2>&1; then
  echo "A training process is still running (robust_train.py)."
  echo "Let it finish first, then re-run this script."
  echo "Override with: FORCE=1 ./init_scale_test.sh"
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

INIT_SCALE="${INIT_SCALE:-0.1}"
echo "near-identity init scale = ${INIT_SCALE}"
echo

run_one () {
  local tag="$1"; shift
  if [ -f "runs/$tag/metrics.json" ]; then echo "[skip] $tag"; return; fi
  echo "[run ] $tag"
  python3 robust_train.py --tag "$tag" "$@" || { echo "[FAIL] $tag"; return; }
  python3 evaluate.py "runs/$tag" --no-plots
}

# seed 2 first -- it is the one that failed, and it decides the hypothesis
for s in 2 0 1 3 4; do
  echo "=================== seed $s ==================="
  run_one "init_strong_L2_s$s" --qubits 4 --reupload 2 --entangler strong \
                               --seed "$s" --init-scale "$INIT_SCALE"

  if [ "$s" = "2" ]; then
    echo
    echo "--- rescue verdict for the previously-failed seed 2 ---"
    python3 - <<'PY'
import json, os

OLD, NEW = "runs/conv_strong_L2_s2", "runs/init_strong_L2_s2"
FAIL_LOSS = 1e-2      # pre-registered: final loss above this = non-converged

def read(d):
    if not os.path.exists(os.path.join(d, "metrics.json")):
        return None
    m = json.load(open(os.path.join(d, "metrics.json")))
    t = json.load(open(os.path.join(d, "meta.json")))
    return m["rel_l2"], t["final_total_loss"], t.get("adam_epochs_used")

old, new = read(OLD), read(NEW)
if not new:
    print("  seed 2 produced no metrics -- the run failed to complete.")
    raise SystemExit

print(f"  default init  : loss {old[1]:.3e}   err {old[0]*100:6.2f}%   {old[2]} epochs"
      if old else "  default init  : (conv_strong_L2_s2 not on disk)")
print(f"  near-identity : loss {new[1]:.3e}   err {new[0]*100:6.2f}%   {new[2]} epochs")
print()
if new[1] < FAIL_LOSS:
    print("  RESCUED. The run that never left the plateau now converges.")
    print("  The failure was an initialisation event, and the standard")
    print("  barren-plateau mitigation fixes it. Let the other seeds run --")
    print("  they say whether this lifts the whole distribution or only")
    print("  rescues the pathological case.")
else:
    print("  NOT RESCUED. Still above the 1e-2 non-convergence threshold.")
    print("  The failure is not about initialisation. Drop the barren-plateau")
    print("  reading of this run and stop the script (Ctrl-C) -- the remaining")
    print("  seeds are testing a hypothesis that has already failed.")
PY
    echo "------------------------------------------------------"
    echo
  fi
done

echo
python3 summarize_init_test.py
