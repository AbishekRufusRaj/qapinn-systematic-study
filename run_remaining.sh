#!/usr/bin/env bash
# =====================================================================
# Everything left to run for the QAPINN paper, in dependency order.
#
#   cd qapinn_systematic_study
#   ./run_remaining.sh              # steps 1-3 (required, ~10 min)
#   ./run_remaining.sh --with-global # also step 4 (~20-40 min)
#
# Every step is idempotent: re-running skips work that is already on
# disk, so a failure part-way through is recovered by running it again.
# Self-tests run BEFORE the expensive steps, on purpose -- a rebuilt
# circuit or a projector that is subtly wrong should fail in seconds,
# not after forty minutes of sampling.
#
# PY defaults to the interpreter that has torch 2.10 + pennylane 0.44.
# Plain `python3` on this machine is homebrew and has neither.
# =====================================================================
set -euo pipefail

PY="${PY:-/usr/local/bin/python3}"
WITH_GLOBAL=0
[[ "${1:-}" == "--with-global" ]] && WITH_GLOBAL=1

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
fail()   { printf '\n\033[31mFAILED at: %s\033[0m\n' "$*" >&2; exit 1; }

banner "0  environment check"
"$PY" - <<'EOF' || { echo "wrong interpreter -- set PY to the one with torch+pennylane"; exit 1; }
import sys, torch, pennylane, numpy
print(f"  python     {sys.version.split()[0]}  ({sys.executable})")
print(f"  torch      {torch.__version__}")
print(f"  pennylane  {pennylane.__version__}")
print(f"  numpy      {numpy.__version__}")
EOF

# ---------------------------------------------------------------------
banner "1  per-mode spectral decomposition (heat)"
# Replaces the early/late split, which measured IC anchoring rather than
# spectral difficulty. Writes mode_metrics.json + mode_coeffs.npz into
# each runs/heat_* directory, plus mode_decomposition_summary.json.
# Forward passes on existing weights -- no training. ~3-8 min.
"$PY" mode_decomposition.py --selftest || fail "mode_decomposition self-test"
"$PY" mode_decomposition.py            || fail "mode_decomposition"

# ---------------------------------------------------------------------
banner "2  XAI layer metrics (only if missing)"
# These three runs already have xai_metrics_*.json and feature_maps_*.png
# on disk, so this normally prints 'already present' and does nothing.
# Delete the JSON to force a rebuild.
for tag in conv_classical_s0 init_strong_L2_s0 qc_qcpinn_s2; do
  if [[ -f "runs/$tag/xai_metrics_$tag.json" ]]; then
    echo "  already present: runs/$tag/xai_metrics_$tag.json"
  else
    echo "  building runs/$tag ..."
    "$PY" xai_feature_maps.py "runs/$tag" || fail "xai_feature_maps $tag"
  fi
done

# ---------------------------------------------------------------------
banner "3  all paper figures"
# 1-5 from loss histories and metrics, 7 from the XAI JSON, 8 from step 1.
# 6 is the per-neuron grid, which xai_feature_maps.py already wrote per run.
"$PY" paper_figures.py || fail "paper_figures"

# ---------------------------------------------------------------------
if [[ "$WITH_GLOBAL" == "1" ]]; then
  banner "4  global-cost control (barren-plateau reconciliation)"
  # Settles whether this project's gradient-variance result really does
  # conflict with arXiv:2604.09957, or whether the two differ only in
  # their baseline. Adds a 'global_sweep' key to
  # gradient_variance_results.json; the existing sweeps are untouched.
  # ~20-40 min. Try --quick first if you want a smoke test.
  "$PY" global_cost_control.py --selftest || fail "global_cost_control self-test"
  "$PY" global_cost_control.py            || fail "global_cost_control"
else
  banner "4  global-cost control -- SKIPPED"
  echo "  optional, ~20-40 min. Run it with:"
  echo "      ./run_remaining.sh --with-global"
  echo "  or a 1-minute smoke test first:"
  echo "      $PY global_cost_control.py --quick"
fi

# ---------------------------------------------------------------------
banner "done"
echo "  figures/           fig1-fig5, fig7  (+ fig8 from step 1, fig9 from step 4)"
echo "  runs/heat_*/       mode_metrics.json, mode_coeffs.npz"
echo "  mode_decomposition_summary.json"
ls -1 figures/*.png 2>/dev/null | sed 's/^/    /'
