#!/usr/bin/env bash
set -u
PY=/usr/local/bin/python3; VENV=external/venv_qcpinn/bin/python; mkdir -p logs_extend
echo "=== 1. QCPINN circuit ablation on Helmholtz (authors' code, 10 seeds) ==="
for s in 0 1 2 3 4 5 6 7 8 9; do echo "$s"; done | xargs -P 4 -I {} sh -c \
 "$VENV replicate_qcpinn.py --model qcpinn --ablate --seed {} --tag abl_qcpinn_s{} > logs_extend/abl_qcpinn_s{}.log 2>&1; \
  echo \"[qc-abl s{}] \$(grep -o 'err_u = [0-9.]*%' logs_extend/abl_qcpinn_s{}.log | tail -1)\""
echo "=== 2. QAPINN circuit ablation on Burgers (ring and strong, 8 seeds) ==="
for e in ring strong; do for s in 0 1 2 3 4 5 6 7; do echo "$e $s"; done; done | xargs -P 4 -L 1 sh -c \
 '[ -f runs/abl_$0_s$1/metrics.json ] || /usr/local/bin/python3 ablate_qapinn.py --entangler $0 --seed $1 --tag abl_$0_s$1 > logs_extend/abl_$0_s$1.log 2>&1; \
  echo "[abl $0 s$1] $(grep -o "rel L2 *[0-9.]*%" logs_extend/abl_$0_s$1.log | tail -1)"'
echo "=== 3. Swap experiment: 2^3 quantum x classical x point stream ==="
bash swap_run.sh
echo "ALL DONE"
