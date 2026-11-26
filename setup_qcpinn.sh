#!/usr/bin/env bash
# setup_qcpinn.sh -- fetch the upstream implementation this study replicates.
#
# The authors' code is MIT-licensed but is NOT redistributed in this
# repository. This clones it at the exact commit every replication run used,
# and builds the pinned environment it needs.
#
# WHY A SEPARATE INTERPRETER
#   The numerics are interpreter-dependent. On identical code and an identical
#   seed, Python 3.9.6 / torch 2.0.1 / PennyLane 0.29 gives 10.00% where
#   Python 3.13 / torch 2.10 gives 6.44%. Every replication and symmetric-
#   control number in the paper was produced under the 3.9 environment. Using
#   a different interpreter will not reproduce them.
set -eu
COMMIT=0dc467be8b16cccaa5101e63fe8cdf8d18bfdcb9
mkdir -p external
if [ ! -d external/QCPINN ]; then
  git clone https://github.com/afrah/QCPINN external/QCPINN
fi
git -C external/QCPINN checkout --detach "$COMMIT"
echo "upstream at $(git -C external/QCPINN rev-parse HEAD)"

if [ ! -d external/venv_qcpinn ]; then
  echo "creating the pinned environment (needs a Python 3.9 interpreter)"
  python3.9 -m venv external/venv_qcpinn
  external/venv_qcpinn/bin/pip install --upgrade pip
  external/venv_qcpinn/bin/pip install "torch==2.0.1" "pennylane==0.29.0" numpy scipy matplotlib
fi
external/venv_qcpinn/bin/python -c "import torch,pennylane;print('torch',torch.__version__,'pennylane',pennylane.__version__)"
echo "ready. Do NOT modify anything under external/QCPINN."
