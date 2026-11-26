# What Moves a Quantum Advantage?

Artefact for **NQComp 2027, Paper ID 583** — *What Moves a Quantum Advantage?
Initialisation, Training Protocol and Circuit Ablation in Hybrid
Quantum-Classical PINNs*.

This repository holds the code, the configurations, the per-run seeds and the
raw run records behind every number in the paper, together with five
verification suites that re-derive those numbers from the raw records.

## The short version

```bash
python3 verify_paper.py          # every claim in PAPER.md
python3 verify_conf_cost.py      # the configuration and cost tables
python3 verify_sym_section.py    # the symmetric control
python3 verify_third_section.py  # the third equation and circuit components
python3 verify_hyper_section.py  # the hyperparameter sweep
```

Each suite reads `runs/**/*.json`, recomputes the statistics with the exact
tests in `stats_analysis.py`, and asserts that the resulting string appears
verbatim in `paper_nqcomp/main.tex`. A number that is right in the analysis
and mistyped in the paper still fails. No training is needed to check the
paper: the raw run records are in this repository.

## What is here

| path | what it is |
|---|---|
| `runs/**/*.json` | raw per-run records: final errors, timings, seeds, configs |
| `verify_*.py` | the five verification suites |
| `stats_analysis.py` | exact permutation tests, Cliff's delta, Fisher exact, Holm |
| `protocol*.py`, `train.py`, `qapinn_model.py` | the trainers |
| `run_hyper_sweep.sh`, `symmetric_protocol4.sh`, … | the experiment drivers |
| `paper_nqcomp/` | the manuscript and its figure script |
| `setup_qcpinn.sh` | clones the upstream implementation at its pinned commit |

Loss histories (`*.npy`) and network weights (`*.pt`) are not included: they
total about 230 MB and no published number depends on them.

## Reproducing a run

**Two interpreters are required, and they are not interchangeable.**

The numerics are interpreter-dependent. On identical code and an identical
seed, Python 3.9.6 / torch 2.0.1 / PennyLane 0.29 produces **10.00%** where
Python 3.13 / torch 2.10 produces **6.44%**. Floating-point gradient
accumulation order is load-bearing here.

| what you are running | interpreter |
|---|---|
| anything touching the upstream QCPINN code — the replication, the symmetric control, the hyperparameter sweep | `external/venv_qcpinn/bin/python` (3.9.6) |
| our own models and all analysis | system `python3` (3.13) |

Run `./setup_qcpinn.sh` first. It clones
[afrah/QCPINN](https://github.com/afrah/QCPINN) at commit `0dc467be` and
builds the pinned environment. The upstream code is **not** redistributed
here and must not be modified — every replication run used it unaltered,
adding only a per-run seed.

```bash
./setup_qcpinn.sh
bash run_hyper_sweep.sh                    # 30 runs, ~20 h on an 8-core M2
external/venv_qcpinn/bin/python protocol5_hyper.py --tag demo --model qcpinn --seed 0
```

## Attribution

This study replicates and builds controls around Farea, Khan and Celebi,
*QCPINN* (arXiv:2503.16678), MIT-licensed. See [NOTICE](NOTICE) for the full
statement of what is theirs and what is ours. Their published result
reproduces; we find no discrepancy with their reported measurement.

## Licence

MIT — see [LICENSE](LICENSE). The upstream code carries its own MIT licence,
Copyright (c) 2025 Afrah Farea, and is obtained separately.
