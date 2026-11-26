"""
Every knob in the study, in one place. Nothing else in this folder
hardcodes a hyperparameter -- if you want to change what an experiment
does, change it here or pass it on the command line.

Two data budgets are defined:

  FULL     -- matches the classical Burgers' PINN baseline's data budget
              exactly (4000 collocation points, 300 IC, 300 BC). Use this
              for the headline runs you actually report, so the QAPINN and
              the classical PINN are compared on genuinely equal footing.
              This is the honest comparison; the earlier proof-of-concept
              runs used ~5x less data than the classical baseline, which
              was a fair criticism of them.

  SWEEP    -- a lighter budget used for the systematic sweep, where the
              point is comparing configurations AGAINST EACH OTHER (all
              at the same budget), not against the classical baseline.
              Keeps a 10+ configuration sweep finishable in hours instead
              of days.
"""

import numpy as np

# ---------------------------------------------------------------------
# Problem (viscous Burgers' equation) -- do not change these, they define
# the benchmark and must match the classical baseline exactly
# ---------------------------------------------------------------------
X_MIN, X_MAX = -1.0, 1.0
T_MAX = 1.0
NU = 0.01 / np.pi

# ---------------------------------------------------------------------
# Data budgets
# ---------------------------------------------------------------------
FULL = {
    "n_collocation_uniform": 3000,   # was 600 in the proof of concept
    "n_collocation_shock": 1000,     # was 200
    "n_ic": 300,                     # was 100
    "n_bc": 300,                     # was 100
    "adam_epochs": 8000,
    "lbfgs_steps": 500,
    "lr": 1e-3,
    "w_ic": 10.0,
    "w_bc": 10.0,
}

SWEEP = {
    "n_collocation_uniform": 800,
    "n_collocation_shock": 300,
    "n_ic": 150,
    "n_bc": 150,
    "adam_epochs": 8000,
    "lbfgs_steps": 500,
    "lr": 1e-3,
    "w_ic": 10.0,
    "w_bc": 10.0,
}

# BUDGET HISTORY -- why this is 8000 epochs and not 2500.
# The first sweep ran at 2500 Adam epochs. At that budget the CLASSICAL
# baseline scored 30.6%, against the 11.2% it is known to reach when
# trained properly -- i.e. nothing in that sweep had converged, and the
# apparent "best depth" was really just "whichever circuit got furthest
# in a short run". Deeper circuits have more parameters and were
# penalised for it. Those results are kept in runs_equalbudget_2500/ and
# remain valid as an EQUAL-BUDGET CONVERGENCE-SPEED comparison (several
# QAPINN configs reached less than half the classical error there), but
# they cannot support conclusions about architecture.
#
# Measured cost per configuration at the 8000-epoch budget (CPU simulator):
#   4 qubits, L=1 ~8 min    L=2 ~12 min    L=4 ~22 min    L=6 ~31 min    L=8 ~42 min
#   2 qubits, L=4 ~6 min    strongly-entangling ansatz roughly +5% on top
# The full 10-configuration sweep lands around 3-4 hours.
# Simulating 8 qubits with second-derivative backprop costs ~14 s/epoch,
# which is 12 hours for ONE configuration, so 8 qubits is deliberately
# excluded from the TRAINING sweep. It still appears in the
# gradient-variance analysis below, which needs no training and can
# therefore afford much wider circuits -- a useful division of labour
# worth stating explicitly when presenting: expensive experiments run
# narrow, cheap diagnostics run wide.

# ---------------------------------------------------------------------
# Quantum-layer architecture defaults (the things the sweep varies)
# ---------------------------------------------------------------------
DEFAULT_N_QUBITS = 4
DEFAULT_REUPLOAD_LAYERS = 6
DEFAULT_ENTANGLER_DEPTH = 1

# Classical stack after the quantum layer -- sized to match BurgersPINN
# (8 layers x 20 wide) minus the one layer the quantum circuit replaces
DEFAULT_CLASSICAL_WIDTH = 20
DEFAULT_CLASSICAL_LAYERS = 7

# ---------------------------------------------------------------------
# What the sweep actually sweeps
# ---------------------------------------------------------------------
# Axis 1: re-upload depth at fixed qubit count -- tests the frequency
#         hypothesis directly (max accessible frequency should scale with
#         re-upload depth, per Schuld/Sweke/Meyer)
SWEEP_REUPLOAD = [1, 2, 4, 6, 8]
SWEEP_REUPLOAD_FIXED_QUBITS = 4

# Axis 2: qubit count at fixed re-upload depth -- tests whether extra
#         Hilbert space helps beyond raw frequency budget, and exposes
#         the trainability cost of more qubits
SWEEP_QUBITS = [2, 4]         # 6 and 8 excluded: 6 qubits cost 69 min and stayed
                              # far from convergence; 8 qubits is ~12h for one config.
                              # Both still appear in the gradient-variance analysis,
                              # which needs no training.
SWEEP_QUBITS_FIXED_REUPLOAD = 4

# ---------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------
# Fourier spectrum analysis: how densely to sample one full period of the
# encoding angle, and which configurations to compare
FOURIER_N_SAMPLES = 512
FOURIER_REUPLOAD_LAYERS = [1, 2, 4, 6, 8]
FOURIER_N_QUBITS = 4

# Barren-plateau / gradient-variance analysis
GRADVAR_QUBIT_COUNTS = [2, 3, 4, 5, 6, 8]
GRADVAR_N_SAMPLES = 120     # random initializations per qubit count
GRADVAR_N_POINTS = 200      # collocation points used for the loss estimate
GRADVAR_REUPLOAD_LAYERS = 4

# Error decomposition: how wide the "shock region" is, in x, around x=0
SHOCK_HALF_WIDTH = 0.1

SEED = 0
