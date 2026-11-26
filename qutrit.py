"""
N-controlled Toffoli: qutrit (d=3) circuit vs. qubit (d=2) circuits.

Idea (Gokhale et al., ISCA 2019): use the spare |2> level of a qutrit as
temporary storage, so the AND-tree needs NO ancillas and has O(log N) depth.

All gates here are controlled permutations of basis levels, so we simulate
them exactly on a state vector (numpy) and also count resources.

Run:  python qudit_vs_qubit_toffoli.py
"""
import numpy as np
from itertools import product

# ----------------------------------------------------------------------------
# Tiny mixed-dimension simulator for controlled-permutation gates
# ----------------------------------------------------------------------------
class Circuit:
    def __init__(self, dims):
        self.dims = list(dims)          # dimension of each wire (2 or 3)
        self.gates = []                 # (controls, target, perm)

    def cgate(self, controls, target, perm):
        """controls: list of (wire, level). perm: new level = perm[old level]."""
        self.gates.append((list(controls), target, tuple(perm)))

    # --- resources -----------------------------------------------------------
    def depth(self):
        busy = [0] * len(self.dims)
        for controls, t, _ in self.gates:
            wires = [w for w, _ in controls] + [t]
            layer = max(busy[w] for w in wires) + 1
            for w in wires:
                busy[w] = layer
        return max(busy)

    def gate_histogram(self):
        h = {}
        for c, _, _ in self.gates:
            k = len(c) + 1               # number of wires the gate touches
            h[k] = h.get(k, 0) + 1
        return h

    # --- simulation ----------------------------------------------------------
    def apply(self, psi):
        psi = psi.reshape(self.dims).copy()
        for controls, t, perm in self.gates:
            idx = [slice(None)] * len(self.dims)
            for w, lvl in controls:
                idx[w] = lvl
            new = psi.copy()
            for old in range(self.dims[t]):
                src, dst = list(idx), list(idx)
                src[t], dst[t] = old, perm[old]
                new[tuple(dst)] = psi[tuple(src)]
            psi = new
        return psi.reshape(-1)


def X_plus(d):  return [(k + 1) % d for k in range(d)]   # |k> -> |k+1 mod d>
def X_minus(d): return [(k - 1) % d for k in range(d)]
X2 = [1, 0]                                              # qubit NOT


# ----------------------------------------------------------------------------
# 1) Qutrit N-controlled Toffoli: binary tree over the controls, no ancillas
# ----------------------------------------------------------------------------
def qutrit_toffoli(N):
    """Wires 0..N-1 = controls (qutrits, inputs in {0,1}); wire N = target qubit.
    Heap-ordered tree: node i has children 2i+1, 2i+2.
    After compute, an internal node sits in |2> iff it and its whole subtree are |1>."""
    c = Circuit([3] * N + [2])
    kids = lambda i: [k for k in (2 * i + 1, 2 * i + 2) if k < N]
    true_lvl = lambda i: 2 if kids(i) else 1           # leaves: |1>, internal: |2>

    depth_of = lambda i: int(np.floor(np.log2(i + 1)))
    internal = [i for i in range(N) if kids(i)]
    order = sorted(internal, key=depth_of, reverse=True)  # bottom-up

    compute = [([(k, true_lvl(k)) for k in kids(i)], i) for i in order]
    for ctrls, i in compute:                          # raise node to |2> if AND holds
        c.cgate(ctrls, i, X_plus(3))
    c.cgate([(0, true_lvl(0))], N, X2)                # flip target from the root
    for ctrls, i in reversed(compute):                # uncompute
        c.cgate(ctrls, i, X_minus(3))
    return c


# ----------------------------------------------------------------------------
# 2) Qubit, linear "V-chain" with N-2 ancillas (Nielsen & Chuang)
# ----------------------------------------------------------------------------
def qubit_vchain(N):
    A = N - 2
    c = Circuit([2] * (N + 1 + A))       # controls | target | ancillas
    anc = lambda j: N + 1 + j
    comp = [([(0, 1), (1, 1)], anc(0))]
    for j in range(1, A):
        comp.append(([(j + 1, 1), (anc(j - 1), 1)], anc(j)))
    for ctrls, t in comp:
        c.cgate(ctrls, t, X2)
    c.cgate([(N - 1, 1), (anc(A - 1), 1)], N, X2)
    for ctrls, t in reversed(comp):
        c.cgate(ctrls, t, X2)
    return c


# ----------------------------------------------------------------------------
# 3) Qubit, log-depth AND-tree with ancillas
# ----------------------------------------------------------------------------
def qubit_tree(N):
    # count ancillas first
    layer, n_anc = list(range(N)), 0
    while len(layer) > 2:
        n_anc += len(layer) // 2
        layer = [None] * (len(layer) // 2 + len(layer) % 2)
    c = Circuit([2] * (N + 1 + n_anc))
    nxt = N + 1
    layer, comp = list(range(N)), []
    while len(layer) > 2:
        new = []
        for a, b in zip(layer[0::2], layer[1::2]):
            comp.append(([(a, 1), (b, 1)], nxt)); new.append(nxt); nxt += 1
        if len(layer) % 2: new.append(layer[-1])
        layer = new
    for ctrls, t in comp:
        c.cgate(ctrls, t, X2)
    c.cgate([(layer[0], 1), (layer[1], 1)], N, X2)
    for ctrls, t in reversed(comp):
        c.cgate(ctrls, t, X2)
    return c


# ----------------------------------------------------------------------------
# Verification: random superposition over the qubit subspace of the inputs
# ----------------------------------------------------------------------------
def verify(circ, N, rng):
    dims = circ.dims
    psi = np.zeros(dims, dtype=complex)
    expected = np.zeros(dims, dtype=complex)
    for bits in product([0, 1], repeat=N + 1):        # controls + target
        amp = rng.normal() + 1j * rng.normal()
        idx = list(bits) + [0] * (len(dims) - N - 1)  # ancillas start in |0>
        psi[tuple(idx)] = amp
        out = list(idx)
        if all(bits[:N]):
            out[N] ^= 1
        expected[tuple(out)] = amp
    got = circ.apply(psi.reshape(-1))
    return np.allclose(got, expected.reshape(-1))


# ----------------------------------------------------------------------------
# Resource report
# ----------------------------------------------------------------------------
# Rough cost model for decomposing into 2-qudit gates (assumption, see notes):
#   qubit Toffoli (3-wire)        -> 6 CNOTs, depth ~ 5
#   qutrit 3-wire controlled gate -> ~6 two-qutrit gates (Di & Wei decomposition)
COST_3WIRE = 6

def report(name, c, N):
    h = c.gate_histogram()
    n_wires = len(c.dims)
    anc = n_wires - (N + 1)
    two_q = h.get(2, 0) + COST_3WIRE * h.get(3, 0)
    return f"{name:<22}{n_wires:>7}{anc:>8}{c.depth():>8}{sum(h.values()):>9}{two_q:>10}"


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("Correctness check (statevector simulation):")
    for N in range(3, 8):
        ok = all(verify(f(N), N, rng) for f in (qutrit_toffoli, qubit_vchain, qubit_tree))
        print(f"  N={N}: {'all circuits correct' if ok else 'MISMATCH'}")

    hdr = f"{'circuit':<22}{'wires':>7}{'ancilla':>8}{'depth':>8}{'gates':>9}{'~2q gates':>10}"
    for N in (8, 16, 64, 256, 1024):
        print(f"\nN = {N} controls")
        print(hdr)
        print(report("qutrit tree (no anc)", qutrit_toffoli(N), N))
        print(report("qubit V-chain", qubit_vchain(N), N))
        print(report("qubit log-tree", qubit_tree(N), N))