"""
PDE definitions, so every experiment in this study can run on more than
one equation without touching the Burgers'-only code path.

WHY A SECOND PDE, AND WHY THIS ONE
----------------------------------
"Single PDE" is the first objection a reviewer raises, and it is a fair
one: a design conclusion drawn from one equation might be a fact about
that equation rather than about the architecture. The cheapest fix is not
"more PDEs" in general -- it is ONE well-chosen contrast.

Burgers' is a shock problem: broadband, with the hard content concentrated
in space around x = 0.

The heat equation here is deliberately its opposite: smooth, with the hard
content concentrated in TIME. The initial condition is a sum of sine modes
that decay at rate exp(-alpha k^2 pi^2 t), so the high-frequency content
exists early and is gone by late time. A model that cannot represent high
frequencies fails at small t and looks fine at large t.

That gives the paper a genuine two-by-two: a design rule that holds on a
smooth PDE and fails on a shock (or vice versa) is a much stronger result
than either equation alone, and it is the difference between "we studied
Burgers'" and "we studied where these rules apply".

WHY THE INITIAL CONDITION IS MULTI-MODE
---------------------------------------
The textbook choice, u(x,0) = -sin(pi x), has a closed-form solution but
only ONE Fourier mode. Every architecture in this study can represent one
mode, so that problem discriminates between nothing -- all models would
score near-identically and the experiment would be wasted compute.

Summing modes k = 1, 3, 5 makes the required bandwidth explicit and
tunable, which connects directly to the Schuld/Sweke/Meyer frequency
ceiling this project already measures: you can predict from the encoding
which modes a circuit is capable of representing, then check. `modes` is a
constructor argument precisely so a bandwidth sweep is available later
without new code.

The solution stays exact -- each mode decays independently -- so unlike
Burgers' there is no numerical reference solution and therefore no solver
error in the ground truth. That is worth having in a paper about
measurement error.

alpha = 0.02 is chosen so the modes separate over t in [0, 1]:
    mode 1 retains ~83% of its amplitude at t = 1
    mode 3 falls to  ~17%
    mode 5 falls to   ~0.7%
"""

import numpy as np
import torch

# =====================================================================
# Base
# =====================================================================


class PDE:
    """A PDE the study can train against.

    Subclasses supply the residual operator, the initial condition, an
    exact or reference solution, a collocation sampler, and an error
    decomposition. Everything else in the pipeline is PDE-agnostic.
    """

    name = "abstract"
    x_min, x_max, t_max = -1.0, 1.0, 1.0

    def residual(self, model, x, t):
        raise NotImplementedError

    def ic(self, x):
        raise NotImplementedError

    def bc_value(self, x, t):
        """Dirichlet value on both boundaries. Zero for both PDEs here."""
        return torch.zeros_like(t)

    def ground_truth(self):
        raise NotImplementedError

    def sample_collocation(self, n_uniform, n_focus, generator=None):
        """Collocation points. Uniform, plus a cluster wherever this PDE
        keeps its hard content -- in space for a shock, in time for a
        decaying transient."""
        raise NotImplementedError

    def decompose(self, X, T, err, U_true):
        """Split the error into the region that is hard for this PDE and
        the region that is easy, and report the ratio. Key names differ
        per PDE because the regions mean different things; both include a
        `hard_easy_ratio` so cross-PDE comparison has one common column."""
        raise NotImplementedError

    def _grads(self, model, x, t):
        """u and its derivatives, shared by both residuals."""
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        u = model(x, t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        return u, u_t, u_x, u_xx


# =====================================================================
# Burgers' -- the existing benchmark, restated here so both PDEs go
# through one interface. Numerically identical to the original path.
# =====================================================================


class Burgers(PDE):
    name = "burgers"

    def __init__(self, nu=0.01 / np.pi, shock_half_width=0.1):
        self.nu = nu
        self.shock_half_width = shock_half_width
        self._gt = None

    def residual(self, model, x, t):
        # u must come from the SAME graph as u_t and u_x -- a second forward
        # pass would give the same value but a disconnected node, and costs
        # an extra evaluation of the quantum circuit on every epoch.
        u, u_t, u_x, u_xx = self._grads(model, x, t)
        return u_t + u * u_x - self.nu * u_xx

    def ic(self, x):
        return -torch.sin(np.pi * x)

    def ground_truth(self):
        if self._gt is None:
            from ground_truth import BurgersGroundTruth
            self._gt = BurgersGroundTruth()
        return self._gt

    def sample_collocation(self, n_uniform, n_focus, generator=None):
        # cluster in SPACE around the shock at x = 0
        xu = torch.rand(n_uniform, 1, generator=generator) * (self.x_max - self.x_min) + self.x_min
        xs = (torch.randn(n_focus, 1, generator=generator) * 0.15).clamp(self.x_min, self.x_max)
        x = torch.cat([xu, xs], dim=0)
        t = torch.rand(x.shape[0], 1, generator=generator) * self.t_max
        return x, t

    def decompose(self, X, T, err, U_true):
        hard = np.abs(X) <= self.shock_half_width
        rs = float(np.linalg.norm(err[hard]) / np.linalg.norm(U_true[hard]))
        rm = float(np.linalg.norm(err[~hard]) / np.linalg.norm(U_true[~hard]))
        return {"rel_l2_shock_region": rs,
                "rel_l2_smooth_region": rm,
                "spectral_bias_ratio": rs / rm if rm > 0 else float("nan"),
                "hard_easy_ratio": rs / rm if rm > 0 else float("nan"),
                "region_definition": f"shock band |x| <= {self.shock_half_width}"}


# =====================================================================
# Heat -- smooth, multi-mode, exact solution, hard content early in time
# =====================================================================


class Heat(PDE):
    name = "heat"

    def __init__(self, alpha=0.02, modes=(1, 3, 5), amps=(1.0, 0.5, 0.25),
                 early_t=0.2, late_t=0.5):
        assert len(modes) == len(amps)
        self.alpha = alpha
        self.modes = tuple(modes)
        self.early_t, self.late_t = early_t, late_t

        # normalise so max|u(x,0)| = 1, matching Burgers' amplitude so the
        # two PDEs' relative-L2 numbers live on a comparable scale
        xs = np.linspace(self.x_min, self.x_max, 2001)
        u0 = sum(a * np.sin(k * np.pi * xs) for k, a in zip(modes, amps))
        self.amps = tuple(float(a / np.abs(u0).max()) for a in amps)

    # ---- physics -------------------------------------------------

    def residual(self, model, x, t):
        _u, u_t, _u_x, u_xx = self._grads(model, x, t)
        return u_t - self.alpha * u_xx

    def ic(self, x):
        out = torch.zeros_like(x)
        for k, a in zip(self.modes, self.amps):
            out = out + a * torch.sin(k * np.pi * x)
        return out

    def ground_truth(self):
        """Exact: each mode decays independently. No solver, no solver error."""
        alpha, modes, amps = self.alpha, self.modes, self.amps

        def u(x_query, t_query):
            x_query = np.asarray(x_query, dtype=float)
            t_query = np.asarray(t_query, dtype=float)
            out = np.zeros(np.broadcast(x_query, t_query).shape)
            for k, a in zip(modes, amps):
                out = out + a * np.sin(k * np.pi * x_query) * np.exp(
                    -alpha * (k * np.pi) ** 2 * t_query)
            return out

        return u

    # ---- sampling ------------------------------------------------

    def sample_collocation(self, n_uniform, n_focus, generator=None):
        # cluster in TIME near t = 0, where the high modes still exist.
        # This mirrors what Burgers' does in space, and for the same
        # reason: sample where the residual is hardest to satisfy.
        x = torch.rand(n_uniform + n_focus, 1, generator=generator) * \
            (self.x_max - self.x_min) + self.x_min
        tu = torch.rand(n_uniform, 1, generator=generator) * self.t_max
        tf = (torch.rand(n_focus, 1, generator=generator) ** 2) * (self.early_t * 1.5)
        t = torch.cat([tu, tf], dim=0)
        return x, t

    # ---- error decomposition -------------------------------------

    def decompose(self, X, T, err, U_true):
        """Early time still carries modes 3 and 5; late time is mode 1 only.
        A spectrally limited model therefore fails EARLY and looks fine
        late -- the temporal analogue of failing at the shock."""
        early, late = T <= self.early_t, T >= self.late_t
        re = float(np.linalg.norm(err[early]) / np.linalg.norm(U_true[early]))
        rl = float(np.linalg.norm(err[late]) / np.linalg.norm(U_true[late]))
        return {"rel_l2_early_time": re,
                "rel_l2_late_time": rl,
                "hard_easy_ratio": re / rl if rl > 0 else float("nan"),
                "region_definition": f"early t <= {self.early_t}, late t >= {self.late_t}"}

    # ---- diagnostics ---------------------------------------------

    def mode_amplitudes(self, t):
        """Analytic amplitude of each mode at time t. Lets a run report
        WHICH mode it lost, not just that its error grew."""
        return {k: a * float(np.exp(-self.alpha * (k * np.pi) ** 2 * t))
                for k, a in zip(self.modes, self.amps)}


# =====================================================================

REGISTRY = {"burgers": Burgers, "heat": Heat}


def get(name, **kw):
    if name not in REGISTRY:
        raise ValueError(f"unknown pde '{name}'; have {list(REGISTRY)}")
    return REGISTRY[name](**kw)


if __name__ == "__main__":
    print("Registered PDEs:", list(REGISTRY), "\n")

    h = Heat()
    print(f"Heat: alpha={h.alpha}, modes={h.modes}, amps={tuple(round(a,3) for a in h.amps)}")
    print("  mode amplitude decay (this is what makes the problem discriminating):")
    for t in (0.0, 0.1, 0.2, 0.5, 1.0):
        amps = h.mode_amplitudes(t)
        print(f"    t={t:4.1f}  " + "  ".join(f"k={k}: {v:6.4f}" for k, v in amps.items()))

    gt = h.ground_truth()
    x = np.linspace(-1, 1, 5)
    print("\n  exact solution sanity:")
    print(f"    u(x, 0) max |.| = {np.abs(gt(np.linspace(-1,1,401), 0.0)).max():.4f}  (want 1.0)")
    print(f"    u(-1, t) = {gt(-1.0, 0.4):.2e}, u(+1, t) = {gt(1.0, 0.4):.2e}  (want 0)")

    # IC must agree between the torch and numpy paths, or training fits one
    # function while evaluation scores another
    xt = torch.linspace(-1, 1, 401).reshape(-1, 1)
    ic_torch = h.ic(xt).numpy().ravel()
    ic_numpy = gt(np.linspace(-1, 1, 401), 0.0)
    print(f"    max |ic_torch - u(x,0)| = {np.abs(ic_torch - ic_numpy).max():.2e}  (want ~0)")

    b = Burgers()
    xs, ts = b.sample_collocation(800, 300)
    xh, th = h.sample_collocation(800, 300)
    print(f"\n  Burgers collocation: {tuple(xs.shape)}, "
          f"{float((xs.abs() <= 0.1).float().mean()):.0%} inside the shock band")
    print(f"  Heat    collocation: {tuple(xh.shape)}, "
          f"{float((th <= 0.2).float().mean()):.0%} at early time")
