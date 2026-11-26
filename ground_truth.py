"""Same numerical Burgers'-equation ground truth as
burgers_equation_pinn/ground_truth.py -- duplicated here (not imported
across folders) so this proof-of-concept is fully self-contained and can
be handed over as its own unit. See that file for the full derivation
writeup; this copy keeps only the working code."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

X_MIN, X_MAX = -1.0, 1.0
T_MAX = 1.0
NU = 0.01 / np.pi

NX = 401
NT_OUTPUT = 201


def _rhs(t, u, x, dx, nu):
    u_x = np.zeros_like(u)
    u_xx = np.zeros_like(u)
    u_x[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx ** 2)
    du_dt = -u * u_x + nu * u_xx
    du_dt[0] = 0.0
    du_dt[-1] = 0.0
    return du_dt


def solve_reference(nx: int = NX, nt_output: int = NT_OUTPUT):
    x = np.linspace(X_MIN, X_MAX, nx)
    dx = x[1] - x[0]
    u0 = -np.sin(np.pi * x)
    u0[0] = 0.0
    u0[-1] = 0.0

    t_eval = np.linspace(0, T_MAX, nt_output)

    sol = solve_ivp(
        _rhs, (0, T_MAX), u0, t_eval=t_eval, args=(x, dx, NU),
        method="BDF", rtol=1e-8, atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"reference solver failed: {sol.message}")

    U = sol.y.T
    return x, t_eval, U


class BurgersGroundTruth:
    """Callable wrapper: query the numerical solution at arbitrary (x, t)."""

    def __init__(self):
        x, t, U = solve_reference()
        self.x, self.t, self.U = x, t, U
        self._interp = RegularGridInterpolator((t, x), U, bounds_error=False, fill_value=None)

    def __call__(self, x_query: np.ndarray, t_query: np.ndarray) -> np.ndarray:
        x_query = np.asarray(x_query)
        t_query = np.asarray(t_query)
        shape = np.broadcast(x_query, t_query).shape
        pts = np.stack([
            np.broadcast_to(t_query, shape).ravel(),
            np.broadcast_to(x_query, shape).ravel(),
        ], axis=-1)
        return self._interp(pts).reshape(shape)


if __name__ == "__main__":
    gt = BurgersGroundTruth()
    print("reference grid:", gt.U.shape, "(nt, nx)")
    print("u(x=0, t=0) should be ~0:", gt(0.0, 0.0))
    print("u(-1, t=0.5) [boundary] should be 0:", gt(-1.0, 0.5))
    print("u(1, t=0.5) [boundary] should be 0:", gt(1.0, 0.5))
