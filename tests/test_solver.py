import numpy as np
from blasius import BlasiusFlow
from solver import LinearBLSolver


def test_zero_inlet_stability():
    U_inf = 1.0
    nu = 1e-3
    rho = 1.0
    x = np.linspace(0.1, 0.5, 5)
    y = np.linspace(0.0, 0.02, 6)
    dt = 1e-3
    Nt = 5

    base = BlasiusFlow(U_inf, nu, x, y)
    solver = LinearBLSolver(base, rho, nu, dt, Nt, lambda t, y: np.zeros_like(y))
    frames_u, frames_v, t_snap = solver.run_explicit()

    for frame in frames_u:
        assert np.allclose(frame, base.U0, atol=1e-8)
    for frame in frames_v:
        assert np.allclose(frame, base.V0, atol=1e-8)


def test_drag_positive():
    U_inf = 1.0
    nu = 1e-3
    rho = 1.0
    mu = rho * nu
    x = np.linspace(0.1, 0.5, 5)
    y = np.linspace(0.0, 0.02, 6)

    base = BlasiusFlow(U_inf, nu, x, y)
    drag = base.wall_shear_drag(mu, x)
    assert drag > 0.0
