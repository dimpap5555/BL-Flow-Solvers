import numpy as np
from blasius import BlasiusFlow


def test_boundary_values():
    U_inf = 1.0
    nu = 1e-3
    x = np.linspace(0.1, 0.5, 5)
    y = np.linspace(0.0, 0.02, 6)
    bl = BlasiusFlow(U_inf, nu, x, y)

    # velocity should vanish at the wall
    assert np.allclose(bl.U0[0], 0.0, atol=1e-8)
    assert np.allclose(bl.V0[0], 0.0, atol=1e-8)

    # velocity should approach freestream at the outer boundary
    assert np.allclose(bl.U0[-1], U_inf, rtol=1e-2)
