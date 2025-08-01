import numpy as np

class BlasiusFlow:
    """Compute the steady Blasius boundary layer solution on a given grid."""

    def __init__(self, U_inf, nu, x, y, eta_max=10.0, Neta=1000):
        self.U_inf = U_inf
        self.nu = nu
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.eta_max = eta_max
        self.Neta = Neta
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self._compute_similarity_solution()
        self.U0, self.V0 = self._compute_base_flow()

    def _compute_similarity_solution(self):
        eta = np.linspace(0.0, self.eta_max, self.Neta)
        f = np.zeros((3, self.Neta))
        f[2, 0] = 0.332057336215194  # f''(0)
        d_eta = eta[1] - eta[0]

        def rhs(state):
            f0, f1, f2 = state
            return np.array([f1, f2, -0.5 * f0 * f2])

        for k in range(self.Neta - 1):
            s = f[:, k]
            k1 = rhs(s)
            k2 = rhs(s + 0.5 * d_eta * k1)
            k3 = rhs(s + 0.5 * d_eta * k2)
            k4 = rhs(s + d_eta * k3)
            f[:, k + 1] = s + (d_eta / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        self.eta = eta
        self.f0_eta = f[0]
        self.f1_eta = f[1]

    def _compute_base_flow(self):
        Nx = len(self.x)
        Ny = len(self.y)

        X, Y = np.meshgrid(self.x, self.y, indexing="xy")

        U0 = np.zeros((Ny, Nx))
        V0 = np.zeros((Ny, Nx))

        mask = X > 0
        et = np.zeros_like(X)
        et[mask] = Y[mask] * np.sqrt(self.U_inf / (self.nu * X[mask]))

        f1 = np.interp(et.ravel(), self.eta, self.f1_eta).reshape(et.shape)
        f0 = np.interp(et.ravel(), self.eta, self.f0_eta).reshape(et.shape)

        U0 = self.U_inf * f1
        V0[mask] = 0.5 * np.sqrt(self.nu * self.U_inf / X[mask]) * (et[mask] * f1[mask] - f0[mask])

        U0[~mask] = self.U_inf * self.f1_eta[-1]
        V0[~mask] = 0.0

        return U0, V0

    def wall_shear_drag(self, mu, x):
        dy = self.dy
        dUdy = (self.U0[1, :] - self.U0[0, :]) / dy
        tau_w = mu * dUdy
        return np.trapezoid(tau_w, x)