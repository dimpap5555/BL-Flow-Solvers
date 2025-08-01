import numpy as np

class LinearBLSolver:
    """Linearized boundary layer solver using a given Blasius base flow."""

    def __init__(self, blasius_flow, rho, nu, dt, Nt, inlet_func):
        self.base = blasius_flow
        self.rho = rho
        self.nu = nu
        self.dt = dt
        self.Nt = Nt
        self.inlet_func = inlet_func

        self.x = blasius_flow.x
        self.y = blasius_flow.y
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.U0 = blasius_flow.U0
        self.V0 = blasius_flow.V0
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.time = np.arange(Nt) * dt

    def inlet_profile(self, t):
        return self.inlet_func(t, self.y)

    def run_explicit(self):
        delta = np.zeros((self.Ny, self.Nx))
        frames_u = []
        frames_v = []
        for n, t in enumerate(self.time):
            delta[:, 0] = self.inlet_profile(t)
            new = delta.copy()
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    dudx = (delta[j, i] - delta[j, i - 1]) / self.dx
                    dudy = (delta[j, i] - delta[j - 1, i]) / self.dy
                    adv = -self.U0[j, i] * dudx - self.V0[j, i] * dudy
                    d2udx2 = (delta[j, i + 1] - 2 * delta[j, i] + delta[j, i - 1]) / self.dx ** 2
                    d2udy2 = (delta[j + 1, i] - 2 * delta[j, i] + delta[j - 1, i]) / self.dy ** 2
                    diff = self.nu * (d2udx2 + d2udy2)
                    new[j, i] = delta[j, i] + self.dt * (adv + diff)
            new[0, :] = 0.0
            new[-1, :] = new[-2, :]
            new[:, -1] = new[:, -2]
            delta = new

            vprime = np.zeros_like(delta)
            for i in range(self.Nx):
                for j in range(1, self.Ny):
                    ddux = (delta[j, i] - delta[j - 1, i]) / self.dy
                    vprime[j, i] = vprime[j - 1, i] - self.dx * ddux

            u_abs = self.U0 + delta
            v_abs = self.V0 + vprime
            if n % 5 == 0:
                frames_u.append(u_abs.copy())
                frames_v.append(v_abs.copy())
        frames_u = np.array(frames_u)
        frames_v = np.array(frames_v)
        frames_u = np.nan_to_num(frames_u, nan=0.0, posinf=0.0, neginf=0.0)
        frames_v = np.nan_to_num(frames_v, nan=0.0, posinf=0.0, neginf=0.0)
        nframes = frames_u.shape[0]
        t_snap = np.arange(nframes) * 5 * self.dt
        return frames_u, frames_v, t_snap

    def run_implicit(self, n_iter=20):
        delta = np.zeros((self.Ny, self.Nx))
        frames_u = []
        frames_v = []
        inv_dt = 1.0 / self.dt
        for n, t in enumerate(self.time):
            delta[:, 0] = self.inlet_profile(t)
            rhs = inv_dt * delta
            new = delta.copy()
            for _ in range(n_iter):
                for i in range(1, self.Nx - 1):
                    for j in range(1, self.Ny - 1):
                        aP = (inv_dt + 2 * self.nu / self.dx ** 2 +
                               2 * self.nu / self.dy ** 2 +
                               self.U0[j, i] / self.dx + self.V0[j, i] / self.dy)
                        aW = -self.nu / self.dx ** 2 - self.U0[j, i] / self.dx
                        aE = -self.nu / self.dx ** 2
                        aS = -self.nu / self.dy ** 2 - self.V0[j, i] / self.dy
                        aN = -self.nu / self.dy ** 2
                        new[j, i] = (rhs[j, i] - aW * new[j, i - 1] -
                                     aE * new[j, i + 1] - aS * new[j - 1, i] -
                                     aN * new[j + 1, i]) / aP
            new[0, :] = 0.0
            new[-1, :] = new[-2, :]
            new[:, -1] = new[:, -2]
            delta = new

            vprime = np.zeros_like(delta)
            for i in range(self.Nx):
                for j in range(1, self.Ny):
                    ddux = (delta[j, i] - delta[j - 1, i]) / self.dy
                    vprime[j, i] = vprime[j - 1, i] - self.dx * ddux

            u_abs = self.U0 + delta
            v_abs = self.V0 + vprime
            if n % 5 == 0:
                frames_u.append(u_abs.copy())
                frames_v.append(v_abs.copy())
        frames_u = np.array(frames_u)
        frames_v = np.array(frames_v)
        frames_u = np.nan_to_num(frames_u, nan=0.0, posinf=0.0, neginf=0.0)
        frames_v = np.nan_to_num(frames_v, nan=0.0, posinf=0.0, neginf=0.0)
        nframes = frames_u.shape[0]
        t_snap = np.arange(nframes) * 5 * self.dt
        return frames_u, frames_v, t_snap

    def compute_drag(self, frames_u, mu):
        drag = np.zeros(frames_u.shape[0])
        for k in range(frames_u.shape[0]):
            dud_y = (frames_u[k, 1, :] - frames_u[k, 0, :]) / self.dy
            drag[k] = mu * np.trapezoid(dud_y, self.x)
        return np.nan_to_num(drag)