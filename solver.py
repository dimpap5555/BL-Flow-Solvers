import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

class LinearBLSolver:
    """Linearized boundary layer solver using a given Blasius base flow."""

    def __init__(self, blasius_flow, rho, nu, dt, Nt, inlet_func, verbose=False):
        self.base = blasius_flow
        self.rho = rho
        self.nu = nu
        self.dt = dt
        self.Nt = Nt
        self.inlet_func = inlet_func
        self.verbose = verbose

        self.x = blasius_flow.x
        self.y = blasius_flow.y
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.U0 = blasius_flow.U0
        self.V0 = blasius_flow.V0
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.time = np.arange(Nt) * dt

        self._setup_linear_system()

    def inlet_profile(self, t):
        return self.inlet_func(t, self.y)

    def _setup_linear_system(self):
        """Pre-compute coefficient arrays and sparse matrix."""
        inv_dt = 1.0 / self.dt
        aP = (
            inv_dt
            + 2 * self.nu / self.dx ** 2
            + 2 * self.nu / self.dy ** 2
            + self.U0 / self.dx
            + self.V0 / self.dy
        )
        aW = -self.nu / self.dx ** 2 - self.U0 / self.dx
        aE = np.full_like(self.U0, -self.nu / self.dx ** 2)
        aS = -self.nu / self.dy ** 2 - self.V0 / self.dy
        aN = np.full_like(self.U0, -self.nu / self.dy ** 2)

        # store only interior coefficients
        self.aP = aP[1:-1, 1:-1]
        self.aW = aW[1:-1, 1:-1]
        self.aE = aE[1:-1, 1:-1]
        self.aS = aS[1:-1, 1:-1]
        self.aN = aN[1:-1, 1:-1]

        Nx_i = self.Nx - 2
        Ny_i = self.Ny - 2

        rows = []
        cols = []
        data = []

        def idx(j, i):
            return j * Nx_i + i

        for j in range(Ny_i):
            for i in range(Nx_i):
                k = idx(j, i)
                data.append(self.aP[j, i])
                rows.append(k)
                cols.append(k)
                if i > 0:
                    rows.append(k)
                    cols.append(idx(j, i - 1))
                    data.append(self.aW[j, i])
                if i < Nx_i - 1:
                    rows.append(k)
                    cols.append(idx(j, i + 1))
                    data.append(self.aE[j, i])
                if j > 0:
                    rows.append(k)
                    cols.append(idx(j - 1, i))
                    data.append(self.aS[j, i])
                if j < Ny_i - 1:
                    rows.append(k)
                    cols.append(idx(j + 1, i))
                    data.append(self.aN[j, i])

        N = Nx_i * Ny_i
        self.A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

    def stability_report(self):
        """Return CFL and diffusive stability metrics.

        Stability score guidelines:
        > 1   : expect instability
        0.5-1 : marginally stable
        < 0.5 : safe
        """
        umax = np.max(np.abs(self.U0))
        vmax = np.max(np.abs(self.V0))
        CFL_x = umax * self.dt / self.dx
        CFL_y = vmax * self.dt / self.dy
        Diff_x = 2 * self.nu * self.dt / self.dx ** 2
        Diff_y = 2 * self.nu * self.dt / self.dy ** 2
        score = max(CFL_x, CFL_y, Diff_x, Diff_y)
        if score > 1:
            tag = "UNSTABLE"
        elif score >= 0.5:
            tag = "marginal"
        else:
            tag = "stable"
        if self.verbose:
            print(
                f"CFL_x={CFL_x:.3f}, CFL_y={CFL_y:.3f}, Diff_x={Diff_x:.3f}, "
                f"Diff_y={Diff_y:.3f}, Score={score:.3f} ({tag})"
            )
        return CFL_x, CFL_y, Diff_x, Diff_y, score

    def run_explicit(self):
        delta = np.zeros((self.Ny, self.Nx))
        frames_u = []
        frames_v = []
        if self.verbose:
            self.stability_report()
        for n, t in enumerate(self.time):
            if self.verbose and n % 10 == 0:
                print(f"[explicit] step {n}/{self.Nt}")
            delta[:, 0] = self.inlet_profile(t)
            new = delta.copy()
            dudx = (delta[1:-1, 1:-1] - delta[1:-1, :-2]) / self.dx
            dudy = (delta[1:-1, 1:-1] - delta[:-2, 1:-1]) / self.dy
            adv = -self.U0[1:-1, 1:-1] * dudx - self.V0[1:-1, 1:-1] * dudy
            d2udx2 = (delta[1:-1, 2:] - 2 * delta[1:-1, 1:-1] + delta[1:-1, :-2]) / self.dx ** 2
            d2udy2 = (delta[2:, 1:-1] - 2 * delta[1:-1, 1:-1] + delta[:-2, 1:-1]) / self.dy ** 2
            diff = self.nu * (d2udx2 + d2udy2)
            new[1:-1, 1:-1] = delta[1:-1, 1:-1] + self.dt * (adv + diff)
            new[0, :] = 0.0
            new[-1, :] = new[-2, :]
            new[:, -1] = new[:, -2]
            delta = new

            vprime = np.zeros_like(delta)
            for i in range(1, self.Nx):
                for j in range(1, self.Ny):
                    dudx = (delta[j, i] - delta[j, i - 1]) / self.dx
                    vprime[j, i] = vprime[j - 1, i] - self.dy * dudx

            u_abs = self.U0 + delta
            v_abs = self.V0 + vprime

            frames_u.append(u_abs.copy())
            frames_v.append(v_abs.copy())
        frames_u = np.array(frames_u)
        frames_v = np.array(frames_v)
        frames_u = np.nan_to_num(frames_u, nan=0.0, posinf=0.0, neginf=0.0)
        frames_v = np.nan_to_num(frames_v, nan=0.0, posinf=0.0, neginf=0.0)
        nframes = frames_u.shape[0]
        t_snap = np.arange(nframes) * self.dt
        return frames_u, frames_v, t_snap

    def run_implicit(self, n_iter=20, tol=1e-6):
        delta = np.zeros((self.Ny, self.Nx))
        frames_u = []
        frames_v = []
        inv_dt = 1.0 / self.dt
        Nx_i = self.Nx - 2
        Ny_i = self.Ny - 2
        if self.verbose:
            self.stability_report()
        for n, t in enumerate(self.time):
            if self.verbose and n % 10 == 0:
                print(f"[implicit] step {n}/{self.Nt}")
            delta[:, 0] = self.inlet_profile(t)
            rhs = inv_dt * delta
            new = delta.copy()
            for it in range(n_iter):
                old = new.copy()
                b = rhs[1:-1, 1:-1].copy()

                # apply boundary contributions
                b[:, 0] -= self.aW[:, 0] * delta[1:-1, 0]
                b[:, -1] -= self.aE[:, -1] * delta[1:-1, -1]
                b[0, :] -= self.aS[0, :] * delta[0, 1:-1]
                b[-1, :] -= self.aN[-1, :] * delta[-1, 1:-1]

                sol = spsolve(self.A, b.ravel())
                new[1:-1, 1:-1] = sol.reshape(Ny_i, Nx_i)

                res = np.linalg.norm(new - old)
                if self.verbose and it % 1 == 0:
                    print(f"  iter {it}: residual {res:.2e}")
                if res < tol:
                    break
            new[0, :] = 0.0
            new[-1, :] = new[-2, :]
            new[:, -1] = new[:, -2]
            delta = new

            vprime = np.zeros_like(delta)
            for i in range(1, self.Nx):
                for j in range(1, self.Ny):
                    dudx = (delta[j, i] - delta[j, i - 1]) / self.dx
                    vprime[j, i] = vprime[j - 1, i] - self.dy * dudx

            u_abs = self.U0 + delta
            v_abs = self.V0 + vprime

            frames_u.append(u_abs.copy())
            frames_v.append(v_abs.copy())
        frames_u = np.array(frames_u)
        frames_v = np.array(frames_v)
        frames_u = np.nan_to_num(frames_u, nan=0.0, posinf=0.0, neginf=0.0)
        frames_v = np.nan_to_num(frames_v, nan=0.0, posinf=0.0, neginf=0.0)
        nframes = frames_u.shape[0]
        t_snap = np.arange(nframes) * self.dt
        return frames_u, frames_v, t_snap

    def compute_drag(self, frames_u, mu):
        drag = np.zeros(frames_u.shape[0])
        for k in range(frames_u.shape[0]):
            dud_y = (frames_u[k, 1, :] - frames_u[k, 0, :]) / self.dy
            drag[k] = mu * np.trapezoid(dud_y, self.x)
        return np.nan_to_num(drag)
