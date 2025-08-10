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


class BlowSuctionSolver:
    """Solver for unsteady blowing/suction from a stationary wall.

    This class advances the linearized Navier--Stokes equations in a
    quiescent base flow.  The only forcing comes from a prescribed normal
    velocity at the wall (``y=0``).  A projection method is used to enforce
    incompressibility.

    Parameters
    ----------
    rho : float
        Fluid density.
    nu : float
        Kinematic viscosity.
    x, y : array_like
        1-D arrays defining the computational grid.
    dt : float
        Time step size.
    Nt : int
        Number of time steps to march.
    wall_func : callable
        Function of ``(t, x)`` giving the normal velocity at the wall.
    verbose : bool, optional
        If True, print progress information.
    """

    def __init__(self, rho, nu, x, y, dt, Nt, wall_func, verbose=False):
        self.rho = rho
        self.nu = nu
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.dt = dt
        self.Nt = Nt
        self.wall_func = wall_func
        self.verbose = verbose

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.time = np.arange(Nt) * dt

        self._build_poisson_matrix()
        self._build_helmholtz_matrix()

    def wall_profile(self, t):
        return self.wall_func(t, self.x)

    def _build_poisson_matrix(self):
        """Build Laplacian matrix for the pressure Poisson equation.

        A homogeneous Neumann boundary condition is applied on all walls and
        the pressure at the first interior cell is pinned to zero to remove the
        nullspace of the operator.
        """
        Nx_i = self.Nx - 2
        Ny_i = self.Ny - 2
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2

        rows, cols, data = [], [], []

        def idx(j, i):
            return j * Nx_i + i

        for j in range(Ny_i):
            for i in range(Nx_i):
                k = idx(j, i)
                aP = 0.0
                # west/east
                if i > 0:
                    rows.append(k); cols.append(idx(j, i - 1)); data.append(-1 / dx2)
                    aP += 1 / dx2
                else:
                    aP += 1 / dx2  # Neumann
                if i < Nx_i - 1:
                    rows.append(k); cols.append(idx(j, i + 1)); data.append(-1 / dx2)
                    aP += 1 / dx2
                else:
                    aP += 1 / dx2
                # south/north
                if j > 0:
                    rows.append(k); cols.append(idx(j - 1, i)); data.append(-1 / dy2)
                    aP += 1 / dy2
                else:
                    aP += 1 / dy2
                if j < Ny_i - 1:
                    rows.append(k); cols.append(idx(j + 1, i)); data.append(-1 / dy2)
                    aP += 1 / dy2
                else:
                    aP += 1 / dy2

                rows.append(k); cols.append(k); data.append(aP)

        N = Nx_i * Ny_i
        P = sparse.csr_matrix((data, (rows, cols)), shape=(N, N)).tolil()
        # pin one pressure point to remove singularity
        P[0, :] = 0.0
        P[0, 0] = 1.0
        self.P = P.tocsr()

    def _build_helmholtz_matrix(self):
        """Build matrix for implicit diffusion step (I - dt nu ∇²)."""
        Nx_i = self.Nx - 2
        Ny_i = self.Ny - 2
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        rx = self.nu * self.dt / dx2
        ry = self.nu * self.dt / dy2
        aP = 1 + 2 * rx + 2 * ry
        aW = aE = -rx
        aS = aN = -ry

        rows, cols, data = [], [], []

        def idx(j, i):
            return j * Nx_i + i

        for j in range(Ny_i):
            for i in range(Nx_i):
                k = idx(j, i)
                rows.append(k); cols.append(k); data.append(aP)
                if i > 0:
                    rows.append(k); cols.append(idx(j, i - 1)); data.append(aW)
                if i < Nx_i - 1:
                    rows.append(k); cols.append(idx(j, i + 1)); data.append(aE)
                if j > 0:
                    rows.append(k); cols.append(idx(j - 1, i)); data.append(aS)
                if j < Ny_i - 1:
                    rows.append(k); cols.append(idx(j + 1, i)); data.append(aN)

        N = Nx_i * Ny_i
        self.H = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

    def stability_report(self):
        """Return diffusive stability metric for the explicit scheme."""
        Diff_x = 2 * self.nu * self.dt / self.dx ** 2
        Diff_y = 2 * self.nu * self.dt / self.dy ** 2
        score = Diff_x + Diff_y
        if score > 1:
            tag = "UNSTABLE"
        elif score >= 0.5:
            tag = "marginal"
        else:
            tag = "stable"
        if self.verbose:
            print(
                f"Diff_x={Diff_x:.3f}, Diff_y={Diff_y:.3f}, Score={score:.3f} ({tag})"
            )
        return Diff_x, Diff_y, score

    def run_explicit(self):
        """March the solution using an explicit projection method."""
        u = np.zeros((self.Ny, self.Nx))
        v = np.zeros_like(u)
        p = np.zeros_like(u)
        frames_u = []
        frames_v = []
        if self.verbose:
            self.stability_report()
        for n, t in enumerate(self.time):
            if self.verbose and n % 10 == 0:
                print(f"[blow] step {n}/{self.Nt}")

            v[0, :] = self.wall_profile(t)
            u_star = u.copy()
            v_star = v.copy()

            lap_u = np.zeros_like(u)
            lap_u[1:-1, 1:-1] = (
                    (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / self.dx ** 2
                    + (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / self.dy ** 2
            )
            lap_v = np.zeros_like(v)
            lap_v[1:-1, 1:-1] = (
                    (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / self.dx ** 2
                    + (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / self.dy ** 2
            )
            u_star[1:-1, 1:-1] += self.dt * self.nu * lap_u[1:-1, 1:-1]
            v_star[1:-1, 1:-1] += self.dt * self.nu * lap_v[1:-1, 1:-1]

            u_star[0, :] = 0.0
            u_star[-1, :] = 0.0
            u_star[:, 0] = 0.0
            u_star[:, -1] = 0.0
            v_star[0, :] = self.wall_profile(t)
            v_star[-1, :] = 0.0
            v_star[:, 0] = 0.0
            v_star[:, -1] = 0.0

            div = (
                (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2 * self.dx)
                + (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2 * self.dy)
            )
            rhs = (self.rho / self.dt) * div
            p_int = spsolve(self.P, rhs.ravel())
            p[1:-1, 1:-1] = p_int.reshape(self.Ny - 2, self.Nx - 2)
            p[0, :] = 0.0
            p[-1, :] = 0.0
            p[:, 0] = 0.0
            p[:, -1] = 0.0

            dpdx = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * self.dx)
            dpdy = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * self.dy)
            u_new = u_star.copy()
            v_new = v_star.copy()
            u_new[1:-1, 1:-1] -= self.dt / self.rho * dpdx
            v_new[1:-1, 1:-1] -= self.dt / self.rho * dpdy

            u_new[0, :] = 0.0
            u_new[-1, :] = 0.0
            u_new[:, 0] = 0.0
            u_new[:, -1] = 0.0
            v_new[0, :] = self.wall_profile(t)
            v_new[-1, :] = 0.0
            v_new[:, 0] = 0.0
            v_new[:, -1] = 0.0

            u = u_new
            v = v_new
            frames_u.append(u.copy())
            frames_v.append(v.copy())

        frames_u = np.array(frames_u)
        frames_v = np.array(frames_v)
        return frames_u, frames_v, self.time

    def run(self):
        """Run the solver using an implicit diffusion step."""
        return self.run_implicit()

    def _implicit_step(self, field, bc):
        """Solve (I - dt nu ∇²) field = rhs with boundary conditions ``bc``.

        Parameters
        ----------
        field : ndarray
            Array containing the previous time level values.
        bc : ndarray
            Array with the boundary values for the new time level.
        """
        rx = self.nu * self.dt / self.dx ** 2
        ry = self.nu * self.dt / self.dy ** 2
        rhs = field.copy()
        rhs[0, :] = bc[0, :]
        rhs[-1, :] = bc[-1, :]
        rhs[:, 0] = bc[:, 0]
        rhs[:, -1] = bc[:, -1]
        b = rhs[1:-1, 1:-1].copy()
        b[:, 0] += rx * bc[1:-1, 0]
        b[:, -1] += rx * bc[1:-1, -1]
        b[0, :] += ry * bc[0, 1:-1]
        b[-1, :] += ry * bc[-1, 1:-1]
        sol = spsolve(self.H, b.ravel())
        out = bc.copy()
        out[1:-1, 1:-1] = sol.reshape(self.Ny - 2, self.Nx - 2)
        return out

    def run_implicit(self):
        """March the solution using an implicit diffusion projection method."""
        u = np.zeros((self.Ny, self.Nx))
        v = np.zeros_like(u)
        p = np.zeros_like(u)
        frames_u = []
        frames_v = []
        if self.verbose:
            self.stability_report()
        for n, t in enumerate(self.time):
            if self.verbose and n % 10 == 0:
                print(f"[blow] step {n}/{self.Nt}")

            bc_u = np.zeros_like(u)
            bc_v = np.zeros_like(v)
            bc_v[0, :] = self.wall_profile(t)

            u_star = self._implicit_step(u, bc_u)
            v_star = self._implicit_step(v, bc_v)

            div = (
                    (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2 * self.dx)
                    + (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2 * self.dy)
            )
            rhs = (self.rho / self.dt) * div
            rhs = rhs.ravel()
            rhs[0] = 0.0
            p_int = spsolve(self.P, rhs)
            p[1:-1, 1:-1] = p_int.reshape(self.Ny - 2, self.Nx - 2)
            p[0, 1:-1] = p[1, 1:-1]
            p[-1, 1:-1] = p[-2, 1:-1]
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]

            dpdx = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * self.dx)
            dpdy = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * self.dy)
            u_new = u_star.copy()
            v_new = v_star.copy()
            u_new[1:-1, 1:-1] -= self.dt / self.rho * dpdx
            v_new[1:-1, 1:-1] -= self.dt / self.rho * dpdy

            u_new[0, :] = 0.0
            u_new[-1, :] = 0.0
            u_new[:, 0] = 0.0
            u_new[:, -1] = 0.0
            v_new[0, :] = self.wall_profile(t)
            v_new[-1, :] = 0.0
            v_new[:, 0] = 0.0
            v_new[:, -1] = 0.0

            u = u_new
            v = v_new
            frames_u.append(u.copy())
            frames_v.append(v.copy())

        frames_u = np.array(frames_u)
        frames_v = np.array(frames_v)
        return frames_u, frames_v, self.time