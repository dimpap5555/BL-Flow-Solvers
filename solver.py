import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu

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
    """Fully explicit solver for blowing/suction from a stationary wall.

    The algorithm follows a four-step explicit projection with an
    artificial-compressibility style pressure correction. No Poisson solve
    is required.

    Parameters
    ----------
    rho : float
        Fluid density.
    nu : float
        Kinematic viscosity.
    x, y : array_like
        Grid coordinates.
    dt : float
        Time step size.
    Nt : int
        Number of time steps to integrate.
    wall_func : callable
        Function ``wall_func(t, x)`` prescribing the wall-normal velocity at
        ``y=0``.
    cp : float, optional
        Explicit pressure correction coefficient (0.5-1.0).
    verbose : bool, optional
        Print stability information when True.
    """

    def __init__(self, rho, nu, x, y, dt, Nt, wall_func, cp=0.7, verbose=False):
        self.rho = rho
        self.nu = nu
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.dt = dt
        self.Nt = Nt
        self.wall_func = wall_func
        self.cp = cp
        self.verbose = verbose

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.time = np.arange(Nt) * dt

        # Matrices for the implicit projection scheme
        self._diff_A = None
        self._pois_A = None
        self._pois_lu = None
        self._Gx = self._Gy = None
        self._Du = self._Dv = None
        self._Bbottom = None
        self._aP = self._aW = self._aE = self._aS = self._aN = None
        # store interior sizes for reuse
        self._Nx_i = self._Ny_i = None

    def wall_profile(self, t):
        return self.wall_func(t, self.x)

    def _build_matrix(self, aP, aW, aE, aS, aN):
        """Assemble a 5-point stencil sparse matrix with Dirichlet boundaries."""
        Nx_i = self.Nx - 2
        Ny_i = self.Ny - 2
        rows, cols, data = [], [], []

        def idx(j, i):
            return j * Nx_i + i

        for j in range(Ny_i):
            for i in range(Nx_i):
                k = idx(j, i)
                rows.append(k)
                cols.append(k)
                data.append(aP)
                if i > 0:
                    rows.append(k)
                    cols.append(idx(j, i - 1))
                    data.append(aW)
                if i < Nx_i - 1:
                    rows.append(k)
                    cols.append(idx(j, i + 1))
                    data.append(aE)
                if j > 0:
                    rows.append(k)
                    cols.append(idx(j - 1, i))
                    data.append(aS)
                if j < Ny_i - 1:
                    rows.append(k)
                    cols.append(idx(j + 1, i))
                    data.append(aN)

        N = Nx_i * Ny_i
        return sparse.csr_matrix((data, (rows, cols)), shape=(N, N))


    def _setup_implicit(self, theta):
        """Precompute matrices for the implicit solver.

        Parameters
        ----------
        theta : float
            Crank–Nicolson weighting factor. ``theta=1`` gives backward Euler
            while ``theta=0.5`` corresponds to the unconditionally stable
            Crank–Nicolson scheme.
        """
        rebuild = self._diff_A is None or getattr(self, "_theta", None) != theta

        if rebuild:
            inv_dt = 1.0 / self.dt
            aP = (
                inv_dt
                + 2 * theta * self.nu / self.dx ** 2
                + 2 * theta * self.nu / self.dy ** 2
            )
            aW = -theta * self.nu / self.dx ** 2
            aE = -theta * self.nu / self.dx ** 2
            aS = -theta * self.nu / self.dy ** 2
            aN = -theta * self.nu / self.dy ** 2
            self._aP, self._aW, self._aE, self._aS, self._aN = aP, aW, aE, aS, aN
            self._diff_A = self._build_matrix(aP, aW, aE, aS, aN)
            self._diff_lu = splu(self._diff_A.tocsc())
            self._theta = theta

        # store interior sizes
        Nx_i = self.Nx - 2
        Ny_i = self.Ny - 2
        self._Nx_i = Nx_i
        self._Ny_i = Ny_i

        if self._Gx is None:
            # 1D derivative matrices with one-sided differences at ends
            def d1(n, h):
                rows, cols, data = [], [], []
                for i in range(n):
                    if i == 0:
                        rows += [i, i]
                        cols += [0, 1]
                        data += [-1.0 / h, 1.0 / h]
                    elif i == n - 1:
                        rows += [i, i]
                        cols += [n - 2, n - 1]
                        data += [-1.0 / h, 1.0 / h]
                    else:
                        rows += [i, i]
                        cols += [i - 1, i + 1]
                        data += [-0.5 / h, 0.5 / h]
                return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

            G1x = d1(Nx_i, self.dx)
            G1y = d1(Ny_i, self.dy)
            Ix = sparse.eye(Nx_i, format="csr")
            Iy = sparse.eye(Ny_i, format="csr")
            self._Gx = sparse.kron(Iy, G1x, format="csr")
            self._Gy = sparse.kron(G1y, Ix, format="csr")
            self._Du = -self._Gx.transpose().tocsr()
            self._Dv = -self._Gy.transpose().tocsr()

            A = self._Du @ self._Gx + self._Dv @ self._Gy
            A = A.tolil()
            # Dirichlet on top, left, right
            for i in range(Nx_i):
                k = (Ny_i - 1) * Nx_i + i
                A[k, :] = 0.0
                A[k, k] = 1.0
            for j in range(Ny_i):
                k_left = j * Nx_i
                k_right = j * Nx_i + (Nx_i - 1)
                A[k_left, :] = 0.0
                A[k_left, k_left] = 1.0
                A[k_right, :] = 0.0
                A[k_right, k_right] = 1.0
            self._pois_A = A.tocsc()
            self._pois_lu = splu(self._pois_A)

            # Bottom boundary source operator
            rows, cols, data = [], [], []
            for i in range(Nx_i):
                k = i  # bottom row index j=0
                rows.append(k)
                cols.append(i + 1)
                data.append(-1.0 / self.dy)
            self._Bbottom = sparse.csr_matrix((data, (rows, cols)), shape=(Nx_i * Ny_i, self.Nx))

    def _assemble_poisson_rhs(self, u_star, v_star, wall_v):
        u_int = u_star[1:-1, 1:-1].ravel()
        v_int = v_star[1:-1, 1:-1].ravel()
        div_flat = (self._Du @ u_int) + (self._Dv @ v_int)
        div_flat = div_flat + (self._Bbottom @ wall_v)
        return (self.rho / self.dt) * div_flat

    def stability_report(self):
        """Return diffusion and pressure stability metrics."""
        diff_limit = (self.dx ** 2 * self.dy ** 2) / (2 * self.nu * (self.dx ** 2 + self.dy ** 2))
        diff_score = self.dt / diff_limit

        press_limit = min(self.dx, self.dy) / self.cp
        press_score = self.dt / press_limit

        score = max(diff_score, press_score)
        if score > 1:
            tag = "UNSTABLE"
        elif score >= 0.5:
            tag = "marginal"
        else:
            tag = "stable"
        if self.verbose:
            print(f"Diff={diff_score:.3f}, Press={press_score:.3f}, Score={score:.3f} ({tag})")
        return diff_score, press_score, score

    def run_explicit(self):
        """Advance the solution using the explicit projection scheme.

        Returns
        -------
        frames_u : ndarray
            Time history of the streamwise velocity.
        frames_v : ndarray
            Time history of the wall-normal velocity.
        frames_p : ndarray
            Time history of the pressure field.
        time : ndarray
            Array of time snapshots.
        """
        u = np.zeros((self.Ny, self.Nx))
        v = np.zeros_like(u)
        p = np.zeros_like(u)
        frames_u = []
        frames_v = []
        frames_p = []
        if self.verbose:
            self.stability_report()
        for n, t in enumerate(self.time):
            if self.verbose and n % 10 == 0:
                print(f"[blow] step {n}/{self.Nt}")

            # Step 1: predictor
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

            dpdx = np.zeros_like(u)
            dpdx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * self.dx)
            dpdy = np.zeros_like(v)
            dpdy[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * self.dy)

            u_star[1:-1, 1:-1] += self.dt * (self.nu * lap_u[1:-1, 1:-1] - (1 / self.rho) * dpdx[1:-1, 1:-1])
            v_star[1:-1, 1:-1] += self.dt * (self.nu * lap_v[1:-1, 1:-1] - (1 / self.rho) * dpdy[1:-1, 1:-1])

            # Apply boundary conditions (no-change except blow/suction wall)
            u_star[0, :] = 0.0
            u_star[-1, :] = u_star[-2, :]
            u_star[:, 0] = u_star[:, 1]
            u_star[:, -1] = u_star[:, -2]
            v_star[0, :] = self.wall_profile(t)
            v_star[-1, :] = v_star[-2, :]
            v_star[:, 0] = v_star[:, 1]
            v_star[:, -1] = v_star[:, -2]

            # Step 2: divergence of tentative velocity
            div = (
                (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2 * self.dx)
                + (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2 * self.dy)
            )

            # Step 3: explicit pressure correction
            p[1:-1, 1:-1] -= self.rho * self.cp * self.dt * div
            # remove null space by enforcing zero mean, but do not
            # overwrite boundary values – the pressure increment already
            # satisfied the desired boundary condition
            p -= np.mean(p)

            # Step 4: velocity projection
            dpdx_new = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * self.dx)
            dpdy_new = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * self.dy)
            u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - (self.dt / self.rho) * dpdx_new
            v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - (self.dt / self.rho) * dpdy_new

            u[0, :] = 0.0
            u[-1, :] = u[-2, :]
            u[:, 0] = u[:, 1]
            u[:, -1] = u[:, -2]
            v[0, :] = self.wall_profile(t)
            v[-1, :] = v[-2, :]
            v[:, 0] = v[:, 1]
            v[:, -1] = v[:, -2]

            frames_u.append(u.copy())
            frames_v.append(v.copy())
            frames_p.append(p.copy())

        frames_u = np.array(frames_u)
        frames_v = np.array(frames_v)
        frames_p = np.array(frames_p)
        return frames_u, frames_v, frames_p, self.time

    def run_implicit(self, theta=0.5):
        """Advance the solution using a fully implicit projection scheme."""
        u = np.zeros((self.Ny, self.Nx))
        v = np.zeros_like(u)
        p = np.zeros_like(u)
        frames_u, frames_v, frames_p = [], [], []

        self._setup_implicit(theta)
        Nx_i, Ny_i = self._Nx_i, self._Ny_i

        for n, t in enumerate(self.time):
            wall_v = self.wall_profile(t)

            u_star = u.copy()
            v_star = v.copy()

            u_star[0, :] = 0.0
            v_star[0, :] = wall_v
            u_star[-1, :] = u_star[-2, :]
            u_star[:, 0] = u_star[:, 1]
            u_star[:, -1] = u_star[:, -2]
            v_star[-1, :] = v_star[-2, :]
            v_star[:, 0] = v_star[:, 1]
            v_star[:, -1] = v_star[:, -2]

            rhs_flat = self._assemble_poisson_rhs(u_star, v_star, wall_v)
            phi_int = self._pois_lu.solve(rhs_flat)

            res = self._pois_A @ phi_int - rhs_flat
            res_inf = float(np.max(np.abs(res)))
            rhs_inf = float(np.max(np.abs(rhs_flat))) if rhs_flat.size else 0.0
            if self.verbose:
                print(f"[pois] res_inf={res_inf:.3e}")
            assert res_inf <= 1e-8 * max(1.0, rhs_inf), "poisson residual too large"

            gradx_phi = (self._Gx @ phi_int).reshape(Ny_i, Nx_i)
            grady_phi = (self._Gy @ phi_int).reshape(Ny_i, Nx_i)

            u_new = u_star.copy()
            v_new = v_star.copy()
            u_new[1:-1, 1:-1] -= (self.dt / self.rho) * gradx_phi
            v_new[1:-1, 1:-1] -= (self.dt / self.rho) * grady_phi

            u_new[0, :] = 0.0
            v_new[0, :] = wall_v
            u_new[-1, :] = u_new[-2, :]
            u_new[:, 0] = u_new[:, 1]
            u_new[:, -1] = u_new[:, -2]
            v_new[-1, :] = v_new[-2, :]
            v_new[:, 0] = v_new[:, 1]
            v_new[:, -1] = v_new[:, -2]

            p[1:-1, 1:-1] += phi_int.reshape(Ny_i, Nx_i)

            rhs_chk = self._assemble_poisson_rhs(u_new, v_new, wall_v)
            max_div = np.max(np.abs(rhs_chk)) * self.dt / self.rho
            if self.verbose:
                print(f"[chk] max_div={max_div:.3e}")
                wall_flux = np.trapezoid(v_new[0, :], self.x)
                top_flux = np.trapezoid(v_new[-1, :], self.x)
                left_flux = np.trapezoid(u_new[:, 0], self.y)
                right_flux = np.trapezoid(u_new[:, -1], self.y)
                mass_err = wall_flux + top_flux + left_flux + right_flux
                print(
                    f"[step {n}] wall={wall_flux:.3e}, top={top_flux:.3e}, left={left_flux:.3e}, right={right_flux:.3e}, mass_err={mass_err:.3e}"
                )
                if abs(mass_err) > abs(wall_flux) * 1e-8:
                    print(f"[warn] mass imbalance {mass_err:.3e}")
            tol_div = 1e-6 if n == 0 else 1e-10
            assert max_div <= tol_div, "divergence exceeds tolerance"

            u, v = u_new, v_new
            frames_u.append(u.copy())
            frames_v.append(v.copy())
            frames_p.append(p.copy())

        return np.array(frames_u), np.array(frames_v), np.array(frames_p), self.time
