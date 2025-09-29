import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu

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
        self.cp = 0.2

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.time = np.arange(Nt) * dt

        # 🔷 Prebuild Neumann Poisson matrix for (Ny, Nx) layout (row-major flatten)
        self.A = self._build_neumann_poisson_matrix(self.Ny, self.Nx, self.dx, self.dy)

        # ✅ bake in reference row once (p[0,0]=1) and factorize
        A_ref = self.A.tolil(copy=True)
        A_ref[0, :] = 0.0
        A_ref[0, 0] = 1.0
        self._A_ref = A_ref.tocsc()
        self._lu = splu(self._A_ref)  # 🔒 factorized once; reuse every step

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

    def wall_profile(self, t):
        return self.wall_func(t, self.x)

    def run_explicit(self):
        """Advance the solution using the explicit projection scheme."""
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

            # Step 1: predictor (diffusion + pressure gradient)
            u_star = u.copy()
            v_star = v.copy()
            u_star, v_star = self._apply_velocity_bcs(u_star, v_star, t)

            lap_u = self._laplacian(u)
            lap_v = self._laplacian(v)
            dpdx, dpdy = self._pressure_gradient(p)

            u_star[1:-1, 1:-1] += self.dt * (
                self.nu * lap_u[1:-1, 1:-1] - (1 / self.rho) * dpdx[1:-1, 1:-1]
            )
            v_star[1:-1, 1:-1] += self.dt * (
                self.nu * lap_v[1:-1, 1:-1] - (1 / self.rho) * dpdy[1:-1, 1:-1]
            )
            u_star, v_star = self._apply_velocity_bcs(u_star, v_star, t)

            # Step 2: divergence of tentative velocity
            div = self._divergence(u_star, v_star)[1:-1, 1:-1]

            # Step 3: explicit pressure correction
            p[1:-1, 1:-1] -= self.rho * self.cp * self.dt * div

            # Step 4: velocity projection
            dpdx_new, dpdy_new = self._pressure_gradient(p)
            u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - (self.dt / self.rho) * dpdx_new[1:-1, 1:-1]
            v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - (self.dt / self.rho) * dpdy_new[1:-1, 1:-1]

            u, v = self._apply_velocity_bcs(u, v, t)

            frames_u.append(u.copy())
            frames_v.append(v.copy())

        return np.array(frames_u), np.array(frames_v), self.time

    def _laplacian(self, f):
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (
            (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]) / self.dx ** 2
            + (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / self.dy ** 2
        )
        return lap

    def _pressure_gradient(self, p):
        dpdx = np.zeros_like(p)
        dpdy = np.zeros_like(p)
        dpdx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * self.dx)
        dpdy[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * self.dy)
        return dpdx, dpdy

    def _implicit_diffusion(self, u, n_iter=200, rtol=1e-8):
        u_new = u.copy()
        coef = 1.0 + 2.0 * self.nu * self.dt * (1.0 / self.dx ** 2 + 1.0 / self.dy ** 2)
        inv_coef = 1.0 / coef
        for _ in range(n_iter):
            u_old = u_new.copy()
            u_new[1:-1, 1:-1] = (
                                        u[1:-1, 1:-1] + self.nu * self.dt * (
                                        (u_old[1:-1, 2:] + u_old[1:-1, :-2]) / self.dx ** 2 +
                                        (u_old[2:, 1:-1] + u_old[:-2, 1:-1]) / self.dy ** 2
                                )
                                ) * inv_coef
            # keep BCs hard during the iteration
            # NOTE: pass a dummy time 't' from caller if needed; here assume caller re-applies after return.
            # If not passing t, copy edge interior to enforce Neumann and set wall row explicitly:
            u_new[0, :] = u[0, :]
            u_new[-1, :] = u_new[-2, :]
            u_new[:, 0] = u_new[:, 1]
            u_new[:, -1] = u_new[:, -2]
            if np.max(np.abs(u_new[1:-1, 1:-1] - u_old[1:-1, 1:-1])) < rtol:
                break
        return u_new

    def _divergence(self, u, v):
        div = np.zeros_like(u)
        div[1:-1, 1:-1] = (
            (u[1:-1, 2:] - u[1:-1, :-2])/(2*self.dx) +
            (v[2:, 1:-1] - v[:-2, 1:-1])/(2*self.dy)
        )
        return div

    def _correct_velocity(self, u_star, v_star, p):
        u = u_star.copy()
        v = v_star.copy()
        u[1:-1, 1:-1] -= self.dt/self.rho * (p[1:-1, 2:] - p[1:-1, :-2])/(2*self.dx)
        v[1:-1, 1:-1] -= self.dt/self.rho * (p[2:, 1:-1] - p[:-2, 1:-1])/(2*self.dy)
        return u, v

    def _apply_velocity_bcs(self, u, v, t):
        """
        Prototype BCs:
          - y=0 (wall): u=0, v=Vw(t,x)
          - x=0, x=Lx, y=Ly: Neumann (copy interior)
        """
        Vw = self.wall_profile(t)
        # y=0 wall
        u[0, :] = 0.0
        v[0, :] = Vw
        # y=Ly top (Neumann)
        u[-1, :] = u[-2, :]
        v[-1, :] = v[-2, :]
        # x=0 left (Neumann)
        u[:, 0] = u[:, 1]
        v[:, 0] = v[:, 1]
        # x=Lx right (Neumann)
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        return u, v

    # -------------------- sparse Poisson (Neumann) --------------------
    def _build_neumann_poisson_matrix(self, Ny, Nx, dx, dy):
        """
        5-pt Laplacian on (Ny,Nx) with homogeneous Neumann on all boundaries.
        Row-major flattening: k = j*Nx + i.
        Uses ghost mirroring by adding an extra off-diagonal to the single interior neighbor.
        We later pin p[0,0]=0 to remove singularity.
        """

        dx2, dy2 = dx * dx, dy * dy
        N = Ny * Nx
        rows, cols, data = [], [], []

        def idx(j, i):  # (y,x)
            return j * Nx + i

        for j in range(Ny):
            for i in range(Nx):
                k = idx(j, i)
                sum_off = 0.0

                # --- X neighbors ---
                if Nx > 1:
                    # left neighbor contribution
                    if i > 0:
                        rows.append(k);
                        cols.append(idx(j, i - 1));
                        data.append(1.0 / dx2)
                        sum_off += 1.0 / dx2
                    else:
                        # Neumann at left: mirror to right (extra coupling to i+1)
                        rows.append(k);
                        cols.append(idx(j, i + 1));
                        data.append(1.0 / dx2)
                        sum_off += 1.0 / dx2

                    # right neighbor contribution
                    if i < Nx - 1:
                        rows.append(k);
                        cols.append(idx(j, i + 1));
                        data.append(1.0 / dx2)
                        sum_off += 1.0 / dx2
                    else:
                        # Neumann at right: mirror to left (extra coupling to i-1)
                        rows.append(k);
                        cols.append(idx(j, i - 1));
                        data.append(1.0 / dx2)
                        sum_off += 1.0 / dx2
                # else Nx==1 => no x-coupling

                # --- Y neighbors ---
                if Ny > 1:
                    # down neighbor
                    if j > 0:
                        rows.append(k);
                        cols.append(idx(j - 1, i));
                        data.append(1.0 / dy2)
                        sum_off += 1.0 / dy2
                    else:
                        # Neumann at bottom: mirror to up
                        rows.append(k);
                        cols.append(idx(j + 1, i));
                        data.append(1.0 / dy2)
                        sum_off += 1.0 / dy2

                    # up neighbor
                    if j < Ny - 1:
                        rows.append(k);
                        cols.append(idx(j + 1, i));
                        data.append(1.0 / dy2)
                        sum_off += 1.0 / dy2
                    else:
                        # Neumann at top: mirror to down
                        rows.append(k);
                        cols.append(idx(j - 1, i));
                        data.append(1.0 / dy2)
                        sum_off += 1.0 / dy2
                # else Ny==1 => no y-coupling

                # center diagonal = -sum(off-diagonals)
                rows.append(k);
                cols.append(k);
                data.append(-sum_off)

        # build CSC; duplicates are summed by SciPy
        A = sparse.csc_matrix((data, (rows, cols)), shape=(N, N))

        # Pin reference pressure p[0,0]=0 (replace row with identity)
        A = A.tolil()
        A[0, :] = 0.0
        A[0, 0] = 1.0
        return A.tocsc()

    def _pressure_poisson_sparse(self, rhs, p_prev=None):
        """
        Solve ∇²p = rhs with Neumann everywhere + reference node p[0,0]=0.
        Uses pre-factorized LU from __init__.
        """
        b = rhs.ravel().copy()
        b[0] = 0.0  # match the baked reference row
        p_flat = self._lu.solve(b)
        return p_flat.reshape((self.Ny, self.Nx))

    def run_implicit(self, diffusion_iters=100):
        u = np.zeros((self.Ny, self.Nx))
        v = np.zeros_like(u)
        p = np.zeros_like(u)

        frames_u, frames_v = [], []

        for n, t in enumerate(self.time):
            if self.verbose and n % 10 == 0:
                print(f"[blow-implicit] step {n}/{self.Nt}")

            # 1) BCs
            u, v = self._apply_velocity_bcs(u, v, t)

            # 2) Implicit diffusion (Jacobi sweeps)
            u_star = self._implicit_diffusion(u, n_iter=diffusion_iters)
            v_star = self._implicit_diffusion(v, n_iter=diffusion_iters)
            u_star, v_star = self._apply_velocity_bcs(u_star, v_star, t)

            # 3) Pressure (sparse Neumann)
            div_star = self._divergence(u_star, v_star)
            rhs = (self.rho / self.dt) * div_star
            rhs -= rhs.mean()  # ensure compatibility for Neumann Poisson
            p = self._pressure_poisson_sparse(rhs, p_prev=p)

            # 4) Correction + BCs
            u, v = self._correct_velocity(u_star, v_star, p)
            u, v = self._apply_velocity_bcs(u, v, t)

            frames_u.append(u.copy())
            frames_v.append(v.copy())

        return np.array(frames_u), np.array(frames_v), self.time


class LinearizedJointSolver:
    """
    Fully-implicit linearized Navier–Stokes about a steady Blasius base (U0,V0),
    with wall blow/suction at y=0 and free-flow (Neumann) on all other sides.
    Projection method with Neumann Poisson + reference node.

    Minimal structural changes from BlowSuctionSolver:
      - same projection pipeline
      - same Poisson assembly (Neumann + pin)
      - predictor now solves *implicit* convection–diffusion–production:
            (I - dt C - dt*diag(Ux)) u* = u^n + dt*diag(Uy) v^(k)
            (I - dt C - dt*diag(Vy)) v* = v^n + dt*diag(Vx) u^(k+1)
        where C = ν∇² + U0∂x + V0∂y  (all linear, from base)
      - coupling handled by a few block Gauss–Seidel inner sweeps
    """

    def __init__(self, blasius_flow, rho, nu, dt, Nt, wall_func, verbose=False,
                 gs_sweeps=2):
        # --- base & grid
        self.base = blasius_flow
        self.rho = rho
        self.nu = nu
        self.dt = dt
        self.Nt = Nt
        self.wall_func = wall_func
        self.verbose = verbose
        self.gs_sweeps = gs_sweeps  # inner block GS iterations per step

        self.x = blasius_flow.x
        self.y = blasius_flow.y
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.time = np.arange(Nt) * dt

        # Blasius base fields on grid (assumed provided by your base class)
        # U0,V0, and their spatial derivatives:
        # Ux = ∂U0/∂x, Uy = ∂U0/∂y, Vx = ∂V0/∂x, Vy = ∂V0/∂y
        self.U0 = blasius_flow.U0
        self.V0 = blasius_flow.V0

        # ✅ Derivatives: use if provided, else compute numerically (centered)
        self.Ux = getattr(blasius_flow, "Ux", None)
        self.Uy = getattr(blasius_flow, "Uy", None)
        self.Vx = getattr(blasius_flow, "Vx", None)
        self.Vy = getattr(blasius_flow, "Vy", None)
        if any(g is None for g in (self.Ux, self.Uy, self.Vx, self.Vy)):
            self.Ux, self.Uy = self._grad_xy(self.U0)
            self.Vx, self.Vy = self._grad_xy(self.V0)

        # --- Poisson (Neumann) for pressure (row-major flatten)
        self.A_p = self._build_neumann_poisson_matrix(self.Ny, self.Nx, self.dx, self.dy)
        A_ref = self.A_p.tolil(copy=True)
        A_ref[0, :] = 0.0
        A_ref[0, 0] = 1.0
        self._lu_p = splu(A_ref.tocsc())

        # --- Fully-implicit predictor matrices for u* and v* on the interior
        # Assemble once, LU once, reuse
        (self.Au_csr, self.Av_csr,
         self.aW_u, self.aE_u, self.aS_u, self.aN_u,
         self.aW_v, self.aE_v, self.aS_v, self.aN_v) = self._build_predictor_mats()

        self._lu_u = splu(self.Au_csr.tocsc())
        self._lu_v = splu(self.Av_csr.tocsc())

    # ---------------- utilities ----------------
    def _grad_xy(self, F):
        Fx = np.zeros_like(F);
        Fy = np.zeros_like(F)
        Fx[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2 * self.dx)
        Fy[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2 * self.dy)
        # Neumann copies
        Fx[:, 0] = Fx[:, 1];
        Fx[:, -1] = Fx[:, -2]
        Fy[0, :] = Fy[1, :];
        Fy[-1, :] = Fy[-2, :]
        # 🔒 soften leading-edge spike (x=0 column)
        Fx[:, 0] = 0.0
        return Fx, Fy

    # ------------------------- public API -------------------------

    def stability_report(self):
        """CFL/viscous metrics (accuracy-oriented; not a hard limit due to full implicit predictor)."""
        umax = float(np.max(np.abs(self.U0)))
        vmax = float(np.max(np.abs(self.V0)))
        CFLx = umax * self.dt / self.dx
        CFLy = vmax * self.dt / self.dy
        Difx = 2*self.nu * self.dt / self.dx**2
        Dify = 2*self.nu * self.dt / self.dy**2
        score = max(CFLx, CFLy, Difx, Dify)
        if self.verbose:
            print(f"CFLx={CFLx:.3f}, CFLy={CFLy:.3f}, Difx={Difx:.3f}, Dify={Dify:.3f}, score≈{score:.3f}")
        return CFLx, CFLy, Difx, Dify, score

    def wall_profile(self, t):
        # wall_func expects (t, x) -> normal velocity at wall
        return self.wall_func(t, self.x)

    def run(self, diffusion_tol=1e-10):
        """
        Fully-implicit linear step + projection at each time.
        Returns: frames_u, frames_v, time
        """
        u = np.zeros((self.Ny, self.Nx))
        v = np.zeros_like(u)
        p = np.zeros_like(u)

        frames_u, frames_v = [], []

        for n, t in enumerate(self.time):
            if self.verbose and (n % 10 == 0 or n == self.Nt-1):
                print(f"[joint-implicit] step {n}/{self.Nt-1}")

            # 0) impose BCs on current fields
            u, v = self._apply_velocity_bcs(u, v, t)

            # 1) Fully-implicit predictor via block GS (u*, v*)
            u_star, v_star = self._fully_implicit_predictor(u, v, t)

            # 2) Poisson (Neumann) for pressure
            div_star = self._divergence(u_star, v_star)
            rhs = (self.rho / self.dt) * div_star
            rhs -= rhs.mean()  # Neumann compatibility
            p = self._pressure_poisson_sparse(rhs, p_prev=p)

            # 3) Projection + BCs
            u, v = self._correct_velocity(u_star, v_star, p)
            u, v = self._apply_velocity_bcs(u, v, t)

            frames_u.append(u.copy())
            frames_v.append(v.copy())

        return np.array(frames_u), np.array(frames_v), self.time

    # ---------------------- predictor assembly ----------------------

    def _build_predictor_mats(self):
        """
        Build interior matrices for:
          (I/dt - ν∇² - U0∂x - V0∂y - Ux) u* = u^n + Uy v^(k)
          (I/dt - ν∇² - U0∂x - V0∂y - Vy) v* = v^n + Vx u^(k+1)
        Using a compact 5-pt stencil + first-order upwind for base advection
        (consistent with your LinearBLSolver), interior unknowns only.
        Boundary contributions are applied to RHS each solve.
        """
        inv_dt = 1.0 / self.dt

        # coefficients on full grid
        aP_u = (inv_dt
                + 2*self.nu/self.dx**2 + 2*self.nu/self.dy**2
                + self.U0/self.dx + self.V0/self.dy
                + self.Ux)
        aW_u = -self.nu/self.dx**2 - self.U0/self.dx
        aE_u = np.full_like(self.U0, -self.nu/self.dx**2)
        aS_u = -self.nu/self.dy**2 - self.V0/self.dy
        aN_u = np.full_like(self.U0, -self.nu/self.dy**2)

        aP_v = (inv_dt
                + 2*self.nu/self.dx**2 + 2*self.nu/self.dy**2
                + self.U0/self.dx + self.V0/self.dy
                + self.Vy)
        aW_v = -self.nu/self.dx**2 - self.U0/self.dx
        aE_v = np.full_like(self.U0, -self.nu/self.dx**2)
        aS_v = -self.nu/self.dy**2 - self.V0/self.dy
        aN_v = np.full_like(self.U0, -self.nu/self.dy**2)

        # restrict to interior coefficients
        aP_u_i = aP_u[1:-1, 1:-1]
        aW_u_i = aW_u[1:-1, 1:-1]
        aE_u_i = aE_u[1:-1, 1:-1]
        aS_u_i = aS_u[1:-1, 1:-1]
        aN_u_i = aN_u[1:-1, 1:-1]

        aP_v_i = aP_v[1:-1, 1:-1]
        aW_v_i = aW_v[1:-1, 1:-1]
        aE_v_i = aE_v[1:-1, 1:-1]
        aS_v_i = aS_v[1:-1, 1:-1]
        aN_v_i = aN_v[1:-1, 1:-1]

        Nx_i = self.Nx - 2
        Ny_i = self.Ny - 2
        N = Nx_i * Ny_i

        def assemble(aP, aW, aE, aS, aN):
            rows, cols, data = [], [], []
            def idx(j,i): return j*Nx_i + i
            for j in range(Ny_i):
                for i in range(Nx_i):
                    k = idx(j,i)
                    # center
                    rows.append(k); cols.append(k); data.append(aP[j,i])
                    # west
                    if i > 0:
                        rows.append(k); cols.append(idx(j,i-1)); data.append(aW[j,i])
                    # east
                    if i < Nx_i-1:
                        rows.append(k); cols.append(idx(j,i+1)); data.append(aE[j,i])
                    # south
                    if j > 0:
                        rows.append(k); cols.append(idx(j-1,i)); data.append(aS[j,i])
                    # north
                    if j < Ny_i-1:
                        rows.append(k); cols.append(idx(j+1,i)); data.append(aN[j,i])
            return sparse.csr_matrix((data,(rows,cols)), shape=(N,N))

        Au = assemble(aP_u_i, aW_u_i, aE_u_i, aS_u_i, aN_u_i)
        Av = assemble(aP_v_i, aW_v_i, aE_v_i, aS_v_i, aN_v_i)
        return (Au, Av, aW_u_i, aE_u_i, aS_u_i, aN_u_i,
                aW_v_i, aE_v_i, aS_v_i, aN_v_i)

    # ---------------------- predictor step ----------------------

    def _fully_implicit_predictor(self, u, v, t):
        """
        Block Gauss–Seidel on:
           Au u* = (1/dt) u^n + Uy v^k + boundary terms
           Av v* = (1/dt) v^n + Vx u^{k+1} + boundary terms
        where Au, Av are time-invariant LU-factored matrices.
        """
        inv_dt = 1.0 / self.dt
        Nx_i = self.Nx - 2
        Ny_i = self.Ny - 2

        # start from current fields
        u_k = u.copy()
        v_k = v.copy()

        # precompute boundary arrays for this step (Dirichlet/Neumann)
        Vw = self.wall_profile(t)

        Lx = self.x[-1] - self.x[0] if self.x[-1] > self.x[0] else 1.0
        Qw = np.trapezoid(Vw, self.x)
        Vw = Vw - Qw / Lx

        # ensure boundary values are consistent before coupling iterations
        u_k[0, :] = 0.0
        v_k[0, :] = Vw
        u_k[-1, :] = u_k[-2, :]
        v_k[-1, :] = v_k[-2, :]
        u_k[:, 0] = u_k[:, 1]
        v_k[:, 0] = v_k[:, 1]
        u_k[:, -1] = u_k[:, -2]
        v_k[:, -1] = v_k[:, -2]

        for it in range(self.gs_sweeps):
            # ----- solve for u* with v_k on RHS
            b_u = inv_dt * u_k
            # interior RHS
            b = b_u[1:-1, 1:-1].copy()
            # add coupling + boundary contributions
            b += self.Uy[1:-1, 1:-1] * v_k[1:-1, 1:-1]  # production Uy * v

            # boundaries (same pattern as LinearBLSolver)
            # west/east
            b[:, 0]  -= self.aW_u[:, 0]  * u_k[1:-1, 0]
            b[:, -1] -= self.aE_u[:, -1] * u_k[1:-1, -1]
            # south/north
            b[0, :]  -= self.aS_u[0, :]  * u_k[0, 1:-1]     # u=0 at wall -> term drops, kept for clarity
            b[-1, :] -= self.aN_u[-1, :] * u_k[-1, 1:-1]    # Neumann: copied value

            sol_u = self._lu_u.solve(b.ravel())
            u_new = u_k.copy()
            u_new[1:-1, 1:-1] = sol_u.reshape(Ny_i, Nx_i)
            # hard BCs
            u_new[0, :] = 0.0
            u_new[-1, :] = u_new[-2, :]
            u_new[:, 0] = u_new[:, 1]
            u_new[:, -1] = u_new[:, -2]

            # ----- solve for v* with u_new on RHS
            b_v = inv_dt * v_k
            b = b_v[1:-1, 1:-1].copy()
            b += self.Vx[1:-1, 1:-1] * u_new[1:-1, 1:-1]     # production Vx * u

            b[:, 0] -= self.aW_v[:, 0] * 0.0  # west is Dirichlet 0 now
            b[:, -1] -= self.aE_v[:, -1] * v_k[1:-1, -1]
            b[0, :] -= self.aS_v[0, :] * Vw[1:-1]  # ← use Dirichlet v=Vw at wall
            b[-1, :] -= self.aN_v[-1, :] * v_k[-1, 1:-1]

            sol_v = self._lu_v.solve(b.ravel())
            v_new = v_k.copy()
            v_new[1:-1, 1:-1] = sol_v.reshape(Ny_i, Nx_i)
            # hard BCs
            v_new[0, :] = Vw
            v_new[-1, :] = v_new[-2, :]
            v_new[:, 0] = v_new[:, 1]
            v_new[:, -1] = v_new[:, -2]

            # update iteration state
            u_k, v_k = u_new, v_new

            if self.verbose:
                # cheap convergence monitor (optional)
                du = np.max(np.abs(u_new - u_k))
                dv = np.max(np.abs(v_new - v_k))
                # (here u_k,v_k already updated; if you want, cache prev before update)

        return u_k, v_k

    # -------------------- projection pieces (unchanged) --------------------

    def _divergence(self, u, v):
        div = np.zeros_like(u)
        div[1:-1, 1:-1] = (
            (u[1:-1, 2:] - u[1:-1, :-2])/(2*self.dx) +
            (v[2:, 1:-1] - v[:-2, 1:-1])/(2*self.dy)
        )
        return div

    def _pressure_gradient(self, p):
        dpdx = np.zeros_like(p)
        dpdy = np.zeros_like(p)
        dpdx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * self.dx)
        dpdy[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * self.dy)
        return dpdx, dpdy

    def _correct_velocity(self, u_star, v_star, p):
        u = u_star.copy()
        v = v_star.copy()
        u[1:-1, 1:-1] -= self.dt/self.rho * (p[1:-1, 2:] - p[1:-1, :-2])/(2*self.dx)
        v[1:-1, 1:-1] -= self.dt/self.rho * (p[2:, 1:-1] - p[:-2, 1:-1])/(2*self.dy)
        return u, v

    def _apply_velocity_bcs(self, u, v, t):
        Vw = self.wall_profile(t)

        # y = 0 wall
        u[0, :] = 0.0
        v[0, :] = Vw

        # x = 0  (INLET)  ➜  Dirichlet perturbations = 0
        u[:, 0] = 0.0
        v[:, 0] = 0.0
        v[0, 0] = Vw[0]  # keep wall value at the corner

        # x = Lx (outflow) & y = Ly (top) ➜ Neumann (copy interior)
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        u[-1, :] = u[-2, :]
        v[-1, :] = v[-2, :]

        return u, v

    # -------------------- Poisson (Neumann) --------------------

    def _build_neumann_poisson_matrix(self, Ny, Nx, dx, dy):
        """
        5-pt Laplacian on (Ny,Nx) with homogeneous Neumann on all sides (ghost mirroring).
        Flatten row-major. Singular; we will pin row 0 outside.
        """
        dx2, dy2 = dx*dx, dy*dy
        N = Ny*Nx
        rows, cols, data = [], [], []

        def idx(j,i): return j*Nx + i

        for j in range(Ny):
            for i in range(Nx):
                k = idx(j,i)
                sum_off = 0.0
                # x-
                if i > 0:
                    rows.append(k); cols.append(idx(j,i-1)); data.append(1.0/dx2); sum_off += 1.0/dx2
                else:
                    rows.append(k); cols.append(idx(j,i+1)); data.append(1.0/dx2); sum_off += 1.0/dx2
                if i < Nx-1:
                    rows.append(k); cols.append(idx(j,i+1)); data.append(1.0/dx2); sum_off += 1.0/dx2
                else:
                    rows.append(k); cols.append(idx(j,i-1)); data.append(1.0/dx2); sum_off += 1.0/dx2
                # y-
                if j > 0:
                    rows.append(k); cols.append(idx(j-1,i)); data.append(1.0/dy2); sum_off += 1.0/dy2
                else:
                    rows.append(k); cols.append(idx(j+1,i)); data.append(1.0/dy2); sum_off += 1.0/dy2
                if j < Ny-1:
                    rows.append(k); cols.append(idx(j+1,i)); data.append(1.0/dy2); sum_off += 1.0/dy2
                else:
                    rows.append(k); cols.append(idx(j-1,i)); data.append(1.0/dy2); sum_off += 1.0/dy2

                rows.append(k); cols.append(k); data.append(-sum_off)

        return sparse.csc_matrix((data,(rows,cols)), shape=(N,N))

    def _pressure_poisson_sparse(self, rhs, p_prev=None):
        b = rhs.ravel().copy()
        b[0] = 0.0
        p_flat = self._lu_p.solve(b)
        return p_flat.reshape((self.Ny, self.Nx))



