# ALE Navier–Stokes (2D) — projection solver with mesh motion (travelling-wave floor)
#
# Notes for Dim:
# - This is a clean, modular rewrite of your script.
# - It supports two modes:
#     * "nonlinear ALE":  ∂u/∂t|mesh + ((u - w)·∇)u = -(1/ρ)∇p + ν∇²u
#     * "linearized ALE around rest": drop (u·∇)u but keep -(w·∇)u
# - Mesh motion: x' = x, y' = y + φ(y)*h(x,t), φ(y)=1 - y/Ly (top fixed, bottom follows h).
# - We do *not* remesh. We keep a uniform computational grid and inject w in advection and BCs.
#   This is a practical ALE-lite suitable for sizable amplitudes without extreme slopes.
# - For extreme deformations or overhangs, use a true moving-mesh FEM/FVM library.
#
# Structure:
#   - Geometry/mesh motion: h, dhdt, dhdx, phi, mesh_velocity
#   - Numerics: laplacian, grad, divergence, advection ((q·∇)u), implicit diffusion (Jacobi), pressure Poisson (sparse)
#   - Projection step adapted to ALE (same pressure correction, incompressibility in physical coords)
#   - BCs: no-slip on moving floor (v = w_y), stationary top/sides with Neumann outflow options
#   - Stepper: step_ale(), with linearized flag
#   - Tests/benchmarks at bottom
#
# IMPORTANT: For readability and testing speed, grids are modest by default.


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# -----------------------------
# Parameters (can be changed)
# -----------------------------
Lx, Ly = 0.1, 0.1
Nx, Ny = 64, 64
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing="ij")

# Physical params
nu = 1e-2
rho = 1.0

# Time
dt = 5e-4

# Travelling-wave floor (user-editable)
A_amp = 0.01         # amplitude [m]
k = 2*np.pi/0.05 # wavenumber, e.g., λ=0.05 m
omega = 50.0     # rad/s  (wave speed c = omega/k)
# CFL & diffusion stability guidance: use tests below.


# -----------------------------
# Mesh motion definitions
# -----------------------------
def h(x, t):
    """Floor displacement [m] at x, time t. Travelling sine wave."""
    return A_amp * np.sin(k*x - omega*t)

def dhdt(x, t):
    """Time derivative of floor displacement."""
    return -A_amp * omega * np.cos(k*x - omega*t)

def dhdx(x, t):
    """Space derivative wrt x of floor displacement."""
    return A_amp * k * np.cos(k*x - omega*t)

def phi(y):
    """Blending function φ(y): φ(0)=1 (floor follows), φ(Ly)=0 (top fixed)."""
    return 1.0 - y / Ly

def dphidy(y):
    return -1.0 / Ly

def mesh_velocity(X, Y, t):
    """
    Mesh velocity w = (w_x, w_y) at grid nodes (computational nodes mapped to physical by y' = y + φ(y) h(x,t)).
    In this ALE-lite, we only need w for convective term and BCs.
    """
    w_x = np.zeros_like(X)
    w_y = phi(Y) * dhdt(X, t)
    return w_x, w_y


# -----------------------------
# Discrete operators
# -----------------------------
def laplacian(f, dx, dy):
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2 +
        (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dy**2
    )
    return lap

def gradx(f, dx):
    g = np.zeros_like(f)
    g[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2*dx)
    # Neumann at boundaries
    g[0, :] = (f[1, :] - f[0, :]) / dx
    g[-1, :] = (f[-1, :] - f[-2, :]) / dx
    return g

def grady(f, dy):
    g = np.zeros_like(f)
    g[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2*dy)
    # Neumann at boundaries
    g[:, 0] = (f[:, 1] - f[:, 0]) / dy
    g[:, -1] = (f[:, -1] - f[:, -2]) / dy
    return g

def divergence(u, v, dx, dy):
    """Centered divergence ∂u/∂x + ∂v/∂y on the uniform computational grid."""
    div = np.zeros_like(u)
    div[1:-1, 1:-1] = ((u[2:, 1:-1] - u[:-2, 1:-1])/(2*dx) +
                       (v[1:-1, 2:] - v[1:-1, :-2])/(2*dy))
    # Simple one-sided at boundaries (used only for diagnostics)
    div[0, :] = (u[1, :] - u[0, :]) / dx + (v[0, 1:] - v[0, :-1]).mean() if u.shape[1] > 1 else 0.0
    div[-1, :] = (u[-1, :] - u[-2, :]) / dx + (v[-1, 1:] - v[-1, :-1]).mean() if u.shape[1] > 1 else 0.0
    div[:, 0] = (v[:, 1] - v[:, 0]) / dy + (u[1:, 0] - u[:-1, 0]).mean() if u.shape[0] > 1 else 0.0
    div[:, -1] = (v[:, -1] - v[:, -2]) / dy + (u[1:, -1] - u[:-1, -1]).mean() if u.shape[0] > 1 else 0.0
    return div

def div_interior(u, v, dx, dy):
    dux = (u[2:, 1:-1] - u[:-2, 1:-1])/(2*dx)
    dvy = (v[1:-1, 2:] - v[1:-1, :-2])/(2*dy)
    return dux + dvy

def advect(u, v, qx, qy, dx, dy):
    """
    Compute advection (q·∇) of a scalar-like field u (or v), with q=(qx,qy).
    Here used component-wise for velocity: ((u - w)·∇)u and ((u - w)·∇)v.
    Central differencing (2nd order); for stability at high CFL, use upwind.
    """
    du = qx * gradx(u, dx) + qy * grady(u, dy)
    return du

def advect_upwind(f, qx, qy, dx, dy):
    """(q·∇)f using 1st-order upwind (more diffusive, much more robust)."""
    out = np.zeros_like(f)
    qxi = qx[1:-1, 1:-1]; qyi = qy[1:-1, 1:-1]
    fim = f[:-2, 1:-1]; fi = f[1:-1, 1:-1]; fip = f[2:, 1:-1]
    fjm = f[1:-1, :-2]; fjp = f[1:-1, 2:]
    ddx = np.where(qxi >= 0.0, (fi - fim)/dx, (fip - fi)/dx)
    ddy = np.where(qyi >= 0.0, (fi - fjm)/dy, (fjp - fi)/dy)
    out[1:-1, 1:-1] = qxi*ddx + qyi*ddy
    return out

def minmod(a, b):
    return 0.5*(np.sign(a)+np.sign(b))*np.minimum(np.abs(a), np.abs(b))

def advect_muscl(f, qx, qy, dx, dy, theta=1.0):
    """
    (q·∇)f via MUSCL with minmod limiter.
    theta in [0,2]; 1.0 = classic, 2.0 = SHARP (superbee-like), 0.0 = donor-cell.
    """
    out = np.zeros_like(f)

    # x-direction limited slopes at cell centers
    dfLx = (f[1:-1, 1:-1] - f[:-2, 1:-1])
    dfRx = (f[2:,   1:-1] - f[1:-1, 1:-1])
    sx = minmod(theta*dfLx, theta*dfRx)

    # y-direction limited slopes
    dfLy = (f[1:-1, 1:-1] - f[1:-1, :-2])
    dfRy = (f[1:-1, 2:]   - f[1:-1, 1:-1])
    sy = minmod(theta*dfLy, theta*dfRy)

    # upwind reconstructions (Godunov flux form)
    qxi = qx[1:-1, 1:-1]; qyi = qy[1:-1, 1:-1]

    # face values
    fL = f[1:-1, 1:-1] - 0.5*sx
    fR = f[1:-1, 1:-1] + 0.5*sx
    fB = f[1:-1, 1:-1] - 0.5*sy
    fT = f[1:-1, 1:-1] + 0.5*sy

    # upwinded gradients
    dfdx = np.where(qxi >= 0.0,
                    (f[1:-1, 1:-1] - f[:-2, 1:-1]) / dx,
                    (f[2:, 1:-1]   - f[1:-1, 1:-1]) / dx)
    dfdy = np.where(qyi >= 0.0,
                    (f[1:-1, 1:-1] - f[1:-1, :-2]) / dy,
                    (f[1:-1, 2:]   - f[1:-1, 1:-1]) / dy)

    out[1:-1, 1:-1] = qxi*dfdx + qyi*dfdy
    return out



# -----------------------------
# Linear algebra for pressure
# -----------------------------
def poisson_matrix(Nx, Ny, dx, dy):
    N = Nx * Ny
    main_diag = np.ones(N) * (-2 / dx**2 - 2 / dy**2)
    off_x = np.ones(N-1) / dx**2
    off_y = np.ones(N-Nx) / dy**2
    # Break x-coupling at row edges
    for j in range(1, Ny):
        off_x[j*Nx - 1] = 0.0
    diags = [main_diag, off_x, off_x, off_y, off_y]
    offsets = [0, -1, 1, -Nx, Nx]
    return sp.diags(diags, offsets, shape=(N, N)).tocsc()

AP = poisson_matrix(Nx, Ny, dx, dy)

def pressure_poisson_sparse(rhs, AP, Nx, Ny):
    assert sp.issparse(AP), f"Poisson matrix AP must be sparse, got type={type(AP)}"
    rhs = rhs.flatten().copy()
    A2 = AP.tolil(copy=True)
    A2[0, :] = 0.0
    A2[0, 0] = 1.0
    rhs[0] = 0.0
    p_flat = spla.spsolve(A2.tocsc(), rhs)
    return p_flat.reshape((Nx, Ny))


# -----------------------------
# Diffusion (implicit Jacobi)
# -----------------------------
def implicit_diffusion(u, nu, dt, dx, dy, n_iter=50):
    u_new = u.copy()
    beta = 1.0 / (1.0 + 2*nu*dt*(1/dx**2 + 1/dy**2))
    for _ in range(n_iter):
        u_new[1:-1, 1:-1] = beta * (
            u[1:-1, 1:-1] +
            nu*dt*((u_new[2:, 1:-1] + u_new[:-2, 1:-1])/dx**2 +
                   (u_new[1:-1, 2:] + u_new[1:-1, :-2])/dy**2)
        )
    return u_new


# -----------------------------
# Boundary conditions (ALE)
# -----------------------------
def apply_bc(u, v, t, stage="pre"):
    """
    stage='pre' : enforce full Dirichlet on walls *before* projection
    stage='post': enforce only tangential components *after* projection
    """
    if stage == "pre":
        # moving bottom: u=0, v=wy
        _, wy = mesh_velocity(X[:, :1], Y[:, :1]*0.0, t)
        u[:, 0] = 0.0
        v[:, 0] = wy[:, 0]
        # stationary top & sides (Dirichlet)
        u[:, -1] = 0.0; v[:, -1] = 0.0
        u[0, :]  = 0.0; v[0, :]  = 0.0
        u[-1, :] = 0.0; v[-1, :] = 0.0
        return u, v

    # stage == "post": enforce tangential only
    # bottom: keep v (normal) from projection; enforce u=0 (tangential)
    u[:, 0] = 0.0
    # top: keep v normal; set u=0
    u[:, -1] = 0.0
    # left/right: keep u normal; set v=0 (tangential)
    v[0, :]  = 0.0
    v[-1, :] = 0.0
    return u, v


# -----------------------------
# Projection method (ALE)
# -----------------------------
def correct_velocity(u_star, v_star, p, dx, dy, dt, rho):
    u = u_star.copy()
    v = v_star.copy()
    u[1:-1, 1:-1] -= dt/rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2*dx)
    v[1:-1, 1:-1] -= dt/rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2*dy)
    return u, v

def build_ppe_rhs(u_star, v_star, t, dx, dy, rho, dt):
    rhs = (rho/dt) * divergence(u_star, v_star, dx, dy)

    # bottom (j=0) normal = +y → affects j=1
    _, wy_b = mesh_velocity(X[:, :1], Y[:, :1]*0.0, t)
    g_b = (rho/dt) * (v_star[:, 0] - wy_b[:, 0])             # dp/dy|y=0
    rhs[1:-1, 1] += (2.0/dy) * g_b[1:-1]

    # top (j=Ny-1) normal = +y → affects j=Ny-2
    # v_bc = 0
    g_t = (rho/dt) * (v_star[:, -1] - 0.0)                   # dp/dy|y=Ly
    rhs[1:-1, -2] -= (2.0/dy) * g_t[1:-1]                    # minus: ghost on +y side

    # left (i=0) normal = +x → affects i=1
    # u_bc = 0
    g_l = (rho/dt) * (u_star[0, :] - 0.0)                    # dp/dx|x=0
    rhs[1, 1:-1] += (2.0/dx) * g_l[1:-1]

    # right (i=Nx-1) normal = +x → affects i=Nx-2
    g_r = (rho/dt) * (u_star[-1, :] - 0.0)                   # dp/dx|x=Lx
    rhs[-2, 1:-1] -= (2.0/dx) * g_r[1:-1]                    # minus: ghost on +x side

    return rhs

def project_iteratively(u_in, v_in, p_acc, t, max_iter=3, tol=1e-3):
    u, v, p = u_in.copy(), v_in.copy(), p_acc
    for k in range(max_iter):
        rhs = build_ppe_rhs(u, v, t, dx, dy, rho, dt)
        pk  = pressure_poisson_sparse(rhs, AP, Nx, Ny)
        u, v = correct_velocity(u, v, pk, dx, dy, dt, rho)
        u, v = apply_bc(u, v, t, stage="post")
        p    = p + pk
        # quick check (interior)
        dux = (u[2:,1:-1]-u[:-2,1:-1])/(2*dx)
        dvy = (v[1:-1,2:]-v[1:-1,:-2])/(2*dy)
        if np.linalg.norm(dux+dvy)/dux.size < tol:
            break
    return u, v, p

# -----------------------------
# Time stepping
# -----------------------------
def step_ale(u, v, p, t, dt, nu, rho, linearized=False, second_projection=True):
    wx, wy = mesh_velocity(X, Y, t)
    if linearized:
        qx, qy = -wx, -wy
    else:
        qx, qy = u - wx, v - wy

    # --- upwind advection
    u_adv = u - dt * advect_muscl(u, qx, qy, dx, dy, theta=1.0)
    v_adv = v - dt * advect_muscl(v, qx, qy, dx, dy, theta=1.0)

    # implicit diffusion
    u_star = implicit_diffusion(u_adv, nu, dt, dx, dy, n_iter=100)
    v_star = implicit_diffusion(v_adv, nu, dt, dx, dy, n_iter=100)

    # BCs before projection
    u_star, v_star = apply_bc(u_star, v_star, t)

    u_new, v_new, p = project_iteratively(u_star, v_star, p*0.0, t, max_iter=8, tol=5e-4)

    return u_new, v_new, p


# -----------------------------
# Initialization
# -----------------------------
def init_fields():
    return np.zeros((Nx, Ny)), np.zeros((Nx, Ny)), np.zeros((Nx, Ny))


# -----------------------------
# Diagnostics & Tests
# -----------------------------
def kinetic_energy(u, v):
    return 0.5*np.mean(u**2 + v**2)

def gcl_residual(t):
    """
    Geometric Conservation Law (diagnostic): ∂J/∂t + ∂(J w_x)/∂x + ∂(J w_y)/∂y.
    For mapping x'=x, y'=y+φ(y)h(x,t), Jacobian J = 1 + φ'(y)h(x,t).
    Here evaluated approximately on the computational grid.
    """
    H = h(X, t)
    J = 1.0 + dphidy(Y) * H
    wx, wy = mesh_velocity(X, Y, t)
    # Time derivative of J:
    dJdt = dphidy(Y) * dhdt(X, t)
    # Divergence of J*w
    dJwx_dx = gradx(J*wx, dx)
    dJwy_dy = grady(J*wy, dy)
    res = dJdt + dJwx_dx + dJwy_dy
    return res

def cfl_number(u, v, wx, wy, dt, dx, dy):
    """Max CFL based on relative velocity |u-w|."""
    qx = u - wx
    qy = v - wy
    cflx = np.max(np.abs(qx)) * dt / dx
    cfly = np.max(np.abs(qy)) * dt / dy
    return max(cflx, cfly)

def suggest_dt(u, v, t, dx, dy, cfl_target=0.3):
    wx, wy = mesh_velocity(X, Y, t)
    qx = np.abs(u - wx); qy = np.abs(v - wy)
    s = max(np.max(qx)/dx, np.max(qy)/dy)
    return np.inf if s == 0 else cfl_target / s

# -----------------------------
# Quick Benchmarks (lightweight)
# -----------------------------
def run_quick_bench(nsteps=200, linearized=True, seed=0):
    """
    1) Sanity: with h=0 (turn off motion), ALE reduces to original scheme.
    2) Divergence after projection stays small.
    3) GCL residual stays near machine precision when A→0 and moderate otherwise.
    4) Energy bounded in linearized mode for viscous flow.
    """
    np.random.seed(seed)
    u, v, p = init_fields()

    # Option: temporarily zero floor motion to check reduction
    A_backup = globals()['A_amp']
    globals()['A_amp'] = 0.0

    u, v, p = init_fields()
    div_hist_nomotion = []
    for n in range(10):
        u, v, p = step_ale(u, v, p, t=n*dt, dt=dt, nu=nu, rho=rho, linearized=linearized)
        div_norm = np.linalg.norm(divergence(u, v, dx, dy)) / (Nx*Ny)
        div_hist_nomotion.append(div_norm)

    # Restore A and run with motion
    globals()['A_amp'] = A_backup
    u, v, p = init_fields()
    div_hist_motion = []
    energy_hist = []
    gcl_hist = []

    for n in range(nsteps):
        t = n*dt
        u, v, p = step_ale(u, v, p, t=t, dt=dt, nu=nu, rho=rho, linearized=linearized)
        div_norm = np.linalg.norm(divergence(u, v, dx, dy)) / (Nx*Ny)
        div_hist_motion.append(div_norm)
        energy_hist.append(kinetic_energy(u, v))
        gcl_hist.append(np.mean(np.abs(gcl_residual(t))))

    results = {
        "div_nomotion_mean": float(np.mean(div_hist_nomotion)),
        "div_motion_mean": float(np.mean(div_hist_motion)),
        "energy_mean": float(np.mean(energy_hist)),
        "gcl_mean_abs_residual": float(np.mean(gcl_hist)),
        "final_u_max": float(np.max(np.abs(u))),
        "final_v_max": float(np.max(np.abs(v))),
    }
    return results


# -----------------------------
# If run in notebook, execute quick tests
# -----------------------------
u = np.zeros((Nx, Ny)); v = np.zeros((Nx, Ny)); p = np.zeros((Nx, Ny))
t = 0.0
for n in range(80):
    u, v, p = step_ale(u, v, p, t, dt, nu, rho, linearized=True, second_projection=False)  # uses iterative projector now
    t += dt

dux = (u[2:,1:-1]-u[:-2,1:-1])/(2*dx)
dvy = (v[1:-1,2:]-v[1:-1,:-2])/(2*dy)
print("div L2 (interior):", np.linalg.norm(dux+dvy)/(dux.size))


# ---------- RUN SETTINGS ----------
n_periods    = 2                       # how many wave periods to simulate
T_period     = 2*np.pi/omega           # period from your omega
nsteps       = int(np.ceil(n_periods*T_period/dt))
sample_every = 5                        # record every N steps (animation frames)
linearized   = True                     # True: ALE-linearized; False: full nonlinear

print(f"Steps: {nsteps}, frames ≈ {nsteps//sample_every}")

# init
u, v, p = init_fields()
t = 0.0

frames_speed = []   # |U|
frames_u = []
frames_v = []
tvals = []
div_vals = []

def div_interior(u, v, dx, dy):
    dux = (u[2:, 1:-1] - u[:-2, 1:-1])/(2*dx)
    dvy = (v[1:-1, 2:] - v[1:-1, :-2])/(2*dy)
    return dux + dvy

for n in range(nsteps):
    # (optional) adaptive dt to enforce relative CFL
    # dt_s = suggest_dt(u, v, t, dx, dy, cfl_target=0.25)
    # if dt > dt_s and np.isfinite(dt_s):
    #     dt = dt_s

    u, v, p = step_ale(u, v, p, t, dt, nu, rho, linearized=linearized)  # step
    t += dt

    if n % sample_every == 0:
        speed = v #np.sqrt(u*u + v*v)
        frames_speed.append(speed.copy())
        frames_u.append(u.copy())
        frames_v.append(v.copy())
        tvals.append(t)

        di = div_interior(u, v, dx, dy)
        div_vals.append(np.linalg.norm(di)/di.size)

print("Recorded frames:", len(frames_speed), "| last div L2 (interior):", div_vals[-1])
vmax = max(np.max(F) for F in frames_speed)  # fixed color scale across frames
vmin = max(np.min(F) for F in frames_speed)  # fixed color scale across frames


import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(6,5))
cax = ax.pcolormesh(X, Y, frames_speed[0], shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
cb = fig.colorbar(cax, ax=ax); cb.set_label('|U|')
ax.set_xlabel('x'); ax.set_ylabel('y')
ttl = ax.set_title(f'|U| (computational)  t={tvals[0]:.4f}s  div_int={div_vals[0]:.2e}')

def update_comp(i):
    ax.clear()
    cax = ax.pcolormesh(X, Y, frames_speed[i], shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'|U| (computational)  t={tvals[i]:.4f}s  div_int={div_vals[i]:.2e}')
    return [cax]

ani_comp = animation.FuncAnimation(fig, update_comp, frames=len(frames_speed), interval=60, blit=False)

# Show interactively:
plt.show()

# (Optional) Save:
# Requires ffmpeg in PATH (Windows: install from https://ffmpeg.org or `pip install imageio[ffmpeg]`)
# ani_comp.save("ale_run_computational.mp4", writer='ffmpeg', dpi=150, fps=30)


fig2, ax2 = plt.subplots(figsize=(6,5))

def phys_coords(t):
    return X, Y + phi(Y)*h(X, t)

Xp0, Yp0 = phys_coords(tvals[0])
cax2 = ax2.pcolormesh(Xp0, Yp0, frames_speed[0], shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
cb2 = fig2.colorbar(cax2, ax=ax2); cb2.set_label('|U|')
ax2.set_xlabel('x'); ax2.set_ylabel('y_phys')
ttl2 = ax2.set_title(f'|U| (physical)  t={tvals[0]:.4f}s  div_int={div_vals[0]:.2e}')

def update_phys(i):
    ax2.clear()
    Xp, Yp = phys_coords(tvals[i])
    cax2 = ax2.pcolormesh(Xp, Yp, frames_speed[i], shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_xlabel('x'); ax2.set_ylabel('y_phys')
    ax2.set_title(f'|U| (physical)  t={tvals[i]:.4f}s  div_int={div_vals[i]:.2e}')
    return [cax2]

ani_phys = animation.FuncAnimation(fig2, update_phys, frames=len(frames_speed), interval=60, blit=False)

plt.show()

# (Optional) Save:
# ani_phys.save("ale_run_physical.mp4", writer='ffmpeg', dpi=150, fps=30)


skip = 6  # increase if arrows are too dense
def update_phys_quiver(i):
    ax2.clear()
    Xp, Yp = phys_coords(tvals[i])
    cax2 = ax2.pcolormesh(Xp, Yp, frames_speed[i], shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.quiver(Xp[::skip, ::skip], Yp[::skip, ::skip],
               frames_u[i][::skip, ::skip], frames_v[i][::skip, ::skip], scale=50)
    ax2.set_xlabel('x'); ax2.set_ylabel('y_phys')
    ax2.set_title(f'|U| + quiver  t={tvals[i]:.4f}s  div_int={div_vals[i]:.2e}')
    return [cax2]

# ani_phys_q = animation.FuncAnimation(fig2, update_phys_quiver, frames=len(frames_speed), interval=60, blit=False)
# ani_phys_q.save("ale_run_physical_quiver.mp4", writer='ffmpeg', dpi=150, fps=30)


def vort(u, v, dx, dy):
    # ω_z = ∂v/∂x - ∂u/∂y (centered interior)
    dv_dx = (v[2:,1:-1] - v[:-2,1:-1])/(2*dx)
    du_dy = (u[1:-1,2:] - u[1:-1,:-2])/(2*dy)
    w = np.zeros_like(u)
    w[1:-1,1:-1] = dv_dx - du_dy
    return w

frames_w = [vort(frames_u[i], frames_v[i], dx, dy) for i in range(len(frames_u))]
wmax = max(np.max(np.abs(W)) for W in frames_w)

fig3, ax3 = plt.subplots(figsize=(6,5))
def update_w(i):
    ax3.clear()
    Xp, Yp = phys_coords(tvals[i])
    cax3 = ax3.pcolormesh(Xp, Yp, frames_w[i], shading='auto', cmap='seismic', vmin=-wmax, vmax=wmax)
    ax3.set_xlabel('x'); ax3.set_ylabel('y_phys')
    ax3.set_title(f'Vorticity  t={tvals[i]:.4f}s  div_int={div_vals[i]:.2e}')
    return [cax3]

ani_w = animation.FuncAnimation(fig3, update_w, frames=len(frames_w), interval=60, blit=False)
plt.show()
# ani_w.save("ale_run_vorticity.mp4", writer='ffmpeg', dpi=150, fps=30)



