import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 0.1, 0.1    # Domain size
Nx, Ny = 100, 100      # Grid points
dx, dy = Lx / (Nx-1), Ly / (Ny-1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Time
dt = 0.01
nu = 0.01
rho = 1.0

# Initial fields
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))
p = np.zeros((Nx, Ny))

def laplacian(f, dx, dy):
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2 +
        (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dy**2
    )
    return lap

# Benchmark test: f(x,y) = sin(pi x) sin(pi y)
f = np.sin(np.pi * X) * np.sin(np.pi * Y)
lap_analytic = -2 * np.pi**2 * f
lap_numeric = laplacian(f, dx, dy)
error = np.linalg.norm(lap_numeric - lap_analytic) / np.linalg.norm(lap_analytic)
print(f'Laplacian benchmark error: {error:.2e}')  # Should be << 1

def implicit_diffusion(u, nu, dt, dx, dy, n_iter=100):
    u_new = u.copy()
    for _ in range(n_iter):
        u_new[1:-1, 1:-1] = (
            u[1:-1, 1:-1] +
            nu * dt * (
                (u_new[2:, 1:-1] + u_new[:-2, 1:-1]) / dx**2 +
                (u_new[1:-1, 2:] + u_new[1:-1, :-2]) / dy**2
            )
        ) / (1 + 2*nu*dt*(1/dx**2 + 1/dy**2))
    return u_new

# Benchmark: Diffuse a Gaussian spot, compare center value to analytic
u0 = np.exp(-100*((X-0.5)**2 + (Y-0.5)**2))
u1 = implicit_diffusion(u0, nu, dt, dx, dy, n_iter=100)
print(f"Diffusion step: max(u1) = {np.max(u1):.3f}")


def pressure_poisson(p, div, dx, dy, n_iter=100):
    p_new = p.copy()
    for _ in range(n_iter):
        p_new[1:-1, 1:-1] = (
            (p_new[2:, 1:-1] + p_new[:-2, 1:-1]) * dy**2 +
            (p_new[1:-1, 2:] + p_new[1:-1, :-2]) * dx**2 -
            div[1:-1, 1:-1] * dx**2 * dy**2
        ) / (2 * (dx**2 + dy**2))
        # Neumann BCs: dp/dn = 0 at all edges except one point (p=0 for reference)
        p_new[0, :] = p_new[1, :]
        p_new[-1, :] = p_new[-2, :]
        p_new[:, 0] = p_new[:, 1]
        p_new[:, -1] = p_new[:, -2]
        p_new[0, 0] = 0.0 # Reference pressure
    return p_new

import scipy.sparse as sp
import scipy.sparse.linalg as spla

def poisson_matrix(Nx, Ny, dx, dy):
    N = Nx * Ny
    main_diag = np.ones(N) * (-2 / dx**2 - 2 / dy**2)
    off_x = np.ones(N-1) / dx**2
    off_y = np.ones(N-Nx) / dy**2

    # Remove connections at row boundaries (for 2D grid, not 1D line)
    for i in range(1, Ny):
        off_x[i*Nx-1] = 0

    diags = [main_diag, off_x, off_x, off_y, off_y]
    offsets = [0, -1, 1, -Nx, Nx]
    A = sp.diags(diags, offsets, shape=(N, N)).tocsc()
    return A

def pressure_poisson_sparse(rhs, A, Nx, Ny):
    # Neumann everywhere, fix p[0,0]=0 for uniqueness
    rhs = rhs.copy()
    rhs = rhs.flatten()
    # Fix reference node
    A = A.copy()
    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1
    rhs[0] = 0
    A = A.tocsc()
    p_flat = spla.spsolve(A, rhs)
    p = p_flat.reshape((Nx, Ny))
    return p

# Benchmark: Manufactured solution
p_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
rhs = -2 * np.pi**2 * p_true   # <-- analytic Laplacian
p_test = pressure_poisson(np.zeros_like(p_true), rhs, dx, dy, n_iter=1000)
error = np.linalg.norm(p_test - p_true) / np.linalg.norm(p_true)
print(f'Pressure Poisson benchmark error: {error:.2e}')

# Manufactured solution
p_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
rhs = -2 * np.pi**2 * p_true  # Laplacian of sin(pi x) sin(pi y)

# Build Poisson matrix (if not already)
A = poisson_matrix(Nx, Ny, dx, dy)

# Solve with sparse solver
p_test = pressure_poisson_sparse(rhs, A, Nx, Ny)

# Compute error (ignoring the reference point p[0,0])
mask = np.ones_like(p_true, dtype=bool)
mask[0, 0] = False
error = np.linalg.norm((p_test - p_true)[mask]) / np.linalg.norm(p_true[mask])
print(f'Pressure Poisson (sparse) benchmark error: {error:.2e}')

def correct_velocity(u_star, v_star, p, dx, dy, dt, rho):
    u = u_star.copy()
    v = v_star.copy()
    u[1:-1, 1:-1] -= dt/rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2*dx)
    v[1:-1, 1:-1] -= dt/rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2*dy)
    return u, v

# Benchmark: After correction, should have div(u, v) ~ 0
def divergence(u, v, dx, dy):
    div = np.zeros_like(u)
    div[1:-1, 1:-1] = ((u[2:, 1:-1] - u[:-2, 1:-1])/(2*dx) +
                        (v[1:-1, 2:] - v[1:-1, :-2])/(2*dy))
    return div

def apply_bc(u, v, Vw=None):
    # No-slip floor: u=0, v=Vw (blow/suction)
    u[:, 0] = 0
    if Vw is not None:
        v[:, 0] = Vw
    else:
        v[:, 0] = 0
    # Outlets (Neumann)
    u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]
    u[0, :] = u[1, :]
    v[0, :] = v[1, :]
    u[-1, :] = u[-2, :]
    v[-1, :] = v[-2, :]
    return u, v

##for step in range(20):  # Just a few steps for demo
##    # Step 1: Apply BCs
##    Vw = 0.1 # * np.sin(2 * np.pi * x)  # Example blow profile
##    u, v = apply_bc(u, v, Vw=Vw)
##
##    # Step 2: Diffusion
##    u_star = implicit_diffusion(u, nu, dt, dx, dy)
##    v_star = implicit_diffusion(v, nu, dt, dx, dy)
##
##    # Step 3: Compute divergence for pressure
##    div_star = divergence(u_star, v_star, dx, dy)
##
##    # Step 4: Pressure solve
##    rhs = rho / dt * div_star
##    p = pressure_poisson(p, rhs, dx, dy, n_iter=1000)
##
##    # Step 5: Velocity correction
##    u, v = correct_velocity(u_star, v_star, p, dx, dy, dt, rho)
##
##    # Step 6: Diagnostics
##    div_final = divergence(u, v, dx, dy)
##    print(f"Step {step}: Max divergence after projection = {np.max(np.abs(div_final)):.2e}")


num_steps = 1000  # or however many frames you want
us = []
vs = []
ps = []

# Reset fields before solve if needed
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))
p = np.zeros((Nx, Ny))

layer_depths = []
times = []

for step in range(num_steps):
    t = step * dt
    Vw = 5.0 * np.ones_like(x)  # Constant blowing at the floor
    u, v = apply_bc(u, v, Vw=Vw)
    u_star = implicit_diffusion(u, nu, dt, dx, dy)
    v_star = implicit_diffusion(v, nu, dt, dx, dy)
    div_star = divergence(u_star, v_star, dx, dy)
    rhs = rho / dt * div_star
    p = pressure_poisson_sparse(rhs, A, Nx, Ny)
    u, v = correct_velocity(u_star, v_star, p, dx, dy, dt, rho)
    us.append(u.copy())
    vs.append(v.copy())
    ps.append(p.copy())

    v_profile = v[Nx // 2, :]
    V0 = 5.0
    cutoff = 4.99

    # Find first index where v drops below cutoff
    idx = np.argmax(v_profile < cutoff)
    if idx > 0 and idx < Ny:
        # Linear interpolation between y[idx-1] and y[idx]
        y1, y2 = y[idx-1], y[idx]
        v1, v2 = v_profile[idx-1], v_profile[idx]
        # Slope: (v2 - v1) / (y2 - y1)
        if v2 != v1:  # Prevent division by zero
            depth = y1 + (cutoff - v1) * (y2 - y1) / (v2 - v1)
        else:
            depth = y1
    else:
        depth = 0  # If cutoff never reached (e.g. at very early t)

    layer_depths.append(depth)
    times.append(t)

##for step in range(0, num_steps, 5):
##    v_profile = vs[step][Nx // 2, :]
##    print(f"Step {step}, v_profile: {v_profile}")
##    print(f"Max v: {v_profile.max()}, Min v: {v_profile.min()}")
##    print(f"Cutoff value: {cutoff}")


import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt

for step in range(0, num_steps, 5):
    plt.plot(y, vs[step][Nx // 2, :], label=f't={step*dt:.2f}s')
plt.axhline(cutoff, color='k', linestyle='--', label='cutoff')
plt.xlabel('y')
plt.ylabel('v')
plt.legend()
plt.show()

# Convert to arrays
layer_depths = np.array(layer_depths)
times = np.array(times)

from scipy.optimize import curve_fit

def sqrt_model(t, a, b, c):
    # Avoid sqrt of negative by forcing t-c >= 0
    return a * np.sqrt(np.maximum(nu * (t - c), 0)) + b

# Only fit for times > c (start with c0 = min time in fit, e.g. 9.0)
p0 = [1.0, 0.0, 9.0]  # initial guesses for a, b, c

fit_mask = times > 9.0
fit_y = np.array(layer_depths)[fit_mask]
fit_t = np.array(times)[fit_mask]

params, _ = curve_fit(sqrt_model, fit_t, fit_y, p0=p0)
a, b, c = params
print(f"Best fit: a={a:.2f}, b={b:.2e}, c={c:.2f}")

# Plot result
plt.figure(figsize=(7,5))
plt.plot(times, layer_depths, 'b-', label='Numerical thickness')
plt.plot(times, sqrt_model(times, a, b, c), 'r--', label=fr'Fit: $a\sqrt{{\nu (t - c)}} + b$')
plt.xlabel('Time $t$')
plt.ylabel('Penetration depth $d$')
plt.legend()
plt.title('Layer Thickness Fit with Three Parameters')
plt.tight_layout()
plt.show()

print(f"Best fit coefficient a = {a:.2f}")

fig, ax = plt.subplots(figsize=(6,5))
contour = ax.contourf(X, Y, np.sqrt(us[0]**2 + vs[0]**2), levels=20, cmap='viridis')
cb = fig.colorbar(contour)
cb.set_label('Velocity Magnitude |U|')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Velocity Magnitude |U|')

def animate(frame):
    speed = np.sqrt(us[frame]**2 + vs[frame]**2)
    ax.clear()
    t = frame * dt
    contour = ax.contourf(X, Y, speed, levels=20, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'|U|, step={frame}, t={t:.3f}s')

ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=200, blit=False)
plt.show()
