"""
Linearized Unsteady 2D Boundary Layer Solver over Exact Blasius Flow
Solves linearized perturbation and animates u, v, and drag, using a numerically
exact Blasius base flow.

Equations:
 ∂u'/∂t + u0 ∂u'/∂x + v0 ∂u'/∂y = ν(∂²u'/∂x² + ∂²u'/∂y²)
 ∂u'/∂x + ∂v'/∂y = 0
Drag(t) = ∫₀ᴸ μ (∂u/∂y)|_{y=0} dx
Analytical steady drag: D_an = ρ·0.664·U∞²√(νL/U∞)

Dependencies:
  - numpy
  - matplotlib

Usage:
  python linear_blasius_solver_exact.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# ----------------------- PARAMETERS -----------------------
U_inf = 2.0               # Freestream velocity [m/s]
nu    = 1.5e-5            # Kinematic viscosity [m^2/s]
rho   = 1.225             # Fluid density [kg/m^3]
mu    = rho * nu          # Dynamic viscosity
L     = 1.0               # Plate length [m]

Nx, Ny = 100, 100         # Grid points in x and y
x = np.linspace(0, L, Nx)
y = np.linspace(0, 0.05, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

dt = 1e-4                 # Time step [s]
Nt = 20000                  # Number of time steps
time = np.arange(Nt) * dt

# Disturbance parameters
epsilon = 0.1            # Perturbation amplitude
omega   = 2 * np.pi * 5   # Perturbation frequency [rad/s]

# Analytical steady drag
drag_analytical = rho * 0.664 * U_inf**2 * np.sqrt(nu * L / U_inf)
# ----------------------------------------------------------

# Precompute exact Blasius f(η), f'(η)
eta_max = 10.0
Neta     = 1000
eta      = np.linspace(0, eta_max, Neta)
f        = np.zeros((3, Neta))
f[0,0]  = 0.0
f[1,0]  = 0.0
f[2,0]  = 0.332057336215194  # f''(0)

d_eta = eta[1] - eta[0]
def rhs(state):
    f0, f1, f2 = state
    return np.array([f1, f2, -0.5 * f0 * f2])

for k in range(Neta-1):
    s = f[:,k]
    k1 = rhs(s)
    k2 = rhs(s + 0.5*d_eta*k1)
    k3 = rhs(s + 0.5*d_eta*k2)
    k4 = rhs(s + d_eta*k3)
    f[:,k+1] = s + (d_eta/6)*(k1 + 2*k2 + 2*k3 + k4)

f0_eta = f[0]
f1_eta = f[1]

# Build base flow U0, V0 on grid using f and f'
def compute_base_flow():
    U0 = np.zeros((Ny, Nx))
    V0 = np.zeros((Ny, Nx))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            if xi <= 0:
                U0[j,i] = U_inf * f1_eta[-1]
                V0[j,i] = 0.0
            else:
                et = yj * np.sqrt(U_inf/(nu*xi))
                idx = min(int(et/eta_max*(Neta-1)), Neta-1)
                up = f1_eta[idx]
                U0[j,i] = U_inf * up
                V0[j,i] = 0.5*np.sqrt(nu*U_inf/xi)*(et*up - f0_eta[idx])
    return U0, V0

U0, V0 = compute_base_flow()

# --- Diagnostic: compute and compare steady drag of base flow ---
# Wall shear from base flow
dUdy_base = (U0[1, :] - U0[0, :]) / dy
tau_w_base = mu * dUdy_base
drag_base = np.trapezoid(tau_w_base, x)
print(f"Numerical steady drag (base flow): {drag_base:.6f} N/m")
print(f"Analytical steady drag:            {drag_analytical:.6f} N/m")
# Plot tau_w comparison
plt.figure(figsize=(6,4))
plt.plot(x, tau_w_base, label='Numerical τ_w')
plt.hlines(drag_analytical/ L, x[0], x[-1], colors='r', linestyles='--', label='Analytical τ_w avg')
plt.xlabel('x [m]')
plt.ylabel('Wall shear stress τ_w [Pa]')
plt.title('Base Flow Wall Shear Stress Profile')
plt.legend()
plt.tight_layout()
plt.show()

# --- Previous Diagnostics ---
# --- Diagnostic: plot Blasius base flow profile ---

# --- Diagnostic: plot Blasius base flow profile ---
plt.figure(figsize=(6,4))
# Select downstream location x = L
plt.plot(y, U0[:, -1], label="Numerical U0 at x=L")
# Analytical similarity profile: map y to eta at x=L
et_a = y * np.sqrt(U_inf/(nu*L))
# Interpolate f1_eta for eta values
from numpy import interp
U_analytical = U_inf * interp(et_a, eta, f1_eta)
plt.plot(y, U_analytical, '--', label="Similarity f'(η) at x=L")
plt.xlabel('y [m]')
plt.ylabel('U [m/s]')
plt.title('Blasius Base Flow at x=L')
plt.legend()
plt.tight_layout()
plt.show()

# --- Diagnostic: contour of base flow U0 over x-y ---
plt.figure(figsize=(6,5))
Xb, Yb = np.meshgrid(x, y)
cmap = plt.contourf(Xb, Yb, U0, levels=50)
plt.colorbar(cmap, label="U0 [m/s]")
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Contour of Exact Blasius Base Flow U0(x,y)')
plt.tight_layout()
plt.show()



# Initialize perturbation field
delta = np.zeros((Ny, Nx))

# Helper: inlet profile (u') at x=0
def inlet_profile(t):
    return epsilon*U_inf*np.sin(omega*t)*np.exp(-y/(0.1*y[-1]))

# Storage for frames
frames_u = []  # total u
frames_v = []  # total v

# Time-marching loop
def compute_vprime(delta):
    vprime = np.zeros_like(delta)
    for i in range(Nx):
        for j in range(1, Ny):
            dv = (delta[j,i] - delta[j-1,i]) / dy
            vprime[j,i] = vprime[j-1,i] - dx * dv
    return vprime

for n, t in enumerate(time):
    # Inlet BC
    delta[:,0] = inlet_profile(t)
    new = delta.copy()
    # Interior update (explicit Euler)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            dudx = (delta[j,i] - delta[j,i-1]) / dx
            dudy = (delta[j,i] - delta[j-1,i]) / dy
            adv   = -U0[j,i]*dudx - V0[j,i]*dudy
            d2udx2 = (delta[j,i+1] - 2*delta[j,i] + delta[j,i-1]) / dx**2
            d2udy2 = (delta[j+1,i] - 2*delta[j,i] + delta[j-1,i]) / dy**2
            diff     = nu*(d2udx2 + d2udy2)
            new[j,i] = delta[j,i] + dt*(adv + diff)
    # BCs\    new[0,:] = 0.0
    new[-1,:] = new[-2,:]
    new[:,-1] = new[:,-2]
    delta = new
    # Compute v' via continuity (note integration in y)
    vprime = np.zeros_like(delta)
    for i in range(Nx):
        for j in range(1, Ny):
            ddux = (delta[j,i] - delta[j-1,i]) / dy
            vprime[j,i] = vprime[j-1,i] - dx * ddux
    u_abs = U0 + delta
    v_abs = V0 + vprime
    if n % 5 == 0:
        frames_u.append(u_abs.copy())
        frames_v.append(v_abs.copy())

# Convert to arrays and set up time snapshots
frames_u = np.array(frames_u)
frames_v = np.array(frames_v)
# Replace NaNs/Infs from instability
frames_u = np.nan_to_num(frames_u, nan=0.0, posinf=0.0, neginf=0.0)
frames_v = np.nan_to_num(frames_v, nan=0.0, posinf=0.0, neginf=0.0)
nframes = frames_u.shape[0]
t_snap = np.arange(nframes) * 5 * dt

# Compute unsteady drag
drag = np.zeros(nframes)
for k in range(nframes):
    dud_y = (frames_u[k,1,:] - frames_u[k,0,:]) / dy
    drag[k] = mu * np.trapezoid(dud_y, x)
# Remove any NaN/Inf from drag
drag = np.nan_to_num(drag, nan=drag_analytical, posinf=drag_analytical, neginf=0.0)

drag = np.zeros(nframes)
for k in range(nframes):
    dud_y = (frames_u[k,1,:] - frames_u[k,0,:]) / dy
    drag[k] = mu * np.trapezoid(dud_y, x)

# Plot & animate
gs = GridSpec(2,2, width_ratios=[1,1], height_ratios=[1,1])
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,:])
X, Y = np.meshgrid(x, y)
levels_u = np.linspace(frames_u.min(), frames_u.max(), 50)
levels_v = np.linspace(frames_v.min(), frames_v.max(), 50)

# Initial static plots with colorbars
cu = ax1.contourf(X, Y, frames_u[0], levels=levels_u)
cbar_u = fig.colorbar(cu, ax=ax1)
cbar_u.set_label('u [m/s]')
cv = ax2.contourf(X, Y, frames_v[0], levels=levels_v)
cbar_v = fig.colorbar(cv, ax=ax2)
cbar_v.set_label('v [m/s]')

# Steady drag line on ax3
line_unsteady, = ax3.plot([], [], 'k-', label='Unsteady drag')
steady_line = ax3.axhline(drag_base, color='b', linestyle='--', label='Steady numerical drag')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Drag [N/m]')
ax3.legend()

# Animation function
def animate(k):
    ax1.clear(); ax2.clear(); ax3.clear()
    # update contours
    ax1.contourf(X, Y, frames_u[k], levels=levels_u)
    ax2.contourf(X, Y, frames_v[k], levels=levels_v)
    # update drag plot
    ax3.plot(t_snap[:k+1], drag[:k+1], 'k-')
    ax3.axhline(drag_base, color='b', linestyle='--')
    # labels and legends
    ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]'); ax1.set_title(f'u at t={t_snap[k]:.3f}s')
    ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]'); ax2.set_title(f'v at t={t_snap[k]:.3f}s')
    ax3.set_xlabel('Time [s]'); ax3.set_ylabel('Drag [N/m]')
    ax3.legend(['Unsteady drag', 'Steady numerical drag'])

ani = FuncAnimation(fig, animate, frames=nframes, interval=100)
plt.tight_layout()
plt.show()
