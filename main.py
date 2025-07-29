"""
Linearized Unsteady 2D Boundary Layer Solver over Blasius Flow
Solves the linearized x-momentum perturbation equation
and animates u(x,y,t) and v(x,y,t) on a full 2D grid via continuity.

Equations:
 ∂u'/∂t + u0 ∂u'/∂x + v0 ∂u'/∂y = ν(∂²u'/∂x² + ∂²u'/∂y²)
 ∂u'/∂x + ∂v'/∂y = 0
Base flow u0(x,y), v0(x,y) approximated via tanh(η).

Dependencies:
  - numpy
  - matplotlib

Usage:
  python linear_blasius_solver_xy_uv.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
dt = 1e-3                 # Time step [s]
Nt = 500                  # Number of time steps
U_inf = 2.0               # Freestream velocity [m/s]
nu = 1.5e-5               # Kinematic viscosity [m^2/s]
L = 1.0                   # Plate length [m]
Nx, Ny = 100, 100         # Grid sizes
x = np.linspace(0, L, Nx)
y = np.linspace(0, 0.05, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
time = np.arange(Nt) * dt

# Base flow (simplified Blasius) on grid
def blasius_uv(xi, yi):
    eta = yi * np.sqrt(U_inf / (nu * xi if xi > 0 else U_inf/(nu*1e-3)))
    u0 = U_inf * np.tanh(eta)
    v0 = 0.0
    return u0, v0

U0 = np.zeros((Ny, Nx))
V0 = np.zeros((Ny, Nx))
for i, xi in enumerate(x):
    for j, yj in enumerate(y):
        U0[j, i], V0[j, i] = blasius_uv(xi, yj)

# Perturbation initialization
delta = np.zeros((Ny, Nx))  # u' field

# Disturbance parameters
epsilon = 0.1           # Perturbation amplitude
omega = 2 * np.pi * 5   # Perturbation frequency

def inlet_profile(t):
    return epsilon * U_inf * np.sin(omega * t)# * np.exp(-y / (0.1 * y[-1]))

# Storage for frames
frames_u = []  # absolute u
frames_v = []  # absolute v

# Time integration loop
for n, t in enumerate(time):
    # Inlet BC at x=0 for u'
    delta[:, 0] = inlet_profile(t)
    new = delta.copy()
    # Interior update for u'
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            # upwind advection
            dudx = (delta[j, i] - delta[j, i - 1]) / dx
            dudy = (delta[j, i] - delta[j - 1, i]) / dy
            adv = -U0[j, i] * dudx - V0[j, i] * dudy
            # central diffusion
            d2udx2 = (delta[j, i + 1] - 2 * delta[j, i] + delta[j, i - 1]) / dx**2
            d2udy2 = (delta[j + 1, i] - 2 * delta[j, i] + delta[j - 1, i]) / dy**2
            diff = nu * (d2udx2 + d2udy2)
            new[j, i] = delta[j, i] + dt * (adv + diff)
    # Boundary conditions\
    new[0, :] = 0.0           # no-slip at wall y=0
    new[-1, :] = new[-2, :]   # zero gradient at y_max
    new[:, -1] = new[:, -2]   # zero-gradient at x=L
    delta = new

    # Compute v' from continuity: ∂v'/∂y = -∂u'/∂x
    vprime = np.zeros_like(delta)
    for i in range(Nx):
        for j in range(1, Ny):
            ddux = (delta[j, i] - delta[j, i - 1]) / dx if i > 0 else delta[j, i] / dx
            vprime[j, i] = vprime[j - 1, i] - dy * ddux
    # Absolute velocities
    u_abs = U0 + delta
    v_abs = V0 + vprime

    # Store every 5 steps
    if n % 5 == 0:
        frames_u.append(u_abs.copy())
        frames_v.append(v_abs.copy())

# Convert to arrays
frames_u = np.array(frames_u)
frames_v = np.array(frames_v)

# Meshgrid for plotting
X, Y = np.meshgrid(x, y)

# Contour levels
u_min, u_max = frames_u.min(), frames_u.max()
v_min, v_max = frames_v.min(), frames_v.max()
levels_u = np.linspace(u_min, u_max, 50)
levels_v = np.linspace(v_min, v_max, 50)

# Figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Initial plots
cu = ax1.contourf(X, Y, frames_u[0], levels=levels_u)
cv = ax2.contourf(X, Y, frames_v[0], levels=levels_v)

# Colorbars
fig.colorbar(cu, ax=ax1).set_label("u [m/s]")
fig.colorbar(cv, ax=ax2).set_label("v [m/s]")
ax1.set_title('u velocity')
ax2.set_title('v velocity')
for ax in (ax1, ax2):
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

# Animation function
def animate(k):
    ax1.clear(); ax2.clear()
    cu = ax1.contourf(X, Y, frames_u[k], levels=levels_u)
    cv = ax2.contourf(X, Y, frames_v[k], levels=levels_v)
    ax1.set_title(f'u at t={(k*5*dt):.3f}s')
    ax2.set_title(f'v at t={(k*5*dt):.3f}s')

ani = FuncAnimation(fig, animate, frames=len(frames_u), interval=50)
plt.tight_layout()
plt.show()
