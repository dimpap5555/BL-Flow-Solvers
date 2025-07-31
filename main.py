import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from blasius import BlasiusFlow
from solver import LinearBLSolver


def main():
    # ----------------------- PARAMETERS -----------------------
    U_inf = 2.0
    nu = 1.5e-5
    rho = 1.225
    mu = rho * nu
    L = 1.0

    Nx, Ny = 100, 100
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, 0.05, Ny)

    dt = 1e-4
    Nt = 200

    epsilon = 0.1
    omega = 2 * np.pi * 5

    # ----------------------------------------------------------

    # Build Blasius base flow
    base = BlasiusFlow(U_inf, nu, x, y)
    drag_base = base.wall_shear_drag(mu, x)

    # Solve linearized equations
    solver = LinearBLSolver(base, rho, nu, dt, Nt, epsilon, omega)
    frames_u, frames_v, t_snap = solver.run()
    drag = solver.compute_drag(frames_u, mu)

    # Plot & animate
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    X, Y = np.meshgrid(x, y)
    levels_u = np.linspace(frames_u.min(), frames_u.max(), 50)
    levels_v = np.linspace(frames_v.min(), frames_v.max(), 50)

    cu = ax1.contourf(X, Y, frames_u[0], levels=levels_u)
    fig.colorbar(cu, ax=ax1).set_label('u [m/s]')
    cv = ax2.contourf(X, Y, frames_v[0], levels=levels_v)
    fig.colorbar(cv, ax=ax2).set_label('v [m/s]')

    ax3.axhline(drag_base, color='b', linestyle='--', label='Steady numerical drag')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Drag [N/m]')
    ax3.legend()

    def animate(k):
        ax1.clear(); ax2.clear(); ax3.clear()
        ax1.contourf(X, Y, frames_u[k], levels=levels_u)
        ax2.contourf(X, Y, frames_v[k], levels=levels_v)
        ax3.plot(t_snap[:k+1], drag[:k+1], 'k-')
        ax3.axhline(drag_base, color='b', linestyle='--')
        ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]'); ax1.set_title(f'u at t={t_snap[k]:.3f}s')
        ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]'); ax2.set_title(f'v at t={t_snap[k]:.3f}s')
        ax3.set_xlabel('Time [s]'); ax3.set_ylabel('Drag [N/m]')
        ax3.legend(['Unsteady drag', 'Steady numerical drag'])

    ani = FuncAnimation(fig, animate, frames=len(t_snap), interval=100)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()