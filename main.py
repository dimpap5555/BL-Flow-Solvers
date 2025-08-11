import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from blasius import BlasiusFlow
from solver import LinearBLSolver, BlowSuctionSolver


def main():
    # ----------------------- PARAMETERS -----------------------
    U_inf = 2.0
    nu = 1.5e-5
    rho = 1.225
    mu = rho * nu
    L = 1.0

    Nx, Ny = 500, 100
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, 0.05, Ny)

    dt = 5e-3
    Nt = 100

    epsilon = 0.1
    omega = 2 * np.pi * 5

    def inlet(t, y):
        return epsilon * U_inf * np.sin(omega * t)# * np.exp(-y / (0.1 * y[-1]))

    # ----------------------------------------------------------

    # Build Blasius base flow
    base = BlasiusFlow(U_inf, nu, x, y)
    drag_base = base.wall_shear_drag(mu, x)

    # Solve linearized equations
    solver = LinearBLSolver(base, rho, nu, dt, Nt, inlet, verbose=True)
    frames_u, frames_v, t_snap = solver.run_implicit(n_iter=20, tol=1e-6)
    drag = solver.compute_drag(frames_u, mu)

    period = 2 * np.pi / omega
    mask = t_snap >= (t_snap[-1] - period)
    mean_drag = np.mean(drag[mask])
    amplitude = 0.5 * (np.max(drag[mask]) - np.min(drag[mask]))
    print(f"Mean drag last cycle: {mean_drag:.6f} (steady {drag_base:.6f})")
    print(f"Oscillation amplitude last cycle: {amplitude:.6f}")

    # Plot & animate
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    X, Y = np.meshgrid(x, y)
    levels_u = np.linspace(frames_u.min(), frames_u.max(), 50)
    inlet_cut = 10
    v_slice = frames_v[:, :, inlet_cut:]
    levels_v = np.linspace(v_slice.min(), v_slice.max(), 50)

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

def blow_suction_example():
    """Run a simple demonstration of the BlowSuctionSolver."""
    rho = 1.0
    nu = 1e-3
    x = np.linspace(0.0, 1.0, 100)
    y = np.linspace(0.0, 0.05, 100)
    dt = 1e-2
    Nt = 100

    def wall(t, x):
        return 0.1 * np.sin(10 * np.pi * t + 2 * np.pi * 10 * x) * np.ones_like(x)

    solver = BlowSuctionSolver(rho, nu, x, y, dt, Nt, wall, cp=1.0, verbose=True)
    solver.stability_report()
    frames_u, frames_v, time = solver.run_implicit()

    # Post-processing: visualize the evolving velocity field
    _visualize_blow_suction(frames_u, frames_v, x, y, time)

    print(f"Max wall-normal velocity at final time: {frames_v[-1].max():.3e}")


def _visualize_blow_suction(frames_u, frames_v, x, y, time):
    """Create contour plots and an animation for the blow/suction solver."""
    X, Y = np.meshgrid(x, y)
    levels_u = np.linspace(frames_u.min(), frames_u.max(), 50)
    levels_v = np.linspace(frames_v.min(), frames_v.max(), 50)

    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    cu = ax1.contourf(X, Y, frames_u[0], levels=levels_u)
    fig.colorbar(cu, ax=ax1).set_label('u [m/s]')
    cv = ax2.contourf(X, Y, frames_v[0], levels=levels_v)
    fig.colorbar(cv, ax=ax2).set_label('v [m/s]')

    max_v = np.max(np.abs(frames_v), axis=(1, 2))
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Max |v| [m/s]')
    ax3.plot(time, max_v, 'k-')

    def animate(k):
        ax1.clear(); ax2.clear(); ax3.clear()
        ax1.contourf(X, Y, frames_u[k], levels=levels_u)
        ax2.contourf(X, Y, frames_v[k], levels=levels_v)
        ax3.plot(time[:k+1], max_v[:k+1], 'k-')
        ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]'); ax1.set_title(f'u at t={time[k]:.3e}s')
        ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]'); ax2.set_title(f'v at t={time[k]:.3e}s')
        ax3.set_xlabel('Time [s]'); ax3.set_ylabel('Max |v| [m/s]')

    ani = FuncAnimation(fig, animate, frames=len(time), interval=100)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    blow_suction_example()
    # main()
