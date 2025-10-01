import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from blasius import BlasiusFlow
from solver import LinearBLSolver, BlowSuctionSolver, LinearizedJointSolver


CMAP = "viridis"


def _boundary_layer_snapshot(
    X,
    Y,
    frames_u,
    frames_v,
    t_snap,
    drag,
    drag_base,
    levels_u,
    levels_v,
    explicit,
):
    """Render a static copy of the boundary-layer animation layout."""

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, height_ratios=[1, 1])
    ax_u = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_drag = fig.add_subplot(gs[1, :])

    # fig.suptitle(
    #     "Boundary Layer Solver - {} scheme (snapshot)".format(
    #         "Explicit" if explicit else "Implicit"
    #     ),
    #     fontsize=14,
    # )

    contour_u = ax_u.contourf(X, Y, frames_u[-1], levels=levels_u, cmap=CMAP)
    contour_v = ax_v.contourf(X, Y, frames_v[-1], levels=levels_v, cmap=CMAP)

    fig.colorbar(contour_u, ax=ax_u).set_label("u [m/s]")
    fig.colorbar(contour_v, ax=ax_v).set_label("v [m/s]")

    ax_u.set_title(f"u at t={t_snap[-1]:.3f}s")
    ax_v.set_title(f"v at t={t_snap[-1]:.3f}s")
    ax_u.set_xlabel("x [m]")
    ax_u.set_ylabel("y [m]")
    ax_v.set_xlabel("x [m]")
    ax_v.set_ylabel("y [m]")

    ax_drag.plot(t_snap, drag, "k-", label="Unsteady drag")
    ax_drag.axhline(drag_base, color="b", linestyle="--", label="Steady numerical drag")
    ax_drag.set_xlabel("Time [s]")
    ax_drag.set_ylabel("Drag [N/m]")
    ax_drag.legend()

    fig.tight_layout()
    return fig


def _blow_suction_snapshot(X, Y, frames_u, frames_v, time, levels_u, levels_v, explicit):
    """Render a static copy of the blow/suction animation layout."""

    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Blow/Suction Solver - {} scheme (snapshot)".format(
            "Explicit" if explicit else "Implicit"
        ),
        fontsize=14,
    )

    contour_u = ax_u.contourf(X, Y, frames_u[-1], levels=levels_u, cmap=CMAP)
    contour_v = ax_v.contourf(X, Y, frames_v[-1], levels=levels_v, cmap=CMAP)

    fig.colorbar(contour_u, ax=ax_u).set_label("u [m/s]")
    fig.colorbar(contour_v, ax=ax_v).set_label("v [m/s]")

    ax_u.set_title(f"u at t={time[-1]:.3f}s")
    ax_v.set_title(f"v at t={time[-1]:.3f}s")
    ax_u.set_xlabel("x [m]")
    ax_u.set_ylabel("y [m]")
    ax_v.set_xlabel("x [m]")
    ax_v.set_ylabel("y [m]")

    fig.tight_layout()
    return fig


def run_boundary_layer(explicit: bool = True):
    """Run the boundary-layer solver with an explicit or implicit scheme."""

    # ----------------------- PARAMETERS -----------------------
    if explicit:
        params = {
            "U_inf": 2.0,
            "nu": 1.5e-5,
            "rho": 1.225,
            "L": 1.0,
            "Nx": 100,
            "Ny": 100,
            "dt": 1e-4,
            "Nt": 10000,
            "epsilon": 0.08,
            "freq": 4.0,
            "implicit_iter": None,
        }
    else:
        params = {
            "U_inf": 2.0,
            "nu": 1.5e-5,
            "rho": 1.225,
            "L": 1.0,
            "Nx": 100,
            "Ny": 100,
            "dt": 1.0e-2,
            "Nt": 100,
            "epsilon": 0.08,
            "freq": 4.0,
            "implicit_iter": 15,
        }

    U_inf = params["U_inf"]
    nu = params["nu"]
    rho = params["rho"]
    mu = rho * nu
    L = params["L"]

    Nx, Ny = params["Nx"], params["Ny"]
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, 0.05, Ny)

    dt = params["dt"]
    Nt = params["Nt"]

    epsilon = params["epsilon"]
    omega = 2 * np.pi * params["freq"]

    def inlet(t, coords):
        return epsilon * U_inf * np.sin(omega * t) * np.exp(-coords / (0.1 * coords[-1]))

    # ----------------------------------------------------------

    # Build Blasius base flow
    base = BlasiusFlow(U_inf, nu, x, y)
    drag_base = base.wall_shear_drag(mu, x)

    solver = LinearBLSolver(base, rho, nu, dt, Nt, inlet, verbose=True)
    if explicit:
        frames_u, frames_v, t_snap = solver.run_explicit()
    else:
        n_iter = params["implicit_iter"] or 20
        frames_u, frames_v, t_snap = solver.run_implicit(n_iter=n_iter, tol=1e-6)
    drag = solver.compute_drag(frames_u, mu)

    period = 2 * np.pi / omega
    mask = t_snap >= (t_snap[-1] - period)
    mean_drag = np.mean(drag[mask])
    amplitude = 0.5 * (np.max(drag[mask]) - np.min(drag[mask]))
    print(f"Mean drag last cycle: {mean_drag:.6f} (steady {drag_base:.6f})")
    print(f"Oscillation amplitude last cycle: {amplitude:.6f}")

    X, Y = np.meshgrid(x, y)
    levels_u = np.linspace(frames_u.min(), frames_u.max(), 50)
    inlet_cut = 10
    v_slice = frames_v[:, :, inlet_cut:]
    levels_v = np.linspace(v_slice.min(), v_slice.max(), 50)

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, height_ratios=[1, 1])
    ax_u = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_drag = fig.add_subplot(gs[1, :])

    fig.suptitle(
        "Boundary Layer Solver - {} scheme".format("Explicit" if explicit else "Implicit"),
        fontsize=14,
    )

    contour_u = [ax_u.contourf(X, Y, frames_u[0], levels=levels_u, cmap=CMAP)]
    contour_v = [ax_v.contourf(X, Y, frames_v[0], levels=levels_v, cmap=CMAP)]

    cbar_u = fig.colorbar(contour_u[0], ax=ax_u)
    cbar_v = fig.colorbar(contour_v[0], ax=ax_v)
    cbar_u.set_label("u [m/s]")
    cbar_v.set_label("v [m/s]")

    ax_u.set_xlabel("x [m]")
    ax_u.set_ylabel("y [m]")
    ax_v.set_xlabel("x [m]")
    ax_v.set_ylabel("y [m]")

    (drag_line,) = ax_drag.plot([], [], "k-", label="Unsteady drag")
    ax_drag.axhline(drag_base, color="b", linestyle="--", label="Steady numerical drag")
    ax_drag.set_xlabel("Time [s]")
    ax_drag.set_ylabel("Drag [N/m]")
    ax_drag.legend()

    ax_u.set_title(f"u at t={t_snap[0]:.3f}s")
    ax_v.set_title(f"v at t={t_snap[0]:.3f}s")

    def animate(k):
        contour_u[0] = ax_u.contourf(X, Y, frames_u[k], levels=levels_u, cmap=CMAP)
        cbar_u.update_normal(contour_u[0])

        contour_v[0] = ax_v.contourf(X, Y, frames_v[k], levels=levels_v, cmap=CMAP)
        cbar_v.update_normal(contour_v[0])

        drag_line.set_data(t_snap[: k + 1], drag[: k + 1])
        ax_drag.relim()
        ax_drag.autoscale_view()

        ax_u.set_title(f"u at t={t_snap[k]:.3f}s")
        ax_v.set_title(f"v at t={t_snap[k]:.3f}s")

    ani = FuncAnimation(fig, animate, frames=len(t_snap), interval=100, blit=False)

    fig.tight_layout()

    _boundary_layer_snapshot(
        X,
        Y,
        frames_u,
        frames_v,
        t_snap,
        drag,
        drag_base,
        levels_u,
        levels_v,
        explicit,
    )

    plt.show()
    return ani


def run_blow_suction(explicit: bool = True):
    """Run the blow/suction solver with an explicit or implicit scheme."""

    if explicit:
        params = {
            "rho": 1.0,
            "nu": 1.81e-5,
            "Nx": 100,
            "Ny": 100,
            "dt": 1e-3,
            "Nt": 1000,
            "wall_amp": 5.0,
            "wall_freq": 4 * np.pi,
        }
    else:
        params = {
            "rho": 1.0,
            "nu": 1.81e-5,
            "Nx": 100,
            "Ny": 100,
            "dt": 1e-2,
            "Nt": 100,
            "wall_amp": 5.0,
            "wall_freq": 4 * np.pi,
        }

    rho = params["rho"]
    nu = params["nu"]
    x = np.linspace(0.0, 0.1, params["Nx"])
    y = np.linspace(0.0, 0.1, params["Ny"])
    dt = params["dt"]
    Nt = params["Nt"]

    def wall(t, coords):
        amplitude = params["wall_amp"]
        frequency = params["wall_freq"]
        return amplitude * np.sin(frequency * coords / 0.1 + 4 * np.pi * t) * np.ones_like(coords)

    solver = BlowSuctionSolver(rho, nu, x, y, dt, Nt, wall, verbose=True)
    solver.stability_report()

    if explicit:
        frames_u, frames_v, time = solver.run_explicit()
    else:
        frames_u, frames_v, time = solver.run_implicit()

    X, Y = np.meshgrid(x, y)
    levels_u = np.linspace(frames_u.min(), frames_u.max(), 50)
    levels_v = np.linspace(frames_v.min(), frames_v.max(), 50)

    fig = plt.figure(figsize=(12, 5))
    ax_u = fig.add_subplot(1, 2, 1)
    ax_v = fig.add_subplot(1, 2, 2)

    fig.suptitle(
        "Blow/Suction Solver - {} scheme".format("Explicit" if explicit else "Implicit"),
        fontsize=14,
    )

    contour_u = [ax_u.contourf(X, Y, frames_u[0], levels=levels_u, cmap=CMAP)]
    contour_v = [ax_v.contourf(X, Y, frames_v[0], levels=levels_v, cmap=CMAP)]

    cbar_u = fig.colorbar(contour_u[0], ax=ax_u)
    cbar_v = fig.colorbar(contour_v[0], ax=ax_v)
    cbar_u.set_label("u [m/s]")
    cbar_v.set_label("v [m/s]")

    ax_u.set_xlabel("x [m]")
    ax_u.set_ylabel("y [m]")
    ax_v.set_xlabel("x [m]")
    ax_v.set_ylabel("y [m]")

    ax_u.set_title(f"u at t={time[0]:.3f}s")
    ax_v.set_title(f"v at t={time[0]:.3f}s")

    def animate(k):
        contour_u[0] = ax_u.contourf(X, Y, frames_u[k], levels=levels_u, cmap=CMAP)
        cbar_u.update_normal(contour_u[0])

        contour_v[0] = ax_v.contourf(X, Y, frames_v[k], levels=levels_v, cmap=CMAP)
        cbar_v.update_normal(contour_v[0])

        ax_u.set_title(f"u at t={time[k]:.3f}s")
        ax_v.set_title(f"v at t={time[k]:.3f}s")

    ani = FuncAnimation(fig, animate, frames=len(time), interval=100, blit=False)

    fig.tight_layout()

    _blow_suction_snapshot(
        X,
        Y,
        frames_u,
        frames_v,
        time,
        levels_u,
        levels_v,
        explicit,
    )

    plt.show()
    return ani

def run_joint_fully_implicit():
    """Run the fully-implicit joint solver (Blasius base + wall blow/suction)."""

    # ----------------------- PARAMETERS -----------------------
    params = {
        "U_inf": 2.5,
        "nu": 1.5e-5,
        "rho": 1.225,
        "L": 1.0,
        "Nx": 100,
        "Ny": 100,
        "ymax": 0.05,        # domain height (m)
        "dt": 1.0e-3,
        "Nt": 1001,
        "wall_amp": 0.05,     # wall-normal velocity amplitude [m/s]
        "wall_freq": 10.0,    # Hz
        "gs_sweeps": 4,      # inner block-GS coupling iterations per step
    }

    U_inf = params["U_inf"]
    nu = params["nu"]
    rho = params["rho"]
    mu = rho * nu
    L = params["L"]
    Nx, Ny = params["Nx"], params["Ny"]
    ymax = params["ymax"]

    x = np.linspace(0.0, L, Nx)
    y = np.linspace(0.0, ymax, Ny)

    dt = params["dt"]
    Nt = params["Nt"]

    wall_amp = params["wall_amp"]
    wall_omega = 2.0 * np.pi * params["wall_freq"]

    # wall v(t, x): traveling sinusoid along x + temporal oscillation
    def wall(t, xcoords):
        # choose a gentle streamwise wavenumber over [0, L]
        kx = 4 * 2.0 * np.pi / max(L, 1e-12)
        return wall_amp * np.sin(kx * xcoords + wall_omega * t)

    # ----------------------------------------------------------

    # Build Blasius base flow
    base = BlasiusFlow(U_inf, nu, x, y, eta_max=10.0, Neta=1000)
    drag_base = base.wall_shear_drag(mu, x)  # steady reference drag (N/m)

    # Joint fully-implicit solver (perturbations about base)
    solver = LinearizedJointSolver(base, rho, nu, dt, Nt, wall, verbose=True, gs_sweeps=params["gs_sweeps"])
    solver.stability_report()

    # Run fully-implicit
    frames_u_p, frames_v_p, t_snap = solver.run()   # perturbation fields
    # total fields = base + perturbation
    # (broadcast base [Ny,Nx] over time axis)
    frames_u = frames_u_p + base.U0[None, :, :]
    frames_v = frames_v_p + base.V0[None, :, :]

    # Drag history from total u
    def compute_drag_total(frames_u_tot, mu_val, dy_val):
        # wall shear τ_w = μ * (∂u/∂y)|_wall; integrate over x -> N/m
        dud_y = (frames_u_tot[:, 1, :] - frames_u_tot[:, 0, :]) / dy_val  # shape [Nt, Nx]
        # trapezoid along x for each time
        return np.trapezoid(mu_val * dud_y, x, axis=1)

    drag = compute_drag_total(frames_u, mu, y[1] - y[0])

    # ---- Plot/animation (same style as other runners) ----
    X, Y = np.meshgrid(x, y)
    levels_u = np.linspace(frames_u.min(), frames_u.max(), 50)
    # avoid colorbar dominated by inlet edge; skip first few columns for v scaling if desired
    v_slice = frames_v[:, :, 10:]
    levels_v = np.linspace(v_slice.min(), v_slice.max(), 50)

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, height_ratios=[1, 1])
    ax_u = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_drag = fig.add_subplot(gs[1, :])

    fig.suptitle("Joint Solver (Blasius + Blow/Suction) – Fully Implicit", fontsize=14)

    contour_u = [ax_u.contourf(X, Y, frames_u[0], levels=levels_u, cmap=CMAP)]
    contour_v = [ax_v.contourf(X, Y, frames_v[0], levels=levels_v, cmap=CMAP)]

    cbar_u = fig.colorbar(contour_u[0], ax=ax_u)
    cbar_v = fig.colorbar(contour_v[0], ax=ax_v)
    cbar_u.set_label("u [m/s]")
    cbar_v.set_label("v [m/s]")

    ax_u.set_xlabel("x [m]"); ax_u.set_ylabel("y [m]")
    ax_v.set_xlabel("x [m]"); ax_v.set_ylabel("y [m]")

    (drag_line,) = ax_drag.plot([], [], "k-", label="Unsteady drag")
    ax_drag.axhline(drag_base, color="b", linestyle="--", label="Steady numerical drag")
    ax_drag.set_xlabel("Time [s]"); ax_drag.set_ylabel("Drag [N/m]")
    ax_drag.legend()

    ax_u.set_title(f"u at t={t_snap[0]:.3f}s")
    ax_v.set_title(f"v at t={t_snap[0]:.3f}s")

    def animate(k):
        contour_u[0] = ax_u.contourf(X, Y, frames_u[k], levels=levels_u, cmap=CMAP)
        cbar_u.update_normal(contour_u[0])

        contour_v[0] = ax_v.contourf(X, Y, frames_v[k], levels=levels_v, cmap=CMAP)
        cbar_v.update_normal(contour_v[0])

        drag_line.set_data(t_snap[: k + 1], drag[: k + 1])
        ax_drag.relim(); ax_drag.autoscale_view()

        ax_u.set_title(f"u at t={t_snap[k]:.3f}s")
        ax_v.set_title(f"v at t={t_snap[k]:.3f}s")

    ani = FuncAnimation(fig, animate, frames=len(t_snap), interval=100, blit=False)

    fig.tight_layout()

    # also render a static snapshot like other runners
    _boundary_layer_snapshot(
        X, Y, frames_u, frames_v, t_snap,
        drag, drag_base, levels_u, levels_v, explicit=False
    )

    plt.show()
    return ani


CASE = "joint-full"
"""
Valid options:
* "bl-explicit" – explicit boundary-layer run
* "bl-implicit" – implicit boundary-layer run
* "bs-explicit" – explicit blow/suction run
* "bs-implicit" – implicit blow/suction run
* "joint-full"  – fully-implicit Blasius + blow/suction (new)
"""

def main(case: str = CASE):
    if case == "bl-explicit":
        run_boundary_layer(explicit=True)
    elif case == "bl-implicit":
        run_boundary_layer(explicit=False)
    elif case == "bs-explicit":
        run_blow_suction(explicit=True)
    elif case == "bs-implicit":
        run_blow_suction(explicit=False)
    elif case == "joint-full":
        run_joint_fully_implicit()
    else:
        raise ValueError(
            "Unknown case. Choose one of "
            "'bl-explicit', 'bl-implicit', 'bs-explicit', 'bs-implicit', or 'joint-full'."
        )



if __name__ == "__main__":
    main()