from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from foamlib import FoamCase
import matplotlib.pyplot as plt


def run_particles(
    case: FoamCase,
    out_name: str = "particles",
    n_particles: int = 10,
    d_p: float = 0.0000625,   # very fine sand
    rho_p: float = 1840.0,    # wet sand
    rho_f: float = 1000.0,
    mu: float = 1.0e-3,
    g_vec: np.ndarray | None = None,
):
    if g_vec is None:
        g_vec = np.array([0.0, -9.81, 0.0])
    else:
        g_vec = np.array(g_vec, dtype=float)

    # --------------------------------------------------------
    # 1. CFD times + velocity fields
    # --------------------------------------------------------
    times = [t for t in case[1:]]  # skip time 0
    if not times:
        raise RuntimeError("No time directories found in case after time 0.")

    times = sorted(times, key=lambda t: t.time)
    t_vals = np.array([t.time for t in times], dtype=float)

    print("[run_particles] CFD times:", t_vals[:10], "...", t_vals[-3:])
    print("[run_particles] Nt =", len(t_vals))

    cc = times[0].cell_centers().internal_field
    assert isinstance(cc, np.ndarray)
    cc = np.asarray(cc, dtype=float)

    U_all = np.stack([t["U"].internal_field for t in times], axis=0)
    U_all = np.asarray(U_all, dtype=float)

    # align sizes
    n_cells = min(cc.shape[0], U_all.shape[1])
    if cc.shape[0] != U_all.shape[1]:
        print(
            f"[run_particles] Warning: C has {cc.shape[0]} cells, "
            f"U has {U_all.shape[1]} cells. Using first {n_cells}."
        )
    cc = cc[:n_cells, :]
    U_all = U_all[:, :n_cells, :]

    tree = cKDTree(cc)
    print("[run_particles] cc shape:", cc.shape, "U_all shape:", U_all.shape)

    # domain extents (for bounds + z walls)
    x_min_cc, x_max_cc = cc[:, 0].min(), cc[:, 0].max()
    y_min_cc, y_max_cc = cc[:, 1].min(), cc[:, 1].max()
    z_min_cc, z_max_cc = cc[:, 2].min(), cc[:, 2].max()
    outlet_x = x_max_cc

    # --------------------------------------------------------
    # 2. BlockMesh-based geometry
    # --------------------------------------------------------
    mm = 1.0e-3
    X_INLET_LEFT   = -20.6 * mm
    X_CONTRACT_END = 0.0
    X_MAIN_END     = 206.0 * mm
    X_OUTLET_END   = 290.0 * mm

    Y_BOTTOM_INLET = 0.0
    Y_TOP_INLET    = 25.4 * mm
    Y_BOTTOM_MAIN  = -25.4 * mm
    Y_TOP_MAIN     = 25.4 * mm
    Y_BOTTOM_OUT   = -16.6 * mm
    Y_TOP_OUT      = 16.6 * mm

    def y_bottom_wall(x: float) -> float:
        if x <= X_CONTRACT_END:
            return Y_BOTTOM_INLET
        elif x <= X_MAIN_END:
            return Y_BOTTOM_MAIN
        else:
            s = (x - X_MAIN_END) / (X_OUTLET_END - X_MAIN_END)
            s = np.clip(s, 0.0, 1.0)
            return Y_BOTTOM_MAIN + s * (Y_BOTTOM_OUT - Y_BOTTOM_MAIN)

    def y_top_wall(x: float) -> float:
        if x <= X_MAIN_END:
            return Y_TOP_MAIN
        else:
            s = (x - X_MAIN_END) / (X_OUTLET_END - X_MAIN_END)
            s = np.clip(s, 0.0, 1.0)
            return Y_TOP_MAIN + s * (Y_TOP_OUT - Y_TOP_MAIN)

    # --------------------------------------------------------
    # 3. Fluid velocity interpolator
    # --------------------------------------------------------
    def fluid_velocity_at(xp_: np.ndarray, t: float) -> np.ndarray:
        if not np.all(np.isfinite(xp_)):
            return np.zeros(3)

        if t <= t_vals[0]:
            k0 = k1 = 0
            alpha = 0.0
        elif t >= t_vals[-1]:
            k0 = k1 = len(t_vals) - 1
            alpha = 0.0
        else:
            k1 = int(np.searchsorted(t_vals, t))
            k0 = k1 - 1
            dt = t_vals[k1] - t_vals[k0]
            alpha = (t - t_vals[k0]) / dt

        _, idx = tree.query(xp_)
        idx = int(np.clip(idx, 0, U_all.shape[1] - 1))

        u0 = U_all[k0, idx, :]
        u1 = U_all[k1, idx, :]

        return (1.0 - alpha) * u0 + alpha * u1

    # --------------------------------------------------------
    # 4. Particle properties + forces
    # --------------------------------------------------------
    V_p = np.pi / 6.0 * d_p**3
    m_p = rho_p * V_p
    A_p = np.pi / 4.0 * d_p**2
    Fg = (rho_p - rho_f) * V_p * g_vec

    # precompute a reasonable speed cap: 10x max fluid speed
    U_mag = np.linalg.norm(U_all.reshape(-1, 3), axis=1)
    U_mag_max = float(np.nanmax(U_mag))
    U_cap = 10.0 * U_mag_max
    print(f"[run_particles] max |U| in CFD = {U_mag_max:.3f}, speed cap = {U_cap:.3f}")

    def drag_force(u_f: np.ndarray, u_p: np.ndarray) -> np.ndarray:
        rel = u_f - u_p
        if not np.all(np.isfinite(rel)):
            return np.zeros(3)

        rel_mag = np.linalg.norm(rel)
        if rel_mag < 1e-14:
            return np.zeros(3)

        Re_p = rho_f * d_p * rel_mag / mu
        if Re_p < 1e-12:
            return np.zeros(3)
        Re_p = min(Re_p, 1e5)

        if Re_p < 1000.0:
            C_D = 24.0 / Re_p * (1.0 + 0.15 * Re_p**0.687)
        else:
            C_D = 0.44

        return 0.5 * rho_f * C_D * A_p * rel_mag * rel

    # --------------------------------------------------------
    # 5. Initial seeding
    # --------------------------------------------------------
    xp = np.zeros((n_particles, 3), dtype=float)
    up = np.zeros((n_particles, 3), dtype=float)

    x0 = X_INLET_LEFT + 0.2 * (X_CONTRACT_END - X_INLET_LEFT)
    y0_bottom = y_bottom_wall(x0)
    y0_top    = y_top_wall(x0)

    xp[:, 0] = x0
    xp[:, 1] = np.linspace(y0_bottom, y0_top, n_particles)
    xp[:, 2] = 0.5 * (z_min_cc + z_max_cc)

    print("[run_particles] seeding at x0 =", x0)
    print("[run_particles] seeding y in [", y0_bottom, ",", y0_top, "]")

    active = np.ones(n_particles, dtype=bool)

    # --------------------------------------------------------
    # 6. Time integration with more substeps
    # --------------------------------------------------------
    if len(t_vals) > 1:
        min_dt_cfd = float(np.min(np.diff(t_vals)))
    else:
        min_dt_cfd = 1e-3

    # smaller dt_p (100 substeps per smallest CFD interval)
    target_dt_p = min_dt_cfd / 100.0
    print("[run_particles] min dt CFD =", min_dt_cfd)
    print("[run_particles] target dt_p =", target_dt_p)

    traj_times: list[float] = []
    traj_xp: list[np.ndarray] = []

    current_t = float(t_vals[0])
    traj_times.append(current_t)
    traj_xp.append(xp.copy())

    nan_kills = 0

    for n in range(len(t_vals) - 1):
        t_start = float(t_vals[n])
        t_end   = float(t_vals[n + 1])
        dt_seg  = t_end - t_start
        if dt_seg <= 0.0:
            continue

        n_sub = max(1, int(np.round(dt_seg / target_dt_p)))
        dt_local = dt_seg / n_sub

        current_t = t_start
        for _ in range(n_sub):
            for i in range(n_particles):
                if not active[i]:
                    continue

                u_f = fluid_velocity_at(xp[i], current_t)
                Fd  = drag_force(u_f, up[i])
                a   = (Fd + Fg) / m_p

                up[i] += a * dt_local
                xp[i] += up[i] * dt_local

                # clamp particle speed to avoid runaway
                speed = np.linalg.norm(up[i])
                if speed > U_cap:
                    up[i] *= (U_cap / speed)

                # NaN/inf guard: kill and NaN-out
                if (not np.all(np.isfinite(xp[i]))) or (not np.all(np.isfinite(up[i]))):
                    active[i] = False
                    xp[i, :] = np.nan
                    up[i, :] = np.nan
                    nan_kills += 1
                    continue

                x_i = xp[i, 0]

                # outlet: allow exit only here
                if x_i >= outlet_x:
                    active[i] = False
                    continue

                # inlet left wall (bounce)
                if x_i < X_INLET_LEFT:
                    xp[i, 0] = X_INLET_LEFT
                    if up[i, 0] < 0.0:
                        up[i, 0] *= -1.0
                    x_i = xp[i, 0]

                # geometry-based top/bottom walls
                y_bot = y_bottom_wall(x_i)
                y_top = y_top_wall(x_i)

                if xp[i, 1] < y_bot:
                    xp[i, 1] = y_bot
                    if up[i, 1] < 0.0:
                        up[i, 1] *= -1.0

                if xp[i, 1] > y_top:
                    xp[i, 1] = y_top
                    if up[i, 1] > 0.0:
                        up[i, 1] *= -1.0

                # z walls
                if xp[i, 2] < z_min_cc:
                    xp[i, 2] = z_min_cc
                    if up[i, 2] < 0.0:
                        up[i, 2] *= -1.0
                if xp[i, 2] > z_max_cc:
                    xp[i, 2] = z_max_cc
                    if up[i, 2] > 0.0:
                        up[i, 2] *= -1.0

            current_t += dt_local

        traj_times.append(t_end)
        traj_xp.append(xp.copy())

    traj_times = np.array(traj_times)
    traj_xp = np.stack(traj_xp, axis=0)

    # --------------------------------------------------------
    # 7. Save trajectories
    # --------------------------------------------------------
    out_path = case.path / f"{out_name}.npz"
    np.savez(out_path, times=traj_times, xp=traj_xp)
    print(f"[run_particles] Saved particle data to {out_path}")
    print(f"[run_particles] NaN kills: {nan_kills}")

    # --------------------------------------------------------
    # 8. Plot x–y trajectories (finite only)
    # --------------------------------------------------------
    x_all = traj_xp[:, :, 0]
    y_all = traj_xp[:, :, 1]
    finite_mask = np.isfinite(x_all) & np.isfinite(y_all)

    if not np.any(finite_mask):
        print("[run_particles] No finite particle positions to plot – skipping figure.")
        print(f"[run_particles] Active particles at end: {active.sum()}/{n_particles}")
        return traj_times, traj_xp

    x_finite = x_all[finite_mask]
    y_finite = y_all[finite_mask]

    x_min_traj = float(x_finite.min())
    x_max_traj = float(x_finite.max())
    y_min_traj = float(y_finite.min())
    y_max_traj = float(y_finite.max())

    print("[run_particles] Trajectory x-range:", x_min_traj, "to", x_max_traj)
    print("[run_particles] Trajectory y-range:", y_min_traj, "to", y_max_traj)

    pad_x = 0.02 * (x_max_traj - x_min_traj) if x_max_traj > x_min_traj else 0.0
    pad_y = 0.02 * (y_max_traj - y_min_traj) if y_max_traj > y_min_traj else 0.0

    fig, ax = plt.subplots(figsize=(8, 3))

    for i in range(n_particles):
        ax.plot(traj_xp[:, i, 0], traj_xp[:, i, 1],
                linewidth=1.0, alpha=0.9)

    # scatter final finite positions
    xf_final = traj_xp[-1, :, 0]
    yf_final = traj_xp[-1, :, 1]
    finite_final = np.isfinite(xf_final) & np.isfinite(yf_final)
    ax.scatter(xf_final[finite_final], yf_final[finite_final],
               s=10, color="k", zorder=5)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min_traj - pad_x, x_max_traj + pad_x)
    ax.set_ylim(y_min_traj - pad_y, y_max_traj + pad_y)

    fig.tight_layout()
    fig_path = case.path / f"{out_name}_xy_trajectories.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[run_particles] Saved trajectory plot to {fig_path}")

    print(f"[run_particles] Active particles at end: {active.sum()}/{n_particles}")
    return traj_times, traj_xp
