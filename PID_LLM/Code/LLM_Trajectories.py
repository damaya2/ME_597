import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Robust imports with fallbacks to handle modular CRTBP project setups
try:
    from LLM_Controller import (
        AU_TO_MS2,
        EARTH_POS_KM,
        LU_TO_KM,
        MOON_POS_KM,
        R_EARTH_KM,
        R_MOON_KM,
        TU_TO_HOURS,
        VU_TO_MS,
    )
except ImportError:
    from CRTBP import DU, MU, TU
    from Trajectories import LU_TO_KM, TU_TO_HOURS, VU_TO_MS, AU_TO_MS2

    R_EARTH_KM = 6378.14
    R_MOON_KM = 1737.4
    EARTH_POS_KM = np.array([-MU * LU_TO_KM, 0.0, 0.0])
    MOON_POS_KM = np.array([(1.0 - MU) * LU_TO_KM, 0.0, 0.0])


def set_3d_equal_aspect(ax, x_data, y_data, z_data, padding_km=2000.0):
    """Enforces equal aspect ratios across all 3 spatial axes in Matplotlib 3D."""
    max_range = (
        np.array(
            [
                np.max(x_data) - np.min(x_data),
                np.max(y_data) - np.min(y_data),
                np.max(z_data) - np.min(z_data),
            ]
        ).max()
        / 2.0
    ) + padding_km

    mid_x = (np.max(x_data) + np.min(x_data)) * 0.5
    mid_y = (np.max(y_data) + np.min(y_data)) * 0.5
    mid_z = (np.max(z_data) + np.min(z_data)) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1, 1, 1])


def animate_trajectory_3d(
    sim_results: dict,
    save_path: str = "trajectory_animation.gif",
    fps: int = 10,
):
    traj_id = sim_results.get("traj_id", 1)
    print(
        f"[*] Generating 3D trajectory animation for Trajectory #{traj_id}...",
        flush=True,
    )

    time_hrs = (sim_results["time"] - sim_results["time"][0]) * TU_TO_HOURS
    ref_pos_km = sim_results["state_ref"][:, :3] * LU_TO_KM
    sim_pos_km = sim_results["state_sim"][:, :3] * LU_TO_KM

    num_frames = len(time_hrs)

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Render Celestial Bodies
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    xe = EARTH_POS_KM[0] + (R_EARTH_KM * 2) * np.cos(u) * np.sin(v)
    ye = EARTH_POS_KM[1] + (R_EARTH_KM * 2) * np.sin(u) * np.sin(v)
    ze = EARTH_POS_KM[2] + (R_EARTH_KM * 2) * np.cos(v)
    ax.plot_surface(xe, ye, ze, color="blue", alpha=0.3)

    xm = MOON_POS_KM[0] + (R_MOON_KM * 3) * np.cos(u) * np.sin(v)
    ym = MOON_POS_KM[1] + (R_MOON_KM * 3) * np.sin(u) * np.sin(v)
    zm = MOON_POS_KM[2] + (R_MOON_KM * 3) * np.cos(v)
    ax.plot_surface(xm, ym, zm, color="gray", alpha=0.4)

    # Static Reference Trajectory Background
    ax.plot(
        ref_pos_km[:, 0],
        ref_pos_km[:, 1],
        ref_pos_km[:, 2],
        "k--",
        linewidth=1.2,
        alpha=0.5,
        label="Reference Path",
    )

    # Animated elements
    (line_sim,) = ax.plot(
        [], [], [], "r-", linewidth=2.0, label=f"LLM Traj #{traj_id}"
    )
    (point_sim,) = ax.plot([], [], [], "ro", markersize=7, label="Spacecraft")
    time_text = ax.text2D(
        0.05, 0.95, "", transform=ax.transAxes, fontweight="bold"
    )

    all_x = np.concatenate(
        [ref_pos_km[:, 0], sim_pos_km[:, 0], [EARTH_POS_KM[0], MOON_POS_KM[0]]]
    )
    all_y = np.concatenate(
        [ref_pos_km[:, 1], sim_pos_km[:, 1], [EARTH_POS_KM[1], MOON_POS_KM[1]]]
    )
    all_z = np.concatenate(
        [ref_pos_km[:, 2], sim_pos_km[:, 2], [EARTH_POS_KM[2], MOON_POS_KM[2]]]
    )

    set_3d_equal_aspect(ax, all_x, all_y, all_z, padding_km=2000.0)

    ax.set_title(
        f"Closed-Loop Control Trajectory Progression (Run #{traj_id})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend(loc="upper right")

    def init():
        line_sim.set_data(np.array([]), np.array([]))
        line_sim.set_3d_properties(np.array([]))
        point_sim.set_data(np.array([]), np.array([]))
        point_sim.set_3d_properties(np.array([]))
        time_text.set_text("")
        return line_sim, point_sim, time_text

    def update(frame):
        idx = min(frame, num_frames - 1)

        line_sim.set_data(
            sim_pos_km[: idx + 1, 0], sim_pos_km[: idx + 1, 1]
        )
        line_sim.set_3d_properties(sim_pos_km[: idx + 1, 2])

        point_sim.set_data(
            np.array([sim_pos_km[idx, 0]]), np.array([sim_pos_km[idx, 1]])
        )
        point_sim.set_3d_properties(np.array([sim_pos_km[idx, 2]]))

        time_text.set_text(f"Time: {time_hrs[idx]:.2f} hrs")
        return line_sim, point_sim, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        init_func=init,
        blit=False,
        interval=1000 / fps,
    )

    try:
        if save_path.endswith(".gif"):
            try:
                anim.save(save_path, writer="pillow", fps=fps)
            except Exception:
                anim.save(save_path, writer="imagemagick", fps=fps)
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps)
        print(f"[*] Animation saved to: {save_path}", flush=True)
    except Exception as e:
        print(f"[Warning] Failed to export animation GIF/MP4: {e}", flush=True)

    plt.close(fig)


def plot_single_trajectory_results(
    sim_results: dict,
    save_path: str = "trajectory_results.png",
    save_anim: bool = True,
    anim_path: str = "trajectory_animation.gif",
):
    """Plots multi-panel diagnostic analysis for a single trajectory run."""
    time_hrs = (sim_results["time"] - sim_results["time"][0]) * TU_TO_HOURS
    ref_pos_km = sim_results["state_ref"][:, :3] * LU_TO_KM
    sim_pos_km = sim_results["state_sim"][:, :3] * LU_TO_KM

    pos_err_km = sim_results["pos_err_km"]
    vel_err_ms = sim_results["vel_err_ms"]
    control_ms2 = sim_results["control"] * AU_TO_MS2

    control_time_hrs = time_hrs[: len(control_ms2)]

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig = plt.figure(figsize=(16, 12), dpi=300)

    traj_id = sim_results.get("traj_id", 1)

    # PANEL 1: 3D Trajectory Plot
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(
        ref_pos_km[:, 0],
        ref_pos_km[:, 1],
        ref_pos_km[:, 2],
        "k--",
        linewidth=1.8,
        label="Nominal Reference",
        alpha=0.8,
    )
    ax1.plot(
        sim_pos_km[:, 0],
        sim_pos_km[:, 1],
        sim_pos_km[:, 2],
        "r-",
        linewidth=2.0,
        label=f"LLM-Controlled #{traj_id}",
    )
    ax1.scatter(
        *sim_pos_km[0],
        color="green",
        s=60,
        marker="o",
        label="Start",
        zorder=5,
    )

    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    xe = EARTH_POS_KM[0] + (R_EARTH_KM * 2) * np.cos(u) * np.sin(v)
    ye = EARTH_POS_KM[1] + (R_EARTH_KM * 2) * np.sin(u) * np.sin(v)
    ze = EARTH_POS_KM[2] + (R_EARTH_KM * 2) * np.cos(v)
    ax1.plot_surface(xe, ye, ze, color="blue", alpha=0.4)

    xm = MOON_POS_KM[0] + (R_MOON_KM * 3) * np.cos(u) * np.sin(v)
    ym = MOON_POS_KM[1] + (R_MOON_KM * 3) * np.sin(u) * np.sin(v)
    zm = MOON_POS_KM[2] + (R_MOON_KM * 3) * np.cos(v)
    ax1.plot_surface(xm, ym, zm, color="gray", alpha=0.5)

    all_x = np.concatenate([ref_pos_km[:, 0], sim_pos_km[:, 0]])
    all_y = np.concatenate([ref_pos_km[:, 1], sim_pos_km[:, 1]])
    all_z = np.concatenate([ref_pos_km[:, 2], sim_pos_km[:, 2]])

    set_3d_equal_aspect(ax1, all_x, all_y, all_z, padding_km=1000.0)

    ax1.set_title(
        f"3D Synodic Trajectory (Run #{traj_id})", fontsize=12, fontweight="bold"
    )
    ax1.set_xlabel("X (km)", labelpad=8)
    ax1.set_ylabel("Y (km)", labelpad=8)
    ax1.set_zlabel("Z (km)", labelpad=8)
    ax1.legend(loc="upper right", frameon=True)

    # PANEL 2: XY Plane Projection
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(
        ref_pos_km[:, 0],
        ref_pos_km[:, 1],
        "k--",
        linewidth=1.5,
        label="Reference",
    )
    ax2.plot(
        sim_pos_km[:, 0],
        sim_pos_km[:, 1],
        "r-",
        linewidth=1.8,
        label="LLM Trajectory",
    )

    earth_circle = plt.Circle(
        (EARTH_POS_KM[0], EARTH_POS_KM[1]),
        R_EARTH_KM * 2,
        color="blue",
        alpha=0.3,
        label="Earth Region",
    )
    moon_circle = plt.Circle(
        (MOON_POS_KM[0], MOON_POS_KM[1]),
        R_MOON_KM * 3,
        color="gray",
        alpha=0.4,
        label="Moon Region",
    )
    ax2.add_patch(earth_circle)
    ax2.add_patch(moon_circle)
    ax2.scatter(
        sim_pos_km[0, 0], sim_pos_km[0, 1], color="green", s=50, zorder=5
    )

    ax2.set_title("Synodic XY Projection", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X Position (km)")
    ax2.set_ylabel("Y Position (km)")
    ax2.axis("equal")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    # PANEL 3: Tracking Errors
    ax3 = fig.add_subplot(2, 2, 3)
    color1 = "tab:red"
    ax3.set_xlabel("Time (Hours)", fontweight="bold")
    ax3.set_ylabel("Position Error (km)", color=color1, fontweight="bold")
    line1 = ax3.plot(
        time_hrs[: len(pos_err_km)],
        pos_err_km,
        color=color1,
        linewidth=2.0,
        label="Pos Error (km)",
    )
    ax3.tick_params(axis="y", labelcolor=color1)

    ax3_vel = ax3.twinx()
    color2 = "tab:blue"
    ax3_vel.set_ylabel("Velocity Error (m/s)", color=color2, fontweight="bold")
    line2 = ax3_vel.plot(
        time_hrs[: len(vel_err_ms)],
        vel_err_ms,
        color=color2,
        linewidth=2.0,
        linestyle="--",
        label="Vel Error (m/s)",
    )
    ax3_vel.tick_params(axis="y", labelcolor=color2)

    dist_lines = []
    if "disturbances" in sim_results and sim_results["disturbances"]:
        for idx_dist, dist in enumerate(sim_results["disturbances"]):
            if isinstance(dist, (list, tuple, np.ndarray)) and len(dist) == 3:
                step_idx, t_dist_hrs, d_vel_ms = dist
                kick_cm_s = np.linalg.norm(d_vel_ms) * 100.0
                label_str = (
                    f"ΔV Impulse ({kick_cm_s:.1f} cm/s)" if idx_dist == 0 else ""
                )
            else:
                t_dist_hrs = float(dist)
                label_str = "ΔV Impulse" if idx_dist == 0 else ""

            vline = ax3.axvline(
                x=t_dist_hrs,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label=label_str,
            )
            if idx_dist == 0:
                dist_lines.append(vline)

    lines = line1 + line2 + dist_lines
    labels = [l.get_label() for l in lines if l.get_label() != ""]
    ax3.legend(lines, labels, loc="upper right", frameon=True)
    ax3.set_title("Tracking Error Response", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # PANEL 4: LLM Control Acceleration Commands
    ax4 = fig.add_subplot(2, 2, 4)
    if len(control_ms2) > 0:
        ax4.step(
            control_time_hrs,
            control_ms2[:, 0],
            where="post",
            label=r"$u_x$",
            linewidth=1.5,
        )
        ax4.step(
            control_time_hrs,
            control_ms2[:, 1],
            where="post",
            label=r"$u_y$",
            linewidth=1.5,
        )
        ax4.step(
            control_time_hrs,
            control_ms2[:, 2],
            where="post",
            label=r"$u_z$",
            linewidth=1.5,
        )

    ax4.axhline(
        0.01,
        color="black",
        linestyle=":",
        alpha=0.7,
        label="Max Limit (0.01 m/s²)",
    )
    ax4.axhline(-0.01, color="black", linestyle=":", alpha=0.7)

    ax4.set_title(
        "LLM Control Acceleration Commands", fontsize=12, fontweight="bold"
    )
    ax4.set_xlabel("Time (Hours)", fontweight="bold")
    ax4.set_ylabel("Acceleration (m/s²)", fontweight="bold")

    ax4.set_yscale("symlog", linthresh=1e-7)
    ax4.set_ylim(-1.5e-2, 1.5e-2)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[*] Saved single trajectory plot to: {save_path}", flush=True)
    plt.close(fig)

    if save_anim:
        animate_trajectory_3d(sim_results, save_path=anim_path, fps=10)


def plot_multi_trajectory_comparison(
    trajectories: list, save_path: str = "multi_trajectory_summary.png"
):
    print(
        f"[*] Generating aggregated comparison plot for {len(trajectories)} trajectories...",
        flush=True,
    )

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    # 1. Position Errors
    ax_pos = axes[0, 0]
    for idx, res in enumerate(trajectories):
        t_hrs = (res["time"] - res["time"][0]) * TU_TO_HOURS
        ax_pos.plot(
            t_hrs[: len(res["pos_err_km"])],
            res["pos_err_km"],
            color=colors[idx % len(colors)],
            linewidth=2.0,
            label=f"Traj #{res.get('traj_id', idx+1)} ({res.get('mission_type', 'N/A')})",
        )
    ax_pos.set_title("Position Error Convergence (km)", fontweight="bold")
    ax_pos.set_xlabel("Time (Hours)")
    ax_pos.set_ylabel("Position Error (km)")
    ax_pos.grid(True, linestyle=":", alpha=0.6)
    ax_pos.legend(loc="upper right")

    # 2. Velocity Errors
    ax_vel = axes[0, 1]
    for idx, res in enumerate(trajectories):
        t_hrs = (res["time"] - res["time"][0]) * TU_TO_HOURS
        ax_vel.plot(
            t_hrs[: len(res["vel_err_ms"])],
            res["vel_err_ms"],
            color=colors[idx % len(colors)],
            linewidth=2.0,
            label=f"Traj #{res.get('traj_id', idx+1)}",
        )
    ax_vel.set_title("Velocity Error Convergence (m/s)", fontweight="bold")
    ax_vel.set_xlabel("Time (Hours)")
    ax_vel.set_ylabel("Velocity Error (m/s)")
    ax_vel.grid(True, linestyle=":", alpha=0.6)
    ax_vel.legend(loc="upper right")

    # 3. Control Magnitude
    ax_u = axes[1, 0]
    for idx, res in enumerate(trajectories):
        t_hrs = (res["time"] - res["time"][0]) * TU_TO_HOURS
        u_mags = np.linalg.norm(res["control"], axis=1) * AU_TO_MS2
        ctrl_t_hrs = t_hrs[: len(u_mags)]
        u_mags_clipped = np.maximum(u_mags, 1e-12)

        ax_u.plot(
            ctrl_t_hrs,
            u_mags_clipped,
            color=colors[idx % len(colors)],
            linewidth=1.8,
            label=f"Traj #{res.get('traj_id', idx+1)}",
        )
    ax_u.axhline(
        0.01,
        color="black",
        linestyle=":",
        alpha=0.7,
        label="Max Accel (0.01 m/s²)",
    )
    ax_u.set_title("Control Acceleration Magnitude", fontweight="bold")
    ax_u.set_xlabel("Time (Hours)")
    ax_u.set_ylabel(r"||$u$|| (m/s²)")
    ax_u.set_yscale("log")
    ax_u.set_ylim(bottom=1e-8, top=2e-2)
    ax_u.grid(True, linestyle=":", alpha=0.6)
    ax_u.legend(loc="upper right")

    # 4. Cumulative Delta-V Bar Chart
    ax_dv = axes[1, 1]
    traj_labels = [
        f"Traj #{r.get('traj_id', i+1)}\n({r.get('mission_type', 'N/A')})"
        for i, r in enumerate(trajectories)
    ]
    dvs = [r["total_delta_v_ms"] for r in trajectories]

    bars = ax_dv.bar(
        traj_labels, dvs, color=colors[: len(trajectories)], alpha=0.85
    )
    ax_dv.set_title("Total Delta-V Expenditure (m/s)", fontweight="bold")
    ax_dv.set_ylabel("Total ΔV (m/s)")
    ax_dv.grid(True, linestyle=":", alpha=0.6, axis="y")

    for bar in bars:
        yval = bar.get_height()
        ax_dv.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.02,
            f"{yval:.2f} m/s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[*] Saved multi-trajectory summary plot to: {save_path}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    multi_path = "sim_results_multi.npy"
    single_path = "sim_results.npy"

    trajectories = []

    if os.path.exists(multi_path):
        print(f"[*] Loading multi-trajectory data from '{multi_path}'...", flush=True)
        loaded = np.load(multi_path, allow_pickle=True)
        trajectories = loaded.tolist() if isinstance(loaded, np.ndarray) else loaded
    elif os.path.exists(single_path):
        print(f"[*] Loading single-trajectory data from '{single_path}'...", flush=True)
        loaded = np.load(single_path, allow_pickle=True).item()
        trajectories = [loaded]
    else:
        print("[!] No results file found! Please run LLM_Controller.py first.", flush=True)

    if trajectories:
        if len(trajectories) > 1:
            plot_multi_trajectory_comparison(
                trajectories, save_path="multi_trajectory_summary.png"
            )

        for res in trajectories:
            tid = res.get("traj_id", 1)
            plot_single_trajectory_results(
                res,
                save_path=f"trajectory_results_run_{tid}.png",
                save_anim=True,
                anim_path=f"trajectory_animation_run_{tid}.gif",
            )