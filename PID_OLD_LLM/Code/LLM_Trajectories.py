import json
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Physical constants for unit conversions
LU_TO_KM = 384400.0  # Canonical length to km
TU_TO_HOURS = 375190.0 / 3600.0  # Canonical time to hours (~104.22 hrs/TU)
VU_TO_MS = 1024.54  # Velocity unit to m/s
AU_TO_MS2 = 0.00273  # Canonical acceleration to m/s^2

MU = 0.01215058560962404
EARTH_POS_KM = np.array([-MU * LU_TO_KM, 0.0, 0.0])
MOON_POS_KM = np.array([(1.0 - MU) * LU_TO_KM, 0.0, 0.0])

R_EARTH_KM = 6371.0
R_MOON_KM = 1737.4

EVAL_JSON = "eval_comparison_results.json"
LLM_ONLY_JSON = "llm_evaluation_results.json"


def set_3d_equal_aspect(ax, x_data, y_data, z_data, padding_km=2000.0):
    """Enforces 1:1:1 geometric aspect ratio across all 3 spatial axes."""
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
    """Generates an animated 3D GIF showing spacecraft trajectory progression."""
    mission_name = sim_results.get("mission_type", "Mission").upper()
    print(f"[*] Generating 3D trajectory animation for [{mission_name}]...")

    time_hrs = (sim_results["time"] - sim_results["time"][0]) * TU_TO_HOURS
    ref_pos_km = sim_results["state_ref"][:, :3] * LU_TO_KM
    sim_pos_km = sim_results["state_sim"][:, :3] * LU_TO_KM

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Celestial Body Spheres
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    xe = EARTH_POS_KM[0] + (R_EARTH_KM * 2) * np.cos(u) * np.sin(v)
    ye = EARTH_POS_KM[1] + (R_EARTH_KM * 2) * np.sin(u) * np.sin(v)
    ze = EARTH_POS_KM[2] + (R_EARTH_KM * 2) * np.cos(v)
    ax.plot_surface(xe, ye, ze, color="blue", alpha=0.3)

    xm = MOON_POS_KM[0] + (R_MOON_KM * 3) * np.cos(u) * np.sin(v)
    ym = MOON_POS_KM[1] + (R_MOON_KM * 3) * np.sin(u) * np.sin(v)
    zm = MOON_POS_KM[2] + (R_MOON_KM * 3) * np.cos(v)
    ax.plot_surface(xm, ym, zm, color="gray", alpha=0.4)

    # Static Reference Trajectory Line
    ax.plot(
        ref_pos_km[:, 0],
        ref_pos_km[:, 1],
        ref_pos_km[:, 2],
        "k--",
        linewidth=1.2,
        alpha=0.5,
        label="Reference Path",
    )

    (line_sim,) = ax.plot([], [], [], "r-", linewidth=2.0, label=f"LLM Trajectory")
    (point_sim,) = ax.plot([], [], [], "ro", markersize=7, label="Spacecraft")
    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontweight="bold")

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
        f"Closed-Loop Trajectory Progression ({mission_name})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend(loc="upper right")

    def init():
        line_sim.set_data([], [])
        line_sim.set_3d_properties([])
        point_sim.set_data([], [])
        point_sim.set_3d_properties([])
        time_text.set_text("")
        return line_sim, point_sim, time_text

    def update(frame):
        line_sim.set_data(sim_pos_km[: frame + 1, 0], sim_pos_km[: frame + 1, 1])
        line_sim.set_3d_properties(sim_pos_km[: frame + 1, 2])

        point_sim.set_data([sim_pos_km[frame, 0]], [sim_pos_km[frame, 1]])
        point_sim.set_3d_properties([sim_pos_km[frame, 2]])

        time_text.set_text(f"Time: {time_hrs[frame]:.2f} hrs")
        return line_sim, point_sim, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(time_hrs),
        init_func=init,
        blit=False,
        interval=1000 / fps,
    )

    try:
        anim.save(save_path, writer="pillow", fps=fps)
    except Exception:
        anim.save(save_path, writer="imagemagick", fps=fps)

    plt.close(fig)
    print(f"[*] Animation saved to: {save_path}")


def plot_single_trajectory_results(
    sim_results: dict,
    save_path: str = "trajectory_results.png",
    save_anim: bool = True,
    anim_path: str = "trajectory_animation.gif",
):
    """Generates 4-panel analytical trajectory report."""
    time_hrs = (sim_results["time"] - sim_results["time"][0]) * TU_TO_HOURS
    ref_pos_km = sim_results["state_ref"][:, :3] * LU_TO_KM
    sim_pos_km = sim_results["state_sim"][:, :3] * LU_TO_KM
    mission_name = sim_results.get("mission_type", "Mission").upper()

    pos_err_km = np.linalg.norm(
        (sim_results["state_sim"][:, :3] - sim_results["state_ref"][:, :3])
        * LU_TO_KM,
        axis=1,
    )
    vel_err_ms = np.linalg.norm(
        (sim_results["state_sim"][:, 3:] - sim_results["state_ref"][:, 3:])
        * VU_TO_MS,
        axis=1,
    )

    control_ms2 = np.array(sim_results["control_u"]) * AU_TO_MS2
    control_time_hrs = (
        time_hrs[:-1] if len(control_ms2) < len(time_hrs) else time_hrs
    )

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig = plt.figure(figsize=(16, 12), dpi=300)

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
        label=f"LLM-Controlled",
    )
    ax1.scatter(
        *sim_pos_km[0], color="green", s=60, marker="o", label="Start", zorder=5
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
        f"3D Synodic Trajectory ({mission_name})", fontsize=12, fontweight="bold"
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

    # PANEL 3: Dual-Axis Tracking Errors
    ax3 = fig.add_subplot(2, 2, 3)
    color1 = "tab:red"
    ax3.set_xlabel("Time (Hours)", fontweight="bold")
    ax3.set_ylabel("Position Error (km)", color=color1, fontweight="bold")
    line1 = ax3.plot(
        time_hrs, pos_err_km, color=color1, linewidth=2.0, label="Pos Error (km)"
    )
    ax3.tick_params(axis="y", labelcolor=color1)

    ax3_vel = ax3.twinx()
    color2 = "tab:blue"
    ax3_vel.set_ylabel("Velocity Error (m/s)", color=color2, fontweight="bold")
    line2 = ax3_vel.plot(
        time_hrs,
        vel_err_ms,
        color=color2,
        linewidth=2.0,
        linestyle="--",
        label="Vel Error (m/s)",
    )
    ax3_vel.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper right", frameon=True)
    ax3.set_title("Tracking Error Response", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # PANEL 4: LLM Control Acceleration Commands (Auto-Scaled)
    ax4 = fig.add_subplot(2, 2, 4)

    # Calculate magnitude trace
    u_mag_ms2 = np.linalg.norm(control_ms2, axis=1)

    ax4.step(
        control_time_hrs,
        control_ms2[:, 0],
        where="post",
        label=r"$u_x$",
        linewidth=1.5,
        alpha=0.85,
    )
    ax4.step(
        control_time_hrs,
        control_ms2[:, 1],
        where="post",
        label=r"$u_y$",
        linewidth=1.5,
        alpha=0.85,
    )
    ax4.step(
        control_time_hrs,
        control_ms2[:, 2],
        where="post",
        label=r"$u_z$",
        linewidth=1.5,
        alpha=0.85,
    )
    ax4.step(
        control_time_hrs,
        u_mag_ms2,
        where="post",
        label=r"$||\mathbf{u}||$",
        linewidth=1.8,
        linestyle="--",
        color="black",
    )

    # Dynamic y-axis scaling with 20% margin
    max_val = max(np.max(np.abs(control_ms2)), np.max(u_mag_ms2), 1e-8)
    ax4.set_ylim(-max_val * 1.2, max_val * 1.2)

    # Use scientific notation for small acceleration regimes
    ax4.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    ax4.set_title(
        "LLM Control Acceleration Commands", fontsize=12, fontweight="bold"
    )
    ax4.set_xlabel("Time (Hours)", fontweight="bold")
    ax4.set_ylabel("Acceleration (m/s²)", fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", frameon=True, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[*] Saved trajectory plot to: {save_path}")
    plt.close(fig)

    if save_anim:
        animate_trajectory_3d(sim_results, save_path=anim_path, fps=10)


def load_all_mission_results(file_path: str) -> dict:
    """Parses JSON outputs into numpy structures for each mission."""
    with open(file_path, "r") as f:
        data = json.load(f)

    parsed_missions = {}
    for key, run_data in data.items():
        if isinstance(run_data, dict) and "state_sim" in run_data:
            parsed_missions[key] = {
                "mission_type": key,
                "time": np.array(run_data["time"]),
                "state_sim": np.array(run_data["state_sim"]),
                "state_ref": np.array(run_data["state_ref"]),
                "control_u": np.array(run_data["control_u"]),
                "metrics": run_data.get("metrics", {}),
            }
    return parsed_missions


def plot_summary_bar_chart(all_missions: dict, save_path="overall_summary.png"):
    """Generates a summary comparison bar chart across all evaluated missions."""
    missions = list(all_missions.keys())
    rms_pos = [all_missions[m]["metrics"].get("rms_pos_err_km", 0.0) for m in missions]
    delta_v = [all_missions[m]["metrics"].get("total_delta_v_ms", 0.0) for m in missions]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)

    x = np.arange(len(missions))
    width = 0.35

    rects1 = ax1.bar(x - width/2, rms_pos, width, label='RMS Pos Error (km)', color='crimson', alpha=0.8)
    ax1.set_ylabel('RMS Position Error (km)', color='crimson', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='crimson')

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, delta_v, width, label='Total Delta-V (m/s)', color='teal', alpha=0.8)
    ax2.set_ylabel('Total Delta-V Expenditure (m/s)', color='teal', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='teal')

    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in missions], rotation=15, fontweight='bold')
    ax1.set_title("Multi-Mission Fine-Tuned LLM Performance Summary", fontweight='bold', fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[*] Saved multi-mission performance summary chart to: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    target_file = None

    if os.path.exists(EVAL_JSON):
        target_file = EVAL_JSON
    elif os.path.exists(LLM_ONLY_JSON):
        target_file = LLM_ONLY_JSON

    if target_file:
        print(f"[*] Loading evaluation results from '{target_file}'...")
        all_missions = load_all_mission_results(target_file)
        
        for m_type, sim_data in all_missions.items():
            plot_single_trajectory_results(
                sim_data,
                save_path=f"trajectory_results_{m_type}.png",
                save_anim=True,
                anim_path=f"trajectory_animation_{m_type}.gif",
            )

        plot_summary_bar_chart(all_missions, save_path="overall_mission_summary.png")
    else:
        print(
            f"[!] Neither '{EVAL_JSON}' nor '{LLM_ONLY_JSON}' were found. "
            "Please run 'LLM_Controller.py' first."
        )