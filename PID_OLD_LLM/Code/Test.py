import argparse
import os
import sys
import time
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless test execution
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from CRTBP import MU
from Trajectories import (
    generate_reference_trajectory,
    generate_halo_orbit,
    generate_dro_orbit,
    simulate_controlled_trajectory,
    LU_TO_KM,
    VU_TO_MS,
    TU_TO_SEC,
    AU_TO_MS2,
)

# Robust trapezoidal integration fallback for NumPy compatibility
trapz_fn = getattr(np, "trapezoid", np.trapz)

# Constants matching plotting specifications
TU_TO_HOURS = TU_TO_SEC / 3600.0  # Canonical time to hours (~104.35 hrs/TU)
EARTH_POS_KM = np.array([-MU * LU_TO_KM, 0.0, 0.0])
MOON_POS_KM = np.array([(1.0 - MU) * LU_TO_KM, 0.0, 0.0])

R_EARTH_KM = 6371.0
R_MOON_KM = 1737.4

OUTPUT_DIR = "test_output_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Helper & Plotting Functions
# =============================================================================

def apply_plot_style():
    """Applies clean plotting parameters with version-safe style selection."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")


def set_3d_equal_aspect(ax, x_data, y_data, z_data, padding_km=2000.0):
    """Enforces equal aspect ratios across all 3 spatial axes in Matplotlib 3D."""
    max_range = (
        np.array([
            np.max(x_data) - np.min(x_data),
            np.max(y_data) - np.min(y_data),
            np.max(z_data) - np.min(z_data),
        ]).max() / 2.0
    ) + padding_km

    mid_x = (np.max(x_data) + np.min(x_data)) * 0.5
    mid_y = (np.max(y_data) + np.min(y_data)) * 0.5
    mid_z = (np.max(z_data) + np.min(z_data)) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1, 1, 1])


def plot_single_trajectory_results(
    sim_results: dict,
    save_path: str = "trajectory_results.png",
    max_plot_points: int = 5000,
    u_max_m_s2: float = 0.01,
):
    """Plots 4-panel comprehensive analysis aligned with static frame outputs."""
    apply_plot_style()

    time_raw = sim_results["time"]
    ref_pos_raw = sim_results["state_ref"][:, :3] * LU_TO_KM
    sim_pos_raw = sim_results["state_sim"][:, :3] * LU_TO_KM
    pos_err_raw = sim_results["pos_err_km"]
    vel_err_raw = sim_results["vel_err_ms"]
    control_raw = sim_results["control"] * AU_TO_MS2

    total_pts = len(time_raw)
    if total_pts > max_plot_points:
        stride = int(np.ceil(total_pts / max_plot_points))
        idx_sub = np.arange(0, total_pts, stride)
        if idx_sub[-1] != total_pts - 1:
            idx_sub = np.append(idx_sub, total_pts - 1)
    else:
        idx_sub = np.arange(total_pts)

    time_hrs = (time_raw[idx_sub] - time_raw[0]) * TU_TO_HOURS
    ref_pos_km = ref_pos_raw[idx_sub]
    sim_pos_km = sim_pos_raw[idx_sub]
    pos_err_km = pos_err_raw[idx_sub]
    vel_err_ms = vel_err_raw[idx_sub]

    if len(control_raw) == total_pts:
        control_ms2 = control_raw[idx_sub]
        control_time_hrs = time_hrs
    else:
        ctrl_sub = idx_sub[idx_sub < len(control_raw)]
        control_ms2 = control_raw[ctrl_sub]
        control_time_hrs = time_hrs[:len(ctrl_sub)]

    fig = plt.figure(figsize=(16, 12), dpi=300)
    traj_id = sim_results.get("traj_id", 1)

    # PANEL 1: 3D Trajectory Plot
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(ref_pos_km[:, 0], ref_pos_km[:, 1], ref_pos_km[:, 2], "k--", linewidth=1.8, label="Nominal Reference", alpha=0.8)
    ax1.plot(sim_pos_km[:, 0], sim_pos_km[:, 1], sim_pos_km[:, 2], "r-", linewidth=2.0, label=f"Controlled #{traj_id}")
    ax1.scatter(*sim_pos_km[0], color="green", s=60, marker="o", label="Start", zorder=5)

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
    ax1.set_title(f"3D Synodic Trajectory (Run #{traj_id})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X (km)", labelpad=8)
    ax1.set_ylabel("Y (km)", labelpad=8)
    ax1.set_zlabel("Z (km)", labelpad=8)
    ax1.legend(loc="upper right", frameon=True)

    # PANEL 2: XY Plane Projection
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(ref_pos_km[:, 0], ref_pos_km[:, 1], "k--", linewidth=1.5, label="Reference")
    ax2.plot(sim_pos_km[:, 0], sim_pos_km[:, 1], "r-", linewidth=1.8, label="Controlled Trajectory")

    earth_circle = plt.Circle((EARTH_POS_KM[0], EARTH_POS_KM[1]), R_EARTH_KM * 2, color="blue", alpha=0.3, label="Earth Region")
    moon_circle = plt.Circle((MOON_POS_KM[0], MOON_POS_KM[1]), R_MOON_KM * 3, color="gray", alpha=0.4, label="Moon Region")
    ax2.add_patch(earth_circle)
    ax2.add_patch(moon_circle)
    ax2.scatter(sim_pos_km[0, 0], sim_pos_km[0, 1], color="green", s=50, zorder=5)

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
    line1 = ax3.plot(time_hrs, pos_err_km, color=color1, linewidth=2.0, label="Pos Error (km)")
    ax3.tick_params(axis="y", labelcolor=color1)

    ax3_vel = ax3.twinx()
    color2 = "tab:blue"
    ax3_vel.set_ylabel("Velocity Error (m/s)", color=color2, fontweight="bold")
    line2 = ax3_vel.plot(time_hrs, vel_err_ms, color=color2, linewidth=2.0, linestyle="--", label="Vel Error (m/s)")
    ax3_vel.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper right", frameon=True)
    ax3.set_title("Tracking Error Response", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # PANEL 4: Control Acceleration (Scales dynamically with u_max)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.step(control_time_hrs, control_ms2[:, 0], where="post", label=r"$u_x$", linewidth=1.5)
    ax4.step(control_time_hrs, control_ms2[:, 1], where="post", label=r"$u_y$", linewidth=1.5)
    ax4.step(control_time_hrs, control_ms2[:, 2], where="post", label=r"$u_z$", linewidth=1.5)

    ax4.axhline(u_max_m_s2, color="black", linestyle=":", alpha=0.7, label=f"Max Limit ({u_max_m_s2} m/s²)")
    ax4.axhline(-u_max_m_s2, color="black", linestyle=":", alpha=0.7)

    ax4.set_yscale("symlog", linthresh=1e-7)
    ax4.set_ylim([-u_max_m_s2 * 1.5, u_max_m_s2 * 1.5])
    ax4.set_title("Control Acceleration Commands", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Time (Hours)", fontweight="bold")
    ax4.set_ylabel("Acceleration (m/s²)", fontweight="bold")
    ax4.grid(True, which="both", linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[*] Saved single trajectory plot to: {save_path}", flush=True)
    plt.close(fig)


def plot_multi_trajectory_comparison(
    trajectories: list,
    save_path: str = "multi_trajectory_summary.png",
    max_plot_points: int = 5000,
    u_max_m_s2: float = 0.01,
):
    """Generates aggregated 4-panel comparison plot across multiple mission profiles."""
    print(f"[*] Generating aggregated comparison plot for {len(trajectories)} trajectories...", flush=True)
    apply_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    # 1. Position Errors
    ax_pos = axes[0, 0]
    for idx, res in enumerate(trajectories):
        t_raw = res["time"]
        p_raw = res["pos_err_km"]

        if len(t_raw) > max_plot_points:
            sub = np.linspace(0, len(t_raw) - 1, max_plot_points, dtype=int)
            t_sub, p_sub = t_raw[sub], p_raw[sub]
        else:
            t_sub, p_sub = t_raw, p_raw

        t_hrs = (t_sub - t_sub[0]) * TU_TO_HOURS
        ax_pos.plot(t_hrs, p_sub, color=colors[idx % len(colors)], linewidth=2.0, label=f"Traj #{res.get('traj_id', idx+1)} ({res['mission_type']})")

    ax_pos.set_title("Position Error Convergence (km)", fontweight="bold")
    ax_pos.set_xlabel("Time (Hours)")
    ax_pos.set_ylabel("Position Error (km)")
    ax_pos.grid(True, linestyle=":", alpha=0.6)
    ax_pos.legend(loc="upper right")

    # 2. Velocity Errors
    ax_vel = axes[0, 1]
    for idx, res in enumerate(trajectories):
        t_raw = res["time"]
        v_raw = res["vel_err_ms"]

        if len(t_raw) > max_plot_points:
            sub = np.linspace(0, len(t_raw) - 1, max_plot_points, dtype=int)
            t_sub, v_sub = t_raw[sub], v_raw[sub]
        else:
            t_sub, v_sub = t_raw, v_raw

        t_hrs = (t_sub - t_sub[0]) * TU_TO_HOURS
        ax_vel.plot(t_hrs, v_sub, color=colors[idx % len(colors)], linewidth=2.0, label=f"Traj #{res.get('traj_id', idx+1)} ({res['mission_type']})")

    ax_vel.set_title("Velocity Error Convergence (m/s)", fontweight="bold")
    ax_vel.set_xlabel("Time (Hours)")
    ax_vel.set_ylabel("Velocity Error (m/s)")
    ax_vel.grid(True, linestyle=":", alpha=0.6)
    ax_vel.legend(loc="upper right")

    # 3. Control Magnitude
    ax_u = axes[1, 0]
    for idx, res in enumerate(trajectories):
        t_raw = res["time"]
        u_raw = res["control"]

        if len(t_raw) > max_plot_points:
            sub = np.linspace(0, len(t_raw) - 1, max_plot_points, dtype=int)
            t_sub, u_sub = t_raw[sub], u_raw[sub]
        else:
            t_sub, u_sub = t_raw, u_raw

        t_hrs = (t_sub - t_sub[0]) * TU_TO_HOURS
        u_mags = np.linalg.norm(u_sub, axis=1) * AU_TO_MS2
        ctrl_t_hrs = t_hrs[:-1] if len(u_mags) < len(t_hrs) else t_hrs
        ax_u.plot(ctrl_t_hrs, u_mags, color=colors[idx % len(colors)], linewidth=1.8, label=f"Traj #{res.get('traj_id', idx+1)} ({res['mission_type']})")

    ax_u.axhline(u_max_m_s2, color="black", linestyle=":", alpha=0.7, label=f"Max Accel ({u_max_m_s2} m/s²)")
    ax_u.set_yscale("symlog", linthresh=1e-7)
    ax_u.set_ylim([1e-8, u_max_m_s2 * 1.5])
    ax_u.set_title("Control Acceleration Magnitude", fontweight="bold")
    ax_u.set_xlabel("Time (Hours)")
    ax_u.set_ylabel(r"||$u$|| (m/s²)")
    ax_u.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_u.legend(loc="upper right")

    # 4. Cumulative Delta-V Bar Chart
    ax_dv = axes[1, 1]
    traj_labels = [f"Traj #{r.get('traj_id', i+1)}\n({r['mission_type']})" for i, r in enumerate(trajectories)]
    dvs = [r["total_delta_v_ms"] for r in trajectories]

    bars = ax_dv.bar(traj_labels, dvs, color=colors[: len(trajectories)], alpha=0.85)
    ax_dv.set_title("Total Delta-V Expenditure (m/s)", fontweight="bold")
    ax_dv.set_ylabel("Total ΔV (m/s)")
    ax_dv.grid(True, linestyle=":", alpha=0.6, axis="y")

    for bar in bars:
        yval = bar.get_height()
        ax_dv.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.2f} m/s", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[*] Saved multi-trajectory summary plot to: {save_path}", flush=True)
    plt.close(fig)


# =============================================================================
# Test Runner Workflow
# =============================================================================

def run_test_for_mission(
    mission_type,
    traj_id=1,
    print_stride=100,
    max_output_points=5000,
    pert_delay_hrs=500.0,
    u_max_m_s2=0.01,
):
    """Executes a two-phase test run across traditional, Halo, or DRO orbits."""
    print(f"\n=======================================================", flush=True)
    print(f"[+] Starting Test Run #{traj_id} for mission profile: '{mission_type}'", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Generate reference trajectory from advanced or classic families
    if mission_type in ["earth_orbit", "moon_orbit", "transfer"]:
        m_type, sol_ref, ref_fn = generate_reference_trajectory(mission_type)
        t_start = sol_ref.t[0]
        t_end = sol_ref.t[-1] - 1e-4
    elif mission_type == "halo_l1":
        m_type, nodes, P_period = generate_halo_orbit(libration_point="L1", Ax_km=15000.0)
        t_nodes = np.linspace(0, P_period, len(nodes))
        spline = CubicSpline(t_nodes, nodes, axis=0)
        ref_fn = lambda t: spline(np.clip(t, 0, P_period))
        t_start, t_end = 0.0, P_period - 1e-4
    elif mission_type == "halo_l2":
        m_type, nodes, P_period = generate_halo_orbit(libration_point="L2", Ax_km=12000.0)
        t_nodes = np.linspace(0, P_period, len(nodes))
        spline = CubicSpline(t_nodes, nodes, axis=0)
        ref_fn = lambda t: spline(np.clip(t, 0, P_period))
        t_start, t_end = 0.0, P_period - 1e-4
    elif mission_type == "dro":
        m_type, nodes, P_period = generate_dro_orbit(x_amplitude_km=40000.0)
        t_nodes = np.linspace(0, P_period, len(nodes))
        spline = CubicSpline(t_nodes, nodes, axis=0)
        ref_fn = lambda t: spline(np.clip(t, 0, P_period))
        t_start, t_end = 0.0, P_period - 1e-4
    else:
        raise ValueError(f"Unsupported mission type: {mission_type}")

    total_hrs = (t_end - t_start) * TU_TO_HOURS
    actual_pert_delay_hrs = min(pert_delay_hrs, total_hrs * 0.4)
    t_pert = t_start + (actual_pert_delay_hrs / TU_TO_HOURS)

    # Phase 1: Nominal Flight
    print(f"[*] Phase 1: Running nominal trajectory tracking from t=0.0 to t={actual_pert_delay_hrs:.1f} hrs...", flush=True)
    x0_sim_phase1 = np.squeeze(ref_fn(t_start)).copy()

    sol_sim_p1, u_hist_p1, safety_p1 = simulate_controlled_trajectory(
        mu_val=MU,
        initial_state=x0_sim_phase1,
        reference_orbit_fn=ref_fn,
        t_span=(t_start, t_pert),
        u_max_m_s2=u_max_m_s2,
        enable_perturbations=True,
        print_stride=print_stride,
    )

    # Inject perturbation kick while retaining integral controller state xc
    print(f"[*] Injecting 2.0 km position & 0.01 m/s velocity disturbance kick at t = {actual_pert_delay_hrs:.1f} hrs...", flush=True)
    pos_pert_canon = 2.0 / LU_TO_KM
    vel_pert_canon = 0.01 / VU_TO_MS

    disturbance_kick = np.array([
        pos_pert_canon, -pos_pert_canon, 0.5 * pos_pert_canon,
        vel_pert_canon, -vel_pert_canon, 0.5 * vel_pert_canon
    ])

    # Preserve continuous state and controller integrator accumulator state (xc)
    last_state_phase1 = sol_sim_p1.y[:6, -1].copy() + disturbance_kick

    # Phase 2: Recovery Flight
    print(f"[*] Phase 2: Running disturbance recovery tracking...", flush=True)
    sol_sim_p2, u_hist_p2, safety_p2 = simulate_controlled_trajectory(
        mu_val=MU,
        initial_state=last_state_phase1,
        reference_orbit_fn=ref_fn,
        t_span=(t_pert, t_end),
        u_max_m_s2=u_max_m_s2,
        enable_perturbations=True,
        print_stride=print_stride,
    )

    # Combine phases safely, trimming overlapping boundary point at t_pert
    t_array = np.concatenate([sol_sim_p1.t[:-1], sol_sim_p2.t])
    state_sim = np.hstack([sol_sim_p1.y[:6, :-1], sol_sim_p2.y[:6, :]]).T
    u_history = np.vstack([u_hist_p1[:-1], u_hist_p2])
    safety_flags = np.concatenate([safety_p1[:-1], safety_p2])

    total_steps = len(t_array)

    # Post-process reference states and errors
    state_ref = np.zeros_like(state_sim)
    for idx, t_i in enumerate(t_array):
        state_ref[idx, :] = np.squeeze(ref_fn(t_i))

    pos_err_vec = state_sim[:, :3] - state_ref[:, :3]
    vel_err_vec = state_sim[:, 3:] - state_ref[:, 3:]

    pos_err_km = np.linalg.norm(pos_err_vec, axis=1) * LU_TO_KM
    vel_err_ms = np.linalg.norm(vel_err_vec, axis=1) * VU_TO_MS

    u_mags_canon = np.linalg.norm(u_history, axis=1)
    total_delta_v_ms = float(trapz_fn(u_mags_canon * AU_TO_MS2, t_array * TU_TO_SEC))

    sim_result = {
        "traj_id": traj_id,
        "mission_type": mission_type,
        "time": t_array,
        "state_sim": state_sim,
        "state_ref": state_ref,
        "pos_err_km": pos_err_km,
        "vel_err_ms": vel_err_ms,
        "control": u_history,
        "total_delta_v_ms": total_delta_v_ms,
        "disturbances": [actual_pert_delay_hrs],
    }

    valid_steps = int(np.sum(safety_flags))
    print(f"\n[*] Integration Complete: {valid_steps}/{total_steps} physical bounds checks passed.", flush=True)
    print(f"[*] Total ΔV Expended: {total_delta_v_ms:.2f} m/s", flush=True)

    assert valid_steps > 0, "All bounds checks failed."
    assert total_delta_v_ms > 0, "Delta-V calculation failed or evaluated to zero."
    assert sim_result["state_sim"].shape == sim_result["state_ref"].shape, "Shape mismatch between sim and ref states."

    return sim_result


def main():
    parser = argparse.ArgumentParser(description="Execute CR3BP trajectory simulation test suite.")
    parser.add_argument("--pert-delay", type=float, default=500.0, help="Delay in hours before applying perturbation kick.")
    parser.add_argument("--u-max", type=float, default=0.01, help="Maximum thruster acceleration bound in m/s^2.")
    args = parser.parse_args()

    # Suite including standard Earth/Moon transfers, L1/L2 Halos, and DROs
    mission_types = ["earth_orbit", "moon_orbit", "transfer", "halo_l1", "dro"]
    all_results = []

    for idx, m_type in enumerate(mission_types, start=1):
        res = run_test_for_mission(
            m_type,
            traj_id=idx,
            print_stride=50,
            max_output_points=5000,
            pert_delay_hrs=args.pert_delay,
            u_max_m_s2=args.u_max,
        )
        all_results.append(res)

        img_path = os.path.join(OUTPUT_DIR, f"trajectory_results_run_{idx}.png")

        plot_single_trajectory_results(
            res,
            save_path=img_path,
            max_plot_points=5000,
            u_max_m_s2=args.u_max,
        )

    summary_path = os.path.join(OUTPUT_DIR, "multi_trajectory_summary.png")
    plot_multi_trajectory_comparison(
        all_results,
        save_path=summary_path,
        max_plot_points=5000,
        u_max_m_s2=args.u_max,
    )

    npy_path = "sim_results_multi.npy"
    np.save(npy_path, np.array(all_results, dtype=object))
    print(f"\n[+] Saved binary results array to '{npy_path}' successfully.", flush=True)


if __name__ == "__main__":
    main()