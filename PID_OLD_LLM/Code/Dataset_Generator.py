import json
import numpy as np
from datetime import datetime
from scipy.interpolate import CubicSpline

from CRTBP import MU, DU, TU
from Trajectories import (
    generate_reference_trajectory as generate_base_ref_trajectory,
    generate_halo_orbit,
    generate_dro_orbit,
    simulate_controlled_trajectory,
    verify_physical_bounds,
    verify_and_get_dynamic_metadata,
    MISSION_DESCRIPTIONS,
    LU_TO_KM,
    VU_TO_MS,
    AU_TO_MS2,
)

MISSION_LABELS = {
    "earth_orbit": "Earth Orbit",
    "moon_orbit": "Moon Orbit",
    "transfer": "Earth-to-Moon Transfer",
    "halo_l1": "L1 Halo Orbit",
    "halo_l2": "L2 Halo Orbit",
    "dro": "Distant Retrograde Orbit",
}

def generate_unified_reference_trajectory(mission_type=None):
    if mission_type is None:
        mission_type = np.random.choice([
            "earth_orbit", "moon_orbit", "transfer", 
            "halo_l1", "halo_l2", "dro"
        ])

    if mission_type in ["earth_orbit", "moon_orbit", "transfer"]:
        return generate_base_ref_trajectory(mission_type=mission_type)

    elif mission_type in ["halo_l1", "halo_l2"]:
        lib_point = "L1" if mission_type == "halo_l1" else "L2"
        ax_km = np.random.uniform(10000.0, 35000.0)
        _, X_nodes, P_period = generate_halo_orbit(libration_point=lib_point, Ax_km=ax_km)
        
        t_nodes = np.linspace(0.0, P_period, X_nodes.shape[0])
        spline = CubicSpline(t_nodes, X_nodes, axis=0)
        
        def ref_fn(t):
            t_arr = np.asarray(t, dtype=np.float64)
            t_eval = np.clip(t_arr, 0.0, P_period)
            return spline(t_eval)
            
        class ReferenceSolMock:
            def __init__(self, t_nodes, X_nodes):
                self.t = t_nodes
                self.y = X_nodes.T

        return mission_type, ReferenceSolMock(t_nodes, X_nodes), ref_fn

    elif mission_type == "dro":
        ax_km = np.random.uniform(20000.0, 50000.0)
        _, X_nodes, P_period = generate_dro_orbit(x_amplitude_km=ax_km)
        
        t_nodes = np.linspace(0.0, P_period, X_nodes.shape[0])
        spline = CubicSpline(t_nodes, X_nodes, axis=0)
        
        def ref_fn(t):
            t_arr = np.asarray(t, dtype=np.float64)
            t_eval = np.clip(t_arr, 0.0, P_period)
            return spline(t_eval)

        class ReferenceSolMock:
            def __init__(self, t_nodes, X_nodes):
                self.t = t_nodes
                self.y = X_nodes.T

        return mission_type, ReferenceSolMock(t_nodes, X_nodes), ref_fn


def generate_prompt_text(
    state,
    desired_state,
    final_desired_state,
    progress_percentage,
    mission_type,
    mission_objective,
):
    x, y, z, vx, vy, vz = state
    ref_x, ref_y, ref_z, ref_vx, ref_vy, ref_vz = desired_state

    x_km, y_km, z_km = x * LU_TO_KM, y * LU_TO_KM, z * LU_TO_KM
    ref_x_km, ref_y_km, ref_z_km = ref_x * LU_TO_KM, ref_y * LU_TO_KM, ref_z * LU_TO_KM
    
    vx_ms, vy_ms, vz_ms = vx * VU_TO_MS, vy * VU_TO_MS, vz * VU_TO_MS
    ref_vx_ms, ref_vy_ms, ref_vz_ms = ref_vx * VU_TO_MS, ref_vy * VU_TO_MS, ref_vz * VU_TO_MS

    pos_mag_km = np.linalg.norm([x_km, y_km, z_km])
    vel_mag_ms = np.linalg.norm([vx_ms, vy_ms, vz_ms])

    pos_az = np.degrees(np.arctan2(y, x)) % 360
    pos_el = np.degrees(np.arctan2(z, np.hypot(x, y)))

    vel_az = np.degrees(np.arctan2(vy, vx)) % 360
    vel_el = np.degrees(np.arctan2(vz, np.hypot(vx, vy)))

    pos_err_vec = np.array([ref_x - x, ref_y - y, ref_z - z])
    vel_err_vec = np.array([ref_vx - vx, ref_vy - vy, ref_vz - vz])

    pos_err_km = np.linalg.norm(pos_err_vec * LU_TO_KM)
    vel_err_ms = np.linalg.norm(vel_err_vec * VU_TO_MS)

    err_pos_az = np.degrees(np.arctan2(pos_err_vec[1], pos_err_vec[0])) % 360
    err_pos_el = np.degrees(np.arctan2(pos_err_vec[2], np.hypot(pos_err_vec[0], pos_err_vec[1])))

    err_vel_az = np.degrees(np.arctan2(vel_err_vec[1], vel_err_vec[0])) % 360
    err_vel_el = np.degrees(np.arctan2(vel_err_vec[2], np.hypot(vel_err_vec[0], vel_err_vec[1])))

    final_pos_lu = np.linalg.norm(final_desired_state[:3])
    if final_pos_lu > 4.0 or np.isnan(final_pos_lu):
        final_pos_lu = 1.0
    final_pos_km = final_pos_lu * LU_TO_KM

    mission_label = MISSION_LABELS.get(mission_type, mission_type)
    article = "an" if mission_label[0].lower() in "aeiou" else "a"

    return (
        f"You are an autonomous spacecraft controller operating in Earth-Moon synodic space, "
        f"currently {progress_percentage:.1f}% complete with {article} {mission_label} maneuver ({mission_objective}). "
        f"Your current continuous physical state is as follows: "
        f"Position: X-component is {x:.6f} canonical units ({x_km:,.0f} km from the barycenter), "
        f"Y-component is {y:.6f} canonical units ({y_km:,.0f} km), and "
        f"Z-component is {z:.6f} canonical units ({z_km:,.0f} km). "
        f"This results in a total position magnitude of {pos_mag_km:,.0f} km, "
        f"directed at an azimuth of {pos_az:.2f} degrees and an elevation of {pos_el:.2f} degrees. "
        f"Velocity: X-component is {vx:.6f} canonical units ({vx_ms:.1f} m/s), "
        f"Y-component is {vy:.6f} canonical units ({vy_ms:.1f} m/s), and "
        f"Z-component is {vz:.6f} canonical units ({vz_ms:.1f} m/s). "
        f"This yields a total synodic speed of {vel_mag_ms:.1f} m/s, "
        f"pointing toward an azimuth of {vel_az:.1f} degrees and an elevation of {vel_el:.2f} degrees. "
        f"Your target waypoint requires an X-position of {ref_x:.6f} canonical units ({ref_x_km:,.0f} km), "
        f"a Y-position of {ref_y:.6f} canonical units ({ref_y_km:,.0f} km), "
        f"a Z-position of {ref_z:.6f} canonical units ({ref_z_km:,.0f} km), "
        f"an X-velocity of {ref_vx:.6f} canonical units ({ref_vx_ms:.1f} m/s), "
        f"a Y-velocity of {ref_vy:.6f} canonical units ({ref_vy_ms:.1f} m/s), and "
        f"a Z-velocity of {ref_vz:.6f} canonical units ({ref_vz_ms:.1f} m/s). "
        f"This yields a position tracking error magnitude of {pos_err_km:,.0f} km "
        f"(directed toward an azimuth of {err_pos_az:.1f} degrees and an elevation of {err_pos_el:.1f} degrees) "
        f"and a velocity tracking error magnitude of {vel_err_ms:.2f} m/s "
        f"(directed toward an azimuth of {err_vel_az:.1f} degrees and an elevation of {err_vel_el:.1f} degrees). "
        f"Your final mission destination magnitude is {final_pos_km:,.0f} km ({final_pos_lu:.6f} canonical units). "
        f"Determine the precise thrust acceleration vector needed to correct this tracking offset and maintain nominal trajectory."
    )


def generate_response_text(u):
    ux, uy, uz = u
    ux_ms2, uy_ms2, uz_ms2 = ux * AU_TO_MS2, uy * AU_TO_MS2, uz * AU_TO_MS2
    acc_mag_ms2 = np.linalg.norm([ux_ms2, uy_ms2, uz_ms2])

    acc_az = np.degrees(np.arctan2(uy, ux)) % 360
    acc_el = np.degrees(np.arctan2(uz, np.hypot(ux, uy)))

    return (
        f"Command a continuous thrust acceleration vector of "
        f"{ux_ms2:+.4e} m/s2 ({ux:.6f} canonical units) along the X-axis, "
        f"{uy_ms2:+.4e} m/s2 ({uy:.6f} canonical units) along the Y-axis, and "
        f"{uz_ms2:+.4e} m/s2 ({uz:.6f} canonical units) along the Z-axis. "
        f"This applies a total physical thrust acceleration magnitude of {acc_mag_ms2:.4e} m/s2, "
        f"oriented toward an azimuth of {acc_az:.1f} degrees and an elevation of {acc_el:.1f} degrees."
    )


def process_and_save_dataset(num_trajectories=100, step_stride=2):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"crtbp_llm_dataset_{timestamp}.jsonl"
    
    print(f"Starting dataset generation. Output: '{output_filename}'")
    total_samples = 0

    with open(output_filename, "w", encoding="utf-8") as f:
        for traj_idx in range(num_trajectories):
            mission_type, sol_ref, ref_fn = generate_unified_reference_trajectory()
            mission_objective = MISSION_DESCRIPTIONS.get(mission_type, "Stationkeeping maneuver.")
            
            t_span = (sol_ref.t[0], sol_ref.t[-1])
            final_desired_state = ref_fn(t_span[1])
            total_duration = t_span[1] - t_span[0]

            pos_noise = 0.00005 if "moon" in mission_type or "halo" in mission_type else 0.0003
            vel_noise = 0.0001 if "moon" in mission_type or "halo" in mission_type else 0.0003

            x0 = ref_fn(t_span[0]) + np.array([
                np.random.uniform(-pos_noise, pos_noise),
                np.random.uniform(-pos_noise, pos_noise),
                np.random.uniform(-pos_noise / 2, pos_noise / 2),
                np.random.uniform(-vel_noise, vel_noise),
                np.random.uniform(-vel_noise, vel_noise),
                np.random.uniform(-vel_noise / 2, vel_noise / 2),
            ])

            sol_ctrl, u_history, safety_flags = simulate_controlled_trajectory(
                mu_val=MU,
                initial_state=x0,
                reference_orbit_fn=ref_fn,
                t_span=t_span,
                u_max_m_s2=0.01,
                enable_perturbations=True,
                print_stride=1000
            )

            traj_samples = 0
            for i in range(0, sol_ctrl.t.size, step_stride):
                if not safety_flags[i]:
                    continue

                t_curr = sol_ctrl.t[i]
                state_curr = sol_ctrl.y[:6, i]
                u_curr = u_history[i]
                desired_state_curr = np.squeeze(ref_fn(t_curr))

                dyn_mission_type, dyn_mission_obj = verify_and_get_dynamic_metadata(state_curr)
                progress_percentage = (t_curr / total_duration) * 100.0 if total_duration > 0 else 100.0

                prompt_text = generate_prompt_text(
                    state_curr,
                    desired_state_curr,
                    final_desired_state,
                    progress_percentage,
                    dyn_mission_type,
                    dyn_mission_obj,
                )
                response_text = generate_response_text(u_curr)

                sample_entry = {
                    "messages": [
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": response_text}
                    ],
                    "mission_type": dyn_mission_type,
                    "progress_pct": progress_percentage,
                }

                f.write(json.dumps(sample_entry, ensure_ascii=False) + "\n")
                traj_samples += 1

            total_samples += traj_samples

    print(f"Saved {total_samples} samples into '{output_filename}'.")


if __name__ == "__main__":
    process_and_save_dataset(num_trajectories=600, step_stride=2)