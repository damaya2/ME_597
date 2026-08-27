import json
import numpy as np
from datetime import datetime
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from CRTBP import MU, DU, TU, crtbp_eom
from Trajectories import (
    LU_TO_KM,
    VU_TO_MS,
    AU_TO_MS2,
    generate_reference_trajectory,
    generate_halo_orbit,
    generate_dro_orbit,
    simulate_controlled_trajectory,
)

MISSION_LABELS = {
    "earth_orbit": "Earth Orbit",
    "moon_orbit": "Moon Orbit",
    "transfer": "Earth-to-Moon Transfer",
    "halo_l1": "L1 Halo Orbit",
    "halo_l2": "L2 Halo Orbit",
    "dro": "Distant Retrograde Orbit",
}

MISSION_DESCRIPTIONS = {
    "earth_orbit": "Earth-centered parking orbit stationkeeping and maintenance.",
    "moon_orbit": "Moon-centered polar orbit stationkeeping and tracking.",
    "transfer": "Trans-lunar injection trajectory tracking and midpoint alignment.",
    "halo_l1": "L1 Halo orbit stationkeeping in the Earth-Moon system.",
    "halo_l2": "L2 Halo orbit stationkeeping for far-side communications.",
    "dro": "Distant Retrograde Orbit (DRO) long-term stability maintenance.",
}

def verify_and_get_dynamic_metadata(state, fallback_mission_type=None):
    """
    Infers mission domain based on state vector coordinates in canonical units,
    falling back to the initial trajectory classification if ambiguous.
    """
    x, y, z = state[:3]
    r_earth = np.sqrt((x + MU)**2 + y**2 + z**2)
    r_moon = np.sqrt((x - (1.0 - MU))**2 + y**2 + z**2)
    
    if r_earth < 0.15:
        return "earth_orbit", MISSION_DESCRIPTIONS["earth_orbit"]
    elif r_moon < 0.10:
        return "moon_orbit", MISSION_DESCRIPTIONS["moon_orbit"]
    elif 0.75 <= x <= 0.90 and abs(y) < 0.2:
        return "halo_l1", MISSION_DESCRIPTIONS["halo_l1"]
    elif 1.10 <= x <= 1.25 and abs(y) < 0.2:
        return "halo_l2", MISSION_DESCRIPTIONS["halo_l2"]
    elif abs(x - 0.85) < 0.35 and abs(y) > 0.15:
        return "dro", MISSION_DESCRIPTIONS["dro"]
    elif fallback_mission_type in MISSION_DESCRIPTIONS:
        return fallback_mission_type, MISSION_DESCRIPTIONS[fallback_mission_type]
    else:
        return "transfer", MISSION_DESCRIPTIONS["transfer"]

def generate_unified_reference_trajectory(mission_type=None):
    if mission_type is None:
        mission_types = ["earth_orbit", "moon_orbit", "transfer", "halo_l1", "halo_l2", "dro"]
        probabilities = [0.13, 0.13, 0.35, 0.13, 0.13, 0.13]
        mission_type = np.random.choice(mission_types, p=probabilities)

    # 1. Handle Halo and DRO orbits using their specific generators in Trajectories.py
    if mission_type in ["halo_l1", "halo_l2", "dro"]:
        if mission_type == "halo_l1":
            _, nodes, P = generate_halo_orbit(libration_point="L1")
        elif mission_type == "halo_l2":
            _, nodes, P = generate_halo_orbit(libration_point="L2")
        elif mission_type == "dro":
            _, nodes, P = generate_dro_orbit()

        # Propagate dense continuous trajectory from the corrected node seed
        x0_corrected = nodes[0]
        t_span = (0.0, 2.0 * P)  # Simulate 2 full periods
        
        sol = solve_ivp(
            crtbp_eom,
            t_span,
            x0_corrected,
            args=(MU,),
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
            max_step=0.005,
        )

        spline = CubicSpline(sol.t, sol.y.T, axis=0)
        t_min, t_max = sol.t[0], sol.t[-1]

        def ref_fn(t):
            t_arr = np.asarray(t, dtype=np.float64)
            t_eval = np.clip(t_arr, t_min, t_max)
            return spline(t_eval)

        return mission_type, sol, ref_fn

    # 2. Handle standard orbits (earth_orbit, moon_orbit, transfer) via generate_reference_trajectory
    return generate_reference_trajectory(mission_type=mission_type)

def generate_prompt_text(
    state,
    desired_state,
    final_desired_state,
    progress_percentage,
    mission_type,
    mission_objective,
):
    x, y, z, vx, vy, vz = state[:6]
    ref_x, ref_y, ref_z, ref_vx, ref_vy, ref_vz = desired_state[:6]

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
    ux, uy, uz = u[:3]
    ux_ms2, uy_ms2, uz_ms2 = ux * AU_TO_MS2, uy * AU_TO_MS2, uz * AU_TO_MS2
    acc_mag_ms2 = np.linalg.norm([ux_ms2, uy_ms2, uz_ms2])

    acc_az = np.degrees(np.arctan2(uy, ux)) % 360
    hypot_xy = np.hypot(ux, uy)
    acc_el = np.degrees(np.arctan2(uz, hypot_xy)) if hypot_xy > 1e-12 or abs(uz) > 1e-12 else 0.0

    return (
        f"Command a continuous thrust acceleration vector of "
        f"{ux_ms2:+.4e} m/s2 ({ux:.6f} canonical units) along the X-axis, "
        f"{uy_ms2:+.4e} m/s2 ({uy:.6f} canonical units) along the Y-axis, and "
        f"{uz_ms2:+.4e} m/s2 ({uz:.6f} canonical units) along the Z-axis. "
        f"This applies a total physical thrust acceleration magnitude of {acc_mag_ms2:.4e} m/s2, "
        f"oriented toward an azimuth of {acc_az:.1f} degrees and an elevation of {acc_el:.1f} degrees."
    )

def process_and_save_dataset(num_trajectories=600, step_stride=2, min_kick_step=50):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"crtbp_llm_dataset_{timestamp}.jsonl"
    
    print(f"Starting dataset generation with mid-trajectory perturbations. Output: '{output_filename}'")
    total_samples = 0

    with open(output_filename, "w", encoding="utf-8") as f:
        for traj_idx in range(num_trajectories):
            mission_type, sol_ref, ref_fn = generate_unified_reference_trajectory()
            mission_objective = MISSION_DESCRIPTIONS.get(mission_type, "Stationkeeping maneuver.")
            
            t_span = (sol_ref.t[0], sol_ref.t[-1])
            final_desired_state = np.ravel(ref_fn(t_span[1]))[:6]
            total_duration = t_span[1] - t_span[0]

            # Ensure the trajectory has enough timesteps to allow a kick >= min_kick_step
            num_ref_steps = len(sol_ref.t)
            if num_ref_steps <= min_kick_step + 50:
                kick_idx = 0
            else:
                kick_idx = np.random.randint(min_kick_step, num_ref_steps - 50)

            t_kick = sol_ref.t[kick_idx]
            nominal_state_at_kick = np.ravel(ref_fn(t_kick))[:6]

            # Generate error values
            init_pos_err = np.random.uniform(1.5, 3.0) / LU_TO_KM
            init_vel_err = np.random.uniform(0.1, 0.3) / VU_TO_MS

            # Inject the error impulse at t_kick
            perturbed_state_at_kick = nominal_state_at_kick + np.array([
                np.random.uniform(-init_pos_err, init_pos_err),
                np.random.uniform(-init_pos_err, init_pos_err),
                np.random.uniform(-init_pos_err / 2, init_pos_err / 2),
                np.random.uniform(-init_vel_err, init_vel_err),
                np.random.uniform(-init_vel_err, init_vel_err),
                np.random.uniform(-init_vel_err / 2, init_vel_err / 2),
            ])

            # Simulate controlled recovery starting from t_kick to t_end
            t_span_ctrl = (t_kick, t_span[1])
            sol_ctrl, u_history, safety_flags = simulate_controlled_trajectory(
                mu_val=MU,
                initial_state=perturbed_state_at_kick,
                reference_orbit_fn=ref_fn,
                t_span=t_span_ctrl,
                u_max_m_s2=0.01,
                enable_perturbations=True,
                print_stride=1000
            )

            traj_samples = 0
            max_steps = min(sol_ctrl.t.size, len(u_history), len(safety_flags))

            for i in range(0, max_steps, step_stride):
                if not safety_flags[i]:
                    continue

                t_curr = sol_ctrl.t[i]
                state_curr = sol_ctrl.y[:6, i]
                u_curr = u_history[i]
                desired_state_curr = np.ravel(ref_fn(t_curr))[:6]

                dyn_mission_type, dyn_mission_obj = verify_and_get_dynamic_metadata(
                    state_curr, fallback_mission_type=mission_type
                )
                
                elapsed = t_curr - t_span[0]
                progress_percentage = (elapsed / total_duration) * 100.0 if total_duration > 0 else 100.0
                progress_percentage = np.clip(progress_percentage, 0.0, 100.0)

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
                    "trajectory_id": int(traj_idx),
                    "mission_type": str(dyn_mission_type),
                    "progress_pct": float(progress_percentage),
                    "kick_step": int(kick_idx),
                    "debug_pos_km": [float(state_curr[0] * LU_TO_KM), float(state_curr[1] * LU_TO_KM), float(state_curr[2] * LU_TO_KM)],
                    "debug_target_km": [float(desired_state_curr[0] * LU_TO_KM), float(desired_state_curr[1] * LU_TO_KM), float(desired_state_curr[2] * LU_TO_KM)],
                }

                f.write(json.dumps(sample_entry, ensure_ascii=False) + "\n")
                traj_samples += 1

            total_samples += traj_samples

    print(f"Saved {total_samples} samples into '{output_filename}'.")

if __name__ == "__main__":
    process_and_save_dataset(num_trajectories=600, step_stride=2)