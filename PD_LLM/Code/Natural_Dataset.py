import json
from datetime import datetime
import numpy as np
from scipy.integrate import solve_ivp
from CRTBP import AU, DU, VU, canonical_to_physical, crtbp_controlled_eom, mu
from verify import verify_physical_bounds, verify_and_get_dynamic_metadata
from Desired_State import (
    SAFETY_EVENTS,
    generate_reference_trajectory,
    get_mission_description,
)

# Standardized mapping dictionary for mission key formats
MISSION_LABELS = {
    "earth_orbit": "Earth Orbit",
    "moon_orbit": "Moon Orbit",
    "transfer": "Earth-to-Moon Transfer",
    "Earth Orbit": "Earth Orbit",
    "Moon Orbit": "Moon Orbit",
    "Cislunar Transfer": "Earth-to-Moon Transfer",
}


# =============================
# CONTROL LAW
# =============================
def control_law(state, desired_state):
    pos_err = state[0:3] - desired_state[0:3]
    vel_err = state[3:6] - desired_state[3:6]
    Kp, Kv = 1.0, 2.0
    return -Kp * pos_err - Kv * vel_err


# =========================================================
# PROMPT GENERATOR (Continuous Natural Language Prose)
# =========================================================
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

    # Convert canonical state and waypoints using CRTBP conversion helper
    (x_km, y_km, z_km), (vx_ms, vy_ms, vz_ms) = canonical_to_physical(state)
    (ref_x_km, ref_y_km, ref_z_km), (ref_vx_ms, ref_vy_ms, ref_vz_ms) = (
        canonical_to_physical(desired_state)
    )

    pos_mag_km = np.linalg.norm([x_km, y_km, z_km])
    vel_mag_ms = np.linalg.norm([vx_ms, vy_ms, vz_ms])

    # Direction Angles
    pos_az = np.degrees(np.arctan2(y, x)) % 360
    pos_el = np.degrees(np.arctan2(z, np.hypot(x, y)))

    vel_az = np.degrees(np.arctan2(vy, vx)) % 360
    vel_el = np.degrees(np.arctan2(vz, np.hypot(vx, vy)))

    # Tracking Errors
    pos_err_vec = np.array([ref_x - x, ref_y - y, ref_z - z])
    vel_err_vec = np.array([ref_vx - vx, ref_vy - vy, ref_vz - vz])

    pos_err_km = np.linalg.norm(pos_err_vec * DU)
    vel_err_ms = np.linalg.norm(vel_err_vec * VU)

    err_pos_az = np.degrees(np.arctan2(pos_err_vec[1], pos_err_vec[0])) % 360
    err_pos_el = np.degrees(
        np.arctan2(pos_err_vec[2], np.hypot(pos_err_vec[0], pos_err_vec[1]))
    )

    err_vel_az = np.degrees(np.arctan2(vel_err_vec[1], vel_err_vec[0])) % 360
    err_vel_el = np.degrees(
        np.arctan2(vel_err_vec[2], np.hypot(vel_err_vec[0], vel_err_vec[1]))
    )

    # Final Destination Sanitization (caps unbounded LU escapes at 4.0 LU / ~1.53M km)
    final_pos_lu = np.linalg.norm(final_desired_state[:3])
    if final_pos_lu > 4.0 or np.isnan(final_pos_lu):
        final_pos_lu = 1.0  # Fallback to Earth-Moon distance (~384,400 km)
    final_pos_km = final_pos_lu * DU

    # Robust label lookup
    mission_label = MISSION_LABELS.get(mission_type, mission_type)
    article = "an" if mission_label[0].lower() in "aeiou" else "a"

    prompt = (
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

    return prompt


# =========================================================
# RESPONSE GENERATOR (Continuous Thrust Acceleration)
# =========================================================
def generate_response_text(u):
    ux, uy, uz = u

    # Acceleration Metric Conversions using imported AU
    ux_ms2, uy_ms2, uz_ms2 = ux * AU, uy * AU, uz * AU
    acc_mag_ms2 = np.linalg.norm([ux_ms2, uy_ms2, uz_ms2])

    # Thrust Vector Angles
    acc_az = np.degrees(np.arctan2(uy, ux)) % 360
    acc_el = np.degrees(np.arctan2(uz, np.hypot(ux, uy)))

    response = (
        f"Solution: Command a continuous thrust acceleration vector of "
        f"{ux_ms2:+.4e} m/s^2 ({ux:.6f} canonical units) along the X-axis, "
        f"{uy_ms2:+.4e} m/s^2 ({uy:.6f} canonical units) along the Y-axis, and "
        f"{uz_ms2:+.4e} m/s^2 ({uz:.6f} canonical units) along the Z-axis. "
        f"This applies a total physical thrust acceleration magnitude of {acc_mag_ms2:.4e} m/s^2, "
        f"oriented toward an azimuth of {acc_az:.1f} degrees and an elevation of {acc_el:.1f} degrees."
    )

    return response


# =============================
# GENERATE ONE TRAJECTORY
# =============================
def generate_trajectory(
    x0, t_span, reference_sol, mission_objective, mission_type, step_stride=1
):
    sol = solve_ivp(
        crtbp_controlled_eom,
        t_span,
        x0,
        args=(reference_sol, control_law, mu),
        method="RK45",
        rtol=1e-5,
        atol=1e-7,
        max_step=0.02,
        events=SAFETY_EVENTS,
    )

    states = sol.y.T
    times = sol.t
    dataset = []

    # Query actual max valid time from interpolated reference solution
    t_end = min(t_span[1], reference_sol.sol.t_max)
    final_desired_state = reference_sol.sol(t_end)
    total_duration = t_end - t_span[0]

    for i in range(0, len(states), step_stride):
        s = states[i]
        t_curr = times[i]

        if t_curr > t_end:
            break

        desired_state = reference_sol.sol(t_curr)
        pos_err = s[:3] - desired_state[:3]
        vel_err = s[3:6] - desired_state[3:6]
        u = control_law(s, desired_state)

        # ---------------------------------------------------------
        # STEP 1: Verify physical bounds (Filters >2,000 km, >100 m/s, or >0.01 m/s^2 control)
        # ---------------------------------------------------------
        if not verify_physical_bounds(s, pos_err, vel_err, u):
            continue

        # ---------------------------------------------------------
        # STEP 2: Dynamically verify true spatial region & assign labels
        # ---------------------------------------------------------
        dyn_mission_type, dyn_mission_obj = verify_and_get_dynamic_metadata(s)

        progress_percentage = (
            (t_curr / total_duration) * 100.0 if total_duration > 0 else 100.0
        )

        prompt_text = generate_prompt_text(
            s,
            desired_state,
            final_desired_state,
            progress_percentage,
            dyn_mission_type,
            dyn_mission_obj,
        )
        response_text = generate_response_text(u)

        dataset.append({
            "mission_type": dyn_mission_type,
            "mission_objective": dyn_mission_obj,
            "progress_pct": progress_percentage,
            "current_state": s.tolist(),
            "desired_state": desired_state.tolist(),
            "final_state": final_desired_state.tolist(),
            "control_vector": u.tolist(),
            "prompt": prompt_text,
            "response": response_text,
            "text": f"{prompt_text} {response_text}",
        })

    return dataset


# =============================
# MAIN DATASET GENERATION
# =============================
if __name__ == "__main__":
    max_t = 12 * np.pi

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"crtbp_natural_language_{timestamp}.jsonl"

    print(f"Starting dataset generation. Output file: '{output_filename}'")

    total_samples = 0

    with open(output_filename, "w", encoding="utf-8") as f:
        for traj_idx in range(500):
            mission_type, reference_sol = generate_reference_trajectory()
            mission_objective = get_mission_description(mission_type)

            t_end = min(max_t, reference_sol.sol.t_max)
            t_span = (0.0, t_end)
            ideal_x0 = reference_sol.sol(0.0)

            # Mode-specific noise scaling
            if mission_type == "moon_orbit":
                pos_noise, vel_noise = 0.00005, 0.0001  # ~19 km, ~0.1 m/s
            else:
                pos_noise, vel_noise = 0.0003, 0.0003   # ~115 km, ~0.3 m/s

            # Sanity loop to ensure valid non-colliding x0
            for _ in range(10):
                x0 = ideal_x0 + np.array([
                    np.random.uniform(-pos_noise, pos_noise),
                    np.random.uniform(-pos_noise, pos_noise),
                    np.random.uniform(-pos_noise / 2, pos_noise / 2),
                    np.random.uniform(-vel_noise, vel_noise),
                    np.random.uniform(-vel_noise, vel_noise),
                    np.random.uniform(-vel_noise / 2, vel_noise / 2),
                ])
                r_earth = np.linalg.norm(x0[:3] + np.array([mu, 0, 0]))
                r_moon = np.linalg.norm(x0[:3] - np.array([1 - mu, 0, 0]))
                
                # Earth radius ~0.0165 LU, Moon radius ~0.00452 LU
                if r_earth > 0.017 and r_moon > 0.0046:
                    break

            traj = generate_trajectory(
                x0, t_span, reference_sol, mission_objective, mission_type, step_stride=1
            )

            for item in traj:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

            f.flush()

            total_samples += len(traj)
            print(
                f"Completed Trajectory {traj_idx + 1}/500 | Saved {len(traj)} steps |"
                f" Cumulative: {total_samples}"
            )

    print(
        f"\nDone! Successfully saved {total_samples} total samples to"
        f" '{output_filename}'"
    )