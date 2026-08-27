import os
import re
import time
import numpy as np
import torch
from peft import PeftModel
from scipy.integrate import solve_ivp
from transformers import AutoModelForCausalLM, AutoTokenizer

# Workspace imports
from CRTBP import (
    AU,
    DU,
    TU,
    VU,
    canonical_to_physical,
    crtbp_controlled_eom,
    crtbp_eom,
    mu,
    physical_to_canonical,
)
from Desired_State import (
    MAX_ESCAPE_RADIUS,
    R_EARTH_CANONICAL,
    R_MOON_CANONICAL,
    SAFETY_EVENTS,
    generate_reference_trajectory,
    get_mission_description,
)
from Natural_Dataset import generate_prompt_text
from verify import verify_and_get_dynamic_metadata, verify_physical_bounds

# Physical constants for unit conversion
LU_TO_KM = 384400.0
TU_IN_SECONDS = 375190.06
TU_TO_HOURS = TU_IN_SECONDS / 3600.0
VU_TO_MS = 1024.54
AU_TO_MS2 = 0.00273

MU = 0.01215058560962404

# Safety & Physics Limits
MAX_PHYSICAL_ACCEL_MS2 = 0.01  # Hard thruster hardware limit (0.01 m/s^2)
MAX_CANONICAL_ACCEL = MAX_PHYSICAL_ACCEL_MS2 / AU_TO_MS2  # ~3.663 canonical acceleration units


def parse_canonical_acceleration_from_text(text: str) -> np.ndarray:
    """
    Parses LLM output text for canonical acceleration values and validates 
    them against physical bounds to prevent unit-scale hallucinations.
    """
    accel = np.zeros(3, dtype=np.float64)
    parsed_successfully = False

    # 1. Primary Regex: Look for target parenthetical pattern e.g., '(0.000441 canonical units)'
    try:
        pattern = r"\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*canonical units\s*\)"
        matches = re.findall(pattern, text)
        if len(matches) >= 3:
            accel = np.array(
                [float(matches[0]), float(matches[1]), float(matches[2])],
                dtype=np.float64,
            )
            parsed_successfully = True
    except Exception:
        pass

    # 2. Fallback Regex: Extract raw numerical floats if parenthetical format is missing
    if not parsed_successfully:
        try:
            numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
            if len(numbers) >= 3:
                accel = np.array(
                    [float(numbers[0]), float(numbers[1]), float(numbers[2])],
                    dtype=np.float64,
                )
                parsed_successfully = True
        except Exception:
            pass

    # 3. Dynamic Unit Verification & Auto-Scaling
    # If LLM output values exceeding canonical max by orders of magnitude (e.g., outputs m/s^2 directly),
    # scale them down to canonical units rather than applying raw unscaled floats.
    max_val = np.max(np.abs(accel))
    if max_val > MAX_CANONICAL_ACCEL * 2.0:
        if max_val <= 10.0:
            # Model outputted m/s^2 physical units instead of canonical units
            accel = accel / AU_TO_MS2
        else:
            # Complete hallucination / extreme value -> nullify command safely
            print(f"[Warning] Extreme control command detected ({max_val:.2f}). Zeroing output.", flush=True)
            return np.zeros(3, dtype=np.float64)

    # 4. Strict Saturation Clamping (Enforces 0.01 m/s^2 bound in Python)
    accel = np.clip(accel, -MAX_CANONICAL_ACCEL, MAX_CANONICAL_ACCEL)
    
    return accel


class LLMController:
    """Wrapper class to load and query the fine-tuned Llama-3.1 model."""

    def __init__(
        self,
        base_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        adapter_path: str = "./results_llama3_1_8b_crtbp_sft_v2/final_adapter",
        device: str = "cuda",
    ):
        print(f"[*] Loading base model: {base_model_id}", flush=True)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        print(f"[*] Loading fine-tuned LoRA adapter from: {adapter_path}", flush=True)
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

    def predict_control(
        self,
        current_state: np.ndarray,
        reference_state: np.ndarray,
        final_desired_state: np.ndarray,
        progress_percentage: float,
        mission_type: str,
        mission_desc: str,
        max_new_tokens: int = 256,
    ) -> tuple[np.ndarray, str]:

        raw_prompt = generate_prompt_text(
            state=current_state,
            desired_state=reference_state,
            final_desired_state=final_desired_state,
            progress_percentage=progress_percentage,
            mission_type=mission_type,
            mission_objective=mission_desc,
        )

        inputs = self.tokenizer(raw_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        a_canonical = parse_canonical_acceleration_from_text(generated_text)
        return a_canonical, generated_text


def run_closed_loop_simulation(
    controller: LLMController,
    mission_type: str = "transfer",
    dt_step: float = 0.002,
    total_steps: int = 1000,
    tcm_interval_steps: int = 5,  # Reduced from 20 to 5 (~0.5 hours) to prevent error drift
    pos_perturbation_km: np.ndarray = np.array([3.0, -2.0, 1.0]),
    vel_perturbation_ms: np.ndarray = np.array([0.005, -0.002, 0.001]),
    disturbance_prob: float = 0.05,
    min_step_spacing: int = 20,
    dist_vel_std_ms: float = 0.002,
    max_kick_ms: float = 0.005,
    traj_id: int = 1,
):
    print("\n========================================================", flush=True)
    print(f" STARTING CLOSED-LOOP CONTINUOUS SIMULATION [Traj {traj_id}] | Mission: {mission_type}", flush=True)
    print(f" LLM Update Interval: Every {tcm_interval_steps} steps (~{(tcm_interval_steps * dt_step * TU_TO_HOURS):.2f} hours)", flush=True)
    print("========================================================\n", flush=True)

    # 1. Generate reference trajectory using Desired_State generator
    selected_mission, ref_sol = generate_reference_trajectory(mission_type=mission_type)
    mission_desc = get_mission_description(selected_mission)

    t_start = ref_sol.t[0]
    t_end = min(t_start + total_steps * dt_step, ref_sol.t[-1])
    total_duration = t_end - t_start
    final_ref_state = ref_sol.sol(t_end)

    # 2. Apply initial position/velocity perturbations
    pos_pert_canonical = pos_perturbation_km / DU
    vel_pert_canonical = vel_perturbation_ms / VU

    ref_state_t0 = ref_sol.sol(t_start)
    current_state = ref_state_t0.copy()
    current_state[:3] += pos_pert_canonical
    current_state[3:] += vel_pert_canonical

    history_time = [t_start]
    history_state_sim = [current_state.copy()]
    history_state_ref = [ref_state_t0.copy()]
    history_control = [np.zeros(3, dtype=np.float64)]
    history_pos_err_km = [np.linalg.norm(pos_perturbation_km)]
    history_vel_err_ms = [np.linalg.norm(vel_perturbation_ms)]
    history_disturbances = []

    total_delta_v_ms = 0.0
    current_t = t_start

    NO_DISTURBANCE_STEPS = 20
    last_disturbance_step = -min_step_spacing
    active_a_pred = np.zeros(3, dtype=np.float64)

    for step in range(total_steps):
        if current_t >= t_end:
            print("[!] Reached maximum trajectory time span.", flush=True)
            break

        steps_since_last = step - last_disturbance_step

        # Apply stochastic environmental disturbances
        if (
            step >= NO_DISTURBANCE_STEPS
            and steps_since_last >= min_step_spacing
            and np.random.rand() < disturbance_prob
        ):
            direction = np.random.normal(size=3)
            direction /= np.linalg.norm(direction)

            kick_mag_ms = np.clip(
                np.abs(np.random.normal(0.002, dist_vel_std_ms)),
                0.0005,
                max_kick_ms,
            )
            d_vel_ms = direction * kick_mag_ms
            current_state[3:] += d_vel_ms / VU

            last_disturbance_step = step
            time_hrs = (current_t - t_start) * TU_TO_HOURS
            history_disturbances.append((step, time_hrs, d_vel_ms))

        ref_state_t = ref_sol.sol(current_t)
        progress_pct = (
            ((current_t - t_start) / total_duration) * 100.0
            if total_duration > 0
            else 100.0
        )

        # Query LLM control model at set intervals
        if step % tcm_interval_steps == 0:
            active_a_pred, raw_llm_output = controller.predict_control(
                current_state=current_state,
                reference_state=ref_state_t,
                final_desired_state=final_ref_state,
                progress_percentage=progress_pct,
                mission_type=selected_mission,
                mission_desc=mission_desc,
            )

            print(
                f"[LLM Update @ Step {step:04d}] Command Accel: [{active_a_pred[0]:.6e}, {active_a_pred[1]:.6e}, {active_a_pred[2]:.6e}] | "
                f"Raw snippet: {raw_llm_output.strip()[:60]}...",
                flush=True,
            )

        # Step integration under predicted control vector
        t_span_step = (current_t, current_t + dt_step)

        def controlled_dynamics(t, x):
            dsdt = np.array(crtbp_eom(t, x, MU), dtype=np.float64)
            dsdt[3:6] += active_a_pred
            return dsdt

        sol_step = solve_ivp(
            controlled_dynamics,
            t_span_step,
            current_state,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            events=SAFETY_EVENTS,
        )

        if sol_step.status != 0 or any(len(ev) > 0 for ev in sol_step.t_events):
            print(f"[CRITICAL] Terminal safety event triggered at t = {sol_step.t[-1]:.4f}! (Impact/Escape)", flush=True)
            break

        current_t = sol_step.t[-1]
        current_state = sol_step.y[:, -1]

        # Calculate tracking errors and delta-v expenditure
        applied_dv_step_ms = np.linalg.norm(active_a_pred) * dt_step * VU
        total_delta_v_ms += applied_dv_step_ms

        pos_err_km = np.linalg.norm(current_state[:3] - ref_sol.sol(current_t)[:3]) * DU
        vel_err_ms = np.linalg.norm(current_state[3:] - ref_sol.sol(current_t)[3:]) * VU

        # Save data step for LMM_Trajectory compatibility
        history_time.append(current_t)
        history_state_sim.append(current_state.copy())
        history_state_ref.append(ref_sol.sol(current_t).copy())
        history_control.append(active_a_pred.copy())
        history_pos_err_km.append(pos_err_km)
        history_vel_err_ms.append(vel_err_ms)

        if (step + 1) % 50 == 0 or step == total_steps - 1:
            print(
                f"Step {step+1:04d}/{total_steps} | t={current_t:.3f} | "
                f"Pos Err: {pos_err_km:7.2f} km | Vel Err: {vel_err_ms:6.2f} m/s | "
                f"Cum ΔV: {total_delta_v_ms:6.3f} m/s",
                flush=True,
            )

    return {
        "traj_id": traj_id,
        "mission_type": selected_mission,
        "initial_pos_pert_km": pos_perturbation_km,
        "initial_vel_pert_ms": vel_perturbation_ms,
        "time": np.array(history_time),
        "state_sim": np.array(history_state_sim),  # Controlled Path
        "state_ref": np.array(history_state_ref),  # Reference Trajectory
        "control": np.array(history_control),
        "pos_err_km": np.array(history_pos_err_km),
        "vel_err_ms": np.array(history_vel_err_ms),
        "total_delta_v_ms": total_delta_v_ms,
        "disturbances": history_disturbances,
    }


if __name__ == "__main__":
    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    ADAPTER_PATH = "./results_llama3_1_8b_crtbp_sft_v2/final_adapter"

    abs_adapter_path = os.path.abspath(ADAPTER_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(abs_adapter_path):
        controller = LLMController(
            base_model_id=BASE_MODEL,
            adapter_path=abs_adapter_path,
            device=device,
        )

        eval_scenarios = [
            {"mission": "transfer", "pos_pert_km": np.array([3.0, -2.0, 1.0]), "vel_pert_ms": np.array([0.005, -0.002, 0.001])},
            {"mission": "earth_orbit", "pos_pert_km": np.array([-2.5, 3.5, -1.0]), "vel_pert_ms": np.array([-0.004, 0.006, -0.002])},
            {"mission": "moon_orbit", "pos_pert_km": np.array([1.5, -1.0, 0.5]), "vel_pert_ms": np.array([0.002, -0.003, 0.001])},
            {"mission": "transfer", "pos_pert_km": np.array([-3.5, -3.0, 1.5]), "vel_pert_ms": np.array([-0.006, 0.002, -0.001])},
            {"mission": "earth_orbit", "pos_pert_km": np.array([2.5, -3.0, -1.0]), "vel_pert_ms": np.array([0.004, 0.003, 0.002])},
        ]

        all_trajectories = []

        for idx, scenario in enumerate(eval_scenarios, start=1):
            np.random.seed(42 + idx)

            traj_result = run_closed_loop_simulation(
                controller=controller,
                mission_type=scenario["mission"],
                dt_step=0.002,
                total_steps=1000,
                tcm_interval_steps=5,  # Faster closed-loop control updates
                pos_perturbation_km=scenario["pos_pert_km"],
                vel_perturbation_ms=scenario["vel_pert_ms"],
                traj_id=idx,
            )
            all_trajectories.append(traj_result)

        # Output file for LMM_Trajectory.py
        save_file = "sim_results_multi.npy"
        np.save(save_file, all_trajectories)
        print(f"\n[*] Trajectory simulation data successfully exported to '{save_file}'.", flush=True)
        print("[*] Run 'python LMM_Trajectory.py' to generate comparison plots.", flush=True)