import os
import re
import numpy as np
import torch
from peft import PeftModel
from scipy.integrate import solve_ivp
from transformers import AutoModelForCausalLM, AutoTokenizer

from CRTBP import DU, MU, TU, crtbp_eom
from Dataset_Generator import (
    generate_prompt_text,
    generate_unified_reference_trajectory,
    verify_and_get_dynamic_metadata,
)

# Same continuous disturbance model used during dataset generation
from Trajectories import unmodeled_perturbations

# Units
LU_TO_KM = DU
TU_TO_HOURS = TU / 3600.0
VU_TO_MS = (DU * 1000.0) / TU
AU_TO_MS2 = (DU * 1000.0) / (TU**2)

# Celestial Constants
R_EARTH_KM = 6378.14
R_MOON_KM = 1737.4
EARTH_POS_KM = np.array([-MU * LU_TO_KM, 0.0, 0.0])
MOON_POS_KM = np.array([(1.0 - MU) * LU_TO_KM, 0.0, 0.0])

# Thrust Limits
MAX_PHYSICAL_ACCEL_MS2 = 0.01
MAX_CANONICAL_ACCEL = MAX_PHYSICAL_ACCEL_MS2 / AU_TO_MS2


def parse_llm_canonical_accel(text: str):
    """Extract canonical acceleration vector from fine-tuned response format.

    Returns: (accel: np.ndarray, parsed_successfully: bool)
    """
    accel = np.zeros(3, dtype=np.float64)
    pattern = (
        r"\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*canonical units\)\s*along the"
        r" ([XYZ])-axis"
    )
    matches = re.findall(pattern, text)
    axis_map = {"X": 0, "Y": 1, "Z": 2}
    parsed = False

    if len(matches) == 3:
        for value, axis in matches:
            accel[axis_map[axis]] = float(value)
        parsed = True
    else:
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if len(numbers) >= 3:
            try:
                accel = np.array(
                    [float(numbers[0]), float(numbers[1]), float(numbers[2])],
                    dtype=np.float64,
                )
                parsed = True
            except Exception:
                accel[:] = 0.0

    magnitude = np.linalg.norm(accel)
    if magnitude > MAX_CANONICAL_ACCEL:
        accel = (accel / magnitude) * MAX_CANONICAL_ACCEL

    return accel, parsed


class PureLLMController:

    def __init__(
        self,
        base_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        adapter_path: str = "./results_llama3_1_8b_crtbp_fast_v1/checkpoint-25000",
        device: str = "cuda",
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        print(f"Loading Base Model: {base_model_id}", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        print(f"Loading Fine-Tuned LoRA Adapter: {adapter_path}", flush=True)
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

        self.previous_command = np.zeros(3, dtype=np.float64)
        self.command_alpha = 0.25
        self.max_command_rate = 2.0e-5
        self.parser_failures = 0

    def reset_controller_state(self):
        self.previous_command = np.zeros(3, dtype=np.float64)
        self.parser_failures = 0

    def smooth_and_limit_command(self, command: np.ndarray) -> np.ndarray:
        filtered = (
            self.command_alpha * command
            + (1.0 - self.command_alpha) * self.previous_command
        )
        delta = filtered - self.previous_command
        delta_norm = np.linalg.norm(delta)

        if delta_norm > self.max_command_rate:
            filtered = (
                self.previous_command
                + (delta / delta_norm) * self.max_command_rate
            )

        self.previous_command = filtered.copy()
        return filtered

    def predict_control(
        self,
        current_state: np.ndarray,
        reference_state: np.ndarray,
        final_desired_state: np.ndarray,
        progress_percentage: float,
        mission_type: str,
    ) -> tuple[np.ndarray, str]:

        dyn_mission_type, dyn_mission_obj = verify_and_get_dynamic_metadata(
            current_state, fallback_mission_type=mission_type
        )

        user_content = generate_prompt_text(
            state=current_state,
            desired_state=reference_state,
            final_desired_state=final_desired_state,
            progress_percentage=progress_percentage,
            mission_type=dyn_mission_type,
            mission_objective=dyn_mission_obj,
        )

        messages = [{"role": "user", "content": user_content}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        encoded_inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,  # Fast generation
                do_sample=False,
                use_cache=True,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_text = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        command, parsed = parse_llm_canonical_accel(generated_text)

        if not parsed:
            self.parser_failures += 1
            print(
                "[WARNING] LLM output parsing failed. Holding previous valid"
                " command.",
                flush=True,
            )
            command = self.previous_command.copy()

        command = self.smooth_and_limit_command(command)
        return command, generated_text


def run_pure_llm_evaluation(
    controller: PureLLMController,
    mission_type: str = "halo_l1",
    traj_id: int = 1,
    dt_step=0.00075,
    total_steps: int = 1500,
    perturbation_step: int = 1000,
    llm_update_stride: int = 10, 
):
    mission_type, sol_ref, ref_fn = generate_unified_reference_trajectory(
        mission_type=mission_type
    )

    t_start, t_end = sol_ref.t[0], sol_ref.t[-1]
    duration = t_end - t_start
    final_ref_state = np.ravel(ref_fn(t_end))[:6]
    current_state = np.ravel(ref_fn(t_start))[:6].copy()
    current_t = t_start
    active_a_llm = np.zeros(3, dtype=np.float64)

    time_hist = [current_t]
    state_sim_hist = [current_state.copy()]
    state_ref_hist = [np.ravel(ref_fn(current_t))[:6]]
    control_hist = []
    pos_err_hist = [0.0]
    vel_err_hist = [0.0]
    disturbances = []

    print(f"\n=================================================", flush=True)
    print(
        f"--- Running Mission #{traj_id}: {mission_type.upper()} ---",
        flush=True,
    )
    print(f"=================================================", flush=True)

    for step in range(total_steps):
        if current_t >= t_end:
            break

        if step == perturbation_step:
            pos_kick_km = np.random.uniform(-1.5, 1.5, size=3)
            vel_kick_ms = np.random.uniform(-0.1, 0.1, size=3)

            current_state[:3] += pos_kick_km / LU_TO_KM
            current_state[3:] += vel_kick_ms / VU_TO_MS

            t_dist_hrs = (current_t - t_start) * TU_TO_HOURS
            disturbances.append((step, t_dist_hrs, vel_kick_ms.tolist()))

            print(
                f"\n[!] Step {step:04d} Perturbation Applied:"
                f"\n    Pos Offset: {pos_kick_km} km"
                f"\n    Vel Offset: {vel_kick_ms} m/s\n",
                flush=True,
            )

        ref_state_curr = np.ravel(ref_fn(current_t))[:6]

        position_error_km = (
            np.linalg.norm(current_state[:3] - ref_state_curr[:3]) * LU_TO_KM
        )

        adaptive_stride = 2 if position_error_km > 5.0 else llm_update_stride

        if step % adaptive_stride == 0:
            progress_pct = ((current_t - t_start) / duration) * 100.0
            active_a_llm, _ = controller.predict_control(
                current_state=current_state,
                reference_state=ref_state_curr,
                final_desired_state=final_ref_state,
                progress_percentage=progress_pct,
                mission_type=mission_type,
            )

        control_hist.append(active_a_llm.copy())

        def llm_driven_dynamics(t, x):
            dsdt = np.array(crtbp_eom(t, x, MU), dtype=np.float64)
            disturbance = unmodeled_perturbations(t, x)
            dsdt[3:6] += active_a_llm + disturbance
            return dsdt

        t_span_step = (current_t, current_t + dt_step)
        sol_step = solve_ivp(
            llm_driven_dynamics,
            t_span_step,
            current_state,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            max_step=0.005,
        )

        current_t = sol_step.t[-1]
        current_state = sol_step.y[:, -1]

        ref_state_next = np.ravel(ref_fn(current_t))[:6]
        pos_err_km = (
            np.linalg.norm(current_state[:3] - ref_state_next[:3]) * LU_TO_KM
        )
        vel_err_ms = (
            np.linalg.norm(current_state[3:] - ref_state_next[3:]) * VU_TO_MS
        )

        time_hist.append(current_t)
        state_sim_hist.append(current_state.copy())
        state_ref_hist.append(ref_state_next)
        pos_err_hist.append(pos_err_km)
        vel_err_hist.append(vel_err_ms)

        if step % 100 == 0:
            print(
                f"Step {step:04d}/{total_steps} | Pos Err: {pos_err_km:8.2f} km"
                f" | Vel Err: {vel_err_ms:6.2f} m/s | LLM Accel:"
                f" [{active_a_llm[0]:+.4e}, {active_a_llm[1]:+.4e},"
                f" {active_a_llm[2]:+.4e}]",
                flush=True,
            )

    dt_sec = dt_step * TU
    total_dv_ms = sum(
        np.linalg.norm(u * AU_TO_MS2) * dt_sec for u in control_hist
    )

    sim_results = {
        "traj_id": traj_id,
        "mission_type": mission_type,
        "time": np.array(time_hist),
        "state_sim": np.array(state_sim_hist),
        "state_ref": np.array(state_ref_hist),
        "control": np.array(control_hist),
        "pos_err_km": np.array(pos_err_hist),
        "vel_err_ms": np.array(vel_err_hist),
        "total_delta_v_ms": total_dv_ms,
        "disturbances": disturbances,
    }

    return sim_results


if __name__ == "__main__":
    controller = PureLLMController()
    mission_types = ["earth_orbit", "moon_orbit", "transfer", "halo_l1", "dro"]
    all_results = []

    for idx, mission in enumerate(mission_types, start=1):
        controller.reset_controller_state()

        res = run_pure_llm_evaluation(
            controller=controller,
            mission_type=mission,
            traj_id=idx,
            dt_step=0.00075,
            total_steps=1500,
            perturbation_step=1000,
            llm_update_stride=10,
        )

        all_results.append(res)

    np.save("sim_results_multi.npy", all_results, allow_pickle=True)
    np.save("sim_results.npy", all_results[0], allow_pickle=True)

    print(
        "\n[*] All 5 CRTBP Mission Simulations Finished Successfully.",
        flush=True,
    )