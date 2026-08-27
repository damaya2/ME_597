import argparse
import json
import os
import re
import numpy as np
import torch
from peft import PeftModel
from scipy.integrate import solve_ivp
from transformers import AutoModelForCausalLM, AutoTokenizer

from CRTBP import MU, crtbp_eom
from Dataset_Generator import (
    MISSION_LABELS,
    generate_prompt_text,
    generate_unified_reference_trajectory,
)
from Trajectories import (
    AU_TO_MS2,
    LU_TO_KM,
    TU_TO_SEC,
    VU_TO_MS,
    unmodeled_perturbations,
    verify_and_get_dynamic_metadata,
    verify_physical_bounds,
)

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./results_llama3_1_8b_crtbp_fast_v1/final_adapter"
OUTPUT_RESULTS_FILE = "eval_comparison_results.json"


class LLMControllerAdapter:
    """
    Inference wrapper for fine-tuned Llama-3.1-8B CRTBP controller.
    Applies artificial gain scaling to extract directional thrust vector.
    """

    def __init__(self, base_model_name, adapter_path, device="cuda", u_max_m_s2=0.01):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.u_max_m_s2 = u_max_m_s2
        self.u_max_canon = u_max_m_s2 / AU_TO_MS2

        hf_token = os.getenv("HF_TOKEN")

        print(f"Loading Base Tokenizer: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True, token=hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = "left"  # Crucial for generation tasks

        print(f"Loading Base Model in BF16 on {self.device}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=self.device,
            trust_remote_code=True,
            token=hf_token,
        )

        print(f"Applying Fine-Tuned Adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

        # Direct string regex pattern matchers
        self.regex_canonical = re.compile(
            r"([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*canonical",
            re.IGNORECASE,
        )

    def predict_control_action(
        self, state, desired_state, final_desired_state, progress_pct, mission_type
    ):
        """Generates control vector u_canon using dynamic error expansion."""
        
        # Scaling gain factors
        K_pos = 100.0  # Expand position perturbation by 100x
        K_vel = 10.0   # Expand velocity perturbation by 10x

        pos_err = state[:3] - desired_state[:3]
        vel_err = state[3:] - desired_state[3:]

        # Virtual state offset
        scaled_state = np.copy(state)
        scaled_state[:3] = desired_state[:3] + pos_err * K_pos
        scaled_state[3:] = desired_state[3:] + vel_err * K_vel

        # Keep original mission metadata to prevent misclassification
        _, dyn_mission_obj = verify_and_get_dynamic_metadata(state)

        prompt = generate_prompt_text(
            state=scaled_state,
            desired_state=desired_state,
            final_desired_state=final_desired_state,
            progress_percentage=progress_pct,
            mission_type=mission_type,  # Force consistent mission classification
            mission_objective=dyn_mission_obj,
        )

        messages = [{"role": "user", "content": prompt}]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted_prompt, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Robust generation slicing
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        response_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        u_canon_raw = self._parse_thrust_vector(response_text)

        # Proportional directional scaling
        u_norm_raw = np.linalg.norm(u_canon_raw)
        if u_norm_raw > 1e-12:
            # Preserve direction, scale magnitude proportionally down
            u_dir = u_canon_raw / u_norm_raw
            # Target small proportional correction
            target_mag = min(self.u_max_canon, (u_norm_raw / K_pos))
            u_canon = u_dir * target_mag
        else:
            u_canon = np.zeros(3, dtype=np.float64)

        # Hard saturation
        u_norm = np.linalg.norm(u_canon)
        if self.u_max_canon is not None and u_norm > self.u_max_canon:
            u_canon = u_canon * (self.u_max_canon / u_norm)

        return u_canon, response_text

    def _parse_thrust_vector(self, text):
        """Extracts numerical values directly from LLM completion string."""
        u_vec = np.zeros(3, dtype=np.float64)
        
        # Look for 3 consecutive numbers or numbers near axis tags
        matches = re.findall(r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?", text)
        if len(matches) >= 3:
            try:
                # Extract first three floating point matches
                u_vec[0] = float(matches[0])
                u_vec[1] = float(matches[1])
                u_vec[2] = float(matches[2])
                return u_vec
            except ValueError:
                pass

        return np.zeros(3, dtype=np.float64)


def simulate_llm_closed_loop(
    adapter, ref_fn, t_span, x0, mission_type, perturbation_offset, perturb_step=100, dt_step=0.02, u_max_m_s2=0.01
):
    """Executes discrete-time LLM feedback integration in CRTBP environment."""
    t_eval = np.arange(t_span[0], t_span[1], dt_step)
    n_steps = len(t_eval)

    state_history = np.zeros((n_steps, 6))
    ref_history = np.zeros((n_steps, 6))
    u_history = np.zeros((n_steps, 3))
    safety_flags = np.zeros(n_steps, dtype=bool)

    current_state = x0.copy()
    total_duration = t_span[1] - t_span[0]
    final_desired_state = np.squeeze(ref_fn(t_span[1]))

    print(f"\n--- Starting Closed-Loop Evaluation ({mission_type}) ---")

    for idx, t in enumerate(t_eval):
        if idx == perturb_step:
            print(f" -> [Step {idx:03d}] Injecting perturbation offset into current state...")
            current_state += perturbation_offset

        ref_state = np.squeeze(ref_fn(t))
        progress_pct = (t / total_duration) * 100.0 if total_duration > 0 else 100.0

        u_canon, _ = adapter.predict_control_action(
            state=current_state,
            desired_state=ref_state,
            final_desired_state=final_desired_state,
            progress_pct=progress_pct,
            mission_type=mission_type,
        )

        state_history[idx] = current_state
        ref_history[idx] = ref_state
        u_history[idx] = u_canon

        pos_err = current_state[:3] - ref_state[:3]
        vel_err = current_state[3:] - ref_state[3:]

        safety_flags[idx] = verify_physical_bounds(
            state=current_state,
            pos_err=pos_err,
            vel_err=vel_err,
            control_vec=u_canon,
            u_max_m_s2=u_max_m_s2,
        )

        if idx < n_steps - 1:
            def closed_loop_rhs(t_sub, y):
                natural_derivs = crtbp_eom(t_sub, y, MU)
                a_dist = unmodeled_perturbations(t_sub, y)
                d_state = natural_derivs.copy()
                d_state[3:] += u_canon + a_dist
                return d_state

            sol = solve_ivp(
                closed_loop_rhs,
                [t, t_eval[idx + 1]],
                current_state,
                method="RK45",
                rtol=1e-7,
                atol=1e-9,
            )
            current_state = sol.y[:, -1]

        if idx % 20 == 0 or idx == n_steps - 1:
            err_km = np.linalg.norm(pos_err) * LU_TO_KM
            u_ms2 = np.linalg.norm(u_canon) * AU_TO_MS2
            print(
                f"[{mission_type} | Step {idx:03d}/{n_steps:03d}] t={t:5.2f} | Pos Err: {err_km:8.4f} km | Thrust: {u_ms2:.4e} m/s²"
            )

    return t_eval, state_history, ref_history, u_history, safety_flags


def compute_performance_metrics(t_eval, state_hist, ref_hist, u_hist):
    """Calculates RMS/Mean position error, velocity error, and Delta-V expenditure."""
    pos_err_km = np.linalg.norm((state_hist[:, :3] - ref_hist[:, :3]) * LU_TO_KM, axis=1)
    vel_err_ms = np.linalg.norm((state_hist[:, 3:] - ref_hist[:, 3:]) * VU_TO_MS, axis=1)

    u_mag_ms2 = np.linalg.norm(u_hist * AU_TO_MS2, axis=1)
    dt_sec = np.diff(t_eval, prepend=t_eval[0]) * TU_TO_SEC
    total_delta_v_ms = np.sum(u_mag_ms2 * dt_sec)

    return {
        "rms_pos_err_km": float(np.sqrt(np.mean(pos_err_km**2))),
        "rms_vel_err_ms": float(np.sqrt(np.mean(vel_err_ms**2))),
        "mean_pos_err_km": float(np.mean(pos_err_km)),
        "max_pos_err_km": float(np.max(pos_err_km)),
        "total_delta_v_ms": float(total_delta_v_ms),
    }


def generate_mission_trajectory(mission_type):
    """Generates reference trajectory and fine station-keeping perturbation noise."""
    m_type, sol_ref, ref_fn = generate_unified_reference_trajectory(mission_type=mission_type)
    t_span = (float(sol_ref.t[0]), float(min(sol_ref.t[-1], 1.5)))

    pos_noise = 2.6e-6  
    vel_noise = 1.0e-6  

    x0_ref = ref_fn(t_span[0])
    
    x0_noisy = x0_ref + np.array([
        np.random.uniform(-pos_noise, pos_noise),
        np.random.uniform(-pos_noise, pos_noise),
        np.random.uniform(-pos_noise / 2, pos_noise / 2),
        np.random.uniform(-vel_noise, vel_noise),
        np.random.uniform(-vel_noise, vel_noise),
        np.random.uniform(-vel_noise / 2, vel_noise / 2),
    ])

    perturbation_offset = np.array([
        np.random.uniform(-pos_noise, pos_noise),
        np.random.uniform(-pos_noise, pos_noise),
        np.random.uniform(-pos_noise / 2, pos_noise / 2),
        np.random.uniform(-vel_noise, vel_noise),
        np.random.uniform(-vel_noise, vel_noise),
        np.random.uniform(-vel_noise / 2, vel_noise / 2),
    ])

    return ref_fn, t_span, x0_noisy, perturbation_offset


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Fine-Tuned LLM Performance Across All 6 Mission Types"
    )
    parser.add_argument("--model-name", type=str, default=BASE_MODEL_NAME)
    parser.add_argument("--adapter-path", type=str, default=ADAPTER_PATH)
    parser.add_argument("--u-max", type=float, default=0.01)
    parser.add_argument("--perturb-step", type=int, default=100)
    args = parser.parse_args()

    adapter = LLMControllerAdapter(
        base_model_name=args.model_name,
        adapter_path=args.adapter_path,
        u_max_m_s2=args.u_max,
    )

    mission_types = [
        "earth_orbit",
        "moon_orbit",
        "transfer",
        "halo_l1",
        "halo_l2",
        "dro",
    ]
    all_results = {}

    for m_type in mission_types:
        print(f"\n=================================================")
        print(f"        RUNNING MISSION TYPE: {m_type}")
        print(f"=================================================")

        ref_fn, t_span, x0_clean, perturbation_offset = generate_mission_trajectory(m_type)

        t_llm, state_llm, ref_llm, u_llm, safe_llm = simulate_llm_closed_loop(
            adapter=adapter,
            ref_fn=ref_fn,
            t_span=t_span,
            x0=x0_clean,
            mission_type=m_type,
            perturbation_offset=perturbation_offset,
            perturb_step=args.perturb_step,
            dt_step=0.02,
            u_max_m_s2=args.u_max,
        )

        metrics = compute_performance_metrics(t_llm, state_llm, ref_llm, u_llm)

        all_results[m_type] = {
            "mission_type": m_type,
            "metrics": metrics,
            "time": t_llm.tolist(),
            "state_sim": state_llm.tolist(),
            "state_ref": ref_llm.tolist(),
            "control_u": u_llm.tolist(),
            "safety_flags": safe_llm.tolist(),
        }

        print(f"\n[{m_type}] METRICS SUMMARY:")
        print(f"  RMS Position Error : {metrics['rms_pos_err_km']:.4f} km")
        print(f"  Max Position Error : {metrics['max_pos_err_km']:.4f} km")
        print(f"  RMS Velocity Error : {metrics['rms_vel_err_ms']:.4f} m/s")
        print(f"  Total Delta-V      : {metrics['total_delta_v_ms']:.4f} m/s")
        print(f"  Safety Check Pass  : {int(np.sum(safe_llm))}/{len(safe_llm)} steps")

    with open(OUTPUT_RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll 6 mission evaluations completed! Results saved to {OUTPUT_RESULTS_FILE}.")


if __name__ == "__main__":
    main()