import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Direct imports from your dataset generation files
from CRTBP import AU, DU, VU, canonical_to_physical, crtbp_eom, mu
from verify import verify_physical_bounds, verify_and_get_dynamic_metadata
from Desired_State import (
    SAFETY_EVENTS,
    generate_reference_trajectory,
    get_mission_description,
)
from Natural_Dataset import control_law, generate_trajectory

OUTPUT_DIR = "training_data_audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_single_trajectory_audit(samples, summary, save_path):
    """Generates a 4-panel diagnostic plot for a single trajectory run."""
    pos_errs_km = []
    vel_errs_ms = []
    acc_mags_ms2 = []
    times = []

    for i, s in enumerate(samples):
        curr = np.array(s["current_state"])
        ref = np.array(s["desired_state"])
        u = np.array(s["control_vector"])

        pos_errs_km.append(np.linalg.norm((curr[:3] - ref[:3]) * DU))
        vel_errs_ms.append(np.linalg.norm((curr[3:] - ref[3:]) * VU))
        acc_mags_ms2.append(np.linalg.norm(u * AU))
        
        # Safe fallback: use 'step' key if present, otherwise default to frame index
        times.append(s.get("step", i))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    traj_id = summary["traj_id"]
    mission_type = summary["mission_type"]

    # Panel 1: Position Error over Steps/Time
    axes[0, 0].plot(times, pos_errs_km, color="#1f77b4", linewidth=1.8)
    axes[0, 0].set_title(f"Trajectory #{traj_id:02d} ({mission_type}): Position Error", fontweight="bold")
    axes[0, 0].set_xlabel("Sample Step Index")
    axes[0, 0].set_ylabel("Position Error (km)")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Velocity Error over Steps/Time
    axes[0, 1].plot(times, vel_errs_ms, color="#2ca02c", linewidth=1.8)
    axes[0, 1].set_title(f"Trajectory #{traj_id:02d} ({mission_type}): Velocity Error", fontweight="bold")
    axes[0, 1].set_xlabel("Sample Step Index")
    axes[0, 1].set_ylabel("Velocity Error (m/s)")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Control Acceleration Magnitude vs Limit Threshold
    axes[1, 0].plot(times, acc_mags_ms2, color="#d62728", linewidth=1.8)
    axes[1, 0].axhline(0.01, color="black", linestyle="--", linewidth=1.5, label="Max Bounds Threshold (0.01 m/s²)")
    axes[1, 0].set_title(f"Trajectory #{traj_id:02d} ({mission_type}): Target Acceleration", fontweight="bold")
    axes[1, 0].set_xlabel("Sample Step Index")
    axes[1, 0].set_ylabel("Thrust Acceleration (m/s²)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 4: Trajectory Summary Metrics Breakdown
    axes[1, 1].axis("off")
    summary_text = (
        f"TRAJECTORY #{traj_id:02d} METRIC SUMMARY\n"
        f"----------------------------------------\n"
        f"Mission Profile    : {mission_type}\n"
        f"Total Samples      : {summary['sample_count']} frames\n"
        f"Max Position Error : {summary['max_pos_err_km']:.2f} km\n"
        f"Final Residual     : {summary['final_pos_err_km']:.2f} km\n"
        f"Mean Velocity Err  : {summary['mean_vel_err_ms']:.4f} m/s\n"
        f"Peak Acceleration  : {summary['max_acc_ms2']:.4e} m/s²\n"
    )
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family="monospace", va="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Saved individual trajectory plot: {save_path}")


def audit_dataset_physics(num_trajectories=15):
    """Audits tracking quality, boundary compliance, and stability,

    generating both per-trajectory plots and an overall summary plot.
    """
    print("=================================================================")
    print(f"[+] AUDITING TRAINING DATA PHYSICS & CONTROLLER ({num_trajectories} Trajectories)")
    print("=================================================================")

    all_audited_samples = []
    trajectory_summaries = []

    for traj_idx in range(1, num_trajectories + 1):
        mission_type, reference_sol = generate_reference_trajectory()
        mission_obj = get_mission_description(mission_type)
        t_end = min(4.0 * np.pi, reference_sol.sol.t_max)
        t_span = (0.0, t_end)

        ideal_x0 = reference_sol.sol(0.0)
        pos_noise, vel_noise = (
            (0.00005, 0.0001) if mission_type == "moon_orbit" else (0.0003, 0.0003)
        )

        x0 = ideal_x0 + np.array([
            np.random.uniform(-pos_noise, pos_noise),
            np.random.uniform(-pos_noise, pos_noise),
            np.random.uniform(-pos_noise / 2, pos_noise / 2),
            np.random.uniform(-vel_noise, vel_noise),
            np.random.uniform(-vel_noise, vel_noise),
            np.random.uniform(-vel_noise / 2, vel_noise / 2),
        ])

        samples = generate_trajectory(
            x0, t_span, reference_sol, mission_obj, mission_type, step_stride=1
        )

        if len(samples) == 0:
            print(f"[*] Trajectory #{traj_idx:02d} ({mission_type}): FAILED - Entirely filtered by safety bounds.")
            continue

        pos_errs = [np.linalg.norm((np.array(s["current_state"])[:3] - np.array(s["desired_state"])[:3]) * DU) for s in samples]
        vel_errs = [np.linalg.norm((np.array(s["current_state"])[3:] - np.array(s["desired_state"])[3:]) * VU) for s in samples]
        u_mags = [np.linalg.norm(np.array(s["control_vector"]) * AU) for s in samples]

        summary = {
            "traj_id": traj_idx,
            "mission_type": mission_type,
            "sample_count": len(samples),
            "max_pos_err_km": np.max(pos_errs),
            "final_pos_err_km": pos_errs[-1],
            "mean_vel_err_ms": np.mean(vel_errs),
            "max_acc_ms2": np.max(u_mags),
        }
        trajectory_summaries.append(summary)
        all_audited_samples.extend(samples)

        # Plot individual trajectory report
        single_plot_path = os.path.join(OUTPUT_DIR, f"trajectory_physics_audit_run_{traj_idx:02d}.png")
        plot_single_trajectory_audit(samples, summary, single_plot_path)

        print(
            f"[*] Trajectory #{traj_idx:02d} ({mission_type:12s}): "
            f"Samples={len(samples):3d} | Max Pos Err={summary['max_pos_err_km']:7.1f} km | "
            f"Final Pos Err={summary['final_pos_err_km']:6.1f} km | "
            f"Max Acc={summary['max_acc_ms2']:.4e} m/s²"
        )

    print(f"\n[+] Audit Complete: Analyzed {len(all_audited_samples)} total training frames across {len(trajectory_summaries)} trajectories.")

    # Render global dataset summary dashboard
    plot_audit_diagnostics(all_audited_samples, trajectory_summaries, os.path.join(OUTPUT_DIR, "training_physics_audit.png"))


def plot_audit_diagnostics(samples, summaries, save_path):
    """Generates the aggregated multi-trajectory diagnostic dashboard."""
    pos_errs_km = []
    vel_errs_ms = []
    acc_mags_ms2 = []
    acc_components = []

    for s in samples:
        curr = np.array(s["current_state"])
        ref = np.array(s["desired_state"])
        u = np.array(s["control_vector"])

        pos_errs_km.append(np.linalg.norm((curr[:3] - ref[:3]) * DU))
        vel_errs_ms.append(np.linalg.norm((curr[3:] - ref[3:]) * VU))

        u_phys = u * AU
        acc_mags_ms2.append(np.linalg.norm(u_phys))
        acc_components.append(u_phys)

    acc_components = np.array(acc_components)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Panel 1: Position Error Distribution
    axes[0, 0].hist(pos_errs_km, bins=35, color="#1f77b4", edgecolor="black", alpha=0.8)
    axes[0, 0].set_title("Position Tracking Error Distribution (km)", fontweight="bold")
    axes[0, 0].set_xlabel("Position Error (km)")
    axes[0, 0].set_ylabel("Frame Count")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Velocity Error Distribution
    axes[0, 1].hist(vel_errs_ms, bins=35, color="#2ca02c", edgecolor="black", alpha=0.8)
    axes[0, 1].set_title("Velocity Tracking Error Distribution (m/s)", fontweight="bold")
    axes[0, 1].set_xlabel("Velocity Error (m/s)")
    axes[0, 1].set_ylabel("Frame Count")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Target Control Acceleration Magnitude vs Filter Boundary
    axes[1, 0].hist(acc_mags_ms2, bins=35, color="#d62728", edgecolor="black", alpha=0.8)
    axes[1, 0].axvline(0.01, color="black", linestyle="--", linewidth=1.5, label="Max Bounds Threshold (0.01 m/s²)")
    axes[1, 0].set_title("Target Acceleration Magnitudes in Dataset (m/s²)", fontweight="bold")
    axes[1, 0].set_xlabel("Thrust Acceleration (m/s²)")
    axes[1, 0].set_ylabel("Frame Count")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 4: Settled Position Residual Offset by Trajectory Run
    traj_labels = [f"#{s['traj_id']}" for s in summaries]
    final_errs = [s["final_pos_err_km"] for s in summaries]
    bars = axes[1, 1].bar(traj_labels, final_errs, color="#9467bd", alpha=0.85, edgecolor="black")
    axes[1, 1].set_title("Final Settled Position Residual Offset (km)", fontweight="bold")
    axes[1, 1].set_xlabel("Trajectory Index")
    axes[1, 1].set_ylabel("Final Residual (km)")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6, axis="y")

    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, height + 1.0, f"{height:.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Audit visual report saved to: {save_path}")


if __name__ == "__main__":
    audit_dataset_physics(num_trajectories=15)


# import json
# import random
# import re
# import sys
# import numpy as np
# import torch
# from peft import PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from tqdm import tqdm
# from CRTBP import AU

# # ==========================================
# # CONFIGURATION
# # ==========================================
# TEST_DATASET_PATH = "crtbp_natural_language_20260804_035216.jsonl"  # Path to test jsonl
# BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# ADAPTER_PATH = "./results_llama3_1_8b_crtbp_sft_v2/final_adapter" 

# NUM_TEST_SAMPLES = 100  # Number of samples to test (set to None for all)
# RANDOM_SEED = 42        # Set seed for reproducible random sampling

# # ==========================================
# # REGEX PARSER FOR MODEL RESPONSE
# # ==========================================
# def parse_predicted_control(response_text):
#     """
#     Extracts canonical u = [ux, uy, uz] from response text using Regex.
#     Looks for pattern: '(0.000441 canonical units)'
#     """
#     pattern = r"\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*canonical units\)"
#     matches = re.findall(pattern, response_text)
    
#     if len(matches) >= 3:
#         try:
#             ux = float(matches[0])
#             uy = float(matches[1])
#             uz = float(matches[2])
#             return np.array([ux, uy, uz])
#         except ValueError:
#             return None
#     return None

# # ==========================================
# # MAIN EVALUATION
# # ==========================================
# def evaluate():
#     print("Loading tokenizer and model...")
#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )

#     base_model = AutoModelForCausalLM.from_pretrained(
#         BASE_MODEL,
#         quantization_config=bnb_config,
#         device_map="auto",
#         torch_dtype=torch.bfloat16,
#     )

#     model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
#     model.eval()

#     # Read full test dataset into memory
#     print(f"Reading test data from {TEST_DATASET_PATH}...")
#     with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
#         all_samples = [json.loads(line) for line in f]

#     # Select random samples
#     if NUM_TEST_SAMPLES and NUM_TEST_SAMPLES < len(all_samples):
#         random.seed(RANDOM_SEED)
#         test_samples = random.sample(all_samples, NUM_TEST_SAMPLES)
#         print(f"Randomly selected {NUM_TEST_SAMPLES} samples out of {len(all_samples)} total samples (seed={RANDOM_SEED}).\n")
#     else:
#         test_samples = all_samples
#         print(f"Evaluating all {len(test_samples)} samples in dataset.\n")

#     canonical_errors = []
#     physical_errors_ms2 = []
#     parsed_count = 0

#     pbar = tqdm(test_samples, desc="Evaluating", file=sys.stdout, ascii=True)

#     for i, sample in enumerate(pbar):
#         prompt = sample["prompt"]
#         gt_u = np.array(sample["control_vector"])  # True [ux, uy, uz] in canonical units

#         messages = [
#             {"role": "system", "content": "You are an expert in astrodynamics and circular restricted three-body problems."},
#             {"role": "user", "content": prompt},
#         ]

#         input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=512,
#                 temperature=0.1,  # Low temp for deterministic numeric evaluation
#                 do_sample=False,
#             )

#         response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
#         pred_u = parse_predicted_control(response)

#         if pred_u is not None:
#             parsed_count += 1
#             # Canonical Error |u_pred - u_gt|
#             c_err = np.abs(pred_u - gt_u)
#             canonical_errors.append(c_err)

#             # Physical Error (m/s^2)
#             p_err = c_err * AU
#             physical_errors_ms2.append(p_err)

#             # --- PRINT SUCCESSFUL SAMPLES ---
#             pbar.write(f"\n--- Sample {i+1} [SUCCESS] ---")
#             pbar.write(f"Ground Truth Canonical u: {gt_u}")
#             pbar.write(f"Predicted Canonical u:    {pred_u}")
#             pbar.write(f"Absolute Error (m/s^2):   {p_err}")

#         else:
#             # --- PRINT FAILED SAMPLES AND RAW OUTPUT ---
#             pbar.write(f"\n--- Sample {i+1} [FAILED PARSE] ---")
#             pbar.write(f"Ground Truth Canonical u: {gt_u}")
#             pbar.write(f"Raw Model Output:\n{response}")

#         # Live success rate calculation on progress bar
#         success_rate = (parsed_count / (i + 1)) * 100
#         pbar.set_postfix({"Parsed": f"{parsed_count}/{i+1}", "Success Rate": f"{success_rate:.1f}%"})

#     # Summary Metrics
#     if parsed_count > 0:
#         mae_canonical = np.mean(canonical_errors, axis=0)
#         mae_physical = np.mean(physical_errors_ms2, axis=0)
        
#         print("\n" + "="*50)
#         print("EVALUATION RESULTS")
#         print("="*50)
#         print(f"Successfully Parsed Responses: {parsed_count}/{len(test_samples)} ({parsed_count/len(test_samples)*100:.1f}%)")
#         print(f"MAE Canonical Units [X, Y, Z]:  [{mae_canonical[0]:.6f}, {mae_canonical[1]:.6f}, {mae_canonical[2]:.6f}]")
#         print(f"MAE Physical (m/s^2) [X, Y, Z]: [{mae_physical[0]:.4e}, {mae_physical[1]:.4e}, {mae_physical[2]:.4e}]")
#         print(f"Total Vector Mag MAE (m/s^2):   {np.linalg.norm(mae_physical):.4e}")
#         print("="*50)
#     else:
#         print("Failed to parse numerical output from model responses. Check regex pattern.")

# if __name__ == "__main__":
#     evaluate()