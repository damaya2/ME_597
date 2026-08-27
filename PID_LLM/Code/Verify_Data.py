import os
import json
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

LU_TO_KM = 384400.0
MU = 0.01215058560962404
EARTH_X_KM = -MU * LU_TO_KM
MOON_X_KM = (1.0 - MU) * LU_TO_KM

def verify_full_trajectories_from_dataset(num_trajectories_to_plot=6, output_dir="dataset_plots"):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob("crtbp_llm_dataset_*.jsonl"))
    if not files:
        print("No dataset files found!")
        return
    
    filename = files[-1]
    print(f"Loading generated dataset: {filename}")

    trajectories = defaultdict(list)
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            traj_id = sample.get("trajectory_id", 0)
            trajectories[traj_id].append(sample)

    unique_ids = list(trajectories.keys())
    print(f"Total complete trajectories found in dataset: {len(unique_ids)}")

    selected_ids = random.sample(unique_ids, min(num_trajectories_to_plot, len(unique_ids)))

    fig = plt.figure(figsize=(16, 10))

    for idx, traj_id in enumerate(selected_ids, 1):
        traj_samples = trajectories[traj_id]
        traj_samples = sorted(traj_samples, key=lambda s: s["progress_pct"])

        pos_history = np.array([s["debug_pos_km"] for s in traj_samples])
        target_history = np.array([s["debug_target_km"] for s in traj_samples])
        
        m_type = traj_samples[0]["mission_type"]
        kick_step = traj_samples[0].get("kick_step", None)

        ax = fig.add_subplot(2, 3, idx, projection='3d')

        # Reference and Controlled Trajectories
        ax.plot(target_history[:, 0], target_history[:, 1], target_history[:, 2], 
                'k--', label='Reference Path', alpha=0.7)

        ax.plot(pos_history[:, 0], pos_history[:, 1], pos_history[:, 2], 
                'r-', label='Actual Flight Path', alpha=0.8)

        # Start and End Markers
        ax.scatter([pos_history[0, 0]], [pos_history[0, 1]], [pos_history[0, 2]], 
                   color='green', s=40, label='Start')
        ax.scatter([pos_history[-1, 0]], [pos_history[-1, 1]], [pos_history[-1, 2]], 
                   color='darkred', s=40, label='End')

        # Celestial Bodies in Barycentric Synodic Space
        ax.scatter([EARTH_X_KM], [0], [0], color='blue', s=60, label='Earth')
        ax.scatter([MOON_X_KM], [0], [0], color='gray', s=40, label='Moon')

        # Safe aspect ratio bounding box calculation
        all_pts = np.vstack([pos_history, target_history])
        x_min, x_max = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
        y_min, y_max = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])
        z_min, z_max = np.min(all_pts[:, 2]), np.max(all_pts[:, 2])

        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        if max_range < 1e-3:
            max_range = 1.0

        x_mid = 0.5 * (x_max + x_min)
        y_mid = 0.5 * (y_max + y_min)
        z_mid = 0.5 * (z_max + z_min)

        ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
        ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
        ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])
        ax.set_box_aspect([1, 1, 1])

        kick_info = f" | Kick Step: {kick_step}" if kick_step is not None else ""
        ax.set_title(f"Traj ID {traj_id}: {m_type}\n({len(traj_samples)} steps{kick_info})", fontsize=9)
        ax.set_xlabel("X (km)", fontsize=8)
        ax.set_ylabel("Y (km)", fontsize=8)
        ax.set_zlabel("Z (km)", fontsize=8)
        ax.legend(fontsize=6, loc='upper right')

    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "verified_trajectories.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Verification plots successfully saved to: {save_path}")

if __name__ == "__main__":
    verify_full_trajectories_from_dataset(num_trajectories_to_plot=6)