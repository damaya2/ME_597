#!/bin/bash
#SBATCH --account=bhsm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=sim_llama3_crtbp
#SBATCH --output=logs/sim_llama_%j.out
#SBATCH --error=logs/sim_llama_%j.err

# Create logs folder if it doesn't exist
mkdir -p logs

# Environment variables
export HF_TOKEN="your_huggingface_token_here"
export HUGGING_FACE_HUB_TOKEN="your_huggingface_token_here"
export TRANSFORMERS_ALLOW_INSECURE_PYTORCH_LOAD=1
export MPLBACKEND=Agg

echo "================================================="
echo "Starting CRTBP Guidance Simulation on $(hostname) at $(date)"
echo "================================================="

# 1. Run Simulation Pipeline
python3 -u LLM_Guidance.py

# Check if simulation completed and produced results
if [ -f "sim_results_multi.npy" ] || [ -f "sim_results.npy" ]; then
    echo "================================================="
    echo "Simulation finished successfully. Generating plots..."
    echo "================================================="
    
    # 2. Run Visualization Script
    python3 -u LLM_Trajectory.py
    
    echo "Visualization pipeline completed!"
else
    echo "[!] ERROR: Neither sim_results_multi.npy nor sim_results.npy was generated."
    exit 1
fi

echo "Job finished at $(date)"
