#!/bin/bash
#SBATCH --account=bhsm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=sim_llama3
#SBATCH --output=logs/sim_llama_%j.out
#SBATCH --error=logs/sim_llama_%j.err

# Create logs directory if it does not exist
mkdir -p logs

# Environment Variables
export HF_TOKEN="your_huggingface_token_here"
export HUGGING_FACE_HUB_TOKEN="your_huggingface_token_here"
export TRANSFORMERS_ALLOW_INSECURE_PYTORCH_LOAD=1
export MPLBACKEND=Agg

echo "================================================="
echo "Starting CRTBP Guidance Simulation on $(hostname) at $(date)"
echo "================================================="

# 1. Run Controller Simulation
python3 -u LLM_Controller.py

# Check if simulation output exists before triggering visualizer
if [ -f "eval_comparison_results.json" ] || [ -f "llm_evaluation_results.json" ]; then
    echo "================================================="
    echo "Simulation finished successfully. Generating plots..."
    echo "================================================="
    
    # 2. Run Plotting and Animation Generation
    python3 -u LLM_Trajectories.py

    echo "Visualization pipeline completed successfully!"
else
    echo "[!] ERROR: Neither eval_comparison_results.json nor llm_evaluation_results.json was generated."
    exit 1
fi

echo "Job finished at $(date)"
