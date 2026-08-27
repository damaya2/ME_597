#!/bin/bash
#SBATCH --account=bhsm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=test_pipeline
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# 1. Ensure required output directories exist
mkdir -p logs
mkdir -p test_output_results

# 2. Activate environment (uncomment and adjust if using conda/venv)
# source /path/to/venv/bin/activate
# conda activate my_env

# 3. Environment variables for headless Matplotlib and CPU thread management
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "================================================"
echo "Starting Test Pipeline on $(hostname) at $(date)"
echo "================================================"

# 4. Execute Test script
python3 -u Test.py

# 5. Output Verification
if [ -f "sim_results_multi.npy" ]; then
    echo "================================================="
    echo "Pipeline execution finished successfully!"
    echo "Generated output array: sim_results_multi.npy"
    echo "Visualizations stored in: test_output_results/"
    echo "================================================="
else
    echo "[!] ERROR: sim_results_multi.npy was not generated."
    exit 1
fi

echo "Job completed at $(date)"
