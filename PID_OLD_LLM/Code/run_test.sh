#!/bin/bash
#SBATCH --account=bhsm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=test_pipeline
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# Create required output directories matching Test.py specifications
mkdir -p logs
mkdir -p test_output_results

# Ensure Matplotlib renders headlessly on cluster nodes
export MPLBACKEND=Agg

echo "================================================="
echo "Starting CRTBP Pipeline on $(hostname) at $(date)"
echo "================================================="

# Execute pipeline (removed unsupported --fast flag)
# You can pass --pert-delay <hours> if you wish to override the default 500.0 hrs
python3 -u Test.py

# Check execution success
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
