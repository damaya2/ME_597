#!/bin/bash
#SBATCH --account=bhsm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --job-name=generate_dataset
#SBATCH --output=logs/data_gen_%j.out
#SBATCH --error=logs/data_gen_%j.err

mkdir -p logs
mkdir -p dataset_plots

export MPLBACKEND=Agg

echo "================================================="
echo "Starting Dataset Generation on $(hostname) at $(date)"
echo "================================================="

python3 -u Dataset_Generator.py

LATEST_DATASET=$(ls -t crtbp_llm_dataset_*.jsonl 2>/dev/null | head -n 1)

if [ -n "$LATEST_DATASET" ]; then
    echo "================================================="
    echo "Dataset generation finished successfully!"
    echo "Generated output dataset: $LATEST_DATASET"
    echo "Running trajectory verification..."
    echo "================================================="
    
    python3 -u Verify_Data.py
    
    echo "================================================="
    echo "Verification complete!"
    echo "Visualizations stored in: dataset_plots/"
    echo "================================================="
else
    echo "[!] ERROR: No 'crtbp_llm_dataset_*.jsonl' file was generated."
    exit 1
fi

echo "Job completed at $(date)"
