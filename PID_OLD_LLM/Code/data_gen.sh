#!/bin/bash
#SBATCH --account=bhsm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=data_gen
#SBATCH --output=logs/data_gen_%j.out
#SBATCH --error=logs/data_gen_%j.err

mkdir -p logs

echo "Starting CRTBP Data Generation on $(hostname) at $(date)"
python3 -u Dataset_Generator.py
echo "Job finished at $(date)"
