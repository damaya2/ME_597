#!/bin/bash
#SBATCH --account=bhsm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval_llama
#SBATCH --output=logs/eval_llama_%j.out
#SBATCH --error=logs/eval_llama_%j.err

mkdir -p logs

export HF_TOKEN="your_huggingface_token_here"
export HUGGING_FACE_HUB_TOKEN="your_huggingface_token_here"

echo "Starting evaluation on $(hostname) at $(date)"
python3 -u test.py
echo "Evaluation finished at $(date)"
