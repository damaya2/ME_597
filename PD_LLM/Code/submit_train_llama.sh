#!/bin/bash
#SBATCH --account=bhsm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=llama3_crtbp
#SBATCH --output=logs/sft_llama_%j.out
#SBATCH --error=logs/sft_llama_%j.err

mkdir -p logs

# Wipe stale token cache
rm -rf ~/.cache/huggingface/token

export HF_TOKEN="your_huggingface_token_here"
export HUGGING_FACE_HUB_TOKEN="your_huggingface_token_here"
export TRANSFORMERS_ALLOW_INSECURE_PYTORCH_LOAD=1

echo "Starting Llama-3.1-8B-Instruct SFT on $(hostname) at $(date)"
python3 -u Natural_Training.py
echo "Job finished at $(date)"
