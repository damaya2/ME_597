#!/bin/bash
#SBATCH --account=bhsm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=llama3_sft_v3
#SBATCH --output=logs/sft_v3_%j.out
#SBATCH --error=logs/sft_v3_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Environment & Authentication Setting# Environment & Authentication Settings
export HF_TOKEN="your_huggingface_token_here"
export HUGGING_FACE_HUB_TOKEN="your_huggingface_token_here"
export TRANSFORMERS_ALLOW_INSECURE_PYTORCH_LOAD=1

# Print Node Info
echo "Starting Fresh Llama-3.1-8B-Instruct 3-Epoch SFT on $(hostname) at $(date)"
echo "GPU allocated:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Execute Training Script across all 4 GPUs
torchrun --nproc_per_node=4 Training_LLM.py

echo "Job finished at $(date)"
