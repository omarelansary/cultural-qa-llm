#!/bin/bash
# Slurm submission script for LoRA fine-tuning on an HPC cluster.
# Update partition/module/paths for your site before submitting.

#SBATCH --job-name=finetune_mcq_saq_lora
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# Request a modest amount of RAM to reduce queue time.
# Start low and increase only if the job OOMs.
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --output=output/logs/%x_%j.out
#SBATCH --error=output/logs/%x_%j.err

set -euo pipefail

# 1) Move to repo root (important when Slurm runs from $HOME).
cd /home/omel305g/cultural-qa-llm

# 2) Load modules (update for your site). Use `module spider Python` to find versions.
# module purge
# module load <required_modules>
# module load Python/<python_version>

# # 3) Activate the virtual environment created earlier.
# source .venv/bin/activate

# 4) Optional: set caches to local scratch to reduce shared FS load.
# export HF_HOME=${SLURM_TMPDIR:-/tmp}/hf_cache
# export TRANSFORMERS_CACHE=$HF_HOME
# mkdir -p "$HF_HOME"

# 5) Run fine-tuning.
python src/finetune_mcq_saq_lora.py \
  --mcq_csv data/train_dataset_mcq.csv \
  --saq_csv data/train_dataset_saq.csv \
  --output_dir output/lora/run_lr2e-4_bs4_ep1/ \
  --model_name meta-llama/Meta-Llama-3-8B \
  --test_size 0.05 \
  --max_length 512 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-4
