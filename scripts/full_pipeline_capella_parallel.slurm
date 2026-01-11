#!/bin/bash
# Slurm submission script to fine-tune, predict, and build a Codabench submission.

#SBATCH --job-name=full_pipeline_parallel_lr1e-4_bs8_ep2
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --output=output/logs/%x_%j.out
#SBATCH --error=output/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=omar_essameldeen_ahmad.elansary@mailbox.tu-dresden.de

set -euo pipefail

cd /home/omel305g/cultural-qa-llm
# Activate your environment.
source .venv/bin/activate

set -a
source .env
set +a

# Set caches to a persistent location to avoid re-downloading.
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=/home/omel305g/cultural-qa-llm/.hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

python -c "import os; print('HF_TOKEN set:', bool(os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')))"


export WANDB_MODE=offline
export WANDB_PROJECT=cultural_qa_llm
export WANDB_DIR=/home/omel305g/cultural-qa-llm/output/wandb
mkdir -p "$WANDB_DIR"


echo "===== ENVIRONMENT READY ====="

RUN_TAG="parallel_lr1e-4_bs8_ep2"
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
FT_FINAL_OUT_DIR="output/lora/${RUN_TAG}/mcq_saq/"
MCQ_OUT="submission/latest/submission_mcq_${RUN_TAG}.tsv"
SAQ_OUT="submission/latest/submission_saq_${RUN_TAG}.tsv"

has_file () { test -f "$1"; }

# Ensure directories exist
mkdir -p output/logs
mkdir -p "${FT_FINAL_OUT_DIR}"
mkdir -p submission/latest

echo "===== Directories READY ====="



# -------------------------
# Stage 1: SAQ fine-tuning starting from MCQ adapter
# -------------------------
echo "===== STARTING STAGE 1: MCQ and SAQ Parallel TRAINING ====="
if has_file "${FT_FINAL_OUT_DIR}/adapter_config.json"; then
  echo "Final adapter already exists -> skipping SAQ training: ${FT_FINAL_OUT_DIR}"
else
  echo "Final adapter not found -> starting SAQ training"
  python src/finetune_mcq_saq_lora_multi.py \
    --mcq_csv data/train_dataset_mcq.csv \
    --saq_csv data/train_dataset_saq.csv \
    --output_dir "${FT_FINAL_OUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --task both \
    --test_size 0.05 \
    --max_length 512 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4
fi

echo "===== Finished STAGE 1: MCQ and SAQ Parallel TRAINING ====="

# -------------------------
# Prediction using FINAL adapter (MCQ->SAQ)
# -------------------------
echo "===== STARTING STAGE 2: MCQ PREDICTION ====="
if has_file "${MCQ_OUT}"; then
  echo "MCQ predictions already exist -> skipping: ${MCQ_OUT}"
else
  echo "MCQ predictions not found -> starting prediction"
  python -m src.predict \
    --task mcq \
    --data_path data/test_dataset_mcq.csv \
    --model_path "${FT_FINAL_OUT_DIR}" \
    --base_model "${MODEL_NAME}" \
    --output_file "${MCQ_OUT}"
fi

echo "===== Finished STAGE 2: MCQ PREDICTION ====="

echo "===== STARTING STAGE 3: SAQ PREDICTION ====="
if has_file "${SAQ_OUT}"; then
  echo "SAQ predictions already exist -> skipping: ${SAQ_OUT}"
else
  echo "SAQ predictions not found -> starting prediction"
  python -m src.predict \
  --task saq \
  --data_path data/test_dataset_saq.csv \
  --model_path "${FT_FINAL_OUT_DIR}" \
  --base_model "${MODEL_NAME}" \
  --output_file "${SAQ_OUT}"
fi


echo "===== Finished STAGE 3: SAQ PREDICTION ====="
echo "===== STARTING STAGE 4: BUILDING SUBMISSION ====="
# -------------------------
# Build submission
# -------------------------
python -m submission.make_submission \
  --in_dir submission/latest \
  --mcq "$(basename "${MCQ_OUT}")" \
  --saq "$(basename "${SAQ_OUT}")" \
  --require_both

echo "===== Finished STAGE 4: SUBMISSION READY ====="
echo "===== FULL PIPELINE COMPLETED SUCCESSFULLY ====="
