#!/bin/bash
# Slurm submission script to fine-tune, predict, and build a Codabench submission.

#SBATCH --job-name=full_pipeline_lr1e-4_bs8_ep2
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --output=output/logs/%x_%j.out
#SBATCH --error=output/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hussam.chihadeh@mailbox.tu-dresden.de

set -euo pipefail

cd /home/huch135g/cultural-qa-llm
# Activate your environment.
source .venv/bin/activate

RUN_TAG="run_lr1e-4_bs8_ep2"
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
OUT_DIR="output/lora/${RUN_TAG}/"
MCQ_OUT="submission/latest/submission_mcq_${RUN_TAG}.tsv"
SAQ_OUT="submission/latest/submission_saq_${RUN_TAG}.tsv"

python src/finetune_mcq_saq_lora.py \
  --mcq_csv data/train_dataset_mcq.csv \
  --saq_csv data/train_dataset_saq.csv \
  --output_dir "${OUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --test_size 0.05 \
  --max_length 512 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-4

python -m src.predict \
  --task mcq \
  --data_path data/test_dataset_mcq.csv \
  --model_path "${OUT_DIR}" \
  --base_model "${MODEL_NAME}" \
  --output_file "${MCQ_OUT}"

python -m src.predict \
  --task saq \
  --data_path data/test_dataset_saq.csv \
  --model_path "${OUT_DIR}" \
  --base_model "${MODEL_NAME}" \
  --output_file "${SAQ_OUT}"

python -m submission.make_submission \
  --in_dir submission/latest \
  --mcq "$(basename "${MCQ_OUT}")" \
  --saq "$(basename "${SAQ_OUT}")" \
  --require_both
