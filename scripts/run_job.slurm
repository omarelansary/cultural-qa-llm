#!/bin/bash
#SBATCH --job-name=llama_finetune
#SBATCH --output=output/logs/%j.out  # Automate logging: %j is job ID
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=4

source activate llama_project

# Run the python script with arguments
python src/train.py --task mcq --learning_rate 2e-5 --batch_size 8

echo "Job finished! Check output/logs/ for results."