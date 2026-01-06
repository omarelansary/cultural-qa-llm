#!/bin/bash
#SBATCH --job-name=saq_zeroshot_prompt_tuned
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=output/logs/%x_%j.out
#SBATCH --error=output/logs/%x_%j.err

source .venv/bin/activate

# Run the python script with arguments
python -m src.predict --task saq --data_path data/test_dataset_saq.csv --model_path meta-llama/Meta-Llama-3-8B --base_model meta-llama/Meta-Llama-3-8B --output_file submission/latest/submission_saq_zs_1.tsv

echo "Job finished! Check output/logs/ for results."