import argparse
import pathlib
import sys

# Ensure project root on sys.path when running as a script (python src/train.py)
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import wandb  # <--- 1. IMPORT WANDB
from src.data_loader import load_raw_data, CulturalDataset
from src.model import load_llama_model
from src.utils import setup_environment
from transformers import Trainer, TrainingArguments

def main():
    # 1. SETUP ENV (Login to HF and WandB)
    setup_environment()

    # 2. LOAD CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mcq_baseline.yaml")
    # ... your other args
    args = parser.parse_args()
    
    # Load YAML
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # <--- 3. INITIALIZE WANDB HERE --->
    # This creates the project dashboard before training starts
    wandb.init(
        project="llama-cultural-qa", # Matches your W&B project name
        config=config,               # Log your config so you know what params you used
        name=config.get("run_name"),  # Optional: name this specific run
        mode="offline"  # It stops the "Network Error" since nodes on HPC maybe not connected to the internet
    )

    # NEW (CORRECT) ORDER
    print("Step 1: Loading Model...")
    model, tokenizer = load_llama_model(config['model_name'])

    print("Step 2: Loading Data...")
    raw_data = load_raw_data(config['data_path']) 
    # Now pass the tokenizer to the dataset!
    dataset = CulturalDataset(raw_data, tokenizer)

    print("Step 3: Training...")
    
    # <--- 4. TELL TRAINER TO REPORT TO WANDB --->
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        report_to="wandb",  # <--- CRITICAL: This turns on auto-logging
        run_name="my-llama-run",
        **config['train_args']
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args
    )
    
    trainer.train()

    print("Step 4: Saving...")
    model.save_pretrained("artifacts/final_model")
    
    # <--- 5. FINISH RUN --->
    wandb.finish()

if __name__ == "__main__":
    main()
