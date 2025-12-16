import pandas as pd
import os
from dotenv import load_dotenv
from huggingface_hub import login

def save_mcq_submission(ids, predictions, filename="output/mcq_prediction.tsv"):
    # predictions should be a list of lists: [False, True, False, False]
    df = pd.DataFrame(predictions, columns=['A', 'B', 'C', 'D'])
    df.insert(0, 'MCQID', ids)
    df.to_csv(filename, sep='\t', index=False)
    print(f"✅ Submission saved to {filename}")


def setup_environment():
    """
    Loads environment variables and logs into Hugging Face.
    Call this at the start of your scripts.
    """
    # 1. Load the .env file
    load_dotenv()
    
    # 2. Check for Token
    hf_token = os.getenv("HF_TOKEN")
    
    if hf_token:
        print(f"✅ Found HF_TOKEN, logging in...")
        login(token=hf_token)
    else:
        print("⚠️ Warning: No HF_TOKEN found in .env. Model download might fail.")

    # 3. Optional: Setup WandB here too if you use it
    # wandb_key = os.getenv("WANDB_API_KEY")
    # if wandb_key:
    #     wandb.login(key=wandb_key)