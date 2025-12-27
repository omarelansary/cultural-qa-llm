import pandas as pd
import os
from dotenv import load_dotenv
from huggingface_hub import login

def save_mcq_submission(ids, predictions, filename="output/mcq_prediction.tsv", id_col="MCQID"):
    """
    predictions: list of [bool,bool,bool,bool] corresponding to A,B,C,D.
    Writes a deterministic TSV with fixed schema: <id_col>, A, B, C, D
    """
    if len(ids) != len(predictions):
        raise ValueError(f"[MCQ SAVE] ids length ({len(ids)}) != predictions length ({len(predictions)})")

    df = pd.DataFrame(predictions, columns=["A", "B", "C", "D"])
    df.insert(0, id_col, ids)

    # Validate exactly one True per row before saving
    counts = df[["A", "B", "C", "D"]].astype(int).sum(axis=1)
    if not (counts == 1).all():
        bad = int((counts != 1).sum())
        raise ValueError(f"[MCQ SAVE] {bad} rows do not have exactly one True")

    df.to_csv(
        filename,
        sep="\t",
        index=False,
        columns=[id_col, "A", "B", "C", "D"],
        lineterminator="\n"
    )
    print(f"✅ Submission saved to {filename}")


def validate_saq_submission(path: str):
    df = pd.read_csv(path, sep="\t", engine="python", keep_default_na=False, na_filter=False)
    expected = ["ID", "answer"]
    if list(df.columns) != expected:
        raise ValueError(f"[SAQ FORMAT] Expected {expected}, got {list(df.columns)}")
    if (df["ID"].astype(str).str.strip() == "").any():
        raise ValueError("[SAQ FORMAT] Missing ID values")
    if df["answer"].isna().any():
        raise ValueError("[SAQ FORMAT] NaN answers present (empty string ok, NaN not)")

def validate_mcq_submission(path: str, expected_id_col="MCQID"):
    df = pd.read_csv(path, sep="\t", engine="python")
    expected_cols = [expected_id_col, "A", "B", "C", "D"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"[MCQ FORMAT] Expected {expected_cols}, got {list(df.columns)}")

    # must be exactly one True per row
    abcd = df[["A","B","C","D"]]
    # allow strings "True"/"False" or 0/1
    abcd = abcd.replace({"True": True, "False": False, "true": True, "false": False})
    
    if df[expected_id_col].isna().any():
        raise ValueError("[MCQ FORMAT] Missing MCQID values")
    
    abcd = abcd.astype(int)
    counts = abcd.sum(axis=1)

    if not (counts == 1).all():
        bad = int((counts != 1).sum())
        raise ValueError(f"[MCQ FORMAT] {bad} rows do not have exactly one marked option")


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
        try:
            print("✅ Found HF_TOKEN, logging in...")
            login(token=hf_token)
        except Exception as e:
            print(f"⚠️ HF login failed: {e}")
    else:
        print("⚠️ Warning: No HF_TOKEN found in .env. Model download might fail.")

    # 3. Optional: Setup WandB here too if you use it
    # wandb_key = os.getenv("WANDB_API_KEY")
    # if wandb_key:
    #     wandb.login(key=wandb_key)