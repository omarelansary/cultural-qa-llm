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
    """
    Validates SAQ submission file for structure, empty values, and formatting corruption.
    """
    try:
        # Load as string to check formatting strictly
        df = pd.read_csv(path, sep="\t", engine="python", dtype=str, keep_default_na=False)
    except Exception as e:
        raise ValueError(f"[SAQ FORMAT] Could not read file. Check separators/encoding. Error: {e}")

    expected = ["ID", "answer"]
    
    # 1. Column Check
    if list(df.columns) != expected:
        raise ValueError(f"[SAQ FORMAT] Expected columns {expected}, got {list(df.columns)}")

    # 2. Missing IDs
    if (df["ID"].str.strip() == "").any():
        raise ValueError("[SAQ FORMAT] Found rows with missing ID values.")

    # 3. TSV Corruption Check (Tabs/Newlines in Answer)
    # If the extractor failed to remove newlines, the TSV structure breaks.
    suspicious_chars = df["answer"].str.contains(r"[\n\t\r]", regex=True)
    if suspicious_chars.any():
        bad_ids = df.loc[suspicious_chars, "ID"].tolist()
        raise ValueError(f"[SAQ FORMAT] Answers contain illegal newlines or tabs (corrupts TSV). IDs: {bad_ids[:5]}...")

    # 4. Length Warning (Optional but recommended)
    # SAQ answers should be short. If > 50 chars, the model might be ranting.
    long_answers = df[df["answer"].str.len() > 50]
    if not long_answers.empty:
        print(f"⚠️  WARNING: {len(long_answers)} answers are longer than 50 characters. Check IDs: {long_answers['ID'].head(3).tolist()}")

    # 5. Empty Answers Check (Strictness depends on your contest rules)
    # If empty strings are allowed, you can remove this. 
    # Current setting: Warn if empty.
    empty_answers = df[df["answer"].str.strip() == ""]
    if not empty_answers.empty:
        print(f"⚠️  WARNING: {len(empty_answers)} answers are empty strings.")

    print(f"✅ Validation Passed: {len(df)} rows loaded correctly.")
    
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