import argparse
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import re
from .utils import setup_environment, save_mcq_submission, validate_saq_submission, validate_mcq_submission

def load_model_for_inference(model_path, base_model_name):
    """
    Smart loader:
    - If model_path == base_model_name -> Loads Base Model (Zero-Shot)
    - If model_path is a folder -> Loads Base Model + LoRA Adapter (Fine-Tuned)
    """
    print(f"🔄 Loading Base Model: {base_model_name}...")
    
    # 1. Load Base Model (Quantized for memory)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 2. Check if we need to load LoRA adapters
    if model_path != base_model_name:
        print(f"🔧 Loading LoRA Adapters from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print("🚀 Using Pure Base Model (Zero-Shot Mode)")
        model = base_model

    return model, tokenizer

def load_system_prompt(task):
    """Load system prompt from prompts/ directory"""
    BASE_DIR = Path(__file__).resolve().parents[1]  # project root if predict.py is in src/
    prompt_file = BASE_DIR / "prompts" / f"{task}_system_prompt.txt"
    
    if prompt_file.exists():
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content + "\n\n" if content else ""
    return ""

def get_saq_question_text(row, mode="auto"):
    """Make the code support three modes (so it’s not confusing and you can align with eval):

    - auto (recommended default): use en_question if present else question
    - en: always use en_question (fallback to question if missing)
    - native: always use question (fallback to en_question if missing)"""

    q_native = str(row["question"]).strip() if "question" in row and pd.notna(row["question"]) else ""
    q_en = str(row["en_question"]).strip() if "en_question" in row and pd.notna(row["en_question"]) else ""

    if mode == "en":
        return q_en or q_native
    if mode == "native":
        return q_native or q_en

    # mode == "auto"
    return q_en or q_native

def format_prompt(row, task, saq_mode="auto"):
    """Creates the prompt based on the task (robust to your actual schemas)."""
    if task == "mcq":
        # MCQ file has a preformatted 'prompt'
        if "prompt" not in row:
            raise KeyError(f"[MCQ] Missing 'prompt'. Available columns: {list(row.index)}")
        return row["prompt"]
    elif task == "saq":
        # SAQ file has 'question' and 'en_question'
        q = get_saq_question_text(row, mode=saq_mode)
        if not q:
            raise KeyError(f"[SAQ] Missing usable question text. Available columns: {list(row.index)}")
        return f"{q}\nAnswer:"

    return ""

def build_model_inputs(tokenizer, user_prompt: str, system_prompt: str = ""):
    """
    Best practice:
    - If tokenizer supports chat templates, use them (system/user roles).
    - Otherwise, fall back to plain text concatenation.
    Returns tokenized inputs ready for model.generate.
    """
    system_prompt = (system_prompt or "").strip()
    user_prompt = (user_prompt or "").strip()

    has_template = getattr(tokenizer, "chat_template", None)
    if has_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return tokenizer(rendered, return_tensors="pt")
    else:
        # Base-model fallback: treat "system prompt" as an instruction prefix
        if system_prompt:
            text = f"{system_prompt}\n\n{user_prompt}"
        else:
            text = user_prompt
        return tokenizer(text, return_tensors="pt")

def decode_new_tokens(tokenizer, inputs, outputs):
    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def extract_mcq_choice(text: str) -> str:
    txt = (text or "").strip()

    # Try parse JSON object anywhere in output
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "answer_choice" in obj:
                val = str(obj["answer_choice"]).strip().upper()
                if val and val[0] in "ABCD":
                    return val[0]
        except Exception:
            pass

    # Fallback: first standalone A/B/C/D
    m = re.search(r"\b([ABCD])\b", txt.upper())
    if m:
        return m.group(1)

    return "D"


def main():
    setup_environment()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["mcq", "saq"])
    parser.add_argument("--saq_question_field",type=str,default="auto",choices=["auto", "en", "native"],help="SAQ: auto=en_question if available else question; en=prefer en_question; native=prefer question.")
    parser.add_argument("--data_path", type=str, required=True, help="data/test_dataset_mcq.csv")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B", help="Path to trained adapter OR base model name")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--output_file", type=str, default="submission.tsv")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N rows for quick tests")
    args = parser.parse_args()

    # 1. Load
    model, tokenizer = load_model_for_inference(args.model_path, args.base_model)
    df = pd.read_csv(args.data_path, sep=None, engine="python")
    
    if args.limit is not None:
        df = df.head(args.limit)
    
    system_prompt = load_system_prompt(args.task)
    
    print("Loaded columns:", list(df.columns))

    if args.task == "mcq" and "prompt" not in df.columns:
        raise ValueError(f"--task mcq but dataset has no 'prompt' column. Columns: {list(df.columns)}")

    if args.task == "saq" and ("question" not in df.columns and "en_question" not in df.columns):
        raise ValueError(f"--task saq but dataset has no 'question'/'en_question' column. Columns: {list(df.columns)}")
    

    predictions = []
    ids = []

    # 2. Predict Loop
    print("🔮 Starting Inference...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        user_prompt = format_prompt(row, args.task, saq_mode=args.saq_question_field)
        inputs = build_model_inputs(tokenizer, user_prompt, system_prompt=system_prompt)
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Extract the new part (remove the prompt)
        answer = decode_new_tokens(tokenizer, inputs, outputs)
        
        # Save raw answer for now (you can add post-processing logic here)
        predictions.append(answer)
        # Handle ID column name differences (MCQID vs ID)
        id_col = "MCQID" if "MCQID" in row.index else "ID"
        ids.append(row[id_col])

    # 3. Format & Save
    if args.task == "mcq":
        # Basic logic: Check which letter appears first in the answer
        final_preds = []
        for ans in predictions:
            choice = extract_mcq_choice(ans)
            final_preds.append([choice == "A", choice == "B", choice == "C", choice == "D"])
            
        save_mcq_submission(ids, final_preds, args.output_file)
        validate_mcq_submission(args.output_file, expected_id_col="MCQID")
    
    else:
        # SAQ logic (Save directly)
        out_df = pd.DataFrame({'ID': ids, 'answer': predictions})
        out_df.to_csv(args.output_file, sep='\t', index=False, columns=["ID", "answer"], lineterminator="\n")
        validate_saq_submission(args.output_file)
        print(f"✅ SAQ Submission saved to {args.output_file}")


if __name__ == "__main__":
    main()