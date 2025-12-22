import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.utils import setup_environment, save_mcq_submission  # Assumes you have save_saq_submission too

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
    tokenizer.pad_token = tokenizer.eos_token
    
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

def format_prompt(row, task):
    """Creates the prompt based on the task"""
    if task == "mcq":
        return f"""Question: {row['question']}
Options:
A. {row['A']}
B. {row['B']}
C. {row['C']}
D. {row['D']}
Answer with the correct letter only (A, B, C, or D).
Answer:"""
    elif task == "saq":
        return f"Question: {row['question']}\nAnswer with a short phrase only.\nAnswer:"
    return ""

def main():
    setup_environment()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["mcq", "saq"])
    parser.add_argument("--data_path", type=str, required=True, help="Path to test.tsv")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B", help="Path to trained adapter OR base model name")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--output_file", type=str, default="submission.tsv")
    args = parser.parse_args()

    # 1. Load
    model, tokenizer = load_model_for_inference(args.model_path, args.base_model)
    df = pd.read_csv(args.data_path, sep='\t')
    
    predictions = []
    ids = []

    # 2. Predict Loop
    print("🔮 Starting Inference...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = format_prompt(row, args.task)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5,  # Short generation for answers
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode answer
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the new part (remove the prompt)
        answer = generated_text[len(prompt):].strip()
        
        # Save raw answer for now (you can add post-processing logic here)
        predictions.append(answer)
        # Handle ID column name differences (MCQID vs ID)
        id_col = 'MCQID' if 'MCQID' in row else 'ID'
        ids.append(row[id_col])

    # 3. Format & Save
    if args.task == "mcq":
        # Basic logic: Check which letter appears first in the answer
        final_preds = []
        for ans in predictions:
            ans_clean = ans.upper()
            row_pred = [False, False, False, False] # A, B, C, D
            if "A" in ans_clean: row_pred[0] = True
            elif "B" in ans_clean: row_pred[1] = True
            elif "C" in ans_clean: row_pred[2] = True
            elif "D" in ans_clean: row_pred[3] = True
            else: row_pred[3] = True # Default to D if unsure (or random)
            final_preds.append(row_pred)
            
        save_mcq_submission(ids, final_preds, args.output_file)
    
    else:
        # SAQ logic (Save directly)
        out_df = pd.DataFrame({'ID': ids, 'answer': predictions})
        out_df.to_csv(args.output_file, sep='\t', index=False)
        print(f"✅ SAQ Submission saved to {args.output_file}")

if __name__ == "__main__":
    main()