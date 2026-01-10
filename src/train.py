import argparse
import ast
import inspect
import json
import os
import pathlib
import re
import sys

# Ensure project root on sys.path when running as a script (python src/train.py)
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
import wandb
from transformers import Trainer, TrainingArguments

from src.data_loader import load_raw_data, CulturalDataset
from src.model import load_llama_model
from src.utils import setup_environment

USER_PREFIX = "### User Question:\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mcq_baseline.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_system_prompt(task: str) -> str:
    prompt_file = ROOT / "prompts" / f"{task}_system_prompt.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    return ""


def combine_prompts(system_prompt: str, user_prompt: str) -> str:
    system_prompt = (system_prompt or "").strip()
    if system_prompt:
        return f"{system_prompt}\n\n{user_prompt}"
    return user_prompt


def parse_choice_dict(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}


def format_choices(choices: dict) -> str:
    if not choices:
        return ""
    lines = []
    for key in sorted(choices.keys()):
        lines.append(f"{key}. {choices[key]}")
    return "\n".join(lines)


def build_mcq_user_prompt(row) -> str:
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        question = str(row.get("question", "")).strip()
        choices = format_choices(parse_choice_dict(row.get("choices")))
        parts = [p for p in [question, choices] if p]
        prompt = "\n".join(parts)
    return f"{USER_PREFIX}{prompt}\n\n"


def build_saq_user_prompt(row) -> str:
    q_native = str(row.get("question", "")).strip()
    q_en = str(row.get("en_question", "")).strip()
    question = q_en or q_native
    return f"{USER_PREFIX}{question}\n\nAnswer: "


def build_prompt_only(row, task: str, system_prompt: str) -> str:
    if task == "mcq":
        user_prompt = build_mcq_user_prompt(row)
    elif task == "saq":
        user_prompt = build_saq_user_prompt(row)
    else:
        raise ValueError(f"Unknown task: {task}")
    return combine_prompts(system_prompt, user_prompt)


def pick_saq_target_from_annotations(annotations_str: str) -> str:
    try:
        anns = ast.literal_eval(annotations_str)
    except (ValueError, SyntaxError):
        return ""

    if not anns:
        return ""

    best = max(anns, key=lambda d: d.get("count", 0))

    en_answers = best.get("en_answers") or []
    answers = best.get("answers") or []

    if isinstance(en_answers, list) and en_answers and en_answers[0]:
        return str(en_answers[0]).strip()
    if isinstance(answers, list) and answers and answers[0]:
        return str(answers[0]).strip()

    return ""


def format_mcq_target(value) -> str:
    letter = str(value or "").strip().upper()
    if not letter:
        return ""
    return json.dumps({"answer_choice": letter})


def build_training_text(row, task: str, system_prompt: str) -> str:
    prompt_only = build_prompt_only(row, task, system_prompt)

    if task == "mcq":
        target = format_mcq_target(row.get("answer_idx", ""))
    elif task == "saq":
        target = pick_saq_target_from_annotations(row.get("annotations", ""))
    else:
        raise ValueError(f"Unknown task: {task}")

    return f"{prompt_only}{target}"


def should_use_wandb(report_to) -> bool:
    if isinstance(report_to, str):
        return report_to.lower() == "wandb"
    if isinstance(report_to, (list, tuple)):
        return "wandb" in report_to
    return False


def build_training_args(**kwargs) -> TrainingArguments:
    allowed = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**filtered)


def split_train_eval(df, eval_split: float, seed: int):
    if eval_split <= 0 or len(df) == 0:
        return df, None
    eval_size = int(len(df) * eval_split)
    eval_size = max(1, eval_size)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    eval_df = df.iloc[:eval_size].reset_index(drop=True)
    train_df = df.iloc[eval_size:].reset_index(drop=True)

    if len(train_df) == 0:
        return df, None

    return train_df, eval_df


def extract_mcq_choice(text: str):
    txt = (text or "").strip()

    m = re.search(r"\{.*?\}", txt, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "answer_choice" in obj:
                val = str(obj["answer_choice"]).strip().upper()
                if val and val[0] in "ABCD":
                    return val[0]
        except Exception:
            pass

    m = re.search(r"\b([ABCD])\b", txt.upper())
    if m:
        return m.group(1)

    return None


def extract_saq_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""

    match = re.search(r"answer\s*:\s*(.*)", text, re.IGNORECASE)
    if match:
        clean_text = match.group(1)
    else:
        clean_text = text

    clean_text = clean_text.split("\n")[0].strip()
    if clean_text.endswith("."):
        clean_text = clean_text[:-1]
    clean_text = re.split(r"\s+(?:or|/)\s+", clean_text, flags=re.IGNORECASE)[0]

    return clean_text.strip()


def normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def compute_mcq_accuracy(pred_texts, gold_letters) -> float:
    if not gold_letters:
        return 0.0
    correct = 0
    for pred, gold in zip(pred_texts, gold_letters):
        letter = extract_mcq_choice(pred)
        if letter and gold and letter == gold:
            correct += 1
    return correct / len(gold_letters)


def compute_exact_match(pred_texts, gold_texts) -> float:
    if not gold_texts:
        return 0.0
    correct = 0
    for pred, gold in zip(pred_texts, gold_texts):
        pred_norm = normalize_answer(extract_saq_answer(pred))
        gold_norm = normalize_answer(gold)
        if pred_norm == gold_norm:
            correct += 1
    return correct / len(gold_texts)


def generate_predictions(model, tokenizer, rows, task, system_prompt, max_new_tokens: int):
    model.eval()
    device = next(model.parameters()).device
    preds = []

    with torch.no_grad():
        for row in rows:
            prompt_only = build_prompt_only(row, task, system_prompt)
            inputs = tokenizer(prompt_only, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
            preds.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

    return preds


def main():
    setup_environment()
    args = parse_args()
    config = load_config(args.config)

    train_args = dict(config.get("train_args", {}))
    report_to = train_args.pop("report_to", config.get("report_to", "wandb"))

    if should_use_wandb(report_to):
        wandb.init(
            project="llama-cultural-qa",
            config=config,
            name=config.get("run_name"),
            mode="offline",
        )

    task = str(config.get("task", "mcq")).lower()
    system_prompt = load_system_prompt(task)

    print("Step 1: Loading Model...")
    model, tokenizer = load_llama_model(
        config["model_name"],
        quantization=config.get("quantization", "4bit"),
        lora=config.get("lora"),
    )

    print("Step 2: Loading Data...")
    raw_data = load_raw_data(config["data_path"])
    max_length = int(config.get("max_length", train_args.get("max_length", 512)))

    seed = int(train_args.get("seed", config.get("seed", 42)))
    train_args.setdefault("seed", seed)
    eval_split = float(config.get("eval_split", 0.05))
    train_df, eval_df = split_train_eval(raw_data, eval_split, seed)

    train_dataset = CulturalDataset(
        train_df,
        tokenizer,
        max_length=max_length,
        build_text_fn=lambda row: build_training_text(row, task, system_prompt),
        build_prompt_fn=lambda row: build_prompt_only(row, task, system_prompt),
    )

    eval_dataset = None
    if eval_df is not None:
        eval_dataset = CulturalDataset(
            eval_df,
            tokenizer,
            max_length=max_length,
            build_text_fn=lambda row: build_training_text(row, task, system_prompt),
            build_prompt_fn=lambda row: build_prompt_only(row, task, system_prompt),
        )

    if eval_dataset is not None:
        train_args.setdefault("evaluation_strategy", "steps")
        train_args.setdefault("eval_steps", train_args.get("logging_steps", 50))

    print("Step 3: Training...")
    training_args = build_training_args(
        output_dir=config["output_dir"],
        report_to=report_to,
        run_name=config.get("run_name"),
        **train_args,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train()

    print("Step 4: Saving...")
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    eval_metrics = None
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        metrics_path = os.path.join(config["output_dir"], "eval_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)

        eval_max_samples = config.get("eval_max_samples")
        eval_rows = eval_df.to_dict(orient="records")
        if eval_max_samples is not None:
            eval_rows = eval_rows[: int(eval_max_samples)]

        eval_max_new_tokens = int(config.get("eval_max_new_tokens", 64))
        preds = generate_predictions(
            model,
            tokenizer,
            eval_rows,
            task,
            system_prompt,
            eval_max_new_tokens,
        )

        if task == "mcq":
            gold = [str(row.get("answer_idx", "")).strip().upper() for row in eval_rows]
            metrics = {
                "mcq_accuracy": compute_mcq_accuracy(preds, gold),
                "num_eval": len(gold),
            }
        else:
            gold = [pick_saq_target_from_annotations(row.get("annotations", "")) for row in eval_rows]
            metrics = {
                "saq_exact_match": compute_exact_match(preds, gold),
                "num_eval": len(gold),
            }

        metrics["eval_loss"] = eval_metrics.get("eval_loss") if eval_metrics else None
        gen_metrics_path = os.path.join(config["output_dir"], "gen_eval_metrics.json")
        with open(gen_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    if should_use_wandb(report_to):
        wandb.finish()


if __name__ == "__main__":
    main()
