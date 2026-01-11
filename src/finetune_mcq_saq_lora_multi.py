#!/usr/bin/env python3
"""
LoRA fine-tuning (canonical SAQ + MCQ) with:
- Load MCQ/SAQ CSVs
- Build prompts
- Canonical SAQ target (most frequent; deterministic tie-break)
- Train/val split
- Tokenization with LOSS MASKING (train only on answer tokens)
- PEFT LoRA + HF Trainer

Usage example:
single run, mixed tasks
python finetune_mcq_saq_lora_multi.py \
  --mcq_csv train_dataset_mcq.csv \
  --saq_csv train_dataset_saq.csv \
  --task both \
  --model_name meta-llama/Meta-Llama-3-8B \
  --output_dir lora/mixed_mcq_saq \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-4

Separate runs:  
MCQ fine-tuning (base → MCQ adapter)
    python finetune_mcq_saq_lora_multi.py \
    --mcq_csv train_dataset_mcq.csv \
    --saq_csv train_dataset_saq.csv \
    --output_dir lora/latest/ \
    --task mcq \
    --model_name meta-llama/Meta-Llama-3-8B \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4

SAQ fine-tuning (MCQ → MCQ+SAQ adapter)
    python finetune_mcq_saq_lora_multi.py \
    --mcq_csv train_dataset_mcq.csv \
    --saq_csv train_dataset_saq.csv \
    --output_dir lora/latest/ \
    --task saq \
    --model_name meta-llama/Meta-Llama-3-8B \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4
"""

import argparse
import ast
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mcq_csv", type=str, required=True)
    p.add_argument("--saq_csv", type=str, required=True)
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default=None)

    p.add_argument("--task", choices=["mcq", "saq", "both"], default="both")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.05)

    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)

    p.add_argument("--max_length", type=int, default=512)

    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=10)

    # LoRA
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Logging/saving
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)

    return p.parse_args()


# -------------------------
# Canonical SAQ answer
# -------------------------
def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def pick_saq_canonical_from_annotations(annotations_str: str) -> str:
    """
    annotations_str looks like:
      "[{'answers': [...], 'en_answers': [...], 'count': 4}, ...]"

    Canonical choice:
      - pick max count
      - deterministic tie-break: choose lexicographically smallest candidate text
      - prefer en_answers over answers
    """
    anns = ast.literal_eval(annotations_str) if isinstance(annotations_str, str) else annotations_str
    if not anns:
        return ""

    # Build candidate texts per entry (prefer English)
    candidates: List[Tuple[int, str]] = []
    for d in anns:
        c = int(d.get("count", 0))
        en = d.get("en_answers") or []
        orig = d.get("answers") or []
        chosen_list = en if (isinstance(en, list) and len(en) > 0 and en[0]) else orig
        if isinstance(chosen_list, list) and chosen_list:
            txt = _norm_text(chosen_list[0])
            if txt:
                candidates.append((c, txt))

    if not candidates:
        return ""

    max_c = max(c for c, _ in candidates)
    tied = [txt for c, txt in candidates if c == max_c]

    # Deterministic tie-break:
    return sorted(tied)[0]


# -------------------------
# Prompt building
# -------------------------
RESPONSE_MARKER = "### Response:\n"


def build_mcq_prompt(country: str, raw_prompt: str) -> str:
    """
    Your MCQ CSV already has a prompt that ends with something like "Answer:".
    We keep it, but force a consistent response marker so masking is easy.
    """
    country = _norm_text(country)
    raw_prompt = str(raw_prompt).rstrip()

    return (
        "### Instruction:\n"
        "Answer the multiple-choice question by selecting the correct option letter.\n\n"
        f"### Context:\nCountry: {country}\n\n"
        "### Question:\n"
        f"{raw_prompt}\n\n"
        f"{RESPONSE_MARKER}"
    )


def build_saq_prompt(country: str, question_en: str) -> str:
    country = _norm_text(country)
    q = _norm_text(question_en)
    return (
        "### Instruction:\n"
        "Return ONLY one line starting with \"Answer: \" followed by the best answer.\n\n"
        f"### Context:\nCountry: {country}\n\n"
        "### Question:\n"
        f"{q}\n\n"
        f"{RESPONSE_MARKER}"
    )


# -------------------------
# Loss-masked tokenization
# -------------------------
def tokenize_with_answer_mask(
    tokenizer,
    prompt_only: str,
    full_text: str,
    max_length: int,
) -> Dict[str, List[int]]:
    """
    Tokenize full_text, but mask labels for the prompt part (train only on answer tokens).
    We compute prompt token length by tokenizing prompt_only separately.
    """
    # Prompt length (no padding)
    prompt_ids = tokenizer(
        prompt_only,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
        return_attention_mask=False,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    # Full sequence
    tok = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
        return_attention_mask=True,
    )
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]

    labels = input_ids.copy()

    # Mask prompt tokens
    # If prompt_len >= max_length, we cannot train on answer (it got truncated away).
    if prompt_len >= max_length:
        labels = [-100] * len(labels)
    else:
        for i in range(prompt_len):
            labels[i] = -100

    # Mask padding
    for i, m in enumerate(attention_mask):
        if m == 0:
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# -------------------------
# Metrics (simple & useful)
# -------------------------
def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def compute_mcq_accuracy(pred_texts: List[str], gold_letters: List[str]) -> float:
    good = 0
    total = 0
    for pred, gold in zip(pred_texts, gold_letters):
        g = gold.strip().upper()

        p = ""
        # Try JSON parse first
        try:
            obj = json.loads(pred.strip())
            if isinstance(obj, dict) and "answer_choice" in obj:
                p = str(obj["answer_choice"]).strip().upper()
        except Exception:
            pass

        # Fallback: find A/B/C/D anywhere
        if p not in ["A", "B", "C", "D"]:
            m = re.search(r"([ABCD])", pred.upper())
            p = m.group(1) if m else ""

        total += 1
        good += int(p == g)

    return good / total if total else 0.0


def compute_exact_match(pred_texts: List[str], gold_texts: List[str]) -> float:
    good = 0
    total = 0
    for pred, gold in zip(pred_texts, gold_texts):
        total += 1
        good += int(normalize_answer(pred) == normalize_answer(gold))
    return good / total if total else 0.0


@torch.no_grad()
def generate_predictions(model, tokenizer, rows: List[Dict], max_new_tokens: int = 64) -> List[str]:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outs = []
    for r in rows:
        prompt = r["prompt_only"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_tokens = gen[0][inputs["input_ids"].shape[-1] :]
        outs.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())
    return outs


# -------------------------
# Main
# -------------------------
def count_trainable_parameters(model, tag: str, out_dir: str):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (100.0 * trainable / total) if total else 0.0
    print(f"[{tag}] trainable params: {trainable} / {total} ({pct:.2f}%)")
    with open(os.path.join(out_dir, f"{tag}_params.json"), "w", encoding="utf-8") as f:
        json.dump({"trainable": trainable, "total": total, "percent": pct}, f, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -------------------------
    # Load CSVs
    # -------------------------
    mcq_df = pd.read_csv(args.mcq_csv)
    saq_df = pd.read_csv(args.saq_csv)

    # -------------------------
    # Build canonical training rows
    # Each row -> prompt_only + full_text + task + gold
    # -------------------------
    rows = []

    # MCQ: expect columns like prompt, answer_idx, country
    if args.task in ["mcq", "both"]:
        for _, r in mcq_df.iterrows():
            country = r.get("country", "")
            raw_prompt = r.get("prompt", "")
            answer_letter = str(r.get("answer_idx", "")).strip().upper()  # A/B/C/D
            answer_json = json.dumps({"answer_choice": answer_letter}, ensure_ascii=False)
            prompt_only = build_mcq_prompt(country=country, raw_prompt=raw_prompt)
            full_text = prompt_only + answer_json

            rows.append(
                {
                    "task": "mcq",
                    "prompt_only": prompt_only,
                    "full_text": full_text,
                    "gold_letter": answer_letter,
                    "gold_text": answer_json,  # for uniformity
                }
            )

    # SAQ: expect en_question, annotations, country
    if args.task in ["saq", "both"]:
        for _, r in saq_df.iterrows():
            country = r.get("country", "")
            q_en = r.get("en_question", None)
            if q_en is None or str(q_en).strip() == "":
                q_en = r.get("question", "")

            canonical = pick_saq_canonical_from_annotations(r.get("annotations", "[]"))
            if not canonical:
                continue  # important: skip empties

            target = f"Answer: {canonical}"

            prompt_only = build_saq_prompt(country=country, question_en=q_en)
            full_text = prompt_only + target

            rows.append(
                {
                    "task": "saq",
                    "prompt_only": prompt_only,
                    "full_text": full_text,
                    "gold_letter": "",  # not used
                    "gold_text": target,
                }
            )

    # Fail fast if nothing was built (avoids obscure downstream errors)
    if len(rows) == 0:
        raise ValueError(
            f"No training rows were built. Check --task={args.task} and your CSV contents/column names."
        )

    ds = Dataset.from_list(rows)

    # Optional caps (useful for quick tests)
    if args.max_train_samples is not None or args.max_eval_samples is not None:
        # cap later after split; here we just keep full dataset
        pass

    # -------------------------
    # Split train / val
    # -------------------------
    split = ds.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)
    train_raw = split["train"]
    eval_raw = split["test"]

    if args.max_train_samples is not None:
        train_raw = train_raw.select(range(min(args.max_train_samples, len(train_raw))))
    if args.max_eval_samples is not None:
        eval_raw = eval_raw.select(range(min(args.max_eval_samples, len(eval_raw))))

    with open(os.path.join(args.output_dir, "data_split_sizes.json"), "w", encoding="utf-8") as f:
        json.dump({"train": len(train_raw), "eval": len(eval_raw)}, f, indent=2)

    # -------------------------
    # Tokenizer + base model
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    base_model.config.use_cache = False

    # -------------------------
    # LoRA config (define BEFORE using)
    # -------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # -------------------------
    # Load existing LoRA or create new one (DO NOT overwrite later)
    # -------------------------
    if args.lora_path and not os.path.isdir(args.lora_path):
        raise FileNotFoundError(f"--lora_path does not exist or is not a directory: {args.lora_path}")

    if args.lora_path:
        print(f"Loading existing LoRA from {args.lora_path}")
        lora_model = PeftModel.from_pretrained(base_model, args.lora_path)
    else:
        lora_model = get_peft_model(base_model, lora_config)

    enable_input_require_grads = getattr(lora_model, "enable_input_require_grads", None)
    if callable(enable_input_require_grads):
        enable_input_require_grads()

    count_trainable_parameters(base_model, "base_model", args.output_dir)
    count_trainable_parameters(lora_model, "lora_model", args.output_dir)

    # -------------------------
    # Tokenize with answer-only loss
    # -------------------------
    def map_tokenize(batch):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for prompt_only, full_text in zip(batch["prompt_only"], batch["full_text"]):
            tok = tokenize_with_answer_mask(tokenizer, prompt_only, full_text, args.max_length)
            out["input_ids"].append(tok["input_ids"])
            out["attention_mask"].append(tok["attention_mask"])
            out["labels"].append(tok["labels"])
        return out

    train_ds = train_raw.map(map_tokenize, batched=True, remove_columns=train_raw.column_names, desc="Tokenizing train")
    eval_ds = eval_raw.map(map_tokenize, batched=True, remove_columns=eval_raw.column_names, desc="Tokenizing eval")

    # Filter out examples where answer got truncated away (all labels == -100)
    def has_any_supervision(example):
        return any(x != -100 for x in example["labels"])

    train_ds = train_ds.filter(has_any_supervision, desc="Filter train (non-empty supervision)")
    eval_ds = eval_ds.filter(has_any_supervision, desc="Filter eval (non-empty supervision)")

    with open(os.path.join(args.output_dir, "post_tokenize_sizes.json"), "w", encoding="utf-8") as f:
        json.dump({"train": len(train_ds), "eval": len(eval_ds)}, f, indent=2)

    # -------------------------
    # TrainingArguments + Trainer
    # -------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=args.save_total_limit,
        report_to="none",
        bf16=torch.cuda.is_available(),
        fp16=False,
        dataloader_drop_last=False,
        gradient_checkpointing=False,
    )

    data_collator = default_data_collator

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # -------------------------
    # Train
    # -------------------------
    print("Starting LoRA fine-tuning...")
    trainer.train()

    # Save adapter + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter + tokenizer to: {args.output_dir}")

    # -------------------------
    # Simple post-train evaluation by generation (MCQ acc + SAQ exact match)
    # -------------------------
    # We evaluate on the *raw eval split* (prompts + gold), not tokenized eval.
    eval_rows = eval_raw.to_list()

    preds = generate_predictions(lora_model, tokenizer, eval_rows, max_new_tokens=64)

    # Split by task
    mcq_preds, mcq_gold = [], []
    saq_preds, saq_gold = [], []

    for r, ptxt in zip(eval_rows, preds):
        if r["task"] == "mcq":
            mcq_preds.append(ptxt)
            mcq_gold.append(r["gold_letter"])
        else:
            saq_preds.append(ptxt)
            saq_gold.append(r["gold_text"])

    metrics = {
        "mcq_accuracy": compute_mcq_accuracy(mcq_preds, mcq_gold) if mcq_gold else None,
        "saq_exact_match": compute_exact_match(saq_preds, saq_gold) if saq_gold else None,
        "num_eval_mcq": len(mcq_gold),
        "num_eval_saq": len(saq_gold),
    }

    with open(os.path.join(args.output_dir, "gen_eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save a few examples for inspection
    examples_out = []
    for i, (r, ptxt) in enumerate(zip(eval_rows[:50], preds[:50])):
        examples_out.append(
            {
                "task": r["task"],
                "prompt": r["prompt_only"],
                "prediction": ptxt,
                "gold": r["gold_letter"] if r["task"] == "mcq" else r["gold_text"],
            }
        )
    with open(os.path.join(args.output_dir, "eval_generations_preview.json"), "w", encoding="utf-8") as f:
        json.dump(examples_out, f, indent=2)

    print("Done. Generation-based eval metrics:")
    print(metrics)


if __name__ == "__main__":
    main()