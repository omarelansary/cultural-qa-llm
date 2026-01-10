import pandas as pd
import torch
from torch.utils.data import Dataset


def load_raw_data(path):
    print(f"[INFO] Reading data from {path}...")
    return pd.read_csv(path, sep=None, engine="python")


class CulturalDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_length=128,
        build_text_fn=None,
        build_prompt_fn=None,
        text_column=None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.build_text_fn = build_text_fn
        self.build_prompt_fn = build_prompt_fn
        self.text_column = text_column

    def __len__(self):
        return len(self.data)

    def _get_text(self, row):
        if self.build_text_fn is not None:
            text = self.build_text_fn(row)
            if not isinstance(text, str):
                raise ValueError("build_text_fn must return a string.")
            return text

        if self.text_column is not None:
            return str(row[self.text_column])

        if "question" in row and "target" in row:
            return f"Question: {row['question']}\nAnswer: {row['target']}"

        raise KeyError("No build_text_fn or valid text_column provided for CulturalDataset.")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = self._get_text(row)
        prompt_text = None
        if self.build_prompt_fn is not None:
            prompt_text = self.build_prompt_fn(row)
            if not isinstance(prompt_text, str):
                raise ValueError("build_prompt_fn must return a string.")

        if prompt_text is not None:
            prompt_ids = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
            )["input_ids"].squeeze()
            prompt_len = int(prompt_ids.shape[-1])
        else:
            prompt_len = 0

        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        labels = input_ids.clone()
        if prompt_len >= self.max_length:
            labels[:] = -100
        elif prompt_len > 0:
            labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
