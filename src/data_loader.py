import pandas as pd
import torch
from torch.utils.data import Dataset

def load_raw_data(path):
    print(f"📂 Reading data from {path}...")
    return pd.read_csv(path, sep='\t')

class CulturalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Map letters to integers if needed for labels, or handle in loop

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Simple formatting: "Question: ... Answer: ..."
        text = f"Question: {row['question']}\nAnswer: {row['target']}"
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # For CausalLM, labels are usually the same as input_ids
        input_ids = encodings['input_ids'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': input_ids.clone() # Simple auto-regressive training
        }