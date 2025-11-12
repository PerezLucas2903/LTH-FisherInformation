import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

def encode_text(text):
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

class LanguageModelingDataset(Dataset):
    def __init__(self, data, seq_len=128):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y
    
if __name__ == "__main__":
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    train_text = "\n".join(dataset["train"]["text"])
    valid_text = "\n".join(dataset["validation"]["text"])

        
    train_tokens = encode_text(train_text)
    valid_tokens = encode_text(valid_text)
    print(f"Train tokens: {len(train_tokens):,}, Valid tokens: {len(valid_tokens):,}")

    seq_len = 128
    batch_size = 64

    train_ds = LanguageModelingDataset(train_tokens, seq_len)
    valid_ds = LanguageModelingDataset(valid_tokens, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, drop_last=True)

    for x_batch, y_batch in train_loader:
        print("Input batch shape:", x_batch.shape)
        print("Target batch shape:", y_batch.shape)
        break