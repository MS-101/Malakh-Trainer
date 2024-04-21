import json
import torch
from torch.utils.data import Dataset


class MvsFSBitboardDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        x = torch.tensor(sample.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.float32)
        y = torch.tensor(sample.iloc[13], dtype=torch.float32)
        return x, y

class MvsFSImageDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        x = torch.tensor(json.loads(sample.iloc[0]), dtype=torch.float32)
        y = torch.tensor(sample.iloc[1], dtype=torch.float32)
        return x, y