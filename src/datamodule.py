import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import MvsFSDataset


class DataModule:
    def __init__(self, filename):
        data = pd.read_csv(filename)

        self.ratio = 0.8
        train_data, val_data = train_test_split(data, train_size=self.ratio, random_state=42)

        self.train_dataset = MvsFSDataset(train_data)
        self.val_dataset = MvsFSDataset(val_data)

        self.batch_size = 64
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
