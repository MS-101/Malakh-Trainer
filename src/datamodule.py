import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import MvsFSBitboardDataset, MvsFSImageDataset


class DataModuleBitboards:
    def __init__(self, filename, ratio, batch_size):
        data = pd.read_csv(filename)

        self.ratio = ratio
        train_data, val_data = train_test_split(data, train_size=self.ratio, random_state=10)

        self.train_dataset = MvsFSBitboardDataset(train_data)
        self.val_dataset = MvsFSBitboardDataset(val_data)

        self.batch_size = batch_size
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

class DataModuleImages:
    def __init__(self, filename, ratio, batch_size):
        data = pd.read_csv(filename)

        self.ratio = ratio
        train_data, val_data = train_test_split(data, train_size=self.ratio, random_state=10)

        self.train_dataset = MvsFSImageDataset(train_data)
        self.val_dataset = MvsFSImageDataset(val_data)

        self.batch_size = batch_size
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