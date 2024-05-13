"""!
@file   datamodule.py
@brief  Dátové moduly majú na starosť rozdelenie vstupných dát na trénovacie, validačné a testovacie dáta.

@author Martin Šváb
@date   Máj 2024
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import MvsFSBitboardDataset, MvsFSImageDataset


class DataModuleBitboards:
    """!
    Dátový modul pre bitboardovú formu MvsFS datasetu.
    """

    def __init__(self, filename, ratio, batch_size):
        """!
        Konštruktor dátového module bitboardovej formy MvsFS datasetu.

        @param filename: Názov vstupného csv súboru.
        @param ratio: Pomer medzi trénovacími a validačnými dátami (validačné a testovacie dáta majú pomer 0.5).
        @param batch_size: Veĺkosť dávky pri načítavaní dát.
        """

        data = pd.read_csv(filename)

        self.ratio = ratio
        train_data, val_data = train_test_split(data, train_size=self.ratio, random_state=10)
        val_data, test_data = train_test_split(val_data, train_size=0.5, random_state=10)

        self.train_dataset = MvsFSBitboardDataset(train_data)
        self.val_dataset = MvsFSBitboardDataset(val_data)
        self.test_dataset = MvsFSBitboardDataset(test_data)

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
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

class DataModuleImages:
    """!
    Dátový modul pre obrázkovú formu MvsFS datasetu.
    """

    def __init__(self, filename, ratio, batch_size):
        """!
        Konštruktor dátového module obrázkovej formy MvsFS datasetu.

        @param filename: Názov vstupného csv súboru.
        @param ratio: Pomer medzi trénovacími a validačnými dátami (validačné a testovacie dáta majú pomer 0.5).
        @param batch_size: Veĺkosť dávky pri načítavaní dát.
        """

        data = pd.read_csv(filename)

        self.ratio = ratio
        train_data, val_data = train_test_split(data, train_size=self.ratio, random_state=10)
        val_data, test_data = train_test_split(val_data, train_size=0.5, random_state=10)

        self.train_dataset = MvsFSImageDataset(train_data)
        self.val_dataset = MvsFSImageDataset(val_data)
        self.test_dataset = MvsFSImageDataset(test_data)

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
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
