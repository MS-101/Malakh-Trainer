"""!
@file   dataset.py
@brief  Definícia vstupného datasetu MvsFS.

@author Martin Šváb
@date   Máj 2024
"""

import json
import torch
from torch.utils.data import Dataset


class MvsFSBitboardDataset(Dataset):
    """!
    Dataset MvsFS v bitboardovej forme pre MLP architektúru.
    """

    def __init__(self, data):
        """!
        Konštruktor bitboardovej formy MvsFS datasetu.

        @param data: Dáta MvsFS datasetu.
        """

        self.data = data
    
    def __len__(self):
        """!
        Interná funkcia na výpočet veľkosti datasetu.
        
        @return Veľkosť datasetu
        """

        return len(self.data)
    
    def __getitem__(self, idx):
        """!
        Interná funkcia na výber položky datasetu.

        @param idx: Identifikátor položky.
        @return Vstupné vlastnosti a ich label.
        """

        sample = self.data.iloc[idx]
        x = torch.tensor(sample.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.float32)
        y = torch.tensor(sample.iloc[13], dtype=torch.float32)
        return x, y

class MvsFSImageDataset(Dataset):
    """!
    Dataset MvsFS v obrázkovej forme pre CNN architektúru-
    """

    def __init__(self, data):
        """!
        Konštruktor obrázkovej formy MvsFS datasetu.

        @param data: Dáta MvsFS datasetu.
        """

        self.data = data
    
    def __len__(self):
        """!
        Interná funkcia na výpočet veľkosti datasetu.
        
        @return Veľkosť datasetu
        """

        return len(self.data)
    
    def __getitem__(self, idx):
        """!
        Interná funkcia na výber položky datasetu.

        @param idx: Identifikátor položky.
        @return Vstupné vlastnosti a ich label.
        """

        sample = self.data.iloc[idx]
        x = torch.tensor(json.loads(sample.iloc[0]), dtype=torch.float32)
        y = torch.tensor(sample.iloc[1], dtype=torch.float32).unsqueeze(0)
        return x, y
