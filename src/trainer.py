"""!
@file   trainer.py
@brief  Definícia trénovania neurónovej siete.

@author Martin Šváb
@date   Máj 2024
"""

import os
import torch
import torchsummary
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.utils.io import Tee
from contextlib import closing


class EarlyStopper:
    """!
    Trieda zodpovedná za predčasné zastavenie trénovania.
    """

    def __init__(self, patience=1, min_delta=0):
        """
        Konštruktor predčasného ukončovateľa trénovania.

        @param patience: Počet epoch, ktoré je možné vykonať po sebe bez vylepšenia výsledkov trénovania.
        @param min_delta: Minimálny rozdiel medzi najlepším výsledkom a aktuálnym výsledkom potrebný aby sa epocha považovala za neúspešnú.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss):
        """!
        V tejto funkcii sa kontroluje, či aktuálna validačná strata je lepšia ako najlepší možný výsledok.
        Ak áno, tak sa počítadlo trpezlivosti reštartuje na 0.
        Ak nie a chyby je vyššia od najlepšieho výsledku o min_delta hodnotu, tak sa inkrementuje počítadlo trpezlivosti.
        Ak počítadlo trpezlivosti dosiahlo maximálnu hodnotu, tak sa ukončí trénovanie.
        
        @param val_loss: Validačnú strata aktuálnej epochy.
        @return True ak sa trénovanie má predčasne ukončiť, False ak trénovanie má pokračovať.
        """

        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def decide_device():
    """!
    Pomocná funkcia na rozhodnutie použítia CPU alebo cuda režimu.

    @return Ak je cuda dostupné tak hodnota cuda, ak nie je dostupné tak hodnota cpu.
    """
    if (torch.cuda.is_available()): return "cuda"
    return "cpu"

class Trainer:
    """!
    Trieda zodpovedná za trénovanie neurónových sietí.
    """

    def __init__(self, datamodule, model, input_shape, criterion, optimizer, patience, min_delta, max_epoch, output_dir):
        """!
        Konštruktor konfigurovateľného trénera neurónových sietí.

        @param datamodule: Dátový modul obsahujúci MvsFS dataset.
        @param model: Architektúra neurónovej siete.
        @param input_shape: Tvar vstupu do neurónovej siete.
        @param criterion: Loss funkcia.
        @param optimizer: Optimalizačný algoritmus trénovania.
        @param patience: Počet epoch, ktoré je možné vykonať po sebe bez vylepšenia výsledkov trénovania.
        @param min_delta: Minimálny rozdiel medzi najlepším výsledkom a aktuálnym výsledkom potrebný aby sa epocha považovala za neúspešnú.
        @param max_epoch: Maximálny počet vykonaných epoch.
        @param output_dir: Priečinok kde sa uložia výsledky trénovania.
        """

        self.device = torch.device(decide_device())

        self.datamodule = datamodule

        self.model = model.to(self.device)
        self.input_shape = input_shape

        self.criterion = criterion
        self.optimizer = optimizer
        self.early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        self.max_epoch = max_epoch

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def fit(self, checkpoint=None):
        """!
        Táto funkcia spustí trénovanie neurónovej siete.

        @param checkpoint: Názov súboru s checkpointom trénovania. Ak nechceme použíť checkpoint a chceme začať trénovať od začiatku, tak zadáme null hodnotu.
        """

        if checkpoint:
            self.load_checkpoint(filename=checkpoint)
        else:
            self.train_losses = []
            self.val_losses = []

            self.cur_epoch = 0

        with closing(Tee(os.path.join(self.output_dir, "model.txt"))):
            torchsummary.summary(self.model, self.input_shape)

        with closing(Tee(os.path.join(self.output_dir, "epochs.txt"))):
            for epoch in range(self.cur_epoch, self.max_epoch):
                self.cur_epoch = epoch

                train_loss = self.train_epoch(epoch)
                self.train_losses.append(train_loss)

                val_loss = self.val_epoch(epoch)
                self.val_losses.append(val_loss)
                
                self.save_checkpoint(filename=f"epoch_{epoch}.pt")

                if self.early_stopper.early_stop(val_loss):
                    print(f'Early stopping at epoch {epoch}')
                    break

            self.save_plot(filename='loss.png', caption='Loss function', metric_name='Loss', train_values=self.train_losses, val_values=self.val_losses)
            self.save_model()

            self.test_epoch()
          
    def train_epoch(self, epoch):
        """!
        Spustenie trénovacej epochy. V tejto epoche vykonávame spätnú propagáciu.

        @param epoch: Poradie epochy.
        """

        return self.epoch(dataloader=self.datamodule.train_loader, training=True, caption=f'Training epoch {epoch}')

    def val_epoch(self, epoch):
        """!
        Spustenie validačnej epochy. V tejto epoche nevykonávame spätnú propagáciu.

        @param epoch: Poradie epochy.
        """

        return self.epoch(dataloader=self.datamodule.val_loader, training=False, caption=f'Validation epoch {epoch}')

    def test_epoch(self):
        """!
        Spustenie testovacej epochy. V tejto epoche nevykonávame spätnú propagáciu.
        Táto epocha je spustená po ukončení trénovanie pre vyhodnotenie efektivity výsledného modelu.
        """

        return self.epoch(dataloader=self.datamodule.test_loader, training=False, caption=f'Testing epoch')
    
    def epoch(self, dataloader, training, caption):
        """!
        Generická metóda pre všetky epochy.

        @param dataloader: Trieda pomocou ktorej periodicky čítame dávky nášho datasetu.
        @param training: Ak je hodnota True, tak vykonávame spätnú propagáciu. Ak je hodnota False, tak spätnú propagáciu nevykonávame.
        @param caption: Popis aktuálnej epochy.
        """

        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0

        with torch.set_grad_enabled(training):
            with tqdm(dataloader, desc=caption) as progress:
                for x,y in progress:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    if training:
                        self.optimizer.zero_grad()

                    y_hat = self.model(x)
                    loss = self.criterion(y_hat, y)
                    total_loss += loss.item()

                    if training:
                        loss.backward()
                        self.optimizer.step()

        avg_loss = total_loss / len(dataloader)

        print(f'{caption}: loss = {avg_loss}')

        return avg_loss

    def load_checkpoint(self, filename):
        """!
        Načítanie priebežných výsledkov trénovania z checkpointu.

        @param filename: Súbor z ktorého načítame checkpoint.
        """

        dir = os.path.join(self.output_dir, 'checkpoints')
        filename = os.path.join(dir, filename)

        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        self.cur_epoch = checkpoint['cur_epoch']
    
    def save_checkpoint(self, filename):
        """!
        Uloženie priebežných výsledkov trénovania do checkpointu.

        @param filename: Súbor do ktorého uložíme checkpoint.
        """

        dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(dir, exist_ok=True)

        filename = os.path.join(dir, filename)

        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'cur_epoch': self.cur_epoch
        }, filename)

    def save_model(self):
        """!
        Uloženie výsledného modelu do jit formátu.
        Tento model používame v šachovom engine Malakh.
        """

        dir = self.output_dir
        os.makedirs(dir, exist_ok=True)

        filename = os.path.join(self.output_dir, f"model.pt")

        script_model = torch.jit.script(self.model)
        script_model.save(filename)

    def save_plot(self, filename, caption, metric_name, train_values, val_values):
        """!
        Uloženie trénovacích a validačných metrík do grafu po ukončení trénovania.
        Pomocou týchto grafov môžeme vyhodnotiť priebeh trénovania a odpozorovať underfitting alebo overfitting.

        @param filename: Súbor do ktorého uložíme graf.
        @param caption: Názov uloženého grafu.
        @param metric_name: Názov zobrazenej metriky.
        @param train_values: Hodnoty trénovacích epoch trénovania.
        @param val_values: Hodnoty validačných epoch trénovania.
        """

        dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(dir, exist_ok=True)

        filename = os.path.join(dir, filename)

        plt.clf()
        plt.plot(train_values, label='Training')
        plt.plot(val_values, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(caption)
        plt.legend()
        plt.savefig(filename)
