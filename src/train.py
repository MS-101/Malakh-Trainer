import os
import torch
import matplotlib.pyplot as plt


def decide_device():
    if (torch.cuda.is_available()): return "cuda"
    return "cpu"

class Trainer:
    def __init__(self, datamodule, model, criterion, optimizer, epochs, output_dir):
        self.device = torch.device(decide_device())

        self.datamodule = datamodule
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.output_dir = output_dir

    def fit(self):
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)

            val_loss = self.val_epoch(epoch)
            val_losses.append(val_loss)
            
            self.save_checkpoint(epoch)

        self.plot_loss(train_losses=train_losses, val_losses=val_losses)
          
    def train_epoch(self, epoch):
        dataloader = self.datamodule.train_loader

        self.model.train()
        total_loss = 0.0

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch}: Training loss: {avg_loss}')

        return avg_loss

    def val_epoch(self, epoch):
        dataloader = self.datamodule.val_loader

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch}: Validation loss: {avg_loss}')

        return avg_loss
    
    def save_checkpoint(self, epoch):
        filename = os.path.join(self.output_dir, f"epoch_{epoch+1}.pt")
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filename)

    def plot_loss(self, train_losses, val_losses):
        filename = os.path.join(self.output_dir, "loss_plot.png")

        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig(filename)
