import os
import torch
import torchsummary
import matplotlib.pyplot as plt
from tqdm import tqdm


def decide_device():
    if (torch.cuda.is_available()): return "cuda"
    return "cpu"

class Trainer:
    def __init__(self, datamodule, model, criterion, optimizer, max_epoch, output_dir, input_shape):
        self.device = torch.device(decide_device())

        self.datamodule = datamodule
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.output_dir = output_dir
        self.input_shape = input_shape

    def fit(self, checkpoint=None):
        torchsummary.summary(self.model, self.input_shape)

        if checkpoint:
            self.load_checkpoint(filename=checkpoint)
        else:
            self.train_losses = []
            self.val_losses = []

            self.cur_epoch = 0

        for epoch in range(self.cur_epoch, self.max_epoch):
            self.cur_epoch = epoch

            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_loss = self.val_epoch(epoch)
            self.val_losses.append(val_loss)
            
            self.save_checkpoint(filename=f"epoch_{epoch}.pt")

        self.save_model()
        self.save_plot(filename='loss.png', caption='Loss function', metric_name='Loss', train_values=self.train_losses, val_values=self.val_losses)
          
    def train_epoch(self, epoch):
        return self.epoch(dataloader=self.datamodule.train_loader, training=True, caption=f'Training epoch {epoch}')

    def val_epoch(self, epoch):
        return self.epoch(dataloader=self.datamodule.val_loader, training=False, caption=f'Validation epoch {epoch}')
    
    def epoch(self, dataloader, training, caption):
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
        dir = os.path.join(self.output_dir, 'checkpoints')
        filename = os.path.join(dir, filename)

        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        self.cur_epoch = checkpoint['cur_epoch']
    
    def save_checkpoint(self, filename):
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
        dir = self.output_dir
        os.makedirs(dir, exist_ok=True)

        filename = os.path.join(self.output_dir, f"model.pt")

        script_model = torch.jit.script(self.model)
        script_model.save(filename)

    def save_plot(self, filename, caption, metric_name, train_values, val_values):
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
