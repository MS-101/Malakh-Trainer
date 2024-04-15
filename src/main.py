import torch
import torch.nn as nn
import torch.optim as optim
from model import MLP
from datamodule import DataModule
from train import Trainer


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def main():
    datamodule = DataModule(filename='input/red.csv')
    output_dir = 'output/red'
    model = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    epochs = 10

    trainer = Trainer(datamodule=datamodule, model=model, criterion=criterion, optimizer=optimizer, epochs=epochs, output_dir=output_dir)
    trainer.fit()

if __name__ == "__main__":
    main()
