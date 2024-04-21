import torch
import torch.nn as nn
import torch.optim as optim
from model import MLP, CNN
from datamodule import DataModuleBitboards, DataModuleImages
from train import Trainer


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def mlp_experiment(input, output_dir):
    print('Running MLP experiment on ' + input)

    datamodule = DataModuleBitboards(filename=input)
    model = MLP(input_features=13, hidden_features=50, output_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    epochs = 100

    trainer = Trainer(datamodule=datamodule, model=model, criterion=criterion, optimizer=optimizer, epochs=epochs, output_dir=output_dir)
    trainer.fit()

def cnn_experiment(input, output_dir):
    print('Running CNN experiment on ' + input)

    datamodule = DataModuleImages(filename=input)
    model = CNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    epochs = 100

    trainer = Trainer(datamodule=datamodule, model=model, criterion=criterion, optimizer=optimizer, epochs=epochs, output_dir=output_dir)
    trainer.fit()

def main():
    mlp_experiment(input='input/red_bitboards.csv', output_dir='output/red/bitboards')
    mlp_experiment(input='input/blue_bitboards.csv', output_dir='output/blue/bitboards')
    cnn_experiment(input='input/red_images.csv', output_dir='output/red/images')
    cnn_experiment(input='input/blue_images.csv', output_dir='output/blue/images')

if __name__ == "__main__":
    main()
