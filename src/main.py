import torch.nn as nn
import torch.optim as optim
from model import MLP, CNN
from datamodule import DataModuleBitboards, DataModuleImages
from trainer import Trainer


def mlp_experiment(input, output_dir):
    print()
    print('Running MLP experiment on ' + input)
    print()

    # data config
    datamodule = DataModuleBitboards(filename=input, ratio=0.8, batch_size=64)

    # model config
    model = MLP(
        input_features=13,
        hidden_features=50,
        output_features=1,
        layers=3,
        activ=nn.ReLU,
        norm=nn.BatchNorm1d
    )

    # training config
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    max_epoch = 20

    trainer = Trainer(
        datamodule=datamodule,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        max_epoch=max_epoch,
        output_dir=output_dir,
        input_shape=(13,)
    )
    trainer.fit()

def cnn_experiment(input, output_dir):
    print()
    print('Running CNN experiment on ' + input)
    print()

    # data config
    datamodule = DataModuleImages(filename=input, ratio=0.8, batch_size=64)

    # model config
    model = CNN(
        conv_layers=2,
        conv_norm=nn.BatchNorm2d,
        conv_activ=nn.ReLU,
        fc_layers=2,
        fc_norm=nn.BatchNorm1d,
        fc_activ=nn.ReLU
    )

    # training config
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    max_epoch = 20

    trainer = Trainer(
        datamodule=datamodule,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        max_epoch=max_epoch,
        output_dir=output_dir,
        input_shape=(2, 8, 8)
    )
    trainer.fit()

def main():
    mlp_experiment(input='input/red_bitboards.csv', output_dir='output/bitboards/red')
    mlp_experiment(input='input/blue_bitboards.csv', output_dir='output/bitboards/blue')
    cnn_experiment(input='input/red_images.csv', output_dir='output/images/red')
    cnn_experiment(input='input/blue_images.csv', output_dir='output/images/blue')

if __name__ == "__main__":
    main()
