"""!
@file   experiment.py
@brief  Definícia konfigurovateľných experimentov.

@author Martin Šváb
@date   Máj 2024
"""

import torch.nn as nn
import torch.optim as optim
from model import MLP, CNN
from datamodule import DataModuleBitboards, DataModuleImages
from trainer import Trainer


def mlp_experiment(input, output_dir, hidden_features, layers):
    """!
    Experiment na testovanie efektivity MLP architektúry a bitboardovej formy MvsFS datasetu.

    @param input: Názov vstupného csv súboru.
    @param output_dir: Priečinok kde sa uložia výsledky experimentu.
    @param hidden_features: Počet neurónov skrytých vrstiev MLP architektúry.
    @param layers: Počet skrytých vrstiev MLP architektúry.
    """

    print()
    print('Running MLP experiment on ' + input)
    print()

    # data config
    datamodule = DataModuleBitboards(filename=input, ratio=0.8, batch_size=64)

    # model config
    model = MLP(
        input_features=13,
        hidden_features=hidden_features,
        output_features=1,
        layers=layers,
        activ=nn.ReLU
    )

    # training config
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    patience = 5
    min_delta = 1000
    max_epoch = 20

    trainer = Trainer(
        datamodule=datamodule,
        model=model,
        input_shape=(13,),
        criterion=criterion,
        optimizer=optimizer,
        patience=patience,
        min_delta=min_delta,
        max_epoch=max_epoch,
        output_dir=output_dir
    )
    trainer.fit()

def cnn_experiment(input, output_dir, fc_layers):
    """!
    Experiment na testovanie efektivity CNN architektúry a obrázkovej formy MvsFS datasetu.

    @param input: Názov vstupného csv súboru.
    @param output_dir: Priečinok kde sa uložia výsledky experimentu.
    @param fc_layers: Počet plne prepojených vrstiev CNN architektúry.
    """

    print()
    print('Running CNN experiment on ' + input)
    print()

    # data config
    datamodule = DataModuleImages(filename=input, ratio=0.8, batch_size=64)

    # model config
    model = CNN(
        conv_layers=2,
        conv_norm=nn.InstanceNorm2d,
        conv_activ=nn.ReLU,
        fc_layers=fc_layers,
        fc_activ=nn.ReLU
    )

    # training config
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    patience = 5
    min_delta = 1000
    max_epoch = 40

    trainer = Trainer(
        datamodule=datamodule,
        model=model,
        input_shape=(2, 8, 8),
        criterion=criterion,
        optimizer=optimizer,
        patience=patience,
        min_delta=min_delta,
        max_epoch=max_epoch,
        output_dir=output_dir
    )
    trainer.fit()
