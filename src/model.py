import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(13, 50)
        self.hidden1 = nn.Linear(50, 50)
        self.hidden2 = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.relu(self.hidden1(x))
        x = self.hidden2(x)
        return x
