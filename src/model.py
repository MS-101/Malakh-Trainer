import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super().__init__()

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features

        self.input = nn.Sequential(
            nn.Linear(input_features, hidden_features, output_features),
            nn.ReLU()
        )

        self.hidden = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU()
        )

        self.output = nn.Linear(hidden_features, output_features)

    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x
    