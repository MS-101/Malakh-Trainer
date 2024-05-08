import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, layers, activ):
        super().__init__()

        # input layer

        self.input = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            activ()
        )

        # hidden layer

        hidden = []

        for _ in range(layers):
            hidden += [
                nn.Linear(hidden_features, hidden_features),
                activ()
            ]

        self.hidden = nn.Sequential(*hidden)

        # output layer

        self.output = nn.Linear(hidden_features, output_features)

    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        return x

class CNN(nn.Module):
    def __init__(self, conv_layers, conv_norm, conv_activ, fc_layers, fc_activ):
        super().__init__()

        # convolution layer

        in_channels = 2
        out_channels = 8

        conv = []

        for _ in range(conv_layers):
            conv += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                conv_norm(out_channels),
                conv_activ(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]

            in_channels = out_channels
            out_channels *= 2

        self.conv = nn.Sequential(*conv)

        # fully connected layer

        dimension = int(8 / (2**conv_layers))
        self.fc_input = in_channels * dimension * dimension

        in_features = self.fc_input
        out_features = int(in_features / 2)

        fc = []

        for _ in range(fc_layers):
            fc += [
                nn.Linear(in_features, out_features),
                fc_activ()
            ]

            in_features = out_features
            out_features = int(in_features / 2)

        self.fc = nn.Sequential(*fc)

        # output layer

        self.output = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.fc_input)
        x = self.fc(x)
        x = self.output(x)
        return x
