import torch
import torch.nn as nn


class SharedMLP(nn.Module):
    def __init__(self, channels, dropout=None, use_bn=True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Dense(nn.Module):
    def __init__(self, in_features, out_features, use_bn=True, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=not use_bn)
        self.bn = nn.BatchNorm1d(out_features) if use_bn else nn.Identity()
        self.act = activation if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.act(x)
        return x
