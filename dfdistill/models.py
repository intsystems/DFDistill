import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(n_filters, n_filters, 3, padding=1)
        )
    def forward(self, x):
        return x + self.block(x)


class SimpleResNet(nn.Module):
    def __init__(
        self, 
        n_blocks=5, 
        n_filters=32, 
        input_shape=(1, 28, 28), 
        n_classes=10
    ):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(input_shape[0], n_filters, 3, padding=1)])
        for i in range(n_blocks-1):
            self.layers.append(nn.BatchNorm2d(n_filters))
            self.layers.append(nn.SiLU())
            self.layers.append(ResBlock(n_filters))
        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(input_shape[1] * input_shape[2] * n_filters, n_classes)
        )
        self.n_filters = n_filters

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.linear(x)


class SimpleConvNet(nn.Module):
    def __init__(
        self,
        n_layers=5,
        n_filters=32,
        input_shape=(1, 28, 28),
        n_classes=10
    ):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(input_shape[0], n_filters, 3, padding=1)])
        for i in range(n_layers-1):
            self.convs.append(nn.SiLU())
            self.convs.append(nn.Conv2d(n_filters, n_filters, 3, padding=1))
        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(input_shape[1] * input_shape[2] * n_filters, n_classes)
        )
        
    
    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        return self.linear(x)