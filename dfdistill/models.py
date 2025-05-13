import torch.nn as nn
import torch.nn.functional as F

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
    

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32):
        super(Generator, self).__init__()

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z.view(z.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img
