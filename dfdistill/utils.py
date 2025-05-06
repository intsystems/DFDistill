from torchvision.datasets import CIFAR100, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar100(
    path='../../data/', 
    train=True
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 

    return CIFAR100(root=path, train=train, download=True, transform=transform)
    

def get_cifar100_loader(
    batch_size=256,
    data_path='../../data/',
    train=True
):
    return DataLoader(get_cifar100(data_path, train), batch_size=batch_size, shuffle=not train)


def get_cifar10(
    path='../../data/', 
    train=True
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 

    return CIFAR10(root=path, train=train, download=True, transform=transform)
    

def get_cifar10_loader(
    batch_size=256,
    data_path='../../data/',
    train=True
):
    return DataLoader(get_cifar10(data_path, train), batch_size=batch_size, shuffle=not train)