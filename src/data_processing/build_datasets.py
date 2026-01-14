import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def build_dataloaders(dataset_name):
    if dataset_name=='MNIST':
        return build_mnist_dataloaders()
    elif dataset_name=='FashionMNIST':
        return build_fashion_mnist_dataloaders()
    

def build_mnist_dataloaders():
    """Create MNIST dataloaders."""
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"

    transform = transforms.ToTensor()

    mnist_train = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(mnist_train, batch_size=256, shuffle=True)
    fim_loader = DataLoader(mnist_train, batch_size=1, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=20, shuffle=True)

    return train_loader, fim_loader, test_loader



def build_fashion_mnist_dataloaders ():
    """Create FashionMNIST dataloaders."""
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"

    transform = transforms.ToTensor()

    mnist_train = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(mnist_train, batch_size=256, shuffle=True)
    fim_loader = DataLoader(mnist_train, batch_size=1, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=20, shuffle=True)

    return train_loader, fim_loader, test_loader