import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import random_split, TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random


# climb up to the repo root and add <repo>/src to Python's path
repo_root = Path().resolve().parents[0]   # parent of "notebooks"
sys.path.insert(0, str(repo_root / "src"))

from fisher_information.fim import FisherInformationMatrix
from models.image_classification_models import ConvModelMNIST
from models.train_test import *
#from prunning_methods.LTH import *
from fisher_information.NGD import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_loaders(data_dir: str, batch_size: int, device: torch.device, 
                  fim_size: int = 5000, seed: int = 42):
# CIFAR-10 stats
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    
    train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean, std),
    ])
    
    
    test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
    ])
    
    
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    num_classes = 10
    assert fim_size % num_classes == 0, \
        f"fim_size ({fim_size}) must be divisible by num_classes ({num_classes})"

    per_class = fim_size // num_classes

    targets = torch.tensor(train_set.targets)  # shape: [50000]
    g = torch.Generator().manual_seed(seed)
    
    indices_per_class = []
    for c in range(num_classes):
        class_idx = torch.nonzero(targets == c).view(-1)  # indices of samples of class c
        # shuffle indices for this class
        perm = class_idx[torch.randperm(len(class_idx), generator=g)]
        # take per_class samples
        indices_per_class.append(perm[:per_class])

    # concatenate all class indices and shuffle globally
    balanced_indices = torch.cat(indices_per_class)
    balanced_indices = balanced_indices[torch.randperm(len(balanced_indices), generator=g)]

    fim_subset = Subset(train_set, balanced_indices.tolist())
    
    train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True
    )

    fim_loader = DataLoader(
    fim_subset,
    batch_size=1,
    shuffle=True
    )
    
    test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False
    )
    
    return train_loader, fim_loader, test_loader


train_loader, fim_loader, test_loader = build_loaders('./data', 1028, device)


def resnet18_cifar(num_classes: int = 10) -> nn.Module:
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    # Replace the 7x7 stride-2 conv + maxpool with a 3x3 stride-1 conv and no pool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
#print(f"ResNet18 tem {sum([p.numel() for p in resnet18_cifar().parameters()])} parametros")

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_global_seed(42)


fim_args = {"complete_fim": False, 
            "layers":  ['layer1.1.conv1.weight'], 
            "mask":  None, 
            "sampling_type":  'x_skip_y', 
            "sampling_frequency":  (9,81)
            }



LTH_args = {"model": resnet18_cifar(num_classes=10).to(device), 
            "criterion": nn.CrossEntropyLoss(), 
           "train_loader": train_loader,
            "test_loader": test_loader,
            "fim_loader": fim_loader,
            "fim_args": fim_args, 
            "lr" : 1e-3,
            "n_iterations":10, 
            "n_epochs":80, 
            "prunning_percentage":0.1, 
            "no_prunning_layers":None,
            "real_opt": 'singd', # 'adam' or 'singd'
            "structure": "dense", # "diag" or "dense"
            "verbose":True,
            "print_freq":10, 
            "use_scheduler":False, 
            "save_path":None
            }
        
output_dict = train_LTH_adam_vs_ngd(**LTH_args)
torch.save(output_dict, "LTH_NGD_cifar10_output_dict_best_result.pth")