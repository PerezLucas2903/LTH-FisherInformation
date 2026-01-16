import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset



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



def collate_fn(batch):
    # Stack images and labels into single tensors
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return images, labels

def apply_train_transforms(examples):
    train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    # Convert PIL image to tensor after applying transforms
    examples["image"] = [train_transforms(image) for image in examples["image"]]
    return examples

def apply_val_transforms(examples):
    val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    examples["image"] = [val_transforms(image) for image in examples["image"]]
    return examples

def balanced_subset_from_labels(dataset, labels, num_classes, fim_size, seed=42):
    assert fim_size % num_classes == 0
    per_class = fim_size // num_classes

    targets = torch.as_tensor(labels, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)

    # 1) group indices by class with ONE sort
    order = torch.argsort(targets)          # indices that would sort by label
    sorted_targets = targets[order]

    # 2) find class slice boundaries
    counts = torch.bincount(targets, minlength=num_classes)
    if (counts < per_class).any():
        bad = torch.nonzero(counts < per_class).view(-1).tolist()
        raise ValueError(f"Not enough samples for classes: {bad}")

    starts = torch.cumsum(counts, dim=0) - counts  # start offset for each class in `order`

    # 3) sample per_class indices from each class slice (fast: slices are small)
    picked = []
    for c in range(num_classes):
        start = starts[c].item()
        cnt = counts[c].item()
        class_indices = order[start:start + cnt]   # indices in original dataset for class c

        perm = torch.randperm(cnt, generator=g)[:per_class]
        picked.append(class_indices[perm])

    balanced_indices = torch.cat(picked)
    balanced_indices = balanced_indices[torch.randperm(len(balanced_indices), generator=g)]

    return Subset(dataset, balanced_indices.tolist())

def build_tiny_imagenet_loaders(data_dir: str, batch_size: int, device: torch.device, 
                  fim_size: int = 8000, seed: int = 42):
    num_classes = 200

    ds = load_dataset("slegroux/tiny-imagenet-200-clean")
    train_set = ds["train"]
    test_set = ds["validation"]

#     train_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # Define transforms for validation (without data augmentation)
#     val_transforms = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
    
    train_set.set_transform(apply_train_transforms)
    test_set.set_transform(apply_val_transforms)
        
    train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
    )

        
    test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
    )

    labels = train_loader.dataset['label']  # or dataset["label"]
    fim_subset = balanced_subset_from_labels(train_loader.dataset, labels, num_classes, fim_size, seed)

    fim_loader = DataLoader(fim_subset, batch_size=1, shuffle=True)

    fim_loader = DataLoader(
    fim_subset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn
    )
    
    return train_loader, fim_loader, test_loader