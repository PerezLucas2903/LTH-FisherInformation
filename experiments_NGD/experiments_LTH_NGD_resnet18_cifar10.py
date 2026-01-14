import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import random_split, TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt
import sys
import math


# # climb up to the repo root and add <repo>/src to Python's path
# repo_root = Path().resolve().parents[0]   # parent of "notebooks"
# sys.path.insert(0, str(repo_root / "src"))

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

from fisher_information.fim import FisherInformationMatrix
from models.train_test import *
#from prunning_methods.LTH import *
from models.image_classification_models import resnet18
from fisher_information.NGD import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_global_seed(seed: int) -> None:
    """Set the global seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    batch_size: int = 1028,
    fim_size: int = 5000,
    seed: int = 42,
):
    """
    CIFAR-10 loaders:
    - train tf: RandomCrop(32,padding=4) + RandomHorizontalFlip + ToTensor + Normalize
    - test tf: ToTensor + Normalize
    - fim_loader: balanced subset of size fim_size (fim_size/10 per class), batch_size=1
    """
    data_root = repo_root / "data"

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

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_tf
    )

    # Main train/test loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    # Build a balanced subset for FIM
    num_classes = 10
    assert fim_size % num_classes == 0, f"fim_size ({fim_size}) must be divisible by {num_classes}"
    per_class = fim_size // num_classes

    targets = torch.tensor(train_set.targets)  # 50000 labels
    g = torch.Generator().manual_seed(seed)

    fim_indices: List[int] = []
    for c in range(num_classes):
        class_idx = torch.nonzero(targets == c).view(-1)
        perm = class_idx[torch.randperm(class_idx.numel(), generator=g)]
        fim_indices.extend(perm[:per_class].tolist())

    fim_subset = Subset(train_set, fim_indices)
    fim_loader = DataLoader(
        fim_subset,
        batch_size=1,
        shuffle=True,
    )

    return train_loader, fim_loader, test_loader


#train_loader, fim_loader, test_loader = build_loaders('./data', 1028, device)



def run_experiments(
    n_lth_runs: int = 10,
    base_seed: int = 42,
    n_iterations: int = 10,
    prunning_percentage: float = 0.1,
    n_epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 1028,
    fim_size: int = 5000,
) -> Dict[int, List[Tuple[float, FisherInformationMatrix, dict, Dict[str, float], Dict[str, float]]]]:
    """
    Returns:
        results: dict where keys are remaining percentages (100, 90, ..., 10)
                 and values are lists of tuples:
                 (acc, fim_obj, mask, logdet_ratio_dict, logdet_ratio_per_dim_dict)
    """

    train_loader, fim_loader, test_loader = build_dataloaders(
        batch_size=batch_size,
        fim_size=fim_size,
        seed=base_seed,
    )

    # FIM configuration
    fim_args = {
        "complete_fim": False,
        "layers": [
            "layer1.0.conv1.weight",
            "layer1.0.conv2.weight",
            "layer1.1.conv1.weight",
            "layer1.1.conv2.weight",
            "layer2.0.conv1.weight",
        ],
        "mask": None,
        "sampling_type": "x_skip_y",
        "sampling_frequency": (9, 81),
    }

    percentages = list(range(100, 0, -10))  # 100, 90, ..., 10
    results: Dict[int, List[Tuple[float, FisherInformationMatrix, dict, Dict[str, float], Dict[str, float]]]] = {
        p: [] for p in percentages
    }

    for run_idx in range(n_lth_runs):
        seed = base_seed + run_idx
        print(f"========== Starting LTH run {run_idx + 1}/{n_lth_runs} (seed={seed}) ==========", flush=True)
        set_global_seed(seed)

        model = resnet18(num_classes=10).to(device)

        LTH_args = {
            "model": model,
            "criterion": nn.CrossEntropyLoss(),
            "train_loader": train_loader,
            "test_loader": test_loader,
            "fim_loader": fim_loader,
            "fim_args": fim_args,
            "lr": lr,
            "n_iterations": n_iterations,
            "n_epochs": n_epochs,
            "prunning_percentage": prunning_percentage,
            "no_prunning_layers": None,
            "real_opt": 'singd', # 'adam' or 'singd'
            "structure": "diagonal", # "diag" or "dense"
            "verbose": True,
            "print_freq": 10,
            "use_scheduler": False,
            "save_path": None,
        }

        
        output_dict = train_LTH_adam_vs_ngd(**LTH_args)

        mask_list = output_dict["mask_list"]
        acc_list = output_dict["test_acc"]
        cos_dist = output_dict["cos_dist_list"]

        step = int(prunning_percentage * 100)

        for i in range(len(acc_list)):
            remaining = 100 - step * i
            if remaining < 10:
                continue
            if remaining not in results:
                continue

            # fim_obj = fim_list[i]
            acc = float(acc_list[i])
            #mask = mask_list[i]
            cos_dist_value = cos_dist[i]

            # logdet_ratio = float(fim_obj.logdet_ratio)
            # logdet_ratio_per_dim = float(fim_obj.logdet_ratio_per_dim)

            # results[remaining].append(
            #     (acc, fim_obj, mask, cos_dist_value, logdet_ratio, logdet_ratio_per_dim)
            # )
            results[remaining].append(
                (acc, cos_dist_value))

    return results


def main():
    print("repo_root:", repo_root, flush=True)
    print("data_root:", (repo_root / "data"), flush=True)
    print("device:", device, flush=True)

    # defaults
    n_lth_runs = 1
    base_seed = 42
    n_iterations = 10
    prunning_percentage = 0.1
    n_epochs = 100
    lr = 1e-3
    batch_size = 1028
    fim_size = 8000

    results = run_experiments(
        n_lth_runs=n_lth_runs,
        base_seed=base_seed,
        n_iterations=n_iterations,
        prunning_percentage=prunning_percentage,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        fim_size=fim_size,
    )

    results_dir = repo_root / "results_NGD" / "ResNet18-CIFAR10"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "LTH_NGD_cifar10_resnet18.pth"

    print(f"\nSaving results to: {out_path}", flush=True)
    torch.save(results, out_path)


if __name__ == "__main__":
    main()
