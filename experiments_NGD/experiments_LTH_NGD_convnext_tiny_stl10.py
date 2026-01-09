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
from models.image_classification_models import convnext_tiny
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
    batch_size: int,
    fim_size: int = 5000,
    seed: int = 42,
):
    data_root = repo_root / "data"
    # STL10 stats (RGB)
    STL10_MEAN = (0.4467, 0.4398, 0.4066)
    STL10_STD  = (0.2241, 0.2215, 0.2239)

    # Transforms (resize down from 96 -> 64 for speed)
    train_tf = T.Compose([
        T.Resize(64),
        T.RandomCrop(64, padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(STL10_MEAN, STL10_STD),
    ])

    test_tf = T.Compose([
        T.Resize(64),
        T.ToTensor(),
        T.Normalize(STL10_MEAN, STL10_STD),
    ])

    train_set = torchvision.datasets.STL10(
        root=data_root, split="train", download=True, transform=train_tf
    )
    test_set = torchvision.datasets.STL10(
        root=data_root, split="test", download=True, transform=test_tf
    )

    num_classes = 10

    # STL10 train has 5000 labeled samples total.
    if fim_size > len(train_set):
        fim_size = len(train_set)

    # make fim_size divisible by num_classes
    fim_size = (fim_size // num_classes) * num_classes
    assert fim_size > 0, "fim_size became 0; increase it."

    per_class = fim_size // num_classes

    # STL10 targets are a Python list of ints
    targets = torch.tensor(train_set.labels, dtype=torch.long)

    g = torch.Generator().manual_seed(seed)

    indices_per_class = []
    for c in range(num_classes):
        class_idx = torch.nonzero(targets == c).view(-1)
        perm = class_idx[torch.randperm(len(class_idx), generator=g)]
        indices_per_class.append(perm[:per_class])

    balanced_indices = torch.cat(indices_per_class)
    balanced_indices = balanced_indices[torch.randperm(len(balanced_indices), generator=g)]
    fim_subset = Subset(train_set, balanced_indices.tolist())

    # DataLoader settings
    # Windows notebooks can be finicky; 0 is safest. Bump to 2-4 if stable.
    num_workers = 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    fim_loader = DataLoader(
        fim_subset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
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
            "features.0.0.weight",
            "features.1.0.block.0.weight",
            "features.1.1.block.0.weight",
            "features.1.2.block.0.weight"
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

        model = convnext_tiny(num_classes=10).to(device)

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
        fim_list = output_dict["fim_list"]

        step = int(prunning_percentage * 100)

        for i in range(len(fim_list)):
            remaining = 100 - step * i
            if remaining < 10:
                continue
            if remaining not in results:
                continue

            fim_obj = fim_list[i]
            acc = float(acc_list[i])
            mask = mask_list[i]

            logdet_ratio: Dict[str, float] = dict(fim_obj.logdet_ratio)
            logdet_ratio_per_dim: Dict[str, float] = dict(fim_obj.logdet_ratio_per_dim)

            for name, value in logdet_ratio.items():
                if math.isinf(value):
                    print(
                        f"[WARNING] logdet_ratio is {'+inf' if value > 0 else '-inf'} "
                        f"for layer '{name}' at remaining={remaining}% (run={run_idx}, iter={i})",
                        flush=True,
                    )

            results[remaining].append((acc, fim_obj, mask, logdet_ratio, logdet_ratio_per_dim))

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
    n_epochs = 150
    lr = 1e-3
    batch_size = 1024
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

    results_dir = repo_root / "results_NGD" / "ConvNextTiny-stl10"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "LTH_NGD_stl10_convnext_tiny.pth"

    print(f"\nSaving results to: {out_path}", flush=True)
    torch.save(results, out_path)


if __name__ == "__main__":
    main()
