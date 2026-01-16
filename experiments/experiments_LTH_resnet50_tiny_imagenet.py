import random
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
import sys
sys.path.insert(0, str(src_path))

from fisher_information.fim import FisherInformationMatrix
from models.image_classification_models import resnet50
from prunning_methods.LTH import train_LTH
from data_processing.build_datasets import build_tiny_imagenet_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    train_loader, fim_loader, test_loader = build_tiny_imagenet_loaders(
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

        model = resnet50(num_classes=10).to(device)

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
            "verbose": True,
            "print_freq": 10,
            "use_scheduler": False,
            "save_path": None,
        }

        output_dict = train_LTH(**LTH_args)

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

    results_dir = repo_root / "results" / "ResNet50-tinyimagenet"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "LTH_tinyimagenet_resnet50.pth"

    print(f"\nSaving results to: {out_path}", flush=True)
    torch.save(results, out_path)


if __name__ == "__main__":
    main()
