import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
import sys
sys.path.insert(0, str(src_path))

from fisher_information.fim import FisherInformationMatrix
from models.image_classification_models import ConvModelEMNIST
from prunning_methods.LTH import train_LTH  
from fisher_information.NGD import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int) -> None:
    """Set the global seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def build_dataloaders():
    data_root = repo_root / "data"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, 1, [1, 2])),
        transforms.Lambda(lambda x: torch.flip(x, [2])),
    ])

    emnist_train = torchvision.datasets.EMNIST(
        root=data_root, split="letters", train=True, download=True,
        transform=transform, target_transform=lambda y: y - 1
    )
    emnist_test = torchvision.datasets.EMNIST(
        root=data_root, split="letters", train=False, download=True,
        transform=transform, target_transform=lambda y: y - 1
    )
    
    ys = [emnist_train[i][1] for i in range(1000)]  # amostra de 1000
    print("EMNIST letters labels (sample): min =", min(ys), "max =", max(ys), flush=True)

    train_loader = DataLoader(emnist_train, batch_size=256, shuffle=True)
    fim_loader = DataLoader(emnist_train, batch_size=1, shuffle=True)
    test_loader = DataLoader(emnist_test, batch_size=20, shuffle=True)

    return train_loader, fim_loader, test_loader



def run_experiments(
    n_lth_runs: int = 10,
    base_seed: int = 42,
    n_iterations: int = 10,
    prunning_percentage: float = 0.1,
    n_epochs: int = 30,
    lr: float = 1e-3,
) -> Dict[int, List[Tuple[float, FisherInformationMatrix, dict, float, float]]]:
    """
    Runs the Lottery Ticket Hypothesis (LTH) multiple times and aggregates results.

    For each LTH run:
      - A new ConvModelEMNIST is initialized (with a different seed).
      - `train_LTH` is executed with the specified number of iterations and pruning percentage.
      - For each pruning step (100%, 90%, ..., 10%), store:
          (test_acc, fim_obj, mask, logdet_ratio, logdet_ratio_per_dim)

    Returns:
        results: a dictionary where:
            - keys: percentages of remaining parameters (100, 90, ..., 10)
            - values: lists of tuples, one tuple per LTH execution.
    """

    train_loader, fim_loader, test_loader = build_dataloaders()

    # FIM configuration
    fim_args = {
        "complete_fim": True,
        "layers": None,
        "mask": None,
        "sampling_type": "complete",
        "sampling_frequency": None,
    }

    # Prepare structure: for each percentage we store 10 tuples across runs
    percentages = list(range(100, 0, -10))  # [100, 90, ..., 10]
    results: Dict[int, List[Tuple[float, FisherInformationMatrix, dict, float, float]]] = {
        p: [] for p in percentages
    }

    for run_idx in range(n_lth_runs):
        seed = base_seed + run_idx
        print(f"========== Starting LTH run {run_idx + 1}/{n_lth_runs} (seed={seed}) ==========")
        set_global_seed(seed)

        model = ConvModelEMNIST(n_classes=26).to(device)

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
            "structure": "dense", # "diag" or "dense"
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
    # Main experiment configuration
    print("repo_root:", repo_root, flush=True)
    print("data_root:", (repo_root / "data"), flush=True)
    print("device:", device, flush=True)
    n_lth_runs = 1
    base_seed = 42
    n_iterations = 10
    prunning_percentage = 0.1
    n_epochs = 30
    lr = 1e-3

    results = run_experiments(
        n_lth_runs=n_lth_runs,
        base_seed=base_seed,
        n_iterations=n_iterations,
        prunning_percentage=prunning_percentage,
        n_epochs=n_epochs,
        lr=lr,
    )

    # Saving results
    results_dir = repo_root / "results_NGD" / "Conv-EMNIST"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "LTH_NGD_emnist_convmodel.pth"


    print(f"\nSaving results to: {out_path}")
    torch.save(results, out_path)


if __name__ == "__main__":
    main()
