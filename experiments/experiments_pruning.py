import argparse
import copy
import math
import random
import sys
from pathlib import Path

# Required arguments: --method (random, lth, snip, grasp, rigl),
# --model (resnet18, densenet) and --dataset (cifar10, svhn, stl10).
# Optional arguments configure epochs, seeds, learning rate, batch/FIM sizes
# and the output path; run this file with --help to see all available options.

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from models.image_classification_models import densenet121, resnet18
from prunning_methods.GraSP import GraSPPruner, train_grasp
from prunning_methods.LTH import train_LTH
from prunning_methods.RigL import train_RigL
from prunning_methods.SNIP import train_snip_nested
from prunning_methods.random_nested import train_progressive_random_pruning


REMAINING_PERCENTAGES = [
    100, 90, 80, 70, 60, 50,
    40, 30, 20, 10, 5, 3,
]

FIM_LAYERS = {
    "resnet18": [
        "layer1.0.conv1.weight",
        "layer1.0.conv2.weight",
        "layer1.1.conv1.weight",
        "layer1.1.conv2.weight",
        "layer2.0.conv1.weight",
    ],
    "densenet": [
        "features.conv0.weight",
        "features.denseblock1.denselayer2.conv2.weight",
        "features.denseblock4.denselayer15.conv2.weight",
    ],
}

DEFAULT_BATCH_SIZES = {
    ("resnet18", "cifar10"): 1028,
    ("resnet18", "svhn"): 256,
    ("resnet18", "stl10"): 256,
    ("densenet", "cifar10"): 2048,
    ("densenet", "svhn"): 1024,
    ("densenet", "stl10"): 1024,
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    dataset_name: str,
    batch_size: int,
    fim_size: int,
    seed: int,
):
    data_root = repo_root / "data"

    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        train_set = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform,
        )
        test_set = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=test_transform,
        )
        targets = torch.tensor(train_set.targets, dtype=torch.long)

    elif dataset_name == "svhn":
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        train_set = torchvision.datasets.SVHN(
            root=data_root,
            split="train",
            download=True,
            transform=train_transform,
        )
        test_set = torchvision.datasets.SVHN(
            root=data_root,
            split="test",
            download=True,
            transform=test_transform,
        )
        targets = torch.tensor(train_set.labels, dtype=torch.long)

    elif dataset_name == "stl10":
        mean = (0.4467, 0.4398, 0.4066)
        std = (0.2241, 0.2215, 0.2239)
        train_transform = T.Compose([
            T.Resize(64),
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        test_transform = T.Compose([
            T.Resize(64),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        train_set = torchvision.datasets.STL10(
            root=data_root,
            split="train",
            download=True,
            transform=train_transform,
        )
        test_set = torchvision.datasets.STL10(
            root=data_root,
            split="test",
            download=True,
            transform=test_transform,
        )
        targets = torch.tensor(train_set.labels, dtype=torch.long)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}.")

    num_classes = 10
    fim_size = min(fim_size, len(train_set))
    fim_size = (fim_size // num_classes) * num_classes
    if fim_size <= 0:
        raise ValueError("fim_size must contain at least one sample per class.")

    generator = torch.Generator().manual_seed(seed)
    per_class = fim_size // num_classes
    fim_indices = []

    for class_index in range(num_classes):
        class_indices = torch.nonzero(
            targets == class_index,
            as_tuple=False,
        ).reshape(-1)
        permutation = torch.randperm(
            class_indices.numel(),
            generator=generator,
        )
        fim_indices.extend(
            class_indices[permutation[:per_class]].tolist()
        )

    fim_subset = Subset(train_set, fim_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    fim_loader = DataLoader(
        fim_subset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, fim_loader, test_loader


def build_fim_args(model_name: str) -> dict:
    return {
        "complete_fim": False,
        "layers": FIM_LAYERS[model_name],
        "mask": None,
        "sampling_type": "x_skip_y",
        "sampling_frequency": (9, 81),
    }


def build_model(model_name: str, device: torch.device) -> nn.Module:
    if model_name == "resnet18":
        return resnet18(num_classes=10).to(device)
    return densenet121(num_classes=10).to(device)


def add_output_to_results(
    results: dict,
    remaining_percentages: list[int],
    output: dict,
) -> None:
    for remaining, mask, accuracy, fim in zip(
        remaining_percentages,
        output["mask_list"],
        output["test_acc"],
        output["fim_list"],
    ):
        logdet_ratio = dict(fim.logdet_ratio)
        logdet_ratio_per_dim = dict(fim.logdet_ratio_per_dim)

        for layer_name, value in logdet_ratio.items():
            if math.isinf(value):
                print(
                    f"[WARNING] logdet_ratio is "
                    f"{'+inf' if value > 0 else '-inf'} for layer "
                    f"'{layer_name}' at remaining={remaining}%",
                    flush=True,
                )

        results[remaining].append((
            float(accuracy),
            fim,
            mask,
            logdet_ratio,
            logdet_ratio_per_dim,
        ))


def run_one_seed(
    method: str,
    model_name: str,
    seed: int,
    device: torch.device,
    train_loader,
    fim_loader,
    test_loader,
    epochs: int,
    lr: float,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    fim_args = build_fim_args(model_name)
    keep_ratios = [value / 100.0 for value in REMAINING_PERCENTAGES]

    if method == "random":
        model = build_model(model_name, device)
        return train_progressive_random_pruning(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            fim_loader=fim_loader,
            fim_args=fim_args,
            keep_ratios=keep_ratios,
            epochs=epochs,
            lr=lr,
            verbose=True,
            use_scheduler=False,
            print_freq=10,
        )

    if method == "lth":
        model = build_model(model_name, device)
        return train_LTH(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            fim_loader=fim_loader,
            fim_args=fim_args,
            lr=lr,
            remaining_percentages=REMAINING_PERCENTAGES,
            n_epochs=epochs,
            no_prunning_layers=None,
            verbose=True,
            print_freq=10,
            use_scheduler=False,
        )

    if method == "snip":
        model = build_model(model_name, device)
        return train_snip_nested(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            fim_loader=fim_loader,
            fim_args=fim_args,
            keep_ratios=keep_ratios,
            epochs=epochs,
            lr=lr,
            verbose=True,
            use_scheduler=False,
            print_freq=10,
        )

    if method == "grasp":
        initial_model = build_model(model_name, device)
        pruner = GraSPPruner(
            no_pruning_layers=None,
            num_classes=10,
            samples_per_class=25,
            num_iters=1,
            T=200.0,
            reinit=True,
        )
        scores = pruner.compute_scores(
            model=initial_model,
            train_loader=train_loader,
            device=device,
        )

        combined_output = {
            "mask_list": [],
            "test_acc": [],
            "fim_list": [],
        }

        for remaining, keep_ratio in zip(
            REMAINING_PERCENTAGES,
            keep_ratios,
        ):
            print(
                f"\n----- GraSP: {remaining}% remaining "
                f"(seed={seed}) -----",
                flush=True,
            )
            model = copy.deepcopy(initial_model).to(device)
            output = train_grasp(
                model=model,
                criterion=criterion,
                train_loader=train_loader,
                test_loader=test_loader,
                fim_loader=fim_loader,
                fim_args=fim_args,
                keep_ratio=keep_ratio,
                epochs=epochs,
                scores=scores,
                lr=lr,
                num_classes=10,
                samples_per_class=25,
                num_iters=1,
                T=200.0,
                reinit=True,
                no_pruning_layers=None,
                verbose=True,
                use_scheduler=False,
                print_freq=10,
            )
            combined_output["mask_list"].append(output["mask_list"][0])
            combined_output["test_acc"].append(output["test_acc"][0])
            combined_output["fim_list"].append(output["fim_list"][0])

        return combined_output

    return train_RigL(
        model_factory=lambda: build_model(model_name, device),
        criterion_factory=nn.CrossEntropyLoss,
        train_loader=train_loader,
        test_loader=test_loader,
        fim_loader=fim_loader,
        fim_args=fim_args,
        remaining_percentages=REMAINING_PERCENTAGES,
        n_epochs=epochs,
        lr=lr,
        optimizer_name="adam",
        weight_decay=0.0,
        update_interval=100,
        drop_fraction=0.3,
        mask_update_begin_step=0,
        mask_update_end_fraction=0.75,
        mask_init_method="erk",
        erk_power_scale=1.0,
        no_pruning_layers=None,
        verbose=True,
        print_freq=10,
        calculate_fim=True,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one pruning method with one model and one dataset."
        )
    )
    parser.add_argument(
        "--method",
        required=True,
        type=str.lower,
        choices=["random", "lth", "snip", "grasp", "rigl"],
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str.lower,
        choices=["resnet18", "densenet", "densenet121"],
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str.lower,
        choices=["cifar10", "svhn", "stl10"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--fim-size", type=int, default=8000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = (
        "densenet" if args.model == "densenet121" else args.model
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or DEFAULT_BATCH_SIZES[
        (model_name, args.dataset)
    ]

    print("repo_root:", repo_root, flush=True)
    print("device:", device, flush=True)
    print("method:", args.method, flush=True)
    print("model:", model_name, flush=True)
    print("dataset:", args.dataset, flush=True)
    print("batch_size:", batch_size, flush=True)

    train_loader, fim_loader, test_loader = build_dataloaders(
        dataset_name=args.dataset,
        batch_size=batch_size,
        fim_size=args.fim_size,
        seed=args.base_seed,
    )

    results = {value: [] for value in REMAINING_PERCENTAGES}

    for run_index in range(args.seeds):
        seed = args.base_seed + run_index
        print(
            f"\n========== Starting {args.method} run "
            f"{run_index + 1}/{args.seeds} (seed={seed}) ==========",
            flush=True,
        )
        set_global_seed(seed)
        output = run_one_seed(
            method=args.method,
            model_name=model_name,
            seed=seed,
            device=device,
            train_loader=train_loader,
            fim_loader=fim_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
        )
        add_output_to_results(
            results=results,
            remaining_percentages=REMAINING_PERCENTAGES,
            output=output,
        )

    architecture_name = (
        "ResNet18" if model_name == "resnet18" else "DenseNet121"
    )
    results_dir = repo_root / "results" / (
        f"{architecture_name}-{args.dataset.upper()}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    method_name = {
        "random": "random_nested",
        "lth": "LTH",
        "snip": "SNIP",
        "grasp": "GraSP",
        "rigl": "RigL",
    }[args.method]
    model_file_name = (
        "resnet18" if model_name == "resnet18" else "densenet121"
    )
    output_path = args.output or (
        results_dir
        / f"{method_name}_{args.dataset}_{model_file_name}.pth"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {output_path}", flush=True)
    torch.save(results, output_path)


if __name__ == "__main__":
    main()
