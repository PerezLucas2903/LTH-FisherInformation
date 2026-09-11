import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# Required arguments: --model (resnet18, densenet, vgg16) and
# --dataset (cifar10, svhn, stl10).
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

PERCENTAGES = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 3]

METHOD_FILES = {
    "LTH": "LTH",
    "Random pruning": "random_nested",
    "GraSP": "GraSP",
    "SNIP": "SNIP",
    "RigL": "RigL",
}

MODEL_CONFIGS = {
    "resnet18": {
        "folder": "ResNet18",
        "file": "resnet18",
        "title": "ResNet18",
    },
    "densenet": {
        "folder": "DenseNet121",
        "file": "densenet121",
        "title": "DenseNet121",
    },
    "vgg16": {
        "folder": "VGG16",
        "file": "vgg16",
        "title": "VGG16",
    },
}

DATASET_TITLES = {
    "cifar10": "CIFAR10",
    "svhn": "SVHN",
    "stl10": "STL10",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot LogDet/Dim and accuracy curves for all pruning methods."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str.lower,
        choices=["resnet18", "densenet", "densenet121", "vgg16"],
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str.lower,
        choices=["cifar10", "svhn", "stl10"],
    )
    return parser.parse_args()


def build_paths(model_name, dataset_name):
    model_config = MODEL_CONFIGS[model_name]
    dataset_title = DATASET_TITLES[dataset_name]
    results_dir = (
        repo_root
        / "results"
        / f"{model_config['folder']}-{dataset_title}"
    )

    result_files = {
        method_name: results_dir
        / f"{file_prefix}_{dataset_name}_{model_config['file']}.pth"
        for method_name, file_prefix in METHOD_FILES.items()
    }

    output_stem = results_dir / (
        "lth_random_grasp_snip_rigl_"
        f"{model_config['file']}_{dataset_name}_100_to_3"
    )

    return model_config, dataset_title, results_dir, result_files, output_stem


def load_results(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    return torch.load(
        file_path,
        map_location="cpu",
        weights_only=False,
    )


def mean_accuracy(runs):
    return float(np.mean([float(run[0]) for run in runs]))


def run_logdet_per_dim(run):
    value = run[4]

    if isinstance(value, dict):
        layer_values = np.array(
            [float(layer_value) for layer_value in value.values()],
            dtype=float,
        )
        finite_values = layer_values[np.isfinite(layer_values)]

        if len(finite_values) == 0:
            return float("nan")

        return float(np.mean(finite_values))

    return float(value)


def logdet_dim_stats(runs, method_name, percentage):
    values = np.array(
        [run_logdet_per_dim(run) for run in runs],
        dtype=float,
    )
    finite_values = values[np.isfinite(values)]
    removed = len(values) - len(finite_values)

    if removed > 0:
        print(
            f"Warning: {method_name} at {percentage}% remaining: "
            f"ignored {removed} non-finite LogDet/Dim values."
        )

    if len(finite_values) == 0:
        return np.nan, np.nan

    mean = float(np.mean(finite_values))
    std = (
        float(np.std(finite_values, ddof=1))
        if len(finite_values) > 1
        else 0.0
    )
    return mean, std


def load_all_results(result_files):
    all_results = {}

    for method_name, file_path in result_files.items():
        results = load_results(file_path)
        missing = [
            percentage
            for percentage in PERCENTAGES
            if percentage not in results
        ]

        if missing:
            raise KeyError(
                f"{method_name}: missing percentages {missing}. "
                f"Available keys: {sorted(results.keys(), reverse=True)}"
            )

        empty = [
            percentage
            for percentage in PERCENTAGES
            if not results[percentage]
        ]

        if empty:
            raise ValueError(
                f"{method_name}: no runs found for percentages {empty}."
            )

        all_results[method_name] = results

    return all_results


def compute_curves(all_results):
    accuracy_curves = {}
    logdet_curves = {}
    logdet_std_curves = {}

    for method_name, results in all_results.items():
        accuracy_curves[method_name] = []
        logdet_curves[method_name] = []
        logdet_std_curves[method_name] = []

        for percentage in PERCENTAGES:
            runs = results[percentage]
            accuracy_curves[method_name].append(mean_accuracy(runs))

            mean_logdet, std_logdet = logdet_dim_stats(
                runs,
                method_name=method_name,
                percentage=percentage,
            )
            logdet_curves[method_name].append(mean_logdet)
            logdet_std_curves[method_name].append(std_logdet)

    return accuracy_curves, logdet_curves, logdet_std_curves


def print_results(
    experiment_title,
    accuracy_curves,
    logdet_curves,
    logdet_std_curves,
):
    print(f"\n{experiment_title} — mean values")
    print("=" * 110)

    for method_name in METHOD_FILES:
        print(f"\n{method_name}")
        print(
            f"{'Remaining':>10} | {'LogDet/Dim':>15} | "
            f"{'LogDet SD':>15} | {'Accuracy':>12}"
        )
        print("-" * 62)

        for index, percentage in enumerate(PERCENTAGES):
            print(
                f"{percentage:>9}% | "
                f"{logdet_curves[method_name][index]:>15.8f} | "
                f"{logdet_std_curves[method_name][index]:>15.8f} | "
                f"{accuracy_curves[method_name][index]:>12.6f}"
            )


def plot_curves(
    experiment_title,
    output_stem,
    accuracy_curves,
    logdet_curves,
    logdet_std_curves,
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    for method_name in METHOD_FILES:
        mean_values = np.array(logdet_curves[method_name])
        std_values = np.array(logdet_std_curves[method_name])

        line, = axes[0].plot(
            PERCENTAGES,
            mean_values,
            marker="o",
            label=method_name,
        )
        axes[0].fill_between(
            PERCENTAGES,
            mean_values - std_values,
            mean_values + std_values,
            color=line.get_color(),
            alpha=0.12,
            linewidth=0,
        )

        axes[1].plot(
            PERCENTAGES,
            accuracy_curves[method_name],
            marker="o",
            label=method_name,
        )

    axes[0].set_title(f"{experiment_title} — LogDet/Dim")
    axes[0].set_xlabel("Remaining parameters (%)")
    axes[0].set_ylabel(r"$f(H)/\mathrm{dim}$")

    axes[1].set_title(f"{experiment_title} — Accuracy")
    axes[1].set_xlabel("Remaining parameters (%)")
    axes[1].set_ylabel("Accuracy")

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
        axis.invert_xaxis()

    fig.suptitle(f"{experiment_title} Pruning Curves")
    fig.tight_layout()

    output_png = output_stem.with_suffix(".png")
    output_pdf = output_stem.with_suffix(".pdf")

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved PNG to: {output_png}")
    print(f"Saved PDF to: {output_pdf}")


def main():
    args = parse_args()
    model_name = "densenet" if args.model == "densenet121" else args.model
    (
        model_config,
        dataset_title,
        results_dir,
        result_files,
        output_stem,
    ) = build_paths(model_name, args.dataset)

    experiment_title = f"{model_config['title']}-{dataset_title}"

    print("\n" + "#" * 110)
    print(f"Processing {experiment_title}")
    print(f"Results directory: {results_dir}")
    print("#" * 110)

    all_results = load_all_results(result_files)
    accuracy_curves, logdet_curves, logdet_std_curves = compute_curves(
        all_results
    )
    print_results(
        experiment_title,
        accuracy_curves,
        logdet_curves,
        logdet_std_curves,
    )
    plot_curves(
        experiment_title,
        output_stem,
        accuracy_curves,
        logdet_curves,
        logdet_std_curves,
    )


if __name__ == "__main__":
    main()
