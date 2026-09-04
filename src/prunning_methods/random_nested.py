import os
from pathlib import Path
import sys
import copy
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path().resolve().parents[0]  
sys.path.insert(0, str(repo_root / "src"))
from fisher_information.fim import FisherInformationMatrix
from models.train_test import *

def progressive_random_pruning(
    model: nn.Module,
    target_keep_ratio: float,
    previous_mask: dict[str, torch.Tensor] | None = None,
    device=None,
) -> dict[str, torch.Tensor]:
    """
    Create a cumulative random pruning mask.

    Parameters
    ----------
    model : nn.Module
        Model to prune.
    target_keep_ratio : float
        Fraction of original prunable parameters to keep in each layer.
    previous_mask : dict | None
        Mask from the previous pruning step.
    device
        Device used to generate the mask.
    """
    if not 0.0 < target_keep_ratio <= 1.0:
        raise ValueError("target_keep_ratio must be in (0, 1].")

    if device is None:
        device = next(model.parameters()).device

    parameter_names = []
    current_masks = []
    shapes = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        parameter_names.append(name)
        shapes.append(parameter.shape)

        if previous_mask is None:
            current_mask = torch.ones(
                parameter.numel(),
                device=device,
                dtype=parameter.dtype,
            )
        else:
            if name not in previous_mask:
                raise KeyError(
                    f"Previous mask does not contain parameter {name}."
                )

            current_mask = previous_mask[name].reshape(-1).to(
                device=device,
                dtype=parameter.dtype,
            )

        current_masks.append(current_mask)

    if not current_masks:
        raise ValueError("No prunable parameters were found.")

    mask_dict = {}

    for name, shape, old_mask in zip(
        parameter_names,
        shapes,
        current_masks,
    ):
        target_active = int(round(
            target_keep_ratio * old_mask.numel()
        ))
        target_active = max(
            1,
            min(target_active, old_mask.numel()),
        )

        current_active_indices = torch.nonzero(
            old_mask.bool(),
            as_tuple=False,
        ).reshape(-1)
        current_active = current_active_indices.numel()

        if target_active > current_active:
            raise ValueError(
                f"Cannot increase active parameters in {name} from "
                f"{current_active} to {target_active}."
            )

        permutation = torch.randperm(
            current_active,
            device=device,
        )
        selected_active_indices = current_active_indices[
            permutation[:target_active]
        ]

        layer_mask = torch.zeros_like(old_mask)
        layer_mask[selected_active_indices] = 1.0
        layer_mask = layer_mask.reshape(shape)

        # Extra protection against reactivating previously pruned weights.
        layer_mask = layer_mask * old_mask.reshape(shape)

        mask_dict[name] = layer_mask

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in mask_dict:
                parameter.mul_(
                    mask_dict[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )

    active = sum(
        int(mask.sum().item())
        for mask in mask_dict.values()
    )
    total = sum(
        mask.numel()
        for mask in mask_dict.values()
    )

    print(
        f"Random pruning: {active}/{total} active "
        f"({100.0 * active / total:.4f}% remaining)"
    )

    return mask_dict

def train_progressive_random_pruning(
    model,
    criterion,
    train_loader,
    test_loader,
    fim_loader,
    fim_args,
    keep_ratios,
    epochs,
    lr,
    verbose=True,
    use_scheduler=False,
    print_freq=5,
    save_path=None,
):
    device = next(model.parameters()).device

    keep_ratios = sorted(
        {float(ratio) for ratio in keep_ratios},
        reverse=True,
    )

    if any(
        ratio <= 0.0 or ratio > 1.0
        for ratio in keep_ratios
    ):
        raise ValueError(
            "Every keep ratio must be in (0, 1]."
        )

    output_dict = {
        "keep_ratios": [],
        "mask_list": [],
        "test_acc": [],
        "fim_list": [],
        "loss_list": [],
    }

    previous_mask = None

    for iteration, keep_ratio in enumerate(keep_ratios):
        if verbose:
            print("\n" + "=" * 80)
            print(
                f"Random pruning iteration "
                f"{iteration + 1}/{len(keep_ratios)}"
            )
            print(
                f"Target remaining parameters: "
                f"{100.0 * keep_ratio:.1f}%"
            )
            print("=" * 80)

        mask = progressive_random_pruning(
            model=model,
            target_keep_ratio=keep_ratio,
            previous_mask=previous_mask,
            device=device,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )

        model, loss_list = train(
            model,
            criterion,
            optimizer,
            train_loader,
            n_epochs=epochs,
            mask=mask,
            verbose=verbose,
            use_scheduler=use_scheduler,
            print_freq=print_freq,
        )

        # Ensure pruned parameters remain zero after training.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in mask:
                    parameter.mul_(
                        mask[name].to(
                            device=parameter.device,
                            dtype=parameter.dtype,
                        )
                    )

        accuracy = test(
            model,
            test_loader,
        )

        if verbose:
            print(
                f"Accuracy with "
                f"{100.0 * keep_ratio:.1f}% remaining: "
                f"{100.0 * accuracy:.2f}%"
            )

        current_fim_args = dict(fim_args or {})
        current_fim_args["mask"] = mask

        fim = FisherInformationMatrix(
            model,
            criterion,
            optimizer,
            fim_loader,
            **current_fim_args,
        )
        fim._fim_to_cpu()

        mask_cpu = {
            name: tensor.detach().cpu().clone()
            for name, tensor in mask.items()
        }

        output_dict["keep_ratios"].append(
            keep_ratio
        )
        output_dict["mask_list"].append(
            mask_cpu
        )
        output_dict["test_acc"].append(
            float(accuracy)
        )
        output_dict["fim_list"].append(
            fim
        )
        output_dict["loss_list"].append(
            loss_list
        )

        # The next pruning step uses the current cumulative mask.
        previous_mask = mask

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        torch.save(
            output_dict,
            save_path,
        )

    return output_dict
