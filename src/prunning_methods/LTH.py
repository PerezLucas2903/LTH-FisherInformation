import os
from pathlib import Path
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


repo_root = Path().resolve().parents[0]
sys.path.insert(0, str(repo_root / "src"))

from fisher_information.fim import FisherInformationMatrix
from models.train_test import *
from fisher_information.jacobian import *


class LTHPruner:
    def __init__(self, remaining_percentage: float, no_pruning_layers: list = None):
        """Lottery Ticket Hypothesis Pruner.

        params:
            remaining_percentage: percentage of weights that should remain
                                  in each layer (between 0 and 100)
            no_pruning_layers: list of layer names to exclude from pruning
        """
        self.remaining_percentage = remaining_percentage
        self.no_pruning_layers = (
            no_pruning_layers if no_pruning_layers is not None else []
        )

    def prune_weights(
        self,
        model: nn.Module,
        prev_mask: dict,
    ) -> dict:
        """
        Layer-wise magnitude pruning to a target remaining percentage.

        The new mask is always a subset of prev_mask, guaranteeing nested masks.
        """

        mask_dict = {}

        with torch.no_grad():
            for name, param in model.named_parameters():

                if (
                    name.split(".")[-1] != "bias"
                    and name not in self.no_pruning_layers
                ):
                    previous_mask = prev_mask[name].bool()

                    total_weights = param.numel()

                    target_remaining = round(
                        total_weights * self.remaining_percentage / 100.0
                    )

                    # Cannot reactivate weights that were already pruned
                    current_active = int(previous_mask.sum().item())
                    target_remaining = min(target_remaining, current_active)

                    new_mask = torch.zeros_like(
                        param,
                        dtype=torch.float32,
                    )

                    if target_remaining > 0:

                        # Magnitudes only among weights that are still active
                        active_indices = previous_mask.flatten().nonzero(
                            as_tuple=False
                        ).squeeze(1)

                        active_magnitudes = (
                            param.abs().flatten()[active_indices]
                        )

                        # Keep the target_remaining largest weights
                        _, top_indices = torch.topk(
                            active_magnitudes,
                            k=target_remaining,
                            largest=True,
                        )

                        surviving_indices = active_indices[top_indices]

                        new_mask_flat = new_mask.flatten()
                        new_mask_flat[surviving_indices] = 1.0
                        new_mask = new_mask_flat.view_as(param)

                    param *= new_mask
                    mask_dict[name] = new_mask

        return mask_dict

    def apply_mask(
        self,
        model: nn.Module,
        mask: dict,
    ) -> nn.Module:

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in mask:
                    param *= mask[name]

        return model


def reset_weights(
    model: nn.Module,
    initial_state_dict: dict,
    mask: dict = None,
) -> nn.Module:

    model.load_state_dict(
        initial_state_dict,
        strict=True,
    )

    # Rewind surviving weights to initialization
    # and keep previously pruned weights at zero.
    if mask is not None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in mask:
                    param *= mask[name]

    for p in model.parameters():
        p.grad = None

    return model


def train_LTH(
    model,
    criterion,
    train_loader,
    test_loader,
    fim_loader,
    fim_args,
    lr=1e-3,
    remaining_percentages=None,
    n_epochs=20,
    no_prunning_layers=None,
    verbose=True,
    print_freq=5,
    use_scheduler=False,
    save_path=None,
    calculate_fim=True,
    calculate_jacobian=False,
    save_model=False,
) -> dict:

    if remaining_percentages is None:
        remaining_percentages = [
            100, 90, 80, 70, 60, 50,
            40, 30, 20, 10, 5, 3,
        ]

    initial_state_dict = copy.deepcopy(
        model.state_dict()
    )

    # Initially all weights are active
    mask = {
        name: torch.ones_like(param)
        for name, param in model.named_parameters()
        if (
            name.split(".")[-1] != "bias"
            and name not in (no_prunning_layers or [])
        )
    }

    output_dict = {
        "mask_list": [],
        "test_acc": [],
        "fim_list": [],
    }

    if calculate_jacobian:
        output_dict["jacobian_list"] = []

    if save_model:
        output_dict["model_list"] = []

    n_iterations = len(remaining_percentages)

    for it, remaining_percentage in enumerate(
        remaining_percentages
    ):

        if verbose:
            print(
                f"\nLTH Iteration {it + 1}/{n_iterations} "
                f"— {remaining_percentage}% remaining"
            )

        # 1. Rewind surviving weights to their initial values

        model = reset_weights(
            model,
            initial_state_dict,
            mask,
        )

        # New optimizer after rewinding
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )

        # 2. Train CURRENT ticket

        model, loss_list = train(
            model,
            criterion,
            optimizer,
            train_loader,
            n_epochs,
            mask,
            verbose,
            use_scheduler,
            print_freq,
        )

        # 3. Evaluate current ticket

        acc = test(
            model,
            test_loader,
        )

        if verbose:
            print(
                f"Test Accuracy at "
                f"{remaining_percentage}% remaining: "
                f"{acc * 100:.2f}%"
            )

        # 4. Compute FIM for current ticket

        fim_args["mask"] = mask

        if calculate_fim:
            fim = FisherInformationMatrix(
                model,
                criterion,
                optimizer,
                fim_loader,
                **fim_args,
            )

            fim._fim_to_cpu()

            output_dict["fim_list"].append(
                fim
            )

        # Save mask corresponding exactly to this trained model
        output_dict["mask_list"].append(
            copy.deepcopy(mask)
        )

        output_dict["test_acc"].append(
            acc
        )

        if calculate_jacobian:
            jacobian = jacobian_param_l2_norms(
                model,
                train_loader,
            )

            output_dict["jacobian_list"].append(
                jacobian
            )

        if save_model:
            output_dict["model_list"].append(
                copy.deepcopy(
                    model.state_dict()
                )
            )

        # 5. AFTER evaluation, prune to the NEXT target sparsity

        if it < n_iterations - 1:

            next_remaining_percentage = (
                remaining_percentages[it + 1]
            )

            pruner = LTHPruner(
                remaining_percentage=next_remaining_percentage,
                no_pruning_layers=no_prunning_layers,
            )

            mask = pruner.prune_weights(
                model,
                prev_mask=mask,
            )

    if save_path is not None:
        torch.save(
            output_dict,
            save_path,
        )

    return output_dict