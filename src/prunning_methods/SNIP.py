import os
from pathlib import Path
import sys
import copy
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

repo_root = Path().resolve().parents[0]  
sys.path.insert(0, str(repo_root / "src"))
from fisher_information.fim import FisherInformationMatrix
from models.train_test import *


def _forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)


def _forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def compute_snip_scores(
    model: nn.Module, criterion: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device | str,) -> tuple[list[str], list[torch.Tensor]]:
    """Compute SNIP scores once using the model's initial weights."""
    net = copy.deepcopy(model).to(device)
    net.eval()

    inputs, targets = inputs.to(device), targets.to(device)

    parameter_names = []
    prunable_layers = []

    for module_name, layer in net.named_modules():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)) or layer.weight is None:
            continue

        layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight, device=device))
        layer.weight.requires_grad_(False)

        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(_forward_conv2d, layer)
        else:
            layer.forward = types.MethodType(_forward_linear, layer)

        parameter_names.append(f"{module_name}.weight")
        prunable_layers.append(layer)

    if not prunable_layers:
        raise ValueError("No Conv2d or Linear layers were found for SNIP.")

    net.zero_grad(set_to_none=True)
    loss = criterion(net(inputs), targets)
    loss.backward()

    score_tensors = []

    for layer in prunable_layers:
        if layer.weight_mask.grad is None:
            raise RuntimeError("No gradient was computed for a SNIP mask.")

        score_tensors.append(
            layer.weight_mask.grad.detach().abs().clone()
        )

    all_scores = torch.cat([score.reshape(-1) for score in score_tensors])
    norm = all_scores.sum()

    if not torch.isfinite(norm):
        raise RuntimeError("The sum of SNIP scores is not finite.")

    score_tensors = [score / (norm + 1e-8) for score in score_tensors]

    return parameter_names, score_tensors


def build_nested_snip_mask(parameter_names: list[str], score_tensors: list[torch.Tensor], keep_ratio: float) -> dict[str, torch.Tensor]:
    """Build a nested mask from a fixed SNIP ranking."""
    if len(parameter_names) != len(score_tensors):
        raise ValueError(
            "parameter_names and score_tensors must have the same length."
        )

    all_scores = torch.cat([score.reshape(-1) for score in score_tensors])
    total_parameters = all_scores.numel()

    number_to_keep = int(round(keep_ratio * total_parameters))
    number_to_keep = max(1, min(number_to_keep, total_parameters))

    selected_indices = torch.topk(
        all_scores,
        k=number_to_keep,
        largest=True,
        sorted=False,
    ).indices

    flat_mask = torch.zeros_like(all_scores)
    flat_mask[selected_indices] = 1.0

    mask_dict = {}
    offset = 0

    for parameter_name, score in zip(parameter_names, score_tensors):
        number_of_elements = score.numel()
        layer_mask = flat_mask[
            offset : offset + number_of_elements
        ].reshape_as(score)

        mask_dict[parameter_name] = layer_mask.clone()
        offset += number_of_elements

    if offset != total_parameters:
        raise RuntimeError(
            "Mask reconstruction used an incorrect number of elements."
        )

    return mask_dict


@torch.no_grad()
def apply_mask(
    model: nn.Module,
    mask: dict[str, torch.Tensor],
) -> None:
    """Aplica a máscara aos parâmetros do modelo."""
    for name, parameter in model.named_parameters():
        if name in mask:
            parameter.mul_(
                mask[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )


def count_active_weights(
    mask: dict[str, torch.Tensor],
) -> tuple[int, int]:
    """Count active and total prunable weights."""
    active = sum(int(layer_mask.sum().item()) for layer_mask in mask.values())
    total = sum(layer_mask.numel() for layer_mask in mask.values())

    return active, total


def train_snip_nested(
    model: nn.Module,
    criterion: nn.Module,
    train_loader,
    test_loader,
    fim_loader,
    fim_args: dict[str, Any] | None,
    keep_ratios: list[float],
    epochs: int,
    lr: float,
    verbose: bool = True,
    use_scheduler: bool = False,
    print_freq: int = 5,
    save_path: str | Path | None = None,
) -> dict:
    """Train independent SNIP subnetworks using nested pruning masks."""
    if not keep_ratios:
        raise ValueError("keep_ratios cannot be empty.")

    keep_ratios = [float(keep_ratio) for keep_ratio in keep_ratios]

    if any(keep_ratio <= 0.0 or keep_ratio > 1.0 for keep_ratio in keep_ratios):
        raise ValueError("Every keep_ratio must be in the interval (0, 1].")

    keep_ratios = sorted(set(keep_ratios), reverse=True)
    device = next(model.parameters()).device

    initial_state_dict = copy.deepcopy(model.state_dict())
    snip_inputs, snip_targets = next(iter(train_loader))

    parameter_names, score_tensors = compute_snip_scores(
        model=model,
        criterion=criterion,
        inputs=snip_inputs,
        targets=snip_targets,
        device=device,
    )

    masks = {
        keep_ratio: build_nested_snip_mask(
            parameter_names=parameter_names,
            score_tensors=score_tensors,
            keep_ratio=keep_ratio,
        )
        for keep_ratio in keep_ratios
    }

    output_dict = {
        "keep_ratios": [],
        "mask_list": [],
        "test_acc": [],
        "fim_list": [],
        "loss_list": [],
        "initial_state_dict": initial_state_dict,
    }

    for iteration, keep_ratio in enumerate(keep_ratios):
        if verbose:
            print("\n" + "=" * 80)
            print(f"SNIP experiment {iteration + 1}/{len(keep_ratios)}")
            print(f"Target remaining weights: {100.0 * keep_ratio:.1f}%")
            print("=" * 80)

        model.load_state_dict(initial_state_dict, strict=True)

        for parameter in model.parameters():
            parameter.grad = None

        mask = {
            name: layer_mask.clone().to(device)
            for name, layer_mask in masks[keep_ratio].items()
        }

        apply_mask(model, mask)

        active, total = count_active_weights(mask)

        if verbose:
            print(
                f"Active prunable weights: {active}/{total} "
                f"({100.0 * active / total:.4f}%)"
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        apply_mask(model, mask)
        accuracy = test(model, test_loader)

        if verbose:
            print(
                f"\nAccuracy with {100.0 * keep_ratio:.1f}% remaining: "
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

        output_dict["keep_ratios"].append(keep_ratio)
        output_dict["mask_list"].append(mask_cpu)
        output_dict["test_acc"].append(float(accuracy))
        output_dict["fim_list"].append(fim)
        output_dict["loss_list"].append(loss_list)

    model.load_state_dict(initial_state_dict, strict=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_dict, save_path)

    return output_dict
