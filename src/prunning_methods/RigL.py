from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from fisher_information.fim import FisherInformationMatrix
from models.train_test import test


def _is_prunable_parameter(
    name: str,
    parameter: nn.Parameter,
    no_pruning_layers: list[str],
) -> bool:
    """
    Default pruning rule:
      - do not prune biases;
      - do not prune explicitly excluded parameters;
      - prune weight tensors with ndim >= 2 (Linear/Conv and similar layers).

    Using ndim >= 2 instead of checking module classes makes this usable with
    more architectures without hard-coding Conv2d/Linear.
    """
    return (
        parameter.requires_grad
        and parameter.ndim >= 2
        and name.split(".")[-1] != "bias"
        and name not in no_pruning_layers
    )


def get_prunable_parameters(
    model: nn.Module,
    no_pruning_layers: list[str] | None = None,
) -> dict[str, nn.Parameter]:
    no_pruning_layers = no_pruning_layers or []

    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if _is_prunable_parameter(name, parameter, no_pruning_layers)
    }

    if not parameters:
        raise ValueError("No prunable parameters were found.")

    return parameters


def _erk_raw_probability(shape: torch.Size, power_scale: float = 1.0) -> float:
    """
    Erdős-Rényi-Kernel probability used by the original RigL repository:

        p_l ∝ (sum(shape_l) / prod(shape_l)) ** power_scale

    For Conv weights this includes kernel dimensions.
    """
    shape_values = [int(x) for x in shape]
    return (
        sum(shape_values) / math.prod(shape_values)
    ) ** power_scale


def compute_erk_sparsities(
    parameters: dict[str, nn.Parameter],
    target_sparsity: float,
    erk_power_scale: float = 1.0,
    custom_sparsities: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    PyTorch port of the ERK layerwise sparsity allocation used in
    google-research/rigl sparse_utils.get_sparsities_erdos_renyi.

    The allocation preserves approximately the same total parameter count as
    uniform sparsity at target_sparsity, while layers whose probability would
    exceed 1 are made dense.
    """
    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in [0, 1).")

    custom_sparsities = dict(custom_sparsities or {})

    unknown = set(custom_sparsities) - set(parameters)
    if unknown:
        raise ValueError(
            f"custom_sparsities contains unknown parameters: {sorted(unknown)}"
        )

    for name, sparsity in custom_sparsities.items():
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(
                f"Custom sparsity for {name} must be in [0, 1], got {sparsity}."
            )

    if target_sparsity == 0.0:
        return {
            name: custom_sparsities.get(name, 0.0)
            for name in parameters
        }

    dense_layers: set[str] = set()

    while True:
        divisor = 0.0
        rhs = 0.0
        raw_probabilities: dict[str, float] = {}

        for name, parameter in parameters.items():
            n_param = parameter.numel()
            n_zeros = int(math.floor(target_sparsity * n_param))

            if name in dense_layers:
                rhs -= n_zeros
            elif name in custom_sparsities:
                # Same behavior as original code: custom layers are excluded
                # from the epsilon calculation.
                continue
            else:
                n_ones = n_param - n_zeros
                rhs += n_ones

                raw_probability = _erk_raw_probability(
                    parameter.shape,
                    power_scale=erk_power_scale,
                )
                raw_probabilities[name] = raw_probability
                divisor += raw_probability * n_param

        if not raw_probabilities:
            break

        if divisor <= 0:
            raise RuntimeError("Invalid ERK divisor.")

        epsilon = rhs / divisor
        max_probability = max(raw_probabilities.values())

        if epsilon * max_probability <= 1.0:
            break

        # As in the original implementation, layers attaining the maximum
        # probability are forced dense and epsilon is recomputed.
        for name, probability in raw_probabilities.items():
            if probability == max_probability:
                dense_layers.add(name)

    sparsities: dict[str, float] = {}

    for name in parameters:
        if name in custom_sparsities:
            sparsities[name] = custom_sparsities[name]
        elif name in dense_layers:
            sparsities[name] = 0.0
        else:
            probability_one = epsilon * raw_probabilities[name]
            sparsities[name] = 1.0 - probability_one

    return sparsities


def initialize_mask(
    model: nn.Module,
    target_density: float,
    mask_init_method: str = "erk",
    no_pruning_layers: list[str] | None = None,
    erk_power_scale: float = 1.0,
    custom_sparsities: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Initialize a fixed-density sparse topology.

    mask_init_method:
        "erk" / "erdos_renyi_kernel": ERK allocation across layers.
        "uniform" / "random": same target sparsity in every prunable layer.

    Within each layer, active locations are sampled uniformly at random.
    """
    if not 0.0 < target_density <= 1.0:
        raise ValueError("target_density must be in (0, 1].")

    parameters = get_prunable_parameters(model, no_pruning_layers)
    target_sparsity = 1.0 - target_density

    method = mask_init_method.lower()

    if method in {"erk", "erdos_renyi_kernel"}:
        layer_sparsities = compute_erk_sparsities(
            parameters=parameters,
            target_sparsity=target_sparsity,
            erk_power_scale=erk_power_scale,
            custom_sparsities=custom_sparsities,
        )
    elif method in {"uniform", "random"}:
        custom_sparsities = custom_sparsities or {}
        layer_sparsities = {
            name: custom_sparsities.get(name, target_sparsity)
            for name in parameters
        }
    else:
        raise ValueError(
            "mask_init_method must be one of "
            "{'erk', 'erdos_renyi_kernel', 'uniform', 'random'}."
        )

    mask: dict[str, torch.Tensor] = {}

    for name, parameter in parameters.items():
        sparsity = layer_sparsities[name]
        n_total = parameter.numel()

        # Original RigL random-mask code chooses floor(sparsity * n) zeros.
        n_zeros = int(math.floor(sparsity * n_total))
        n_ones = n_total - n_zeros

        flat_mask = torch.zeros(
            n_total,
            device=parameter.device,
            dtype=parameter.dtype,
        )

        if n_ones > 0:
            active_indices = torch.randperm(
                n_total,
                device=parameter.device,
            )[:n_ones]
            flat_mask[active_indices] = 1.0

        mask[name] = flat_mask.view_as(parameter)

    return mask


@torch.no_grad()
def apply_mask(
    model: nn.Module,
    mask: dict[str, torch.Tensor],
) -> None:
    """Force all inactive parameters to zero."""
    for name, parameter in model.named_parameters():
        if name in mask:
            parameter.mul_(
                mask[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )


@torch.no_grad()
def mask_gradients(
    model: nn.Module,
    mask: dict[str, torch.Tensor],
) -> None:
    """
    Restrict optimizer gradients to connections that were active for the
    current forward/backward pass.
    """
    for name, parameter in model.named_parameters():
        if name in mask and parameter.grad is not None:
            parameter.grad.mul_(
                mask[name].to(
                    device=parameter.grad.device,
                    dtype=parameter.grad.dtype,
                )
            )


def clone_dense_gradients(
    model: nn.Module,
    mask: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Save the instantaneous gradients before masking them for the optimizer.

    Since inactive weights are stored as zero parameters (rather than detached
    from the computational graph), their gradients are still available and can
    be used by RigL to select new connections.
    """
    dense_gradients: dict[str, torch.Tensor] = {}

    for name, parameter in model.named_parameters():
        if name not in mask:
            continue

        if parameter.grad is None:
            raise RuntimeError(
                f"No gradient found for prunable parameter {name}."
            )

        dense_gradients[name] = parameter.grad.detach().clone()

    return dense_gradients


def cosine_drop_fraction(
    step: int,
    begin_step: int,
    end_step: int,
    initial_drop_fraction: float,
) -> float:
    """
    Cosine annealing used by the original RigL sparse optimizer.
    """
    if step < begin_step or step > end_step:
        return 0.0

    if end_step <= begin_step:
        return 0.0

    progress = (step - begin_step) / float(end_step - begin_step)
    progress = min(max(progress, 0.0), 1.0)

    return initial_drop_fraction * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _reset_optimizer_state_at_indices(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    indices_mask: torch.Tensor,
) -> None:
    """
    Reset optimizer slots (momentum, Adam exp_avg/exp_avg_sq, etc.) at
    newly activated locations, matching the reset behavior in RigL.

    Scalar optimizer state such as Adam's 'step' is left unchanged.
    """
    state = optimizer.state.get(parameter, {})

    for value in state.values():
        if (
            torch.is_tensor(value)
            and value.shape == parameter.shape
        ):
            value[indices_mask] = 0


class RigLPruner:
    """
    Dynamic sparse topology manager following the RigL update rule.

    At each topology update, independently for every prunable layer:
      1. drop a fraction of currently active weights with lowest |w|;
      2. grow the same number of inactive weights with largest
         instantaneous |dL/dw|;
      3. initialize new weights at zero;
      4. reset optimizer state for newly activated weights.

    This preserves each layer's number of active connections and therefore
    preserves the total density throughout training.
    """

    def __init__(
        self,
        model: nn.Module,
        mask: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        drop_fraction: float = 0.3,
        update_interval: int = 100,
        begin_step: int = 0,
        end_step: int | None = None,
        anneal: str = "cosine",
        noise_std: float = 1e-5,
    ):
        if not 0.0 <= drop_fraction <= 1.0:
            raise ValueError("drop_fraction must be in [0, 1].")
        if update_interval <= 0:
            raise ValueError("update_interval must be positive.")

        self.model = model
        self.mask = mask
        self.optimizer = optimizer
        self.initial_drop_fraction = float(drop_fraction)
        self.update_interval = int(update_interval)
        self.begin_step = int(begin_step)
        self.end_step = end_step
        self.anneal = anneal.lower()
        self.noise_std = float(noise_std)

        self.last_update_step = begin_step - update_interval

    def should_update(self, step: int) -> bool:
        if self.end_step is not None and step > self.end_step:
            return False
        if step < self.begin_step:
            return False
        return step >= self.last_update_step + self.update_interval

    def current_drop_fraction(self, step: int) -> float:
        if self.anneal == "constant":
            return self.initial_drop_fraction

        if self.anneal == "cosine":
            if self.end_step is None:
                raise ValueError(
                    "Cosine annealing requires an explicit end_step."
                )
            return cosine_drop_fraction(
                step=step,
                begin_step=self.begin_step,
                end_step=self.end_step,
                initial_drop_fraction=self.initial_drop_fraction,
            )

        raise ValueError(
            "anneal must be either 'constant' or 'cosine'."
        )

    @torch.no_grad()
    def update_topology(
        self,
        dense_gradients: dict[str, torch.Tensor],
        step: int,
    ) -> dict[str, int]:
        if not self.should_update(step):
            return {}

        drop_fraction = self.current_drop_fraction(step)
        parameter_dict = dict(self.model.named_parameters())
        update_counts: dict[str, int] = {}

        for name, layer_mask in self.mask.items():
            parameter = parameter_dict[name]
            gradient = dense_gradients[name]

            old_mask = layer_mask.bool()
            n_active = int(old_mask.sum().item())

            n_prune = int(n_active * drop_fraction)

            if n_prune <= 0 or n_active == 0:
                update_counts[name] = 0
                continue

            n_inactive = parameter.numel() - n_active
            n_prune = min(n_prune, n_inactive)

            if n_prune <= 0:
                update_counts[name] = 0
                continue

            flat_weight = parameter.detach().flatten()
            flat_gradient = gradient.detach().flatten()
            flat_old_mask = old_mask.flatten()

            active_indices = flat_old_mask.nonzero(
                as_tuple=False
            ).squeeze(1)

            inactive_indices = (~flat_old_mask).nonzero(
                as_tuple=False
            ).squeeze(1)

            # Drop score from original RigL: |masked weight| with a tiny
            # random perturbation to avoid deterministic tie pathologies.
            active_scores = flat_weight[active_indices].abs()

            if self.noise_std > 0:
                active_scores = active_scores + torch.randn_like(
                    active_scores
                ) * self.noise_std

            prune_local = torch.topk(
                active_scores,
                k=n_prune,
                largest=False,
                sorted=False,
            ).indices
            prune_indices = active_indices[prune_local]

            # Grow score from original RigL: instantaneous |gradient| among
            # currently inactive connections.
            grow_scores = flat_gradient[inactive_indices].abs()
            grow_local = torch.topk(
                grow_scores,
                k=n_prune,
                largest=True,
                sorted=False,
            ).indices
            grow_indices = inactive_indices[grow_local]

            new_flat_mask = flat_old_mask.clone()
            new_flat_mask[prune_indices] = False
            new_flat_mask[grow_indices] = True

            new_mask_bool = new_flat_mask.view_as(parameter)

            dropped_bool = old_mask & (~new_mask_bool)
            grown_bool = (~old_mask) & new_mask_bool

            # New RigL connections are zero initialized.
            parameter[dropped_bool] = 0
            parameter[grown_bool] = 0

            self.mask[name] = new_mask_bool.to(
                dtype=parameter.dtype,
                device=parameter.device,
            )

            _reset_optimizer_state_at_indices(
                optimizer=self.optimizer,
                parameter=parameter,
                indices_mask=grown_bool,
            )

            update_counts[name] = n_prune

        self.last_update_step = step
        return update_counts


def count_active_weights(
    mask: dict[str, torch.Tensor],
) -> tuple[int, int]:
    active = sum(int(layer_mask.sum().item()) for layer_mask in mask.values())
    total = sum(layer_mask.numel() for layer_mask in mask.values())
    return active, total


def train_rigl_single_density(
    model: nn.Module,
    criterion: nn.Module,
    train_loader,
    test_loader,
    fim_loader,
    fim_args: dict[str, Any] | None,
    target_density: float,
    n_epochs: int = 30,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    update_interval: int = 100,
    drop_fraction: float = 0.3,
    mask_update_begin_step: int = 0,
    mask_update_end_fraction: float = 0.75,
    mask_init_method: str = "erk",
    erk_power_scale: float = 1.0,
    no_pruning_layers: list[str] | None = None,
    custom_sparsities: dict[str, float] | None = None,
    verbose: bool = True,
    print_freq: int = 5,
    calculate_fim: bool = True,
    calculate_jacobian: bool = False,
    jacobian_fn: Callable | None = None,
    save_model: bool = False,
) -> dict:
    """
    Train one RigL network at one fixed target density.

    Important: unlike LTH, different density levels should be trained
    independently from the same random initialization distribution.
    """
    if n_epochs <= 0:
        raise ValueError("n_epochs must be positive.")

    if not 0.0 < mask_update_end_fraction <= 1.0:
        raise ValueError(
            "mask_update_end_fraction must be in (0, 1]."
        )

    device = next(model.parameters()).device
    initial_state_dict = copy.deepcopy(model.state_dict())

    mask = initialize_mask(
        model=model,
        target_density=target_density,
        mask_init_method=mask_init_method,
        no_pruning_layers=no_pruning_layers,
        erk_power_scale=erk_power_scale,
        custom_sparsities=custom_sparsities,
    )

    apply_mask(model, mask)

    optimizer_name_lower = optimizer_name.lower()

    if optimizer_name_lower == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name_lower == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name_lower in {"momentum", "sgd_momentum"}:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer_name_lower in {"nesterov", "sgd_nesterov"}:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            "optimizer_name must be one of "
            "{'adam', 'sgd', 'momentum', 'nesterov'}."
        )

    total_steps = n_epochs * len(train_loader)
    mask_update_end_step = int(
        math.floor(total_steps * mask_update_end_fraction)
    )

    rigl = RigLPruner(
        model=model,
        mask=mask,
        optimizer=optimizer,
        drop_fraction=drop_fraction,
        update_interval=update_interval,
        begin_step=mask_update_begin_step,
        end_step=mask_update_end_step,
        anneal="cosine",
    )

    loss_list: list[float] = []
    topology_updates: list[dict[str, Any]] = []
    global_step = 0

    active, total = count_active_weights(mask)

    if verbose:
        print(
            f"RigL target density: {100.0 * target_density:.2f}% | "
            f"actual initial density: {100.0 * active / total:.4f}% "
            f"({active}/{total})"
        )
        print(
            f"Mask updates: every {update_interval} steps, "
            f"from step {mask_update_begin_step} "
            f"to {mask_update_end_step}, "
            f"initial drop fraction={drop_fraction}"
        )

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Inactive parameters must be exactly zero for the sparse forward.
            apply_mask(model, rigl.mask)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Dense instantaneous gradients are needed for RigL growth.
            dense_gradients = clone_dense_gradients(
                model=model,
                mask=rigl.mask,
            )

            # The optimizer itself must only update connections that were
            # active during this forward/backward pass.
            old_mask = {
                name: tensor.clone()
                for name, tensor in rigl.mask.items()
            }
            mask_gradients(model, old_mask)

            if (
                target_density < 1.0
                and rigl.should_update(global_step)
            ):
                counts = rigl.update_topology(
                    dense_gradients=dense_gradients,
                    step=global_step,
                )

                topology_updates.append(
                    {
                        "step": global_step,
                        "drop_fraction": rigl.current_drop_fraction(
                            global_step
                        ),
                        "updated_connections": counts,
                    }
                )

            optimizer.step()

            # This removes optimizer updates at connections pruned during the
            # current step and guarantees exact sparsity after every batch.
            apply_mask(model, rigl.mask)

            running_loss += float(loss.item())
            global_step += 1

        epoch_loss = running_loss / max(1, len(train_loader))
        loss_list.append(epoch_loss)

        if verbose and (
            (epoch + 1) % print_freq == 0
            or epoch == 0
            or epoch + 1 == n_epochs
        ):
            active, total = count_active_weights(rigl.mask)
            print(
                f"Epoch {epoch + 1:03d}/{n_epochs} | "
                f"loss={epoch_loss:.6f} | "
                f"density={100.0 * active / total:.4f}%"
            )

    apply_mask(model, rigl.mask)
    accuracy = test(model, test_loader)

    if verbose:
        print(
            f"Final accuracy at {100.0 * target_density:.1f}% density: "
            f"{100.0 * float(accuracy):.2f}%"
        )

    final_mask = {
        name: tensor.detach().cpu().clone()
        for name, tensor in rigl.mask.items()
    }

    output: dict[str, Any] = {
        "target_density": float(target_density),
        "mask": final_mask,
        "test_acc": float(accuracy),
        "loss_list": loss_list,
        "topology_updates": topology_updates,
        "initial_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in initial_state_dict.items()
        },
    }

    current_fim_args = dict(fim_args or {})
    current_fim_args["mask"] = rigl.mask

    if calculate_fim:
        fim = FisherInformationMatrix(
            model,
            criterion,
            optimizer,
            fim_loader,
            **current_fim_args,
        )
        fim._fim_to_cpu()
        output["fim"] = fim

    if calculate_jacobian:
        if jacobian_fn is None:
            raise ValueError(
                "calculate_jacobian=True requires jacobian_fn."
            )
        output["jacobian"] = jacobian_fn(
            model,
            train_loader,
        )

    if save_model:
        output["model_state_dict"] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

    return output


def train_RigL(
    model_factory: Callable[[], nn.Module],
    criterion_factory: Callable[[], nn.Module],
    train_loader,
    test_loader,
    fim_loader,
    fim_args: dict[str, Any] | None,
    remaining_percentages: list[int] | None = None,
    n_epochs: int = 30,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    update_interval: int = 100,
    drop_fraction: float = 0.3,
    mask_update_begin_step: int = 0,
    mask_update_end_fraction: float = 0.75,
    mask_init_method: str = "erk",
    erk_power_scale: float = 1.0,
    no_pruning_layers: list[str] | None = None,
    custom_sparsities: dict[str, float] | None = None,
    verbose: bool = True,
    print_freq: int = 5,
    calculate_fim: bool = True,
    calculate_jacobian: bool = False,
    jacobian_fn: Callable | None = None,
    save_model: bool = False,
    save_path: str | Path | None = None,
    seed: int | None = None,
) -> dict:
    """
    Run independent RigL trainings for several target densities.

    model_factory is intentionally used instead of a single model instance:
    every density gets a fresh model initialization, which avoids accidental
    weight/topology carry-over between sparsity experiments.
    """
    if remaining_percentages is None:
        remaining_percentages = [
            100, 90, 80, 70, 60, 50,
            40, 30, 20, 10, 5, 3,
        ]

    output_dict: dict[str, Any] = {
        "remaining_percentages": [],
        "mask_list": [],
        "test_acc": [],
        "fim_list": [],
        "loss_list": [],
        "topology_updates": [],
    }

    if calculate_jacobian:
        output_dict["jacobian_list"] = []

    if save_model:
        output_dict["model_list"] = []

    for iteration, remaining_percentage in enumerate(
        remaining_percentages
    ):
        if not 0 < remaining_percentage <= 100:
            raise ValueError(
                "remaining_percentages must contain values in (0, 100]."
            )

        if verbose:
            print("\n" + "=" * 80)
            print(
                f"RigL experiment {iteration + 1}/"
                f"{len(remaining_percentages)}"
            )
            print(
                f"Target remaining weights: "
                f"{remaining_percentage:.1f}%"
            )
            print("=" * 80)

        # Re-seed before each density so that, within one experimental run,
        # all sparsity levels start from the same dense initialization and
        # from the same RNG state. This makes comparisons across densities
        # much cleaner.
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        model = model_factory()
        criterion = criterion_factory()

        result = train_rigl_single_density(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            fim_loader=fim_loader,
            fim_args=fim_args,
            target_density=remaining_percentage / 100.0,
            n_epochs=n_epochs,
            lr=lr,
            optimizer_name=optimizer_name,
            momentum=momentum,
            weight_decay=weight_decay,
            update_interval=update_interval,
            drop_fraction=drop_fraction,
            mask_update_begin_step=mask_update_begin_step,
            mask_update_end_fraction=mask_update_end_fraction,
            mask_init_method=mask_init_method,
            erk_power_scale=erk_power_scale,
            no_pruning_layers=no_pruning_layers,
            custom_sparsities=custom_sparsities,
            verbose=verbose,
            print_freq=print_freq,
            calculate_fim=calculate_fim,
            calculate_jacobian=calculate_jacobian,
            jacobian_fn=jacobian_fn,
            save_model=save_model,
        )

        output_dict["remaining_percentages"].append(
            remaining_percentage
        )
        output_dict["mask_list"].append(result["mask"])
        output_dict["test_acc"].append(result["test_acc"])
        output_dict["loss_list"].append(result["loss_list"])
        output_dict["topology_updates"].append(
            result["topology_updates"]
        )

        if calculate_fim:
            output_dict["fim_list"].append(result["fim"])

        if calculate_jacobian:
            output_dict["jacobian_list"].append(
                result["jacobian"]
            )

        if save_model:
            output_dict["model_list"].append(
                result["model_state_dict"]
            )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_dict, save_path)

    return output_dict
