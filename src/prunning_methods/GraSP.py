import os
from pathlib import Path
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

repo_root = Path().resolve().parents[0]
sys.path.insert(0, str(repo_root / "src"))
from fisher_information.fim import FisherInformationMatrix
from models.train_test import *


def grasp_fetch_data(dataloader, num_classes, samples_per_class):
    """
    Fetch a balanced mini-dataset: 'samples_per_class' samples for each class.
    Returns tensors X, y concatenated over classes.
    """
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)

    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x = inputs[idx:idx+1]
            y = targets[idx:idx+1]
            c = y.item()

            if len(datas[c]) == samples_per_class:
                mark[c] = True
                continue

            datas[c].append(x)
            labels[c].append(y)

        if len(mark) == num_classes:
            break

    X = torch.cat([torch.cat(v, 0) for v in datas], dim=0)
    y = torch.cat([torch.cat(v, 0) for v in labels], dim=0).view(-1)
    return X, y


class GraSPPruner:
    def __init__(
        self,
        no_pruning_layers=None,
        num_classes: int = 10,
        samples_per_class: int = 25,
        num_iters: int = 1,
        T: float = 200.0,
        reinit: bool = True,
    ):
        """
        GraSP pruner (Gradient Signal Preservation).

        Scores are computed once and can then be reused to create masks at
        different sparsity levels. Reusing the same scores guarantees nested masks.
        """
        self.no_pruning_layers = no_pruning_layers or []
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.num_iters = num_iters
        self.T = T
        self.reinit = reinit

    def compute_scores(
        self,
        model: nn.Module,
        train_loader,
        device,
    ) -> dict:
        """Compute GraSP scores once for all prunable weights."""
        model.to(device)

        net = copy.deepcopy(model).to(device)
        net.zero_grad()

        # Collect prunable Conv/Linear weights
        prunable_layers = []
        for name, layer in net.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                param_name = name + ".weight"
                if param_name not in self.no_pruning_layers:
                    prunable_layers.append((name, layer))

        # Optional reinitialization used by the original implementation
        if self.reinit:
            for _, layer in prunable_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)

        weights = [layer.weight for _, layer in prunable_layers]

        grad_w = None
        inputs_one = []
        targets_one = []

        # First phase: accumulate gradients
        for _ in range(self.num_iters):
            inputs, targets = grasp_fetch_data(
                train_loader,
                num_classes=self.num_classes,
                samples_per_class=self.samples_per_class,
            )

            N = inputs.shape[0]
            inputs_one.extend([inputs[:N // 2].clone(), inputs[N // 2:].clone()])
            targets_one.extend([targets[:N // 2].clone(), targets[N // 2:].clone()])

            inputs = inputs.to(device)
            targets = targets.to(device)

            for x, y in (
                (inputs[:N // 2], targets[:N // 2]),
                (inputs[N // 2:], targets[N // 2:]),
            ):
                outputs = net(x) / self.T
                loss = F.cross_entropy(outputs, y)
                grad = autograd.grad(loss, weights, create_graph=False)

                if grad_w is None:
                    grad_w = list(grad)
                else:
                    for i in range(len(grad_w)):
                        grad_w[i] += grad[i]

        # Second phase: Hessian-gradient product
        for inputs, targets in zip(inputs_one, targets_one):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = net(inputs) / self.T
            loss = F.cross_entropy(outputs, targets)
            grad_f = autograd.grad(loss, weights, create_graph=True)

            z = sum(
                (grad_w[i].data * grad_f[i]).sum()
                for i in range(len(weights))
            )
            z.backward()

        # GraSP scores: -theta * H g
        return {
            name + ".weight": -layer.weight.data * layer.weight.grad
            for name, layer in prunable_layers
        }

    def mask_from_scores(
        self,
        model: nn.Module,
        scores: dict,
        keep_ratio: float,
        device,
    ) -> dict:
        """
        Build a mask from fixed GraSP scores.

        If the same `scores` dict is reused for different keep_ratios,
        the resulting masks are nested by construction.
        """
        keep_ratio = float(keep_ratio)

        if keep_ratio >= 1.0:
            return {
                name: torch.ones_like(param, device=device)
                for name, param in model.named_parameters()
            }

        all_scores = torch.cat([score.view(-1) for score in scores.values()])
        total_params = all_scores.numel()
        num_params_to_rm = int(total_params * (1.0 - keep_ratio))
        num_params_to_rm = max(0, min(num_params_to_rm, total_params))

        if num_params_to_rm == 0:
            return {
                name: torch.ones_like(param, device=device)
                for name, param in model.named_parameters()
            }

        # GraSP prunes the largest scores
        threshold = torch.topk(
            all_scores,
            num_params_to_rm,
            sorted=True,
        ).values[-1]

        keep_masks = {
            name: (score <= threshold).float()
            for name, score in scores.items()
        }

        mask_dict = {}
        for name, param in model.named_parameters():
            if name.endswith("bias") or name in self.no_pruning_layers:
                mask_dict[name] = torch.ones_like(param, device=device)
            elif name in keep_masks:
                mask_dict[name] = keep_masks[name].to(device)
            else:
                mask_dict[name] = torch.ones_like(param, device=device)

        return mask_dict

    def compute_mask(
        self,
        model: nn.Module,
        train_loader,
        device,
        keep_ratio: float,
        scores: dict = None,
    ) -> dict:
        """
        Convenience wrapper.

        Pass precomputed `scores` when creating multiple sparsity levels so that
        all masks are guaranteed to be nested.
        """
        if scores is None:
            scores = self.compute_scores(model, train_loader, device)

        return self.mask_from_scores(
            model=model,
            scores=scores,
            keep_ratio=keep_ratio,
            device=device,
        )

    @torch.no_grad()
    def apply_mask(self, model: nn.Module, mask_dict: dict) -> nn.Module:
        """Multiply weights by mask."""
        for name, param in model.named_parameters():
            if name in mask_dict:
                param.mul_(mask_dict[name].to(param.device))
        return model


def train_grasp(
    model,
    criterion,
    train_loader,
    test_loader,
    fim_loader,
    fim_args,
    keep_ratio,
    epochs,
    scores=None,
    lr=1e-3,
    num_classes=10,
    samples_per_class=25,
    num_iters=1,
    T=200.0,
    reinit=True,
    no_pruning_layers=None,
    verbose=True,
    use_scheduler=False,
    print_freq=5,
    save_path=None,
) -> dict:
    """
    Train a model after GraSP pruning (one-shot, data-based).
    keep_ratio: final fraction of weights to keep.
    """
    device = next(model.parameters()).device

    # GraSP pruning
    pruner = GraSPPruner(
        no_pruning_layers=no_pruning_layers,
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        num_iters=num_iters,
        T=T,
        reinit=reinit,
    )

    mask = pruner.compute_mask(
        model=model,
        train_loader=train_loader,
        device=device,
        keep_ratio=keep_ratio,
        scores=scores,
    )
    pruner.apply_mask(model, mask)

    # Train sparse model
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

    # Test accuracy
    acc = test(model, test_loader)
    if verbose:
        print(f"\nAccuracy after GraSP training: {acc*100:.2f}%")

    # Compute FIM
    fim_args = dict(fim_args or {})
    fim_args["mask"] = mask
    fim = FisherInformationMatrix(model, criterion, optimizer, fim_loader, **fim_args)
    fim._fim_to_cpu()

    # Save results
    output_dict = {
        "mask_list": [mask],
        "test_acc": [acc],
        "fim_list": [fim],
    }

    if save_path is not None:
        torch.save(output_dict, save_path)

    return output_dict