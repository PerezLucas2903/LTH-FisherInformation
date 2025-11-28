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
        no_pruning_layers: list of parameter names ('layer.weight') to exclude from pruning.
        num_classes: number of classes in the dataset.
        samples_per_class: number of samples per class for GraSP scoring.
        num_iters: how many GraSP iterations to run.
        T: temperature scaling for outputs.
        reinit: whether to reinitialize Linear layers in the copied network.
        """
        self.no_pruning_layers = no_pruning_layers or []
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.num_iters = num_iters
        self.T = T
        self.reinit = reinit

    def compute_mask(
        self,
        model: nn.Module,
        train_loader,
        device,
        keep_ratio: float,
    ) -> dict:
        """
        Compute GraSP pruning mask for the given model.
        keep_ratio: fraction of parameters to keep (0–1).
        Returns: dict param_name -> mask tensor {0,1}.
        """
        model.to(device)

        # Early exit: no pruning
        if keep_ratio >= 1.0:
            mask_dict = {}
            for name, param in model.named_parameters():
                if name.endswith("bias") or name in self.no_pruning_layers:
                    mask_dict[name] = torch.ones_like(param, device=device)
                else:
                    mask_dict[name] = torch.ones_like(param, device=device)
            return mask_dict

        eps = 1e-10
        keep_ratio = float(keep_ratio)

        # Work on a copy of the model
        net = copy.deepcopy(model).to(device)
        net.zero_grad()

        # Collect prunable layers (Conv2d, Linear) and their names
        prunable_layers = []
        for name, layer in net.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                param_name = name + ".weight"
                if param_name in self.no_pruning_layers:
                    continue
                prunable_layers.append((name, layer))

        weights = [layer.weight for (_, layer) in prunable_layers]

        # Optionally reinit Linear layers in the copied network
        if self.reinit:
            for name, layer in prunable_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)

        # Ensure gradients are tracked for weights
        for w in weights:
            w.requires_grad_(True)

        grad_w = None
        inputs_one = []
        targets_one = []

        # ========== First phase: accumulate grad_w ==========
        for it in range(self.num_iters):
            inputs, targets = grasp_fetch_data(
                train_loader,
                num_classes=self.num_classes,
                samples_per_class=self.samples_per_class,
            )

            N = inputs.shape[0]
            din = inputs.clone()
            dtarget = targets.clone()

            # Split data into two halves (as in original GraSP code)
            inputs_one.append(din[: N // 2])
            targets_one.append(dtarget[: N // 2])
            inputs_one.append(din[N // 2 :])
            targets_one.append(dtarget[N // 2 :])

            inputs = inputs.to(device)
            targets = targets.to(device)

            # First half
            outputs = net(inputs[: N // 2]) / self.T
            loss = F.cross_entropy(outputs, targets[: N // 2])
            grad_w_p = autograd.grad(loss, weights, create_graph=False)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

            # Second half
            outputs = net(inputs[N // 2 :]) / self.T
            loss = F.cross_entropy(outputs, targets[N // 2 :])
            grad_w_p = autograd.grad(loss, weights, create_graph=False)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

        # ========== Second phase: accumulate Hessian-gradient product ==========
        for it in range(len(inputs_one)):
            inputs = inputs_one.pop(0).to(device)
            targets = targets_one.pop(0).to(device)

            outputs = net(inputs) / self.T
            loss = F.cross_entropy(outputs, targets)

            grad_f = autograd.grad(loss, weights, create_graph=True)

            z = 0
            for idx in range(len(weights)):
                z += (grad_w[idx].data * grad_f[idx]).sum()
            z.backward()

        # ========== Build grads dict (scores) for original model param names ==========
        grads = {}
        # prunable_layers is in same order as weights/grad_f/grad_w
        for (name, layer) in prunable_layers:
            param_name = name + ".weight"
            grads[param_name] = -layer.weight.data * layer.weight.grad  # -θ ⊙ H g

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([g.view(-1) for g in grads.values()])
        norm_factor = torch.abs(all_scores.sum()) + eps
        all_scores.div_(norm_factor)

        total_params = all_scores.numel()
        num_params_to_rm = int(total_params * (1.0 - keep_ratio))
        num_params_to_rm = max(0, min(num_params_to_rm, total_params))

        if num_params_to_rm == 0:
            # Nothing to prune; full masks
            mask_dict = {}
            for name, param in model.named_parameters():
                mask_dict[name] = torch.ones_like(param, device=device)
            return mask_dict

        # Threshold on scores (we prune the largest scores)
        threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
        acceptable_score = threshold[-1]

        # Build keep masks for prunable weights
        keep_masks = {}
        for param_name, g in grads.items():
            score = g / norm_factor
            keep_masks[param_name] = (score <= acceptable_score).float()

        # Now build full mask_dict (for all params in the original model)
        mask_dict = {}
        for name, param in model.named_parameters():
            if name.endswith("bias") or name in self.no_pruning_layers:
                # Never prune biases or protected layers
                mask_dict[name] = torch.ones_like(param, device=device)
            elif name in keep_masks:
                mask_dict[name] = keep_masks[name].to(device)
            else:
                # Non-conv/linear params: keep them
                mask_dict[name] = torch.ones_like(param, device=device)

        return mask_dict

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
