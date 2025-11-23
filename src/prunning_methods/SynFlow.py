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


class SynFlowPruner:
    def __init__(self, no_pruning_layers=None):
        """
        SynFlow pruner (data-free).
        no_pruning_layers: param names excluded from pruning.
        """
        self.no_pruning_layers = no_pruning_layers or []

    @torch.no_grad()
    def _linearize(self, model: nn.Module):
        """Make all weights positive and store signs."""
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def _nonlinearize(self, model: nn.Module, signs):
        """Restore original weight signs."""
        for name, param in model.state_dict().items():
            param.mul_(signs[name])

    def compute_scores(self, model: nn.Module, train_loader, device) -> dict:
        """Compute SynFlow scores |w * grad| for prunable params."""
        model.eval()
        model.to(device)

        # Make weights positive
        signs = self._linearize(model)

        # Input of ones
        data_iter = iter(train_loader)
        inputs, _ = next(data_iter)
        syn_input = torch.ones_like(inputs[:1]).to(device)

        # Gradients of sum(outputs)
        model.zero_grad(set_to_none=True)
        output = model(syn_input)
        torch.sum(output).backward()

        scores = {}
        for name, param in model.named_parameters():
            if name.endswith("bias") or name in self.no_pruning_layers:
                continue
            if param.grad is None:
                scores[name] = torch.zeros_like(param)
            else:
                scores[name] = (param.grad * param).abs()
                param.grad.zero_()

        # Restore original weights
        self._nonlinearize(model, signs)

        return scores

    def iterative_prune(
        self,
        model: nn.Module,
        train_loader,
        device,
        final_keep_ratio: float,
        n_iterations: int = 100,
    ) -> dict:
        """
        Iterative SynFlow pruning with exponential schedule.
        final_keep_ratio: target fraction of weights to keep at the end.
        n_iterations: number of pruning iterations.
        """
        model.eval()
        model.to(device)

        # Early stop
        if final_keep_ratio >= 1.0:
            full_mask = {}
            for name, param in model.named_parameters():
                if not name.endswith("bias") and name not in self.no_pruning_layers:
                    full_mask[name] = torch.ones_like(param, device=device)
            self.apply_mask(model, full_mask) 
            return full_mask

        current_mask = None
        total_params = None

        for k in range(1, n_iterations + 1):
            # SynFlow scores for current masked model
            scores = self.compute_scores(model, train_loader, device)

            # Count total prunable parameters once
            if total_params is None:
                total_params = sum(s.numel() for s in scores.values())

            # Do not allow already pruned weights to come back
            if current_mask is not None:
                for name in scores:
                    scores[name][current_mask[name] == 0] = -float("inf")

            # Exponential keep schedule: keep_k = final_keep_ratio^(k / n_iterations)
            keep_k = float(final_keep_ratio ** (k / n_iterations))
            keep_k = max(0.0, min(keep_k, 1.0))

            k_keep = max(1, min(int(keep_k * total_params), total_params))

            # Global threshold
            all_scores = torch.cat([s.view(-1) for s in scores.values()])
            threshold = torch.topk(all_scores, k_keep, sorted=True).values[-1]

            # Build new mask and intersect with previous mask
            new_mask = {}
            for name, score in scores.items():
                m = (score >= threshold).float()
                if current_mask is not None:
                    m = m * current_mask[name]
                new_mask[name] = m

            current_mask = new_mask

        # Apply final mask to the model
        self.apply_mask(model, current_mask)
        return current_mask

    @torch.no_grad()
    def apply_mask(self, model: nn.Module, mask_dict: dict) -> nn.Module:
        """Multiply weights by mask."""
        for name, param in model.named_parameters():
            if mask_dict is not None and name in mask_dict:
                param.mul_(mask_dict[name].to(param.device))
        return model


def train_synflow(
    model,
    criterion,
    train_loader,
    test_loader,
    fim_loader,
    fim_args,
    keep_ratio,
    epochs,
    lr=1e-3,
    n_iterations=100,
    no_pruning_layers=None,
    verbose=True,
    use_scheduler=False,
    print_freq=5,
    save_path=None,
) -> dict:
    """
    Train a model after iterative SynFlow pruning (exponential schedule).
    keep_ratio: final fraction of weights to keep.
    n_iterations: number of pruning iterations before training.
    """
    device = next(model.parameters()).device

    # Iterative SynFlow pruning
    pruner = SynFlowPruner(no_pruning_layers=no_pruning_layers)
    mask = pruner.iterative_prune(
        model,
        train_loader,
        device,
        final_keep_ratio=keep_ratio,
        n_iterations=n_iterations,
    )

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
        print(f"\nAccuracy after SynFlow training: {acc*100:.2f}%")

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
