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

def random_pruning(model, keep_ratio, device=None):
    if device is None:
        device = next(model.parameters()).device

    mask_dict = {}

    with torch.no_grad():
        for name, param in model.named_parameters():

            # Skip frozen parameters
            if not param.requires_grad:
                continue

            # Total number of elements in the tensor
            num_params = param.numel()
            k = int(num_params * keep_ratio)
            k = max(1, min(k, num_params))  # ensure at least one parameter is kept

            # Generate random scores for selecting which parameters to keep
            random_scores = torch.rand(num_params, device=device)

            # Select the top-k scores
            _, idx_keep = torch.topk(random_scores, k)
            mask = torch.zeros(num_params, device=device)
            mask[idx_keep] = 1.0
            mask = mask.view_as(param)

            # Store mask
            mask_dict[name] = mask.clone().detach()

            # Apply the mask directly to the model parameters
            param.mul_(mask)

    return mask_dict

def train_random_pruning(model, criterion, train_loader, test_loader, fim_loader, fim_args, keep_ratio, epochs,
    lr, verbose=True, use_scheduler=False, print_freq=5, save_path=None)-> dict:
    
    device = next(model.parameters()).device

    # Prunning
    mask = random_pruning(model, keep_ratio, device)

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, loss_list = train(
        model, criterion, optimizer, train_loader,
        n_epochs=epochs, mask=mask, verbose=verbose,
        use_scheduler=use_scheduler, print_freq=print_freq
    )

    # Test
    acc = test(model, test_loader)
    if verbose:
        print(f"\nAccuracy after random pruning and training: {acc*100:.2f}%")

    # FIM
    fim_args = dict(fim_args or {})
    fim_args['mask'] = mask
    fim = FisherInformationMatrix(model, criterion, optimizer, fim_loader, **fim_args)
    fim._fim_to_cpu()  

    output_dict = {
        'mask_list': [mask],
        'test_acc': [acc],
        'fim_list': [fim],
    }

    if save_path is not None:
        torch.save(output_dict, save_path)

    return output_dict