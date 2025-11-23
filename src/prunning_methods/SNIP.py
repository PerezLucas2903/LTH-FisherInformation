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


def _forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)

def _forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def snip(model, criterion, keep_ratio, train_loader, device):
    model.eval()
    data_iter = iter(train_loader)
    inputs, targets = next(data_iter)
    inputs, targets = inputs.to(device), targets.to(device)

    # Perform SNIP on a copy of the model
    net = copy.deepcopy(model).to(device)
    net.eval()

    grads_abs = []
    name_list = []  

    # Only Conv2d and Linear
    for name, layer in net.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)) and layer.weight is not None:
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)   
            layer.weight.requires_grad = False

            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(_forward_conv2d, layer)
            else:
                layer.forward = types.MethodType(_forward_linear, layer)

            name_list.append(name + ".weight")

    # Forward + backward on a single batch
    net.zero_grad(set_to_none=True)
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    
    for name, layer in net.named_modules():
        if hasattr(layer, "weight_mask"):
            grads_abs.append(layer.weight_mask.grad.detach().abs().clone())

    all_scores = torch.cat([g.view(-1) for g in grads_abs])
    norm = all_scores.sum()

    all_scores = all_scores / (norm + 1e-8) 

    k = int(keep_ratio * all_scores.numel())
    k = max(1, min(k, all_scores.numel()))

    threshold = torch.topk(all_scores, k, sorted=True).values[-1]

    keep_masks = [((g / (norm + 1e-8)) >= threshold).float() for g in grads_abs]

    mask_dict = {name: mask for name, mask in zip(name_list, keep_masks)}

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in mask_dict:
                p.mul_(mask_dict[name].to(p.device))

    return mask_dict

def train_snip(model, criterion, train_loader, test_loader, fim_loader, fim_args, keep_ratio, epochs,
    lr, verbose=True, use_scheduler=False, print_freq=5, save_path=None)-> dict:
    
    device = next(model.parameters()).device

    # Prunning
    mask = snip(model, criterion, keep_ratio, train_loader, device)

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
        print(f"\nAccuracy after SNIP training: {acc*100:.2f}%")

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
