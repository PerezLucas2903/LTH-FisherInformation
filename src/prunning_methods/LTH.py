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
    def __init__(self, pruning_percentage: float, no_pruning_layers: list = None):
        """ Lottery Ticket Hypothesis Pruner.
        params:
            prunning_percentage: float, percentage of weights to prune (between 0 and 1)
            no_prunning_layers: list of layer names to exclude from pruning
        """
        self.pruning_percentage = pruning_percentage
        self.no_pruning_layers = no_pruning_layers if no_pruning_layers is not None else []

    def prune_weights(self, model: nn.Module) -> dict:
        mask_dict = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.split('.')[-1] != 'bias' and name not in self.no_pruning_layers:
                    # Calculate the pruning threshold
                    #threshold = torch.quantile(param.abs(), pruning_percentage)
                    threshold = torch.quantile(param.abs(), self.pruning_percentage)
                    mask = (param.abs() >= threshold).float()
                    param *= mask
                    mask_dict[name] = mask
        return mask_dict
    

    def apply_mask(self, model: nn.Module, mask: dict) -> nn.Module:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in mask.keys():
                    param *= mask[name]
        return model
    
def reset_weights(model: nn.Module, initial_state_dict: dict) -> nn.Module:
    model.load_state_dict(initial_state_dict, strict=True)
    for p in model.parameters():
        p.grad = None
    return model

def _train_LTH_one_iter(model, criterion, optimizer, train_loader, n_epochs=20, prunning_percentage=0.2, no_prunning_layers=None, verbose=True, print_freq=5,use_scheduler=False):
    pruner = LTHPruner(prunning_percentage, no_prunning_layers)
    mask = pruner.prune_weights(model)
    model, loss_list = train(model, criterion, optimizer, train_loader, n_epochs, mask, verbose, use_scheduler, print_freq)
    return model, mask, loss_list

def train_LTH(model, criterion, train_loader, test_loader, fim_loader, fim_args, lr = 1e-3,
              n_iterations=5, n_epochs=20, prunning_percentage=0.2, no_prunning_layers=None, 
              verbose=True, print_freq=5,use_scheduler=False, save_path=None, calculate_fim=True,
              calculate_jacobian=False, save_model=False) -> dict:
    
    initial_state_dict = copy.deepcopy(model.state_dict())
    output_dict = {'mask_list': [], 'test_acc': [], "fim_list" : []}
    if calculate_jacobian:
        output_dict['jacobian_list'] = []
    if save_model:
        output_dict['model_list'] = []

    for it in range(n_iterations):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        current_prunning_percentage = prunning_percentage * it
        if verbose:
            print(f"LTH Iteration {it+1}/{n_iterations}")
            
        model, mask, loss_list = _train_LTH_one_iter(model, criterion, optimizer, train_loader, n_epochs, current_prunning_percentage, no_prunning_layers, verbose, print_freq,use_scheduler)
        acc = test(model, test_loader)
        fim_args['mask'] = mask
        if verbose:
            print(f"Test Accuracy after iteration {it+1}: {acc*100:.2f}%")


        if calculate_fim:
            fim = FisherInformationMatrix(model, criterion, optimizer, fim_loader, **fim_args)
            fim._fim_to_cpu()
            output_dict['fim_list'].append(fim)

        

        output_dict['mask_list'].append(mask)
        output_dict['test_acc'].append(acc)
        #output_dict['fim_list'].append(fim)

        if calculate_jacobian:
            jacobian = jacobian_param_l2_norms(model, train_loader)
            output_dict['jacobian_list'].append(jacobian)


        if save_model:
            output_dict['model_list'].append(copy.deepcopy(model.state_dict()))
            
        model = reset_weights(model, initial_state_dict)

    if save_path is not None:
        torch.save(output_dict, save_path)

    return output_dict