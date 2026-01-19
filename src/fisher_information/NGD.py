"""
hybrid_singd_adam_lth.py

Single-file implementation of your LTH experiment comparing Adam vs SINGD (NGD),
with an automatic fallback that updates SINGD-UNSUPPORTED layers using Adam only.

Key behavior:
- SINGD updates supported layers.
- Any layers SINGD reports as "will not be trained" are automatically updated by
  an Adam "fallback" optimizer that only owns those parameters.

You can drop this file into your project and import the functions you already use:
- train_LTH_adam_vs_ngd(...)
"""

import os
from pathlib import Path
import sys
import copy
import re
import warnings
from typing import Optional, Iterable, Dict, Any, Set, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from singd.optim.optimizer import SINGD  # external NGD / K-FAC-like optimizer

# --- Your repo import (as you had it) ---
repo_root = Path().resolve().parents[0]
sys.path.insert(0, str(repo_root / "src"))
from prunning_methods.LTH import *  # noqa: F401,F403  (LTHPruner, reset_weights, test, FisherInformationMatrix, ...)


# =============================================================================
# Utilities
# =============================================================================

def _flatten_params(params_iterable: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in params_iterable])


def build_flat_mask(mask_dict: dict, model: nn.Module) -> torch.Tensor:
    flats = []
    for name, p in model.named_parameters():
        if name in mask_dict:
            flats.append(mask_dict[name].view(-1))
        else:
            # parameters that were never pruned (e.g. biases) → all ones
            flats.append(torch.ones_like(p).view(-1))
    return torch.cat(flats)


def _get_submodule_safe(model: nn.Module, name: str) -> nn.Module:
    """Torch-safe get_submodule with a fallback for older versions."""
    if hasattr(model, "get_submodule"):
        return model.get_submodule(name)

    cur = model
    for part in name.split("."):
        if not hasattr(cur, part):
            raise AttributeError(f"Model has no submodule '{name}' (failed at '{part}')")
        cur = getattr(cur, part)
    return cur


_UNSUPPORTED_LAYER_RE = re.compile(r"will not be trained in layer\s+([^:]+):")


def _parse_unsupported_layer_names_from_warnings(
    warning_records: Iterable[warnings.WarningMessage],
) -> Set[str]:
    names: Set[str] = set()
    for w in warning_records:
        msg = str(w.message)
        m = _UNSUPPORTED_LAYER_RE.search(msg)
        if m:
            names.add(m.group(1).strip())
    return names


# =============================================================================
# Hybrid optimizer: SINGD + Adam fallback for unsupported layers
# =============================================================================

class HybridSINGD:
    """
    Optimizer-like wrapper:
      - SINGD updates supported layers
      - Adam updates ONLY the parameters belonging to layers SINGD warned are unsupported

    This wrapper supports .zero_grad(set_to_none=...) and .step().
    """

    def __init__(self, singd_opt: SINGD, adam_fallback_opt: Optional[torch.optim.Optimizer] = None):
        self.singd = singd_opt
        self.adam_fallback = adam_fallback_opt

        # Make it look optimizer-like (some code expects param_groups)
        self.param_groups = []
        if hasattr(self.singd, "param_groups"):
            self.param_groups += list(self.singd.param_groups)
        if self.adam_fallback is not None and hasattr(self.adam_fallback, "param_groups"):
            self.param_groups += list(self.adam_fallback.param_groups)

    def zero_grad(self, set_to_none: bool = True):
        self.singd.zero_grad(set_to_none=set_to_none)
        if self.adam_fallback is not None:
            self.adam_fallback.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        out = self.singd.step(closure=closure)
        if self.adam_fallback is not None:
            self.adam_fallback.step(closure=closure)
        return out

    def state_dict(self) -> Dict[str, Any]:
        d = {"singd": self.singd.state_dict()}
        if self.adam_fallback is not None:
            d["adam_fallback"] = self.adam_fallback.state_dict()
        return d

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.singd.load_state_dict(state_dict["singd"])
        if self.adam_fallback is not None and "adam_fallback" in state_dict:
            self.adam_fallback.load_state_dict(state_dict["adam_fallback"])

    def __getattr__(self, item: str):
        """
        Forward unknown attributes to the wrapped SINGD optimizer.
        This helps if external code accesses .defaults, etc.
        """
        return getattr(self.singd, item)


def build_singd_with_adam_fallback(
    model: nn.Module,
    *,
    ngd_lr: float,
    structure: str,
    adam_reference: torch.optim.Optimizer,
    warn_unsupported: bool = True,
    verbose: bool = True,
) -> Tuple[HybridSINGD, Set[str]]:
    """
    Construct SINGD and parse its warnings to detect unsupported layers.
    Then create an Adam optimizer over ONLY those layers' parameters.

    Returns:
      (hybrid_optimizer, unsupported_layer_names)
    """
    # Copy Adam hyperparams from the provided Adam optimizer (group 0)
    g0 = adam_reference.param_groups[0]
    adam_kwargs = dict(
        lr=g0.get("lr", ngd_lr),
        betas=g0.get("betas", (0.9, 0.999)),
        eps=g0.get("eps", 1e-8),
        weight_decay=g0.get("weight_decay", 0.0),
        amsgrad=g0.get("amsgrad", False),
    )

    # Capture SINGD warnings during construction
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always", UserWarning)
        singd_opt = SINGD(
            model,
            lr=ngd_lr,
            structures=(structure, structure),
            warn_unsupported=warn_unsupported,
        )

    unsupported_layers = _parse_unsupported_layer_names_from_warnings(rec)

    # Collect params belonging to those layers (non-recursive: only the module's own params)
    unsupported_params: List[torch.nn.Parameter] = []
    for layer_name in sorted(unsupported_layers):
        try:
            m = _get_submodule_safe(model, layer_name)
        except Exception:
            # If we can't resolve the module path, skip it (better than crashing)
            continue

        for p in m.parameters(recurse=False):
            if p.requires_grad:
                unsupported_params.append(p)

    # Deduplicate by object id
    uniq: Dict[int, torch.nn.Parameter] = {}
    for p in unsupported_params:
        uniq[id(p)] = p
    unsupported_params = list(uniq.values())

    adam_fallback = None
    if len(unsupported_params) > 0:
        adam_fallback = torch.optim.Adam(unsupported_params, **adam_kwargs)

    if verbose and len(unsupported_layers) > 0:
        print("[HybridSINGD] SINGD-unsupported layers -> Adam fallback:")
        for name in sorted(unsupported_layers):
            print(f"  - {name}")

    return HybridSINGD(singd_opt, adam_fallback), unsupported_layers


# =============================================================================
# Training: Adam vs NGD (SINGD) comparison, with masks + LTH loop
# =============================================================================

def train_adam_vs_ngd_singd(
    model: nn.Module,
    criterion,
    adam_optimizer: torch.optim.Optimizer,
    train_loader,
    mask: dict,
    pruner: "LTHPruner",
    n_epochs: int = 20,
    verbose: bool = True,
    print_freq: int = 5,
    ngd_lr: float = 1e-3,
    structure: str = "diag",  # "diag" or "dense"
    real_opt: str = "adam",   # "adam" or "singd"
):
    """
    Train for n_epochs while comparing Adam and SINGD (NGD) updates.

    real_opt:
        "adam"  -> Adam is REAL optimizer, SINGD is VIRTUAL.
        "singd" -> SINGD is REAL optimizer, Adam is VIRTUAL.

    On each batch:
      - compute grads once (loss.backward()).
      - backup parameters + grads at θ_t.
      - VIRTUAL optimizer step -> Δθ_virtual, then restore params + grads.
      - REAL optimizer step -> Δθ_real (kept).
      - re-apply mask.
      - compute cosine similarity/distance between Adam and NGD updates,
        restricted to non-masked coordinates.
    """
    assert real_opt in {"adam", "singd"}, "real_opt must be 'adam' or 'singd'"

    device = next(model.parameters()).device

    # ---- NGD optimizer: SINGD + Adam fallback for SINGD-unsupported layers ----
    ngd_optimizer, _unsupported_layers = build_singd_with_adam_fallback(
        model,
        ngd_lr=ngd_lr,
        structure=structure,
        adam_reference=adam_optimizer,
        warn_unsupported=True,
        verbose=verbose,
    )

    # Flatten mask once and build active index (non-masked positions)
    mask_flat = build_flat_mask(mask, model).to(device)
    active_idx = mask_flat.bool().to('cpu')

    loss_list = []
    cos_sim_list = []
    cos_dist_list = []

    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # choose which opt is real/virtual this batch
            if real_opt == "adam":
                real_optimizer = adam_optimizer
                virtual_optimizer = ngd_optimizer
            else:  # real_opt == "singd"
                real_optimizer = ngd_optimizer
                virtual_optimizer = adam_optimizer

            # 1) zero grads for both (they share p.grad buffers)
            adam_optimizer.zero_grad(set_to_none=True)
            ngd_optimizer.zero_grad(set_to_none=True)

            # 2) forward + backward once
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # 2a) optional: zero grads on pruned weights so neither optimizer updates them
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    if name in mask:
                        p.grad.mul_(mask[name])

            # 3) backup parameters AND grads at θ_t
            with torch.no_grad():
                params_before = [p.data.clone() for p in model.parameters()]
                grads_backup = [
                    p.grad.clone() if p.grad is not None else None
                    for p in model.parameters()
                ]
                params_before_flat = _flatten_params(params_before).to('cpu')

            # ====================== VIRTUAL STEP ======================
            virtual_optimizer.step()

            if mask is not None:
                pruner.apply_mask(model, mask)

            with torch.no_grad():
                params_after_virtual_flat = _flatten_params(model.parameters()).to('cpu')
                delta_virtual = params_after_virtual_flat - params_before_flat

                # restore params + grads back to θ_t
                for p, b, g in zip(model.parameters(), params_before, grads_backup):
                    p.data.copy_(b)
                    if g is not None and p.grad is not None:
                        p.grad.copy_(g)

            # ======================= REAL STEP ========================
            real_optimizer.step()

            if mask is not None:
                pruner.apply_mask(model, mask)

            with torch.no_grad():
                params_after_real_flat = _flatten_params(model.parameters()).to('cpu')
                delta_real = params_after_real_flat - params_before_flat

                # map to "delta_adam" and "delta_ngd" irrespective of who is real
                if real_opt == "adam":
                    delta_adam = delta_real
                    delta_ngd = delta_virtual
                else:  # real_opt == "singd"
                    delta_adam = delta_virtual
                    delta_ngd = delta_real

                # restrict to non-masked coordinates
                delta_adam_active = delta_adam[active_idx]
                delta_ngd_active = delta_ngd[active_idx]

                # cosine similarity / distance
                norm_adam = delta_adam_active.norm()
                norm_ngd = delta_ngd_active.norm()
                if norm_adam > 0 and norm_ngd > 0:
                    cos_sim = F.cosine_similarity(delta_adam_active, delta_ngd_active, dim=0)
                    cos_dist = 1.0 - cos_sim
                    cos_sim_list.append(cos_sim.item())
                    cos_dist_list.append(cos_dist.item())
                else:
                    cos_sim_list.append(float("nan"))
                    cos_dist_list.append(float("nan"))

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / max(num_batches, 1)
        loss_list.append(epoch_loss)

        if verbose and (epoch % print_freq == 0 or epoch == n_epochs - 1):
            print(
                f"Epoch [{epoch+1}/{n_epochs}] - "
                f"train loss: {epoch_loss:.4f} - "
                f"batches: {num_batches}"
            )

    return model, loss_list, cos_sim_list, cos_dist_list


def _train_LTH_one_iter_adam_vs_ngd(
    model,
    criterion,
    optimizer,
    train_loader,
    n_epochs=20,
    prunning_percentage=0.2,
    no_prunning_layers=None,
    verbose=True,
    print_freq=5,
    use_scheduler=False,
    structure: str = "diag",  # "diag" or "dense"
    real_opt: str = "adam",   # "adam" or "singd"
):
    pruner = LTHPruner(prunning_percentage, no_prunning_layers)
    mask = pruner.prune_weights(model)

    model, loss_list, cos_sim_list, cos_dist_list = train_adam_vs_ngd_singd(
        model=model,
        criterion=criterion,
        adam_optimizer=optimizer,
        train_loader=train_loader,
        mask=mask,
        pruner=pruner,
        n_epochs=n_epochs,
        verbose=verbose,
        print_freq=print_freq,
        ngd_lr=optimizer.param_groups[0]["lr"],
        structure=structure,
        real_opt=real_opt,
    )

    return model, mask, loss_list, cos_sim_list, cos_dist_list


def train_LTH_adam_vs_ngd(
    model,
    criterion,
    train_loader,
    test_loader,
    fim_loader,
    fim_args,
    lr=1e-3,
    n_iterations=5,
    n_epochs=20,
    prunning_percentage=0.2,
    no_prunning_layers=None,
    structure: str = "diag",  # "diag" or "dense"
    real_opt: str = "adam",   # "adam" or "singd"
    verbose=True,
    print_freq=5,
    use_scheduler=False,
    save_path=None,
) -> dict:
    """
    Run LTH iterations with pruning, comparing Adam vs SINGD updates each batch,
    with hybrid fallback for unsupported layers when SINGD is involved.

    Returns:
      dict with keys:
        mask_list, test_acc, fim_list, cos_sim_list, cos_dist_list
    """
    initial_state_dict = copy.deepcopy(model.state_dict())
    output_dict = {
        "mask_list": [],
        "test_acc": [],
        "fim_list": [],
        "cos_sim_list": [],
        "cos_dist_list": [],
    }

    for it in range(n_iterations):
        # we still construct an Adam optimizer; if real_opt="singd" this one is virtual
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        current_prunning_percentage = prunning_percentage * it

        if verbose:
            print(f"\n=== LTH Iteration {it+1}/{n_iterations} ===")
            print(f"Current pruning percentage: {current_prunning_percentage:.3f}")

        model, mask, loss_list, cos_sim_list, cos_dist_list = _train_LTH_one_iter_adam_vs_ngd(
            model,
            criterion,
            optimizer,
            train_loader,
            n_epochs,
            current_prunning_percentage,
            no_prunning_layers,
            verbose,
            print_freq,
            use_scheduler,
            structure=structure,
            real_opt=real_opt,
        )

        acc = test(model, test_loader)
        fim_args["mask"] = mask

        if verbose:
            print(f"Test Accuracy after iteration {it+1}: {acc*100:.2f}%")

        #fim = FisherInformationMatrix(model, criterion, optimizer, fim_loader, **fim_args)
        #fim._fim_to_cpu()

        output_dict["mask_list"].append(mask)
        output_dict["test_acc"].append(acc)
        #output_dict["fim_list"].append(fim)
        output_dict["cos_sim_list"].append(cos_sim_list)
        output_dict["cos_dist_list"].append(cos_dist_list)

        # reset model back to original initialization for next LTH iteration
        model = reset_weights(model, initial_state_dict)

    if save_path is not None:
        torch.save(output_dict, save_path)

    return output_dict
