import os
from pathlib import Path
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from singd.optim.optimizer import SINGD  # external NGD / K-FAC-like optimizer

repo_root = Path().resolve().parents[0]
sys.path.insert(0, str(repo_root / "src"))
from prunning_methods.LTH import *  # LTHPruner, reset_weights, test, FisherInformationMatrix, ...


def _flatten_params(params_iterable):
    return torch.cat([p.view(-1) for p in params_iterable])


def build_flat_mask(mask_dict, model):
    flats = []
    for name, p in model.named_parameters():
        if name in mask_dict:
            flats.append(mask_dict[name].view(-1))
        else:
            # parameters that were never pruned (e.g. biases) → all ones
            flats.append(torch.ones_like(p).view(-1))
    return torch.cat(flats)


def train_adam_vs_ngd_singd(
    model: nn.Module,
    criterion,
    adam_optimizer: torch.optim.Optimizer,
    train_loader,
    mask: dict,
    pruner: LTHPruner,
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

    # External NGD optimizer (SINGD)
    ngd_optimizer = SINGD(model, lr=ngd_lr, structures=(structure, structure))

    # Flatten mask once and build active index (non-masked positions)
    mask_flat = build_flat_mask(mask, model).to(device)
    active_idx = mask_flat.bool()

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

            # 2a) optional: zero grads on pruned weights so neither optimizer
            # will try to update them
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
                params_before_flat = _flatten_params(params_before)

            # ====================== VIRTUAL STEP ======================
            virtual_optimizer.step()

            if mask is not None:
                pruner.apply_mask(model, mask)

            with torch.no_grad():
                params_after_virtual_flat = _flatten_params(model.parameters())
                delta_virtual = params_after_virtual_flat - params_before_flat

                # restore params + grads back to θ_t
                for p, b, g in zip(model.parameters(), params_before, grads_backup):
                    p.data.copy_(b)
                    if g is not None:
                        p.grad.copy_(g)

            # ======================= REAL STEP ========================
            real_optimizer.step()

            if mask is not None:
                pruner.apply_mask(model, mask)

            with torch.no_grad():
                params_after_real_flat = _flatten_params(model.parameters())
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

        fim = FisherInformationMatrix(model, criterion, optimizer, fim_loader, **fim_args)
        fim._fim_to_cpu()

        output_dict["mask_list"].append(mask)
        output_dict["test_acc"].append(acc)
        output_dict["fim_list"].append(fim)
        output_dict["cos_sim_list"].append(cos_sim_list)
        output_dict["cos_dist_list"].append(cos_dist_list)

        # reset model back to original initialization for next LTH iteration
        model = reset_weights(model, initial_state_dict)

    if save_path is not None:
        torch.save(output_dict, save_path)

    return output_dict
