import torch
from torch.utils.data import DataLoader
from torch.func import functional_call, jacrev, vmap

def jacobian_param_l2_norms(
    model: torch.nn.Module,
    data,
    *,
    device="cuda",
    batch_size: int = 64,
    microbatch_size: int = 1,
    vmap_chunk_size: int = 1,
    num_workers: int = 0,
    max_samples: int | None = None,
    output_fn=None,  # optional: pick subset of logits etc
):
    """
    Computes per-parameter L2 norm of the Jacobian columns:
        norm_i = || J[:, i] ||_2 = sqrt(sum_over_outputs (d y / d theta_i)^2)

    Returns:
        norms: dict(name -> tensor same shape as param) on CPU
        meta:  dict
    """
    model.eval()
    model = model.to(device)

    # DataLoader if Dataset was passed
    if isinstance(data, DataLoader):
        loader = data
    else:
        loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    param_names = list(params.keys())

    # Single sample -> flattened output (K,)
    def single_out(params, x_single):
        y = functional_call(model, (params, buffers), (x_single.unsqueeze(0),))
        y = y.squeeze(0)
        if output_fn is not None:
            y = output_fn(y)
        return y.reshape(-1)

    jac_single_params = jacrev(single_out, argnums=0)

    # Accumulate diag(J^T J) on CPU (same shape as params)
    diag_jtj = {n: torch.zeros_like(p, device="cpu") for n, p in params.items()}

    def get_x(batch):
        return batch[0] if isinstance(batch, (tuple, list)) else batch

    total_seen = 0
    K_per_sample = None

    with torch.enable_grad():
        for batch in loader:
            if max_samples is not None and total_seen >= max_samples:
                break

            x = get_x(batch)

            # Trim if needed
            if max_samples is not None and total_seen + x.shape[0] > max_samples:
                x = x[: max_samples - total_seen]

            x = x.to(device, non_blocking=False)
            B = x.shape[0]

            mb = microbatch_size
            for start in range(0, B, mb):
                x_mb = x[start : start + mb]  # (mb, ...)

                # Jmb[name]: (mb, K, *param.shape)
                Jmb = vmap(
                    lambda xi: jac_single_params(params, xi),
                    chunk_size=vmap_chunk_size,
                )(x_mb)

                if K_per_sample is None and param_names:
                    K_per_sample = Jmb[param_names[0]].shape[1]

                # Accumulate sum_{mb,K} J^2 into diag(J^T J)
                for n in param_names:
                    diag_jtj[n] += (Jmb[n] ** 2).sum(dim=(0, 1)).detach().cpu()

                del Jmb, x_mb
                torch.cuda.empty_cache()

            total_seen += B
            del x
            torch.cuda.empty_cache()

    # Convert to L2 norms
    norms = {n: torch.sqrt(diag_jtj[n]) for n in param_names}

    meta = {
        "num_samples": total_seen,
        "K_per_sample": K_per_sample,
        "note": "Returned per-parameter column norms ||J[:,i]||_2 without forming J.",
    }
    return norms, meta
