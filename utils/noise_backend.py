"""Experiment-only, removable per-sample DP-SGD backend.

Per-sample gradients are obtained with chunked batched vector-Jacobian
products. This performs one model forward per batch and avoids both a Python
loop over samples and Opacus model/state-dict wrapping. Opacus is used for
noise calibration and RDP accounting.
"""

import logging
from typing import Iterable, Optional

import torch


def _get_client_train_loader(trainer, client_idx: int):
    loaders = getattr(trainer, "fed_train_loader_x_dict", None)
    if isinstance(loaders, dict) and client_idx in loaders:
        return loaders[client_idx]

    data_manager = getattr(trainer, "dm", None)
    loaders = getattr(data_manager, "fed_train_loader_x_dict", None)
    if isinstance(loaders, dict) and client_idx in loaders:
        return loaders[client_idx]

    return getattr(trainer, "train_loader_x", None)


def _ensure_accounting_state(trainer):
    if not hasattr(trainer, "_dp_accountants"):
        trainer._dp_accountants = {}
    if not hasattr(trainer, "_dp_local_sizes"):
        trainer._dp_local_sizes = {}
    if not hasattr(trainer, "_dp_step_count_by_client"):
        trainer._dp_step_count_by_client = {}


def prepare_client_privacy(
    trainer,
    client_idx: int,
    logger: Optional[logging.Logger] = None,
):
    del logger
    if not getattr(trainer, "dp_enable", False):
        return float(getattr(trainer, "dp_sigma", 0.0))

    epsilon = float(getattr(trainer.args, "dp_epsilon", 0.0))
    delta = float(getattr(trainer.args, "dp_delta", 0.0))
    if epsilon <= 0 or delta <= 0:
        raise ValueError(
            "Per-sample DP-SGD requires positive dp_epsilon and dp_delta."
        )

    clip = float(getattr(trainer, "dp_clip", 0.0))
    if clip <= 0:
        raise ValueError("Per-sample DP-SGD requires a positive dp_clip.")

    _ensure_accounting_state(trainer)
    if not hasattr(trainer, "_ldp_sigma_by_client"):
        trainer._ldp_sigma_by_client = {}

    train_loader = _get_client_train_loader(trainer, client_idx)
    if train_loader is None:
        raise RuntimeError(
            f"No training loader is available for client {client_idx}."
        )

    try:
        local_size = int(len(train_loader.dataset))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot determine the local dataset size for client {client_idx}."
        ) from exc
    if local_size <= 0:
        raise RuntimeError(f"Client {client_idx} has an empty training set.")

    batch_size = int(
        getattr(
            train_loader,
            "batch_size",
            getattr(trainer.args, "train_batch_size", 32),
        )
    )
    steps_per_epoch = int(len(train_loader))
    local_epochs = int(getattr(trainer.cfg.OPTIM, "MAX_EPOCH", 1))
    total_rounds = int(
        getattr(
            trainer.cfg.OPTIM,
            "ROUND",
            getattr(trainer.args, "round", 1),
        )
    )
    total_steps = max(
        1,
        total_rounds * local_epochs * steps_per_epoch,
    )
    sample_rate = min(1.0, float(batch_size) / float(local_size))

    configured_sigma = float(getattr(trainer.args, "dp_sigma", 0.0))
    if configured_sigma > 0:
        sigma = configured_sigma
    elif client_idx in trainer._ldp_sigma_by_client:
        sigma = float(trainer._ldp_sigma_by_client[client_idx])
    else:
        from opacus.accountants.utils import get_noise_multiplier

        sigma = float(
            get_noise_multiplier(
                target_epsilon=epsilon,
                target_delta=delta,
                sample_rate=sample_rate,
                steps=total_steps,
            )
        )

    if sigma <= 0:
        raise RuntimeError(
            f"Noise calibration returned an invalid sigma for client {client_idx}."
        )

    from opacus.accountants import RDPAccountant

    trainer._ldp_sigma_by_client[client_idx] = sigma
    trainer._dp_local_sizes[client_idx] = local_size
    trainer._dp_step_count_by_client.setdefault(client_idx, 0)
    trainer._dp_accountants.setdefault(client_idx, RDPAccountant())
    trainer._dp_current_client = int(client_idx)
    trainer.dp_sigma = sigma
    return sigma


def _full_target(parameter):
    return {
        "parameter": parameter,
        "kind": "full",
    }


def _prefix_target(parameter, block_size, shared_blocks):
    if parameter is None:
        raise RuntimeError("The uploaded prompt parameter is missing.")
    if parameter.dim() not in (2, 3):
        raise RuntimeError(
            "The uploaded prompt parameter must be 2-D or 3-D, "
            f"but received shape {tuple(parameter.shape)}."
        )
    if block_size <= 0 or shared_blocks <= 0:
        raise ValueError("Prompt prefix dimensions must be positive.")
    if parameter.dim() == 2 and parameter.shape[0] % block_size:
        raise RuntimeError(
            "The flattened prompt parameter is incompatible with block_size."
        )
    total_blocks = (
        parameter.shape[0]
        if parameter.dim() == 3
        else parameter.shape[0] // block_size
    )
    return {
        "parameter": parameter,
        "kind": "prefix",
        "block_size": int(block_size),
        "shared_blocks": min(int(shared_blocks), int(total_blocks)),
    }


def _select_gradient(gradient, target):
    if target["kind"] == "full":
        return gradient

    count = target["shared_blocks"]
    if target["parameter"].dim() == 3:
        return gradient[:, :count]
    rows = count * target["block_size"]
    return gradient[:, :rows]


def _validate_targets(targets):
    unique = []
    seen = set()
    for target in targets:
        parameter = target["parameter"]
        if parameter is None or not parameter.requires_grad:
            continue
        key = id(parameter)
        if key in seen:
            raise RuntimeError(
                "A parameter was selected more than once for DP-SGD."
            )
        seen.add(key)
        unique.append(target)
    if not unique:
        raise RuntimeError("No trainable uploaded parameter was selected.")
    return unique


def _chunked_private_updates(trainer, per_sample_losses, targets):
    losses = per_sample_losses.reshape(-1)
    if losses.numel() == 0:
        raise RuntimeError("Per-sample DP-SGD received an empty loss vector.")
    if not losses.requires_grad:
        raise RuntimeError("Per-sample losses must retain their autograd graph.")

    targets = _validate_targets(targets)
    parameters = [target["parameter"] for target in targets]
    batch_size = int(losses.numel())
    requested_chunk = int(
        getattr(trainer.args, "dp_microbatch_size", 8)
    )
    chunk_size = max(1, min(batch_size, requested_chunk))
    clip = float(trainer.dp_clip)
    sigma = float(trainer.dp_sigma)
    if clip <= 0 or sigma <= 0:
        raise RuntimeError("DP-SGD requires positive clip and sigma values.")

    accumulators = [
        torch.zeros(
            _select_gradient(
                torch.zeros(
                    (1, *parameter.shape),
                    device=parameter.device,
                    dtype=torch.float32,
                ),
                target,
            ).shape[1:],
            device=parameter.device,
            dtype=torch.float32,
        )
        for parameter, target in zip(parameters, targets)
    ]

    for start in range(0, batch_size, chunk_size):
        stop = min(batch_size, start + chunk_size)
        width = stop - start
        grad_outputs = torch.zeros(
            (width, batch_size),
            device=losses.device,
            dtype=losses.dtype,
        )
        row = torch.arange(width, device=losses.device)
        grad_outputs[row, row + start] = 1

        gradients = torch.autograd.grad(
            outputs=losses,
            inputs=parameters,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
            is_grads_batched=True,
        )

        selected_gradients = []
        norm_sq = torch.zeros(
            width,
            device=losses.device,
            dtype=torch.float32,
        )
        for parameter, target, gradient in zip(
            parameters,
            targets,
            gradients,
        ):
            if gradient is None:
                gradient = torch.zeros(
                    (width, *parameter.shape),
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            selected = _select_gradient(gradient, target).float()
            selected_gradients.append(selected)
            norm_sq += selected.flatten(1).square().sum(dim=1)

        factors = (
            clip / norm_sq.sqrt().clamp_min(1e-12)
        ).clamp(max=1.0)
        for index, selected in enumerate(selected_gradients):
            shape = (width,) + (1,) * (selected.dim() - 1)
            accumulators[index].add_(
                (selected * factors.view(shape)).sum(dim=0)
            )

    noise_std = sigma * clip
    updates = []
    with torch.no_grad():
        for target, accumulator in zip(targets, accumulators):
            accumulator.add_(
                torch.randn_like(accumulator) * noise_std
            )
            accumulator.div_(float(batch_size))
            updates.append(
                {
                    "parameter": target["parameter"],
                    "kind": target["kind"],
                    "gradient": accumulator,
                    "block_size": target.get("block_size"),
                    "shared_blocks": target.get("shared_blocks"),
                }
            )

    _record_accounting_step(trainer, batch_size)
    return updates


def compute_private_gradients(
    trainer,
    per_sample_losses,
    params: Iterable[torch.nn.Parameter],
):
    targets = [_full_target(parameter) for parameter in params]
    return _chunked_private_updates(
        trainer=trainer,
        per_sample_losses=per_sample_losses,
        targets=targets,
    )


def compute_private_prompt_prefix(
    trainer,
    per_sample_losses,
    parameter,
    block_size,
    shared_blocks,
):
    return _chunked_private_updates(
        trainer=trainer,
        per_sample_losses=per_sample_losses,
        targets=[
            _prefix_target(
                parameter=parameter,
                block_size=block_size,
                shared_blocks=shared_blocks,
            )
        ],
    )


@torch.no_grad()
def apply_private_gradients(updates):
    for update in updates:
        parameter = update["parameter"]
        gradient = update["gradient"].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        if update["kind"] == "full":
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad.copy_(gradient)
            continue

        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        count = int(update["shared_blocks"])
        if parameter.dim() == 3:
            parameter.grad[:count].copy_(gradient)
        else:
            rows = count * int(update["block_size"])
            parameter.grad[:rows].copy_(gradient)


def _record_accounting_step(trainer, batch_size):
    _ensure_accounting_state(trainer)
    if not hasattr(trainer, "_dp_current_client"):
        raise RuntimeError(
            "prepare_client_privacy must run before a DP-SGD step."
        )
    client_idx = int(trainer._dp_current_client)
    local_size = int(trainer._dp_local_sizes[client_idx])
    sample_rate = min(1.0, float(batch_size) / float(local_size))
    trainer._dp_accountants[client_idx].step(
        noise_multiplier=float(trainer.dp_sigma),
        sample_rate=sample_rate,
    )
    trainer._dp_step_count_by_client[client_idx] += 1


def get_client_epsilon(trainer, client_idx):
    _ensure_accounting_state(trainer)
    accountant = trainer._dp_accountants.get(int(client_idx))
    if accountant is None:
        return 0.0
    delta = float(getattr(trainer.args, "dp_delta", 0.0))
    return float(accountant.get_epsilon(delta=delta))
