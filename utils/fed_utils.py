import argparse
import logging
import os
import time

import numpy as np
import torch


def setup_logging(args: argparse.Namespace, cfg):
    dataset_name = getattr(cfg.DATASET, "NAME", args.dataset)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    optimizer_name = getattr(cfg.OPTIM, "NAME", "sgd")
    lr = getattr(cfg.OPTIM, "LR", None)
    lr_str = "default" if lr is None else f"{lr:g}"

    log_dir = os.path.join(
        os.getcwd(),
        args.logdir,
        f"eps{args.dp_epsilon}_delta{args.dp_delta}_clip{args.dp_clip}",
        dataset_name,
        f"beta{args.beta}",
        (
            f"N{cfg.DATASET.USERS}_T{cfg.OPTIM.ROUND}_"
            f"E{cfg.OPTIM.MAX_EPOCH}_{optimizer_name}{lr_str}"
        ),
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{args.trainer.lower()}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_dir, log_path


def build_datanumber_client(args: argparse.Namespace, cfg, local_trainer):
    del args
    sizes = []
    loaders = getattr(local_trainer, "fed_train_loader_x_dict", {})
    for client_id in range(cfg.DATASET.USERS):
        if client_id in loaders:
            sizes.append(len(loaders[client_id].dataset))
        else:
            logging.warning(
                "Client %d has no training loader; its aggregation weight is zero.",
                client_id,
            )
            sizes.append(0)
    return sizes


def log_global_stats(name, acc_list, final_rounds=5, round_idx_list=None):
    if not acc_list:
        logging.info("[%s] no accuracy records to summarize.", name)
        return

    if round_idx_list is None or len(round_idx_list) != len(acc_list):
        round_idx_list = list(range(len(acc_list)))

    for round_idx, accuracy in zip(round_idx_list, acc_list):
        logging.info("[%s] round %d: %.4f", name, round_idx, accuracy)

    window = min(len(acc_list), max(1, int(final_rounds)))
    final_values = acc_list[-window:]
    logging.info("[%s] best: %.4f", name, max(acc_list))
    logging.info(
        "[%s] final-%d mean/std: %.4f / %.4f",
        name,
        window,
        float(np.mean(final_values)),
        float(np.std(final_values)),
    )


def _filter_and_reshape_state(state_dict, model_state):
    filtered = {}
    if not isinstance(state_dict, dict):
        return filtered

    for key, value in state_dict.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if getattr(value, "shape", None) == getattr(target, "shape", None):
            filtered[key] = value
            continue
        if (
            hasattr(value, "numel")
            and hasattr(target, "numel")
            and value.numel() == target.numel()
        ):
            filtered[key] = value.reshape_as(target)
    return filtered


def load_client_model_state(model, initial_weights, client_partial_weights=None):
    """Reset the model before overlaying one client's private state."""
    model_state = model.state_dict()
    model.load_state_dict(
        _filter_and_reshape_state(initial_weights, model_state),
        strict=False,
    )
    if isinstance(client_partial_weights, dict) and client_partial_weights:
        model.load_state_dict(
            _filter_and_reshape_state(client_partial_weights, model_state),
            strict=False,
        )


def has_prompt_tensor(value):
    if value is None:
        return False
    if torch.is_tensor(value):
        return value.numel() > 0
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return True


def clear_pfedmoap_nonlocal(local_trainer):
    if hasattr(local_trainer, "download_nonlocal_ctx"):
        local_trainer.download_nonlocal_ctx([])


def prepare_pfedmoap_nonlocal(local_trainer, client_id, prompt_bank, method):
    if not hasattr(local_trainer, "download_nonlocal_ctx"):
        return
    if not has_prompt_tensor(prompt_bank[client_id]):
        clear_pfedmoap_nonlocal(local_trainer)
        return

    try:
        selected = local_trainer.sparse_selection(
            client_id,
            prompt_bank,
            method=method,
        )
    except TypeError:
        selected = local_trainer.sparse_selection(client_id, prompt_bank)

    contexts = [
        prompt_bank[int(other_id)]
        for other_id in selected
        if 0 <= int(other_id) < len(prompt_bank)
        and has_prompt_tensor(prompt_bank[int(other_id)])
    ]
    local_trainer.download_nonlocal_ctx(contexts)


def prepare_pfedmoap_eval_client(local_trainer, client_id):
    prompt_bank = getattr(local_trainer, "_pfedmoap_prompt_bank", None)
    method = getattr(local_trainer, "_pfedmoap_sparse_method", "nearest")
    if not isinstance(prompt_bank, list) or client_id >= len(prompt_bank):
        clear_pfedmoap_nonlocal(local_trainer)
        return

    prepare_pfedmoap_nonlocal(local_trainer, client_id, prompt_bank, method)
    if has_prompt_tensor(prompt_bank[client_id]) and hasattr(
        local_trainer.model,
        "load_ctx",
    ):
        local_trainer.model.load_ctx(prompt_bank[client_id])


def get_ref_param_for_round(
    local_weights,
    initial_weights,
    parameter_key,
    participating_clients,
):
    for client_id in participating_clients:
        weights = local_weights[client_id]
        if isinstance(weights, dict) and parameter_key in weights:
            return weights[parameter_key].clone()

    if parameter_key in initial_weights:
        return initial_weights[parameter_key].clone()

    available = [
        key for key in initial_weights if key.startswith("prompt_learner.")
    ]
    raise KeyError(
        f"{parameter_key!r} is unavailable; prompt keys include {available[:10]}"
    )


def evaluate_and_record(
    trainer_name,
    round_idx,
    test_interval,
    total_rounds,
    local_trainer,
    local_weights,
    test_cross,
    in_client_acc,
    cross_client_acc,
    in_client_rounds,
    cross_client_rounds,
    num_clients,
    initial_weights,
    force_eval=False,
):
    should_test = force_eval or (
        round_idx % test_interval == 0 or round_idx >= total_rounds - 5
    )
    if not should_test:
        logging.info(
            "[Round %d] skip evaluation (interval=%d).",
            round_idx,
            test_interval,
        )
        return

    local_results = []
    logging.info(
        "================== [%s] Round %d In-Client test ==================",
        trainer_name.upper(),
        round_idx,
    )
    for client_id in range(num_clients):
        load_client_model_state(
            local_trainer.model,
            initial_weights,
            local_weights[client_id],
        )
        if trainer_name.lower() == "pfedmoap":
            prepare_pfedmoap_eval_client(local_trainer, client_id)
        local_results.append(local_trainer.test(idx=client_id, split="local"))

    cross_results = []
    if test_cross:
        logging.info(
            "================== [%s] Round %d Cross-Client test ==================",
            trainer_name.upper(),
            round_idx,
        )
        for client_id in range(num_clients):
            load_client_model_state(
                local_trainer.model,
                initial_weights,
                local_weights[client_id],
            )
            if trainer_name.lower() == "pfedmoap":
                prepare_pfedmoap_eval_client(local_trainer, client_id)
            cross_results.append(
                local_trainer.test(idx=client_id, split="neighbor")
            )

    mean_local = (
        sum(result[0] for result in local_results) / len(local_results)
        if local_results
        else 0.0
    )
    mean_cross = (
        sum(result[0] for result in cross_results) / len(cross_results)
        if cross_results
        else None
    )
    in_client_acc.append(mean_local)
    in_client_rounds.append(round_idx)
    if mean_cross is not None:
        cross_client_acc.append(mean_cross)
        cross_client_rounds.append(round_idx)
        harmonic_mean = (
            2.0 * mean_local * mean_cross / (mean_local + mean_cross)
            if mean_local + mean_cross > 0
            else 0.0
        )
        logging.info(
            "[Round %d] In-Client %.4f | Cross-Client %.4f | HM %.4f",
            round_idx,
            mean_local,
            mean_cross,
            harmonic_mean,
        )
    else:
        logging.info("[Round %d] In-Client %.4f", round_idx, mean_local)


def server_aggregate_weights(
    shared_weights,
    participating_clients,
    client_sizes,
):
    tensors = [shared_weights[client_id] for client_id in participating_clients]
    if not tensors or any(tensor is None for tensor in tensors):
        raise ValueError("Missing an uploaded tensor for a participating client.")

    total_samples = float(
        sum(client_sizes[client_id] for client_id in participating_clients)
    )
    if total_samples <= 0:
        raise ValueError("Participating clients contain no training samples.")

    averaged = torch.zeros_like(tensors[0])
    for client_id, tensor in zip(participating_clients, tensors):
        averaged += tensor * (client_sizes[client_id] / total_samples)
    return averaged
