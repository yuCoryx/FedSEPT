import argparse
import copy
import logging
import os
import warnings

import numpy as np
import setproctitle
import torch

from Dassl.dassl.engine import build_trainer
from Dassl.dassl.utils import set_random_seed
from options import setup_cfg
from utils.fed_utils import (
    build_datanumber_client,
    clear_pfedmoap_nonlocal,
    evaluate_and_record,
    get_ref_param_for_round,
    has_prompt_tensor,
    load_client_model_state,
    log_global_stats,
    prepare_pfedmoap_nonlocal,
    server_aggregate_weights,
    setup_logging,
)


SUPPORTED_TRAINERS = (
    "fedsept",
    "promptfl",
    "fedpgp",
    "fedotp",
    "fedpha",
    "pfedmoap",
    "dpfpl",
)

warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"opacus\..*")


def _sample_clients(args, cfg, local_trainer, round_idx):
    available = list(range(cfg.DATASET.USERS))
    loaders = getattr(local_trainer, "fed_train_loader_x_dict", None)
    if isinstance(loaders, dict):
        available = [client_id for client_id in available if client_id in loaders]
    if not available:
        raise RuntimeError("No client has a training loader.")

    if args.frac >= 1.0:
        return available

    count = max(1, int(args.frac * len(available)))
    rng = np.random.default_rng(cfg.SEED + round_idx)
    return sorted(rng.choice(available, count, replace=False).tolist())


def _train_fedsept(
    args,
    cfg,
    local_trainer,
    initial_weights,
    local_weights,
    participating_clients,
    client_sizes,
    round_idx,
):
    shared_key = "prompt_learner.experts_A"
    uploads = [None for _ in range(cfg.DATASET.USERS)]
    private_states = {}

    for client_id in participating_clients:
        load_client_model_state(
            local_trainer.model,
            initial_weights,
            local_weights[client_id],
        )
        local_trainer.train(
            idx=client_id,
            global_epoch=round_idx,
            is_fed=True,
        )
        state = local_trainer.model.state_dict()
        uploads[client_id] = state[shared_key].detach().clone()
        private_states[client_id] = {
            key: value.detach().clone()
            for key, value in state.items()
            if key.startswith("prompt_learner.") and key != shared_key
        }

    global_shared = server_aggregate_weights(
        uploads,
        participating_clients,
        client_sizes,
    )
    for client_id in range(cfg.DATASET.USERS):
        local_weights[client_id][shared_key] = global_shared
        if client_id in private_states:
            local_weights[client_id].update(private_states[client_id])


def _train_promptfl(
    cfg,
    local_trainer,
    initial_weights,
    local_weights,
    participating_clients,
    client_sizes,
    round_idx,
):
    shared_key = "prompt_learner.ctx"
    global_ctx = get_ref_param_for_round(
        local_weights,
        initial_weights,
        shared_key,
        participating_clients,
    )
    uploads = [None for _ in range(cfg.DATASET.USERS)]

    for client_id in participating_clients:
        load_client_model_state(
            local_trainer.model,
            initial_weights,
            local_weights[client_id],
        )
        local_trainer.train(
            idx=client_id,
            global_epoch=round_idx,
            is_fed=True,
            global_weight=global_ctx,
        )
        uploads[client_id] = (
            local_trainer.model.state_dict()[shared_key].detach().clone()
        )

    global_ctx = server_aggregate_weights(
        uploads,
        participating_clients,
        client_sizes,
    )
    for client_id in range(cfg.DATASET.USERS):
        local_weights[client_id][shared_key] = global_ctx


def _train_fedpgp(
    cfg,
    local_trainer,
    initial_weights,
    local_weights,
    participating_clients,
    client_sizes,
    round_idx,
):
    shared_key = "prompt_learner.global_ctx"
    private_keys = (
        "prompt_learner.local_u_ctx",
        "prompt_learner.local_v_ctx",
    )
    uploads = [None for _ in range(cfg.DATASET.USERS)]
    private_states = {}

    for client_id in participating_clients:
        load_client_model_state(
            local_trainer.model,
            initial_weights,
            local_weights[client_id],
        )
        local_trainer.train(
            idx=client_id,
            global_epoch=round_idx,
            is_fed=True,
        )
        state = local_trainer.model.state_dict()
        uploads[client_id] = state[shared_key].detach().clone()
        private_states[client_id] = {
            key: state[key].detach().clone() for key in private_keys
        }

    global_ctx = server_aggregate_weights(
        uploads,
        participating_clients,
        client_sizes,
    )
    for client_id in range(cfg.DATASET.USERS):
        local_weights[client_id][shared_key] = global_ctx
        if client_id in private_states:
            local_weights[client_id].update(private_states[client_id])


def _train_fedotp(
    cfg,
    local_trainer,
    initial_weights,
    local_weights,
    participating_clients,
    client_sizes,
    round_idx,
):
    shared_key = "prompt_learner.ctx"
    prompt_count = int(cfg.TRAINER.FEDOTP.N)
    shared_count = int(cfg.TRAINER.FEDOTP.AVG_N)
    context_length = int(cfg.TRAINER.FEDOTP.N_CTX)
    uploads = [None for _ in range(cfg.DATASET.USERS)]
    private_prompts = {}

    for client_id in participating_clients:
        load_client_model_state(
            local_trainer.model,
            initial_weights,
            local_weights[client_id],
        )
        local_trainer.train(
            idx=client_id,
            global_epoch=round_idx,
            is_fed=True,
        )
        ctx = local_trainer.model.state_dict()[shared_key]
        ctx = ctx.reshape(prompt_count, context_length, -1)
        uploads[client_id] = ctx[:shared_count].detach().clone()
        private_prompts[client_id] = ctx[shared_count:].detach().clone()

    global_prompts = server_aggregate_weights(
        uploads,
        participating_clients,
        client_sizes,
    )
    initial_ctx = initial_weights[shared_key].reshape(
        prompt_count,
        context_length,
        -1,
    )
    for client_id in range(cfg.DATASET.USERS):
        if client_id in private_prompts:
            private = private_prompts[client_id]
        elif shared_key in local_weights[client_id]:
            private = local_weights[client_id][shared_key].reshape(
                prompt_count,
                context_length,
                -1,
            )[shared_count:]
        else:
            private = initial_ctx[shared_count:]
        local_weights[client_id][shared_key] = torch.cat(
            [global_prompts, private],
            dim=0,
        ).reshape(prompt_count * context_length, -1)


def _train_fedpha(
    cfg,
    local_trainer,
    initial_weights,
    local_weights,
    participating_clients,
    client_sizes,
    round_idx,
):
    shared_key = "prompt_learner.ctx_global"
    uploads = [None for _ in range(cfg.DATASET.USERS)]
    private_states = {}

    for client_id in participating_clients:
        load_client_model_state(
            local_trainer.model,
            initial_weights,
            local_weights[client_id],
        )
        local_trainer.train(
            idx=client_id,
            global_epoch=round_idx,
            is_fed=True,
        )
        state = local_trainer.model.state_dict()
        uploads[client_id] = state[shared_key].detach().clone()
        private_key = f"prompt_learner.ctx_local_list.{client_id}"
        private_states[client_id] = {
            private_key: state[private_key].detach().clone()
        }

    global_ctx = server_aggregate_weights(
        uploads,
        participating_clients,
        client_sizes,
    )
    for client_id in range(cfg.DATASET.USERS):
        local_weights[client_id][shared_key] = global_ctx
        if client_id in private_states:
            local_weights[client_id].update(private_states[client_id])


def _train_pfedmoap(
    args,
    cfg,
    local_trainer,
    initial_weights,
    local_weights,
    prompt_bank,
    gating_states,
    global_prompt,
    participating_clients,
    client_sizes,
    round_idx,
):
    method = args.pfedmoap_sparse_selection

    for client_id in participating_clients:
        clear_pfedmoap_nonlocal(local_trainer)
        load_client_model_state(local_trainer.model, initial_weights)
        if gating_states[client_id]:
            local_trainer.model.load_state_dict(
                gating_states[client_id],
                strict=False,
            )
        if round_idx > 0 and has_prompt_tensor(prompt_bank[client_id]):
            prepare_pfedmoap_nonlocal(
                local_trainer,
                client_id,
                prompt_bank,
                method,
            )
        if global_prompt is not None:
            local_trainer.model.load_state_dict(
                {"prompt_learner.ctx": global_prompt},
                strict=False,
            )
        elif has_prompt_tensor(prompt_bank[client_id]):
            local_trainer.model.load_ctx(prompt_bank[client_id])

        local_trainer.train(
            idx=client_id,
            global_epoch=round_idx,
            is_fed=True,
        )
        state = local_trainer.model.state_dict()
        prompt_bank[client_id] = state[
            "prompt_learner.ctx"
        ].detach().clone()
        gating_states[client_id] = {
            key: value.detach().clone()
            for key, value in state.items()
            if "gating" in key
        }

    if hasattr(local_trainer, "reset_distance_cache"):
        local_trainer.reset_distance_cache(
            update_indices=participating_clients
        )
    global_prompt = server_aggregate_weights(
        prompt_bank,
        participating_clients,
        client_sizes,
    )
    for client_id in range(cfg.DATASET.USERS):
        client_prompt = (
            prompt_bank[client_id]
            if has_prompt_tensor(prompt_bank[client_id])
            else global_prompt
        )
        local_weights[client_id]["prompt_learner.ctx"] = client_prompt
        local_weights[client_id].update(gating_states[client_id])

    local_trainer._pfedmoap_prompt_bank = prompt_bank
    local_trainer._pfedmoap_sparse_method = method
    return global_prompt


def _train_dpfpl(
    cfg,
    local_trainer,
    initial_weights,
    local_weights,
    participating_clients,
    client_sizes,
    round_idx,
):
    shared_key = "prompt_learner.global_ctx"
    private_keys = (
        "prompt_learner.local_u_ctx",
        "prompt_learner.local_v_ctx",
        "prompt_learner.local_residual_ctx",
    )
    uploads = [None for _ in range(cfg.DATASET.USERS)]
    private_states = {}

    for client_id in participating_clients:
        load_client_model_state(
            local_trainer.model,
            initial_weights,
            local_weights[client_id],
        )
        local_trainer.train(
            idx=client_id,
            global_epoch=round_idx,
            is_fed=True,
        )
        state = local_trainer.model.state_dict()
        uploads[client_id] = state[shared_key].detach().clone()
        private_states[client_id] = {
            key: state[key].detach().clone() for key in private_keys
        }

    global_ctx = server_aggregate_weights(
        uploads,
        participating_clients,
        client_sizes,
    )
    for client_id in range(cfg.DATASET.USERS):
        local_weights[client_id][shared_key] = global_ctx
        if client_id in private_states:
            local_weights[client_id].update(private_states[client_id])


def main(args: argparse.Namespace) -> None:
    args.trainer = args.trainer.lower()
    if args.trainer not in SUPPORTED_TRAINERS:
        raise ValueError(
            f"Unsupported trainer {args.trainer!r}; choose from "
            f"{', '.join(SUPPORTED_TRAINERS)}."
        )

    cfg = setup_cfg(args)
    test_interval = max(1, args.test_every)
    log_dir, log_path = setup_logging(args, cfg)

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    args.para_dir = log_dir
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    logging.info(
        "Dataset: %s | Trainer: %s | Clients: %d | Seed: %d",
        cfg.DATASET.NAME,
        args.trainer,
        cfg.DATASET.USERS,
        cfg.SEED,
    )
    logging.info("Log file: %s", log_path)
    logging.info("Arguments: %s", args)

    num_clients = int(cfg.DATASET.USERS)
    local_weights = [{} for _ in range(num_clients)]
    prompt_bank = [None for _ in range(num_clients)]
    gating_states = [{} for _ in range(num_clients)]

    local_trainer = build_trainer(args, cfg)
    initial_weights = copy.deepcopy(local_trainer.model.state_dict())
    global_prompt = initial_weights.get("prompt_learner.ctx")
    local_trainer.fed_before_train()
    client_sizes = build_datanumber_client(args, cfg, local_trainer)

    in_client_acc = []
    cross_client_acc = []
    in_client_rounds = []
    cross_client_rounds = []
    total_rounds = int(cfg.OPTIM.ROUND)

    for round_idx in range(total_rounds):
        participating = _sample_clients(
            args,
            cfg,
            local_trainer,
            round_idx,
        )
        logging.info(
            "[Round %d] participating clients: %s",
            round_idx,
            participating,
        )

        if args.trainer == "fedsept":
            _train_fedsept(
                args,
                cfg,
                local_trainer,
                initial_weights,
                local_weights,
                participating,
                client_sizes,
                round_idx,
            )
        elif args.trainer == "promptfl":
            _train_promptfl(
                cfg,
                local_trainer,
                initial_weights,
                local_weights,
                participating,
                client_sizes,
                round_idx,
            )
        elif args.trainer == "fedpgp":
            _train_fedpgp(
                cfg,
                local_trainer,
                initial_weights,
                local_weights,
                participating,
                client_sizes,
                round_idx,
            )
        elif args.trainer == "fedotp":
            _train_fedotp(
                cfg,
                local_trainer,
                initial_weights,
                local_weights,
                participating,
                client_sizes,
                round_idx,
            )
        elif args.trainer == "fedpha":
            _train_fedpha(
                cfg,
                local_trainer,
                initial_weights,
                local_weights,
                participating,
                client_sizes,
                round_idx,
            )
        elif args.trainer == "pfedmoap":
            global_prompt = _train_pfedmoap(
                args,
                cfg,
                local_trainer,
                initial_weights,
                local_weights,
                prompt_bank,
                gating_states,
                global_prompt,
                participating,
                client_sizes,
                round_idx,
            )
        elif args.trainer == "dpfpl":
            _train_dpfpl(
                cfg,
                local_trainer,
                initial_weights,
                local_weights,
                participating,
                client_sizes,
                round_idx,
            )

        evaluate_and_record(
            trainer_name=args.trainer,
            round_idx=round_idx,
            test_interval=test_interval,
            total_rounds=total_rounds,
            local_trainer=local_trainer,
            local_weights=local_weights,
            test_cross=args.test_cross,
            in_client_acc=in_client_acc,
            cross_client_acc=cross_client_acc,
            in_client_rounds=in_client_rounds,
            cross_client_rounds=cross_client_rounds,
            num_clients=num_clients,
            initial_weights=initial_weights,
        )

    log_global_stats(
        "In-Client",
        in_client_acc,
        round_idx_list=in_client_rounds,
    )
    if args.test_cross:
        log_global_stats(
            "Cross-Client",
            cross_client_acc,
            round_idx_list=cross_client_rounds,
        )


def build_parser():
    parser = argparse.ArgumentParser(description="FedSEPT main experiments")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trainer", choices=SUPPORTED_TRAINERS, default="fedsept")
    parser.add_argument("--dataset", choices=("caltech101", "cifar10", "cifar100", "domainnet", "dtd", "food101", "office31", "officehome", "oxford_flowers", "oxford_pets", "pacs"), default="oxford_pets")
    parser.add_argument("--backbone", default="ViT-B/16")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--frac", type=float, default=1.0, help="Fraction of available clients sampled per round.")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--test_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=("adam", "amsgrad", "sgd", "rmsprop", "radam", "adamw"), default="sgd")
    parser.add_argument("--round", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=1)

    parser.set_defaults(test_cross=True)
    parser.add_argument("--cross-test", dest="test_cross", action="store_true")
    parser.add_argument("--no-cross-test", dest="test_cross", action="store_false")

    parser.add_argument("--useall", action="store_true")
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--partition", default="noniid-labeldir")
    parser.add_argument("--num_shots", type=int, default=16)
    parser.add_argument("--n_ctx", type=int, default=16)

    parser.add_argument("--dp_mode", choices=("local", "none"), default="local")
    parser.add_argument("--dp_clip", type=float, default=1.0)
    parser.add_argument("--dp_epsilon", type=float, default=1.0)
    parser.add_argument("--dp_delta", type=float, default=1e-5)
    parser.add_argument("--dp_sigma", type=float, default=0.0, help="Noise multiplier; zero enables per-client auto-tuning.")
    parser.add_argument("--dp_microbatch_size", type=int, default=8, help="Number of vector-Jacobian rows computed together for memory-efficient per-sample DP-SGD.")

    parser.add_argument("--fedsept_num_experts", type=int, default=4)
    parser.add_argument("--fedsept_rank", type=int, default=16)
    parser.add_argument("--fedsept_lambda_div", type=float, default=10.0)
    parser.add_argument("--fedsept_router_num_heads", type=int, default=4)

    parser.add_argument("--promptfl_mu", type=float, default=0.0)
    parser.add_argument("--fedpgp_mu", type=int, default=100)
    parser.add_argument("--fedpgp_temp", type=float, default=0.5)
    parser.add_argument("--fedpgp_rank", type=int, default=4)
    parser.add_argument("--dpfpl_rank", type=int, default=16)

    parser.add_argument("--fedotp_num_prompt", type=int, default=2)
    parser.add_argument("--fedotp_thresh", type=float, default=1e-3)
    parser.add_argument("--fedotp_eps", type=float, default=0.1)
    parser.add_argument("--fedotp_ot", default="Sinkhorn")
    parser.add_argument("--fedotp_top_percent", type=float, default=1.0)
    parser.add_argument("--fedotp_max_iter", type=int, default=100)

    parser.add_argument("--fedpha_alpha", type=float, default=1.0)
    parser.add_argument("--fedpha_ratio", type=float, default=0.8)
    parser.add_argument("--specify", action="store_true")
    parser.add_argument("--prompts_lens", nargs="+", type=int)

    parser.add_argument("--pfedmoap_num_experts", type=int, default=8)
    parser.add_argument("--pfedmoap_gating_heads", type=int, default=4)
    parser.add_argument("--pfedmoap_gating_embed_dim", type=int, default=128)
    parser.add_argument("--pfedmoap_lmbda", type=float, default=0.5)
    parser.add_argument("--pfedmoap_scaling", type=float, default=10.0)
    parser.add_argument("--pfedmoap_sparse_selection", choices=("nearest", "random"), default="nearest")

    parser.add_argument("--logdir", default="logs")

    default_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument("--root", default=default_root)
    parser.add_argument("--resume")
    parser.add_argument("--transforms", nargs="+")
    parser.add_argument("--head", default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    return parser

if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    setproctitle.setproctitle(
        f"{parsed_args.trainer}_{parsed_args.backbone}_{parsed_args.dataset}"
    )
    main(parsed_args)
