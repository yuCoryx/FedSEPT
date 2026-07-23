import argparse
import random

from Dassl.dassl.config import get_cfg_default


def reset_cfg(cfg, args: argparse.Namespace) -> None:
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.resume:
        cfg.RESUME = args.resume
    if args.seed:
        cfg.SEED = args.seed
    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, args: argparse.Namespace) -> None:
    from yacs.config import CfgNode as CN

    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = args.n_ctx
    cfg.TRAINER.PROMPTFL.CSC = False
    cfg.TRAINER.PROMPTFL.CTX_INIT = False
    cfg.TRAINER.PROMPTFL.PREC = "fp16"
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.PROMPTFL.MU = args.promptfl_mu

    cfg.TRAINER.FEDOTP = CN()
    cfg.TRAINER.FEDOTP.N_CTX = args.n_ctx
    cfg.TRAINER.FEDOTP.CSC = False
    cfg.TRAINER.FEDOTP.CTX_INIT = False
    cfg.TRAINER.FEDOTP.PREC = "fp16"
    cfg.TRAINER.FEDOTP.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.FEDOTP.N = args.fedotp_num_prompt
    cfg.TRAINER.FEDOTP.AVG_N = args.fedotp_num_prompt // 2
    cfg.TRAINER.FEDOTP.THRESH = args.fedotp_thresh
    cfg.TRAINER.FEDOTP.EPS = args.fedotp_eps
    cfg.TRAINER.FEDOTP.OT = args.fedotp_ot
    cfg.TRAINER.FEDOTP.TOP_PERCENT = args.fedotp_top_percent
    cfg.TRAINER.FEDOTP.MAX_ITER = args.fedotp_max_iter

    cfg.TRAINER.FEDPGP = CN()
    cfg.TRAINER.FEDPGP.N_CTX = args.n_ctx
    cfg.TRAINER.FEDPGP.N = 1
    cfg.TRAINER.FEDPGP.FEATURE = False
    cfg.TRAINER.FEDPGP.mu = args.fedpgp_mu
    cfg.TRAINER.FEDPGP.temp = args.fedpgp_temp
    cfg.TRAINER.FEDPGP.CSC = False
    cfg.TRAINER.FEDPGP.CTX_INIT = False
    cfg.TRAINER.FEDPGP.PREC = "fp16"
    cfg.TRAINER.FEDPGP.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.FEDPGP.RANK = args.fedpgp_rank

    cfg.TRAINER.FedPHA = CN()
    cfg.TRAINER.FedPHA.N_CTX_GLOBAL = args.n_ctx
    cfg.TRAINER.FedPHA.N = 1
    cfg.TRAINER.FedPHA.CSC = False
    cfg.TRAINER.FedPHA.CTX_INIT = False
    cfg.TRAINER.FedPHA.PREC = "fp16"
    cfg.TRAINER.FedPHA.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.FedPHA.lambda_orthogonal = 1
    cfg.TRAINER.FedPHA.alpha = args.fedpha_alpha
    cfg.TRAINER.FedPHA.ratio = args.fedpha_ratio

    cfg.TRAINER.PFEDMOAP = CN()
    cfg.TRAINER.PFEDMOAP.N_CTX = args.n_ctx
    cfg.TRAINER.PFEDMOAP.CSC = False
    cfg.TRAINER.PFEDMOAP.CTX_INIT = False
    cfg.TRAINER.PFEDMOAP.PREC = "fp16"
    cfg.TRAINER.PFEDMOAP.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.PFEDMOAP.NUM_EXPERTS = min(args.pfedmoap_num_experts, args.num_users)
    cfg.TRAINER.PFEDMOAP.GATING_HEADS = args.pfedmoap_gating_heads
    cfg.TRAINER.PFEDMOAP.GATING_EMBED_DIM = args.pfedmoap_gating_embed_dim
    cfg.TRAINER.PFEDMOAP.LMBDA = args.pfedmoap_lmbda
    cfg.TRAINER.PFEDMOAP.SCALING = args.pfedmoap_scaling
    cfg.TRAINER.PFEDMOAP.SPARSE_SELECTION = args.pfedmoap_sparse_selection

    cfg.TRAINER.DPFPL = CN()
    cfg.TRAINER.DPFPL.N_CTX = args.n_ctx
    cfg.TRAINER.DPFPL.N = 1
    cfg.TRAINER.DPFPL.RANK = args.dpfpl_rank
    cfg.TRAINER.DPFPL.CSC = False
    cfg.TRAINER.DPFPL.CTX_INIT = False
    cfg.TRAINER.DPFPL.PREC = "fp16"
    cfg.TRAINER.DPFPL.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.FEDSEPT = CN()
    cfg.TRAINER.FEDSEPT.N_CTX = args.n_ctx
    cfg.TRAINER.FEDSEPT.CSC = False
    cfg.TRAINER.FEDSEPT.CTX_INIT = False
    cfg.TRAINER.FEDSEPT.PREC = "fp16"
    cfg.TRAINER.FEDSEPT.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.FEDSEPT.NUM_EXPERTS = args.fedsept_num_experts
    cfg.TRAINER.FEDSEPT.RANK = args.fedsept_rank
    cfg.TRAINER.FEDSEPT.LAMBDA_DIV = args.fedsept_lambda_div

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.USERS = args.num_users
    cfg.DATASET.NAME = args.dataset
    cfg.DATASET.USER_PROMPT_LENGTHS = []
    cfg.DATASET.DOMAINNET_NUM_CLASSES = 365
    cfg.DATASET.IID = args.iid
    cfg.DATASET.PARTITION = args.partition
    cfg.DATASET.USEALL = args.useall
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.DATASET.BETA = args.beta
    cfg.DATASET.REPEATRATE = 0.0

    cfg.OPTIM.ROUND = 1
    cfg.OPTIM.GAMMA = args.gamma
    cfg.MODEL.BACKBONE.PRETRAINED = True
    cfg.EPSILON = args.dp_epsilon if args.dp_mode == "local" else 0.0
    cfg.DELTA = args.dp_delta
    cfg.NORM_THRESH = args.dp_clip


def setup_cfg(args: argparse.Namespace):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    if args.dataset:
        cfg.merge_from_file(f"configs/datasets/{args.dataset}.yaml")

    reset_cfg(cfg, args)

    if args.lr is not None:
        cfg.OPTIM.LR = args.lr
    if args.optimizer is not None:
        cfg.OPTIM.NAME = args.optimizer
    if args.round is not None:
        cfg.OPTIM.ROUND = args.round
    if args.epoch is not None:
        cfg.OPTIM.MAX_EPOCH = args.epoch

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size
    cfg.DATALOADER.TEST.NEIGHBOR_BATCH_SIZE = min(args.test_batch_size * 5, 500)

    ds_name_lower = cfg.DATASET.NAME.lower()
    if ds_name_lower == "office31":
        cfg.DATASET.USERS = 6
    elif ds_name_lower in ["officehome", "pacs"]:
        cfg.DATASET.USERS = 8
    elif ds_name_lower == "domainnet":
        cfg.DATASET.USERS = 12
    else:
        cfg.DATASET.USERS = int(args.num_users)

    random.seed(cfg.SEED)

    if cfg.DATASET.NAME.lower() in ["cifar10", "cifar100"]:
        cfg.DATASET.USER_PROMPT_LENGTHS = [
            random.randint(4, 16) for _ in range(cfg.DATASET.USERS)
        ]
    elif cfg.DATASET.NAME in ["Office31", "OfficeHome"] and args.specify:
        if args.prompts_lens is None or len(args.prompts_lens) != cfg.DATASET.USERS:
            raise ValueError(
                "When using --specify, you must provide a --prompts_lens list "
                "with the same number of elements as the number of users."
            )
        cfg.DATASET.USER_PROMPT_LENGTHS = args.prompts_lens

    if not cfg.DATASET.USER_PROMPT_LENGTHS:
        cfg.DATASET.USER_PROMPT_LENGTHS = [
            random.randint(4, 16) for _ in range(cfg.DATASET.USERS)
        ]

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg
