import os.path as osp
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import noise_backend

logger = logging.getLogger(__name__)
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": "promptfl",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }

    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts): 
        w_dtype = self.positional_embedding.dtype
        device = self.positional_embedding.device
        
        x = prompts.to(device=device, dtype=w_dtype) + self.positional_embedding
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x)

        tokenized_prompts = tokenized_prompts.to(device=x.device)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """
    PROMPTFL Prompt：
      - ctx: nn.Parameter(n_ctx, d)
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PROMPTFL.N_CTX
        ctx_dim = clip_model.ln_final.weight.shape[0]

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        
        init_dtype = clip_model.ln_final.weight.dtype

        logger.info("Initializing a generic context")
        ctx_data = torch.empty(n_ctx, ctx_dim, dtype=init_dtype)
        nn.init.normal_(ctx_data, std=0.02)
        self.ctx = nn.Parameter(ctx_data)

        prompt_prefix = " ".join(["X"] * n_ctx)
        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        
        with torch.no_grad():
            tok_emb = clip_model.token_embedding(tokenized_prompts)

        self.register_buffer("token_prefix", tok_emb[:, :1, :])             
        self.register_buffer("token_suffix", tok_emb[:, 1 + n_ctx:, :])     
        self.register_buffer("embedding", tok_emb)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION

    def forward(self):
        
        ctx = self.ctx

        
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        
        prefix_ = prefix.to(dtype=ctx.dtype)
        suffix_ = suffix.to(dtype=ctx.dtype)

        if self.class_token_position == "end":
            prompts = torch.cat([prefix_, ctx, suffix_], dim=1)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix_[i: i + 1, :, :]
                class_i = suffix_[i: i + 1, :name_len, :]
                suffix_i = suffix_[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts_list.append(prompt)
            prompts = torch.cat(prompts_list, dim=0)

        elif self.class_token_position == "front":
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix_[i: i + 1, :, :]
                class_i = suffix_[i: i + 1, :name_len, :]
                suffix_i = suffix_[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts_list.append(prompt)
            prompts = torch.cat(prompts_list, dim=0)

        else:
            raise ValueError(f"Unsupported class_token_position: {self.class_token_position}")

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

    def forward(self, image, idx=None):
        
        img_dtype = next(self.image_encoder.parameters()).dtype
        image_features = self.image_encoder(image.to(dtype=img_dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts.to(device=prompts.device)

        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


class PROMPTFL(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTFL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        self.mu = cfg.TRAINER.PROMPTFL.MU
        classnames = self.dm.dataset.classnames

        logger.info(str(self.dm.dataset))
        logger.info("Loading CLIP (backbone: %s)", cfg.MODEL.BACKBONE.NAME)
        clip_model = load_clip_to_cpu(cfg)

        
        clip_model.float()

        logger.info("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        logger.info("Turning off gradients except prompt_learner")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        logger.info(f"# params: {count_num_param(self.model):,}")
        logger.info(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if (cfg.TRAINER.PROMPTFL.PREC == "amp") else None
        
        dp_mode = str(getattr(self.args, "dp_mode", "none")).lower()
        self.dp_mode = dp_mode
        self.dp_enable = (dp_mode == "local")

        self.dp_clip = float(getattr(self.args, "dp_clip", 1.0))
        self.dp_sigma = float(getattr(self.args, "dp_sigma", 0.0))

        self._ldp_sigma_by_client = {}
        logger.info(
            "[PROMPTFL] privacy_backend=%s",
            self.dp_enable,
        )

    def _dp_param_iter(self, idx: int = -1):
        """Select the uploaded prompt context."""
        for p in self.model.prompt_learner.parameters():
            if p is not None and p.requires_grad:
                yield p

    def _compute_private_updates(
        self,
        per_sample_losses,
        idx: int = -1,
    ):
        return noise_backend.compute_private_gradients(
            trainer=self,
            per_sample_losses=per_sample_losses,
            params=self._dp_param_iter(idx=idx),
        )






    def _prepare_client_privacy(self, idx: int):
        return noise_backend.prepare_client_privacy(
            trainer=self,
            client_idx=idx,
            logger=logger,
        )


    def train(self, idx: int = -1, global_epoch: int = 0, is_fed: bool = False, **kwargs):
        
        if getattr(self, "dp_enable", False):
            self._prepare_client_privacy(idx=idx)
        return super().train(idx=idx, global_epoch=global_epoch, is_fed=is_fed, **kwargs)

    def forward_backward(
        self,
        batch_idx,
        batch,
        idx: int = -1,              
        global_epoch: int = 0,
        is_fed: bool = False,
        global_weight=None,
        use_fedprox=False,
        **kwargs,
    ):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PROMPTFL.PREC
        use_amp = (prec == "amp")

        if use_amp:
            with autocast():
                output = self.model(image, idx=idx)
                per_sample_losses = F.cross_entropy(
                    output,
                    label,
                    reduction="none",
                )
        else:
            output = self.model(image, idx=idx)
            per_sample_losses = F.cross_entropy(
                output,
                label,
                reduction="none",
            )

        if (global_weight is not None) and (self.mu > 0):
            ctx_local = self.model.prompt_learner.ctx
            gw = global_weight.to(device=ctx_local.device, dtype=ctx_local.dtype)
            fed_prox_reg = (self.mu / 2.0) * torch.norm(ctx_local - gw) ** 2
            per_sample_losses = per_sample_losses + fed_prox_reg

        loss = per_sample_losses.mean()
        private_updates = (
            self._compute_private_updates(per_sample_losses, idx=idx)
            if self.dp_enable
            else None
        )
        self.model_zero_grad()

        if use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            if private_updates is not None:
                noise_backend.apply_private_gradients(private_updates)
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.model_backward(loss)
            if private_updates is not None:
                noise_backend.apply_private_gradients(private_updates)
            self.model_update()

        loss_summary = {
            "loss": float(loss.item()),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label
