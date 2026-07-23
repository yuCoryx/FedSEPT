import os.path as osp
import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
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
        "trainer": "dpfpl",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model




def orthogonalize(matrix):
    m = matrix.shape[1]
    for i in range(m):
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2) + 1e-12)
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0, keepdim=True) * col


def factorize_ctx(origin, rank):
    """
    origin: [n_ctx, ctx_dim]
    返回: U [n_ctx, rank], V [rank, ctx_dim], residual [n_ctx, ctx_dim]
    只在初始化/重分解时调用。
    """
    device = origin.device
    dtype = origin.dtype

    with torch.no_grad():
        v = torch.normal(0, 1, size=(origin.shape[1], rank), device=device, dtype=dtype)
        u = torch.matmul(origin, v)
        orthogonalize(u)
        v = torch.matmul(origin.t(), u)
        orthogonalize(v)
        v = v.t()
        residual = origin - torch.matmul(u, v)

    return u, v, residual




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
    DPFPL Prompt（DP 友好版）：
      - global_ctx:      nn.Embedding(n_ctx, d)
      - local_u_ctx:     nn.Embedding(n_ctx, r)
      - local_v_ctx:     nn.Embedding(r, d)
      - local_residual:  nn.Embedding(n_ctx, d)
    ctx = global + U@V + residual

    关键修复：
      - DP 模式：所有 Embedding 权重强制 fp32（避免 embedding grad_sample dtype mismatch）
      - prefix/suffix 与 ctx 拼接前对齐 dtype
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.dp_enabled = (getattr(cfg, "EPSILON", 0.0) > 0)

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DPFPL.N_CTX

        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.rank = cfg.TRAINER.DPFPL.RANK
        self.N = cfg.TRAINER.DPFPL.N

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim

        
        init_dtype = torch.float32  
        init_local_full = torch.empty(self.n_ctx, self.ctx_dim, dtype=init_dtype)
        nn.init.normal_(init_local_full, std=0.02)
        u0, v0, residual0 = factorize_ctx(init_local_full, self.rank)  

        
        global_init = torch.empty(self.n_ctx, self.ctx_dim, dtype=init_dtype)
        nn.init.normal_(global_init, std=0.02)

        self.global_ctx = nn.Parameter(global_init)                     
        self.local_u_ctx = nn.Parameter(u0.to(init_dtype))              
        self.local_v_ctx = nn.Parameter(v0.to(init_dtype))              
        self.local_residual_ctx = nn.Parameter(residual0.to(init_dtype))
        
        
        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            tok_emb = clip_model.token_embedding(tokenized_prompts)

        self.register_buffer("token_prefix", tok_emb[:, :1, :])
        self.register_buffer("token_suffix", tok_emb[:, 1 + n_ctx:, :])
        self.register_buffer("embedding", tok_emb)

        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = cfg.TRAINER.DPFPL.CLASS_TOKEN_POSITION

    def forward(self):
        global_ctx = self.global_ctx                     
        u = self.local_u_ctx                             
        v = self.local_v_ctx                             
        residual = self.local_residual_ctx               

        low_rank = torch.matmul(u, v)                    
        ctx = global_ctx + low_rank + residual           

        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  

        prefix = self.token_prefix
        suffix = self.token_suffix

        
        prefix_ = prefix.to(dtype=ctx.dtype)
        suffix_ = suffix.to(dtype=ctx.dtype)

        if self.class_token_position == "end":
            prompts = torch.cat([prefix_, ctx, suffix_], dim=1)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            all_prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix_[i: i + 1, :, :]
                class_i = suffix_[i: i + 1, :name_len, :]
                suffix_i = suffix_[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                all_prompts.append(prompt)
            prompts = torch.cat(all_prompts, dim=0)

        elif self.class_token_position == "front":
            all_prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix_[i: i + 1, :, :]
                class_i = suffix_[i: i + 1, :name_len, :]
                suffix_i = suffix_[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                all_prompts.append(prompt)
            prompts = torch.cat(all_prompts, dim=0)

        else:
            raise ValueError(f"Unsupported class_token_position: {self.class_token_position}")

        return prompts

    @torch.no_grad()
    def refactor_local_ctx(self):
        """
        使用当前本地 (U,V,residual) 构造 full local prompt，再 factorize_ctx 重新分解并回写。
        建议按“每个本地 epoch/轮”做一次，而非每个 batch
        """
        local_full = torch.matmul(self.local_u_ctx, self.local_v_ctx) + self.local_residual_ctx
        u_new, v_new, residual_new = factorize_ctx(local_full, self.rank)

        self.local_u_ctx.copy_(u_new.to(self.local_u_ctx.dtype))
        self.local_v_ctx.copy_(v_new.to(self.local_v_ctx.dtype))
        self.local_residual_ctx.copy_(residual_new.to(self.local_residual_ctx.dtype))


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
 

class DPFPL(TrainerX):
    """DP-FPL trainer with pluggable protection for uploaded parameters."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DPFPL.PREC in ["fp16", "fp32", "amp"]

    @staticmethod
    def get_share_keys():
        return ["prompt_learner.global_ctx"]

    @staticmethod
    def get_local_keys():
        return [
            "prompt_learner.local_u_ctx",
            "prompt_learner.local_v_ctx",
            "prompt_learner.local_residual_ctx",
        ]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        logger.info("Loading CLIP (backbone: %s)", cfg.MODEL.BACKBONE.NAME)
        clip_model = load_clip_to_cpu(cfg)

        
        clip_model.float()

        logger.info("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        logger.info("Turning off gradients except prompt_learner")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        prec = cfg.TRAINER.DPFPL.PREC
        self.use_amp = (prec == "amp")
        self.scaler = GradScaler() if self.use_amp else None

        
        dp_mode = str(getattr(self.args, "dp_mode", "none")).lower()
        self.dp_mode = dp_mode
        self.dp_enable = (dp_mode == "local")

        self.dp_clip = float(getattr(self.args, "dp_clip", 1.0))
        self.dp_sigma = float(getattr(self.args, "dp_sigma", 0.0))
        self._ldp_sigma_by_client = {}
        logger.info(
            "[DPFPL] privacy_backend=%s",
            self.dp_enable,
        )

    def _dp_param_iter(self, idx: int = -1):
        """Select the uploaded global context."""
        parameter = getattr(self.model.prompt_learner, "global_ctx", None)
        if parameter is not None and parameter.requires_grad:
            yield parameter


    def _compute_private_updates(self, per_sample_losses):
        return noise_backend.compute_private_gradients(
            trainer=self,
            per_sample_losses=per_sample_losses,
            params=self._dp_param_iter(),
        )






    def _prepare_client_privacy(self, idx: int):
        return noise_backend.prepare_client_privacy(
            trainer=self,
            client_idx=idx,
            logger=logger,
        )

 
    def train(self, idx: int = -1, global_epoch: int = 0, is_fed: bool = False, **kwargs):
        if self.dp_enable:
            self._prepare_client_privacy(idx=idx)
        return super().train(idx=idx, global_epoch=global_epoch, is_fed=is_fed, **kwargs)

    def forward_backward(self, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)

        if self.use_amp:
            with autocast():
                logits = self.model(image)
        else:
            logits = self.model(image)

        per_sample_losses = F.cross_entropy(
            logits,
            label,
            reduction="none",
        )
        loss = per_sample_losses.mean()
        private_updates = (
            self._compute_private_updates(per_sample_losses)
            if self.dp_enable
            else None
        )
        self.model_zero_grad()

        if self.use_amp:
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
            "acc": float(compute_accuracy(logits, label)[0].item()),
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            with torch.no_grad():
                prompt_module = getattr(self.model.prompt_learner, "_module", self.model.prompt_learner)
                if hasattr(prompt_module, "refactor_local_ctx"):
                    prompt_module.refactor_local_ctx()
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label
