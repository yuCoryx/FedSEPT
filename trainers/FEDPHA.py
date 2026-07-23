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
        "trainer": "fedpha",
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
    FedPHA Prompt（DP 友好版）：
      - 全局提示：ctx_global      nn.Embedding(n_ctx_global, d)
      - 本地提示：ctx_local_list  ModuleList[nn.Embedding(n_ctx_user_i, d)]

    Implementation notes:
      - cfg.EPSILON>0 时（DP）：强制 ctx_global + ctx_local_list 的 Embedding 权重 float32
      - prefix/suffix 与 ctx 拼接前统一 dtype（避免 embedding grad_sample dtype mismatch）
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)

        
        self.user_prompt_lengths = cfg.DATASET.USER_PROMPT_LENGTHS
        num_users = cfg.DATASET.USERS
        assert len(self.user_prompt_lengths) == num_users, "USER_PROMPT_LENGTHS 的长度必须与 DATASET.USERS 相同"

        self.n_ctx_global = cfg.TRAINER.FedPHA.N_CTX_GLOBAL
        self.N = cfg.TRAINER.FedPHA.N
        self.ratio = cfg.TRAINER.FedPHA.ratio

        ctx_dim = clip_model.ln_final.weight.shape[0]

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        
        init_dtype = clip_model.ln_final.weight.dtype

        
        logger.info("Initializing a generic context (FedPHA)")

        ctx_global_data = torch.empty(self.n_ctx_global, ctx_dim, dtype=init_dtype)
        nn.init.normal_(ctx_global_data, std=0.02)
        self.ctx_global = nn.Parameter(ctx_global_data)

        self.ctx_local_list = nn.ParameterList([
            nn.Parameter(torch.empty(self.user_prompt_lengths[idx], ctx_dim, dtype=init_dtype))
            for idx in range(num_users)
        ])
        for param in self.ctx_local_list:
            nn.init.normal_(param, std=0.02)

        
        max_prompt_length = max(self.n_ctx_global, max(self.user_prompt_lengths))
        self.max_prompt_length = max_prompt_length
        prompt_prefix = " ".join(["X"] * max_prompt_length)

        logger.info('Initial context (local/global): "%s"', prompt_prefix)
        logger.info(
            "Number of context words -> global: %d, per client local lengths: %s",
            self.n_ctx_global, str(self.user_prompt_lengths)
        )

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)

        with torch.no_grad():
            tok_emb = clip_model.token_embedding(tokenized_prompts)

        self.register_buffer("token_prefix", tok_emb[:, :1, :])                         
        self.register_buffer("token_suffix", tok_emb[:, 1 + max_prompt_length:, :])     
        self.register_buffer("embedding", tok_emb)

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.FedPHA.CLASS_TOKEN_POSITION
        self.n_ctx = max_prompt_length  

    def compute_null_space(self, global_ctx, ratio=0.8):
        """
        global_ctx: [n_ctx_global, dim]
        返回：全局子空间上 (1-ratio) 部分对应的正交补基 V2
        """
        global_ctx = global_ctx.view(-1, global_ctx.shape[-1]).to(torch.float32)

        
        try:
            U, S, Vh = torch.linalg.svd(global_ctx, full_matrices=False)
            V = Vh.transpose(-2, -1)  
        except Exception:
            try:
                U, S, V = torch.svd(global_ctx)
            except RuntimeError as e:
                logger.warning("SVD failed on GPU (%s), fallback to CPU", str(e))
                U, S, V = torch.svd(global_ctx.cpu())
                V = V.to(global_ctx.device)

        cutoff = int(S.shape[0] * (1 - ratio))
        V2 = V[:, cutoff:]
        return V2.to(dtype=global_ctx.dtype)

    def pad_to_77(self, prompts):
        cur_len = prompts.shape[1]
        if cur_len < 77:
            pad_len = 77 - cur_len
            pad_zeros = torch.zeros(
                prompts.size(0), pad_len, prompts.size(2),
                dtype=prompts.dtype, device=prompts.device
            )
            prompts = torch.cat([prompts, pad_zeros], dim=1)
        elif cur_len > 77:
            prompts = prompts[:, :77, :]
        return prompts

    def forward(self, idx):
        """
        idx: 当前 client 索引
        返回：
          - prompts: 本地提示 ctx_local
          - prompts_global: 全局提示 ctx_global
          - prompts_projected_local: 本地提示在全局零空间上的投影
        """
        if idx < 0 or idx >= len(self.user_prompt_lengths):
            raise ValueError(f"Invalid idx: {idx}")

        n_ctx_local = self.user_prompt_lengths[idx]
        n_ctx_global = self.n_ctx_global

        
        for param in self.ctx_local_list:
            param.requires_grad_(False)
        self.ctx_local_list[idx].requires_grad_(True)

        
        ctx_global = self.ctx_global  
        ctx_local = self.ctx_local_list[idx]  

        
        null_space = self.compute_null_space(ctx_global, self.ratio)  
        null_space = null_space.to(dtype=ctx_local.dtype)

        ctx_flat = ctx_local.view(-1, ctx_local.shape[-1])
        projected_ctx = torch.mm(ctx_flat, torch.mm(null_space, null_space.T))
        projected_ctx_local = projected_ctx.view(ctx_local.shape)

        
        ctx_local_exp = ctx_local.unsqueeze(0).expand(self.n_cls, -1, -1).contiguous()
        ctx_local_exp = ctx_local_exp.view(self.n_cls * self.N, n_ctx_local, -1)

        ctx_global_exp = ctx_global.unsqueeze(0).expand(self.n_cls, -1, -1).contiguous()
        ctx_global_exp = ctx_global_exp.view(self.n_cls * self.N, n_ctx_global, -1)

        proj_local_exp = projected_ctx_local.unsqueeze(0).expand(self.n_cls, -1, -1).contiguous()
        proj_local_exp = proj_local_exp.view(self.n_cls * self.N, n_ctx_local, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            
            prefix_ = prefix.to(dtype=ctx_local_exp.dtype)
            suffix_ = suffix.to(dtype=ctx_local_exp.dtype)

            prompts = torch.cat([prefix_, ctx_local_exp, suffix_], dim=1)
            prompts_global = torch.cat([prefix_, ctx_global_exp.to(ctx_local_exp.dtype), suffix_], dim=1)
            prompts_projected_local = torch.cat([prefix_, proj_local_exp.to(ctx_local_exp.dtype), suffix_], dim=1)

        elif self.class_token_position in ["middle", "front"]:
            
            half_n_ctx = self.n_ctx // 2
            prompts_list = []

            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :].to(dtype=ctx_local_exp.dtype)
                class_i = suffix[i: i + 1, :name_len, :].to(dtype=ctx_local_exp.dtype)
                suffix_i = suffix[i: i + 1, name_len:, :].to(dtype=ctx_local_exp.dtype)

                if self.class_token_position == "middle":
                    ctx_i_half1 = ctx_local_exp[i: i + 1, :half_n_ctx, :]
                    ctx_i_half2 = ctx_local_exp[i: i + 1, half_n_ctx:, :]
                    prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                else:
                    ctx_i = ctx_local_exp[i: i + 1, :, :]
                    prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)

                prompts_list.append(prompt)

            prompts = torch.cat(prompts_list, dim=0)
            prompts_global = prompts
            prompts_projected_local = prompts

        else:
            raise ValueError(f"Unsupported class_token_position: {self.class_token_position}")

        
        prompts = self.pad_to_77(prompts)
        prompts_global = self.pad_to_77(prompts_global)
        prompts_projected_local = self.pad_to_77(prompts_projected_local)

        return prompts, prompts_global, prompts_projected_local


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.N = cfg.TRAINER.FedPHA.N

    def forward(self, image, idx):
        tokenized_prompts = self.tokenized_prompts

        prompts, prompts_global, prompts_projected_local = self.prompt_learner(idx)
        tokenized_prompts = tokenized_prompts.to(device=prompts.device)

        
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        
        img_dtype = next(self.image_encoder.parameters()).dtype
        image_features = self.image_encoder(image.to(dtype=img_dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if self.training:
            text_features_global = self.text_encoder(prompts_global, tokenized_prompts)
            text_features_global = text_features_global / text_features_global.norm(dim=-1, keepdim=True)

            text_features_projected_local = self.text_encoder(prompts_projected_local, tokenized_prompts)
            text_features_projected_local = text_features_projected_local / text_features_projected_local.norm(dim=-1, keepdim=True)

            logits_global = logit_scale * image_features @ text_features_global.t()
            return logits, text_features_global, text_features, text_features_projected_local, logits_global

        return logits


class FEDPHA(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.FedPHA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        self.lambda_orthogonal = cfg.TRAINER.FedPHA.lambda_orthogonal
        self.alpha = cfg.TRAINER.FedPHA.alpha
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

        
        if cfg.DATASET.NAME == "ImageNet":
            self.device = torch.device("cuda:0")
            device1 = torch.device("cuda")
            self.model.to(self.device)
            self.model.text_encoder.to(device1)
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)
        else:
            self.model.to(self.device)

        
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if (cfg.TRAINER.FedPHA.PREC == "amp") else None
        
        dp_mode = str(getattr(self.args, "dp_mode", "none")).lower()
        self.dp_mode = dp_mode
        self.dp_enable = (dp_mode == "local")

        self.dp_clip = float(getattr(self.args, "dp_clip", 1.0))
        self.dp_sigma = float(getattr(self.args, "dp_sigma", 0.0))
        self._ldp_sigma_by_client = {}
        logger.info(
            "[FEDPHA] privacy_backend=%s",
            self.dp_enable,
        )


    def _dp_param_iter(self, idx: int = -1):
        """Select the uploaded global context."""
        pl = self.model.prompt_learner
        if getattr(pl, "ctx_global", None) is not None and pl.ctx_global.requires_grad:
            yield pl.ctx_global


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
        
        if self.dp_enable:
            self._prepare_client_privacy(idx=idx)

        return super().train(idx=idx, global_epoch=global_epoch, is_fed=is_fed, **kwargs)

    def forward_backward(self, batch_idx, batch, idx=-1, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.FedPHA.PREC
        use_amp = (prec == "amp")

        def _forward_losses():
            (
                output,
                global_features,
                local_features,
                projected_local_features,
                output_global,
            ) = self.model(image, idx)
            pull_loss = F.mse_loss(
                local_features,
                projected_local_features,
            )
            push_loss = F.relu(
                self.alpha
                - torch.norm(
                    local_features - global_features,
                    dim=-1,
                )
            ).mean()
            per_sample_losses = F.cross_entropy(
                output,
                label,
                reduction="none",
            )
            per_sample_losses = (
                per_sample_losses
                + F.cross_entropy(
                    output_global,
                    label,
                    reduction="none",
                )
                + pull_loss
                + push_loss
            )
            return output, per_sample_losses

        if use_amp:
            with autocast():
                output, per_sample_losses = _forward_losses()
        else:
            output, per_sample_losses = _forward_losses()

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
