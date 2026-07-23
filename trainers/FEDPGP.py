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
        "trainer": "fedpgp",
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
    FedPGP Prompt：
      - local_u_ctx: (N, n_ctx, rank)   -> Parameter(N, n_ctx, rank)
      - local_v_ctx: (N, rank, ctx_dim) -> Parameter(N, rank, ctx_dim)
      - global_ctx:  (N, n_ctx, ctx_dim)-> Parameter(N, n_ctx, ctx_dim)
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.FEDPGP.N_CTX
        ctx_init = cfg.TRAINER.FEDPGP.CTX_INIT

        
        ctx_dim = clip_model.ln_final.weight.shape[0]
        rank = cfg.TRAINER.FEDPGP.RANK
        self.N = cfg.TRAINER.FEDPGP.N

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.rank = rank
        self.ctx_dim = ctx_dim

        
        init_dtype = clip_model.ln_final.weight.dtype

        
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            self.n_ctx = n_ctx

            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                tok_emb = clip_model.token_embedding(prompt).to(dtype=init_dtype)
            ctx_vectors = tok_emb[0, 1 : 1 + n_ctx, :]

            prompt_prefix = ctx_init

            local_u_ctx_data = torch.empty(self.N, n_ctx, rank, dtype=init_dtype)
            local_v_ctx_data = torch.empty(self.N, rank, ctx_dim, dtype=init_dtype)
            global_ctx_data = torch.empty(self.N, n_ctx, ctx_dim, dtype=init_dtype)

            nn.init.normal_(local_u_ctx_data, std=0.02)
            nn.init.normal_(local_v_ctx_data, std=0.02)
            global_ctx_data.data = ctx_vectors.unsqueeze(0).expand(self.N, -1, -1).to(init_dtype)
        else:
            prompt_prefix = " ".join(["X"] * n_ctx)

            local_u_ctx_data = torch.empty(self.N, n_ctx, rank, dtype=init_dtype)
            local_v_ctx_data = torch.empty(self.N, rank, ctx_dim, dtype=init_dtype)
            global_ctx_data = torch.empty(self.N, n_ctx, ctx_dim, dtype=init_dtype)

            nn.init.normal_(local_u_ctx_data, std=0.02)
            nn.init.normal_(local_v_ctx_data, std=0.02)
            nn.init.normal_(global_ctx_data, std=0.02)

        logger.info('Initial context: "%s"', prompt_prefix)
        logger.info("n_ctx=%d, rank=%d, N=%d, init_dtype=%s",
                    self.n_ctx, self.rank, self.N, str(init_dtype))

        
        self.local_u_ctx = nn.Parameter(local_u_ctx_data)
        self.local_v_ctx = nn.Parameter(local_v_ctx_data)
        self.global_ctx = nn.Parameter(global_ctx_data)

        
        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)

        with torch.no_grad():
            tok_emb_full = clip_model.token_embedding(tokenized_prompts)
        self.register_buffer("token_prefix", tok_emb_full[:, :1, :])               
        self.register_buffer("token_suffix", tok_emb_full[:, 1 + self.n_ctx :, :]) 
        self.register_buffer("embedding", tok_emb_full)

        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = cfg.TRAINER.FEDPGP.CLASS_TOKEN_POSITION

    def forward(self):
        """
        返回：
          - embedding:      原始手工 prompt（不加可学习上下文）
          - prompts_sigma:  仅使用 global_ctx 的 prompt
          - prompts_UV:     仅使用 U@V 的 prompt
          - prompts:        U@V + global_ctx 的完整 prompt
        """
        U = self.local_u_ctx           
        V = self.local_v_ctx           
        global_ctx = self.global_ctx   

        UV = torch.matmul(U, V)               
        ctx = UV + global_ctx                 

        N, n_ctx, d = ctx.shape

        
        ctx_exp = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1).permute(1, 0, 2, 3).contiguous()
        ctx_exp = ctx_exp.view(N * self.n_cls, n_ctx, d)

        UV_exp = UV.unsqueeze(0).expand(self.n_cls, -1, -1, -1).permute(1, 0, 2, 3).contiguous()
        UV_exp = UV_exp.view(N * self.n_cls, n_ctx, d)

        global_exp = global_ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1).permute(1, 0, 2, 3).contiguous()
        global_exp = global_exp.view(N * self.n_cls, n_ctx, d)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            
            prefix_ = prefix.to(dtype=ctx_exp.dtype)
            suffix_ = suffix.to(dtype=ctx_exp.dtype)

            prompts = torch.cat([prefix_, ctx_exp, suffix_], dim=1)
            prompts_sigma = torch.cat([prefix_, global_exp.to(ctx_exp.dtype), suffix_], dim=1)
            prompts_UV = torch.cat([prefix_, UV_exp.to(ctx_exp.dtype), suffix_], dim=1)

        elif self.class_token_position in ["middle", "front"]:
            
            
            half_n_ctx = self.n_ctx // 2
            prompts_list = []

            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :].to(dtype=ctx_exp.dtype)
                class_i = suffix[i : i + 1, :name_len, :].to(dtype=ctx_exp.dtype)
                suffix_i = suffix[i : i + 1, name_len:, :].to(dtype=ctx_exp.dtype)

                if self.class_token_position == "middle":
                    ctx_i_half1 = ctx_exp[i : i + 1, :half_n_ctx, :]
                    ctx_i_half2 = ctx_exp[i : i + 1, half_n_ctx:, :]
                    prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                else:  
                    ctx_i = ctx_exp[i : i + 1, :, :]
                    prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)

                prompts_list.append(prompt)

            prompts = torch.cat(prompts_list, dim=0)
            prompts_sigma = prompts
            prompts_UV = prompts

        else:
            raise ValueError(f"Unsupported class_token_position: {self.class_token_position}")

        return self.embedding, prompts_sigma, prompts_UV, prompts


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
        image = image.to(dtype=img_dtype)

        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embedding, prompts_sigma, prompts_UV, prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts.to(device=prompts.device)

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if self.training:
            
            text_features_0 = self.text_encoder(embedding, tokenized_prompts)
            text_features_sigma = self.text_encoder(prompts_sigma, tokenized_prompts)
            text_features_UV = self.text_encoder(prompts_UV, tokenized_prompts)

            text_features_0 = text_features_0 / text_features_0.norm(dim=-1, keepdim=True)
            text_features_sigma = text_features_sigma / text_features_sigma.norm(dim=-1, keepdim=True)
            text_features_UV = text_features_UV / text_features_UV.norm(dim=-1, keepdim=True)

            return logits, text_features_0, text_features_sigma, text_features_UV, text_features

        return logits


class FEDPGP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.FEDPGP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        self.mu = cfg.TRAINER.FEDPGP.mu
        self.temp = cfg.TRAINER.FEDPGP.temp

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

        self.scaler = GradScaler() if (cfg.TRAINER.FEDPGP.PREC == "amp") else None
        
        dp_mode = str(getattr(self.args, "dp_mode", "none")).lower()
        self.dp_mode = dp_mode
        self.dp_enable = (dp_mode == "local")

        self.dp_clip = float(getattr(self.args, "dp_clip", 1.0))
        self.dp_sigma = float(getattr(self.args, "dp_sigma", 0.0))
        self._ldp_sigma_by_client = {}
        logger.info(
            "[FEDPGP] privacy_backend=%s",
            self.dp_enable,
        )

    def _dp_param_iter(self, idx: int = -1):
        """Select the uploaded global context."""
        parameter = getattr(self.model.prompt_learner, "global_ctx", None)
        if parameter is not None and parameter.requires_grad:
            yield parameter


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

    def forward_backward(self, batch_idx, batch, idx: int = -1, **kwargs):
        cos = torch.nn.CosineSimilarity(dim=-1)
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.FEDPGP.PREC
        use_amp = (prec == "amp")

        def _forward_losses():
            (
                output,
                text_features_0,
                text_features_sigma,
                _,
                text_features,
            ) = self.model(image, idx=idx)
            positive = cos(text_features_0, text_features_sigma)
            negative = cos(text_features_sigma, text_features)
            contrast_logits = torch.stack(
                (positive, negative),
                dim=1,
            ) / self.temp
            contrast_target = torch.zeros(
                contrast_logits.size(0),
                device=self.device,
                dtype=torch.long,
            )
            contrast_loss = F.cross_entropy(
                contrast_logits,
                contrast_target,
            )
            per_sample_losses = F.cross_entropy(
                output,
                label,
                reduction="none",
            )
            per_sample_losses = (
                per_sample_losses + self.mu * contrast_loss
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
