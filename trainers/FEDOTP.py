import os.path as osp
import logging

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
        "trainer": "fedotp",
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
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.FEDOTP.N_CTX

        
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.N = cfg.TRAINER.FEDOTP.N  

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        
        init_dtype = clip_model.ln_final.weight.dtype

        logging.info("Initializing a generic context")
        prompt_prefix = " ".join(["X"] * n_ctx)
        logging.info(f'Initial context: "{prompt_prefix}"')
        logging.info(f"Number of context words (tokens): {n_ctx}")

        
        ctx_data = torch.empty(self.N, n_ctx, ctx_dim, dtype=init_dtype)
        nn.init.normal_(ctx_data, std=0.02)
        self.ctx = nn.Parameter(ctx_data)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)

        
        with torch.no_grad():
            tok_emb = clip_model.token_embedding(tokenized_prompts)

        self.register_buffer("token_prefix", tok_emb[:, :1, :])              
        self.register_buffer("token_suffix", tok_emb[:, 1 + n_ctx:, :])      
        self.register_buffer("embedding", tok_emb)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.FEDOTP.CLASS_TOKEN_POSITION

    def forward(self):
        
        ctx = self.ctx

        
        if ctx.dim() != 3:
            raise RuntimeError(f"Unexpected ctx dim: {ctx.shape}")

        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)         
        ctx = ctx.permute(1, 0, 2, 3).contiguous()                    
        ctx = ctx.view(self.N * self.n_cls, self.n_ctx, -1)           

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            
            prefix_ = prefix.to(dtype=ctx.dtype)
            suffix_ = suffix.to(dtype=ctx.dtype)
            prompts = torch.cat([prefix_, ctx, suffix_], dim=1)

        elif self.class_token_position == "middle":
            
            half_n_ctx = self.n_ctx // 2
            prefix_ = prefix.to(dtype=ctx.dtype)
            suffix_ = suffix.to(dtype=ctx.dtype)

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
            prefix_ = prefix.to(dtype=ctx.dtype)
            suffix_ = suffix.to(dtype=ctx.dtype)

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
        self.n_cls = len(classnames)

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

        self.N = cfg.TRAINER.FEDOTP.N
        self.dataset = cfg.DATASET.NAME

        self.use_uniform = True
        self.eps = cfg.TRAINER.FEDOTP.EPS
        self.thresh = cfg.TRAINER.FEDOTP.THRESH
        self.OT = cfg.TRAINER.FEDOTP.OT
        self.top_percent = cfg.TRAINER.FEDOTP.TOP_PERCENT
        self.max_iter = cfg.TRAINER.FEDOTP.MAX_ITER

        
        self.device1 = None

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = self.thresh
        for _ in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
        return T

    def entropic_COT_fast(self, a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False):
        dx = torch.ones_like(a)
        dy = torch.ones_like(b)
        stopThr = self.thresh

        K = M
        Kp = torch.matmul(torch.diag_embed(1 / a, dim1=1), K)
        Kq = torch.matmul(torch.diag_embed(1 / b, dim1=1), K.permute(0, 2, 1))

        cpt = 0
        u = dx
        v = dy
        while cpt < numItermax:
            v0 = v
            temp = torch.div(dx, torch.matmul(Kp, v.unsqueeze(-1)).squeeze(-1))
            u = torch.minimum(temp, dx)
            v = torch.div(dy, torch.matmul(Kq, u.unsqueeze(-1)).squeeze(-1))

            cpt += 1
            err = (v - v0).abs().mean()
            if err.item() < stopThr:
                break

        Kprev = torch.matmul(torch.diag_embed(u, dim1=1), K)
        Kprev = torch.matmul(Kprev, torch.diag_embed(v, dim1=1))
        if log:
            return Kprev, {"err": []}
        return Kprev

    def forward(self, image, idx=None):
        b = image.shape[0]

        
        img_dtype = next(self.image_encoder.parameters()).dtype
        image_features = self.image_encoder(image.to(dtype=img_dtype))

        
        if image_features.dim() == 3:
            image_feature_pool = image_features[0]   
            image_features = image_features[1:]      
        elif image_features.dim() == 2:
            image_feature_pool = image_features
            image_features = image_features.unsqueeze(0)  
        else:
            raise ValueError(f"Unexpected image_features dim: {image_features.shape}")

        M = image_features.shape[0]
        d = image_features.shape[-1]

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts.to(device=prompts.device)

        
        if self.dataset == "ImageNet" and (self.device1 is not None):
            text_features = self.text_encoder(prompts.to(self.device1), tokenized_prompts.to(self.device1))
            text_features = text_features.to(prompts.device)
        else:
            text_features = self.text_encoder(prompts, tokenized_prompts)

        text_features = text_features.contiguous().view(self.N, self.n_cls, d)
        text_feature_pool = text_features.mean(dim=0)

        image_features = F.normalize(image_features, dim=2)
        image_feature_pool = F.normalize(image_feature_pool, dim=1)
        text_features = F.normalize(text_features, dim=2)
        text_feature_pool = F.normalize(text_feature_pool, dim=1)

        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()
        sim = sim.view(M, self.N, b * self.n_cls).permute(2, 0, 1)  
        wdist = 1.0 - sim

        xx = torch.zeros(b * self.n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1.0 / M)
        if self.OT == 'Sinkhorn':
            yy = torch.zeros(b * self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1.0 / self.N)
        elif self.OT == 'COT':
            top_percent = min(torch.sum(xx).item(), self.top_percent)
            yy = torch.zeros(b * self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1.0 / self.N) * top_percent
        else:
            raise ValueError(f"Unsupported OT: {self.OT}")

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            if self.OT == 'Sinkhorn':
                T = self.Sinkhorn(KK, xx, yy)
            else:
                T = self.entropic_COT_fast(xx, yy, KK, 0.01, numItermax=self.max_iter)

        if torch.isnan(T).any():
            return None

        sim_op = torch.sum(T * sim, dim=(1, 2)).contiguous().view(b, self.n_cls)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * sim_op
        return logits


class FEDOTP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.FEDOTP.PREC in ["fp16", "fp32", "amp"]

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

        
        if cfg.DATASET.NAME == "ImageNet":
            self.device = torch.device("cuda:0")
            device1 = torch.device("cuda")
            self.model.to(self.device)
            self.model.text_encoder.to(device1)
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)

            
            self.model.device1 = device1
        else:
            self.model.to(self.device)

        
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if (cfg.TRAINER.FEDOTP.PREC == "amp") else None

        
        dp_mode = str(getattr(self.args, "dp_mode", "none")).lower()
        self.dp_mode = dp_mode
        self.dp_enable = (dp_mode == "local")

        self.dp_clip = float(getattr(self.args, "dp_clip", 1.0))
        self.dp_sigma = float(getattr(self.args, "dp_sigma", 0.0))
        self._ldp_sigma_by_client = {}
        logger.info(
            "[FEDOTP] privacy_backend=%s",
            self.dp_enable,
        )

    def _compute_private_updates(
        self,
        per_sample_losses,
        idx: int = -1,
    ):
        prompt = getattr(self.model.prompt_learner, "ctx", None)
        return noise_backend.compute_private_prompt_prefix(
            trainer=self,
            per_sample_losses=per_sample_losses,
            parameter=prompt,
            block_size=int(self.cfg.TRAINER.FEDOTP.N_CTX),
            shared_blocks=int(self.cfg.TRAINER.FEDOTP.AVG_N),
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
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.FEDOTP.PREC
        use_amp = (prec == "amp")


        if use_amp:
            with autocast():
                output = self.model(image, idx=idx)
        else:
            output = self.model(image, idx=idx)

        if output is None:
            return {"loss": 0.0, "acc": 0.0}
        per_sample_losses = F.cross_entropy(
            output,
            label,
            reduction="none",
        )
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
