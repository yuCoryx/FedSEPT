import os
import copy
import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import count_num_param, load_checkpoint, load_pretrained_weights
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import noise_backend


from trainers.PROMPTFL import TextEncoder, load_clip_to_cpu


_tokenizer = _Tokenizer()
logger = logging.getLogger(__name__)


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PFEDMOAP.N_CTX
        ctx_init = cfg.TRAINER.PFEDMOAP.CTX_INIT
        dtype = clip_model.ln_final.weight.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.PFEDMOAP.CSC:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        
        self.token_prefix = nn.Parameter(embedding[:, :1, :], requires_grad=False)
        self.token_suffix = nn.Parameter(embedding[:, 1 + n_ctx :, :], requires_grad=False)
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PFEDMOAP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts_list.append(prompt)
            prompts = torch.cat(prompts_list, dim=0)
        elif self.class_token_position == "front":
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts_list.append(prompt)
            prompts = torch.cat(prompts_list, dim=0)
        else:
            raise ValueError

        return prompts


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, scaling=1.0, dtype=torch.float16):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scaling = scaling
        self.dtype = dtype

        self.W_q = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_k = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_v = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_o = nn.Linear(d_model, d_model, dtype=self.dtype)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def split_heads(self, x):
        bsz, seq_len, _ = x.size()
        return x.view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        bsz, heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output, torch.mean(attn_probs, dim=1)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_class = len(classnames)
        self.cfg = cfg
        self.classnames = classnames
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        
        self.nonlocal_ctx = None
        self.nonlocal_text_features = None
        self.lmbda = cfg.TRAINER.PFEDMOAP.LMBDA

        num_heads = cfg.TRAINER.PFEDMOAP.GATING_HEADS
        gating_embed_dim = cfg.TRAINER.PFEDMOAP.GATING_EMBED_DIM
        self.reduce_times = self.image_encoder.output_dim // gating_embed_dim
        self.gating = MultiheadAttention(
            gating_embed_dim, num_heads, dropout=0.1, scaling=cfg.TRAINER.PFEDMOAP.SCALING, dtype=self.dtype
        )

    def pool(self, t):
        if len(t.shape) == 4:
            return t[:, :, :, :: self.reduce_times]
        if len(t.shape) == 3:
            return t[:, :, :: self.reduce_times]
        if len(t.shape) == 2:
            return t[:, :: self.reduce_times]
        return None

    def _compute_nonlocal_text_features(self):
        if not self.nonlocal_ctx:
            return
        temp_local_state_dict = copy.deepcopy(self.prompt_learner.state_dict())
        self.nonlocal_text_features = []

        ctx_list = self.nonlocal_ctx if isinstance(self.nonlocal_ctx, list) else [self.nonlocal_ctx]
        for ctx in ctx_list:
            self.load_ctx(ctx)
            with torch.no_grad():
                text_features = self.text_encoder(self.prompt_learner(), self.tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = self.pool(text_features)
                self.nonlocal_text_features.append(text_features.detach())
        self.prompt_learner.load_state_dict(temp_local_state_dict)

    def load_ctx(self, ctx):
        temp_dict = self.prompt_learner.state_dict()
        temp_dict["ctx"] = ctx
        self.prompt_learner.load_state_dict(temp_dict)

    def forward(self, image, idx=None):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        local_logits = logit_scale * image_features @ text_features.t()

        if self.nonlocal_text_features:
            q = self.pool(image_features).repeat(self.n_class, 1, 1)
            k = v = torch.stack([self.pool(text_features)] + self.nonlocal_text_features).permute(1, 0, 2)
            new_features = self.gating(q, k, v)[0].permute(1, 2, 0)
            fused = logit_scale * torch.bmm(self.pool(image_features).unsqueeze(1), new_features).squeeze(1)
            return self.lmbda * local_logits + fused
        else:
            return local_logits


class PFEDMOAP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PFEDMOAP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        
        clip_model = load_clip_to_cpu(cfg)
        
        clip_model.float()

        self.model = CustomCLIP(cfg, classnames, clip_model)

        
        for name, param in self.model.named_parameters():
            if ("prompt_learner" not in name) and ("gating" not in name):
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim_p = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched_p = build_lr_scheduler(self.optim_p, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim_p, self.sched_p)

        
        self.optim_g = build_optimizer(self.model.gating, cfg.OPTIM)
        self.sched_g = build_lr_scheduler(self.optim_g, cfg.OPTIM)
        self.register_model("gating", self.model.gating, self.optim_g, self.sched_g)

        self.scaler = GradScaler() if (cfg.TRAINER.PFEDMOAP.PREC == "amp") else None

        
        self.num_experts = cfg.TRAINER.PFEDMOAP.NUM_EXPERTS
        self.shuffled_all_indices = list(range(cfg.DATASET.USERS))
        self.reset_distance_cache()

        
        dp_mode = str(getattr(self.args, "dp_mode", "none")).lower()
        self.dp_mode = dp_mode
        self.dp_enable = (dp_mode == "local")

        self.dp_clip = float(getattr(self.args, "dp_clip", 1.0))
        self.dp_sigma = float(getattr(self.args, "dp_sigma", 0.0))
        self._ldp_sigma_by_client = {}
        logger.info(
            "[PFEDMOAP] privacy_backend=%s",
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
        
        if self.dp_enable:
            self._prepare_client_privacy(idx=idx)
        return super().train(idx=idx, global_epoch=global_epoch, is_fed=is_fed, **kwargs)

    def reset_distance_cache(self, update_indices=None):
        if update_indices is None:
            self.distance_cache = {i: {j: None for j in range(self.cfg.DATASET.USERS)} for i in range(self.cfg.DATASET.USERS)}
        else:
            for idx in update_indices:
                self.distance_cache[idx] = {j: None for j in range(self.cfg.DATASET.USERS)}
                for i in range(self.cfg.DATASET.USERS):
                    self.distance_cache[i][idx] = None

    def download_nonlocal_ctx(self, nonlocal_ctx):
        if nonlocal_ctx is None:
            valid_ctx = []
        elif isinstance(nonlocal_ctx, (list, tuple)):
            valid_ctx = [ctx for ctx in nonlocal_ctx if self._has_ctx(ctx)]
        else:
            valid_ctx = [nonlocal_ctx] if self._has_ctx(nonlocal_ctx) else []

        if not valid_ctx:
            self.model.nonlocal_ctx = None
            self.model.nonlocal_text_features = None
            return

        self.model.nonlocal_ctx = valid_ctx
        self.model._compute_nonlocal_text_features()

    @staticmethod
    def _has_ctx(ctx):
        if ctx is None:
            return False
        if torch.is_tensor(ctx):
            return ctx.numel() > 0
        if isinstance(ctx, (list, tuple)):
            return len(ctx) > 0
        return True

    def _get_dist_from_cache(self, idx, x):
        if x in self.distance_cache[idx]:
            return self.distance_cache[idx][x]
        elif idx in self.distance_cache[x]:
            return self.distance_cache[x][idx]
        return None

    def sparse_selection(self, idx, ctxs, method="random"):
        def random_selection(idx, ctxs):
            selected_indices = []
            for x in self.shuffled_all_indices:
                if x != idx and self._has_ctx(ctxs[x]):
                    selected_indices.append(x)
                if len(selected_indices) == self.num_experts - 1:
                    break
            return selected_indices

        if method == "random":
            return random_selection(idx, ctxs)

        if method == "nearest":
            if not self._has_ctx(ctxs[idx]):
                return random_selection(idx, ctxs)
            trained_indices = [i for i in range(len(ctxs)) if self._has_ctx(ctxs[i])]
            if len(trained_indices) <= self.num_experts:
                return [i for i in trained_indices if i != idx]

            candidate_indices = []
            distances = []
            for a_trained_idx in trained_indices:
                if a_trained_idx == idx:
                    continue
                dist = self._get_dist_from_cache(idx, a_trained_idx)
                if dist is None:
                    dist = torch.norm(ctxs[idx] - ctxs[a_trained_idx])
                    self.distance_cache[idx][a_trained_idx] = dist
                    self.distance_cache[a_trained_idx][idx] = dist
                candidate_indices.append(a_trained_idx)
                distances.append(dist)
            indices_for_smallest_dist = torch.topk(torch.stack(distances), self.num_experts - 1, largest=False)[1]
            return [int(candidate_indices[i.item()]) for i in indices_for_smallest_dist]

        raise ValueError(f"Unknown sparse selection method for experts: {method}")


    def forward_backward(
        self,
        batch_idx,
        batch,
        idx: int = -1,
        global_epoch: int = 0,
        is_fed: bool = False,
        global_weight=None,
        fedprox: bool = False,
        mu: float = 0.5,
        **kwargs,
    ):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PFEDMOAP.PREC
        use_amp = (prec == "amp")


        if use_amp:
            with autocast():
                logits = self.model(image, idx=idx)
        else:
            logits = self.model(image, idx=idx)

        per_sample_losses = F.cross_entropy(
            logits,
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
            self.scaler.unscale_(self.optim_p)
            self.scaler.unscale_(self.optim_g)
            if private_updates is not None:
                noise_backend.apply_private_gradients(private_updates)
            self.scaler.step(self.optim_p)
            self.scaler.step(self.optim_g)
            self.scaler.update()
        else:
            self.model_backward(loss)
            if private_updates is not None:
                noise_backend.apply_private_gradients(private_updates)
            self.model_update()

        loss_summary = {
            "loss": float(loss.item()),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr(["prompt_learner", "gating"])

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

 
