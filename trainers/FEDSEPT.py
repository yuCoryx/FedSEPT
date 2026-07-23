
from typing import Optional
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from utils import noise_backend


logger = logging.getLogger(__name__)


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
        "trainer": "fedsept",
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

    def forward(self, prompts, tokenized_prompts, return_tokens: bool = False):
        w_dtype = self.positional_embedding.dtype
        device = self.positional_embedding.device

        x = prompts.to(device=device, dtype=w_dtype) + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        tokenized_prompts = tokenized_prompts.to(device=x.device)
        eot = tokenized_prompts.argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0], device=x.device), eot] @ self.text_projection

        if return_tokens:
            return pooled, x
        return pooled


class LinearGateHead(nn.Module):
    def __init__(self, in_dim: int, num_experts: int, init_std: float = 1e-2, dtype=None):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_experts, bias=True)
        nn.init.normal_(self.fc.weight, std=init_std)
        nn.init.zeros_(self.fc.bias)
        if dtype is not None:
            self.fc.weight.data = self.fc.weight.data.to(dtype)
            self.fc.bias.data = self.fc.bias.data.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class QueryCrossAttnAggregator(nn.Module):

    def __init__(self, token_dim: int, num_queries: int = 1, num_heads: int = 4,
                 dropout: float = 0.1, dtype=None):
        super().__init__()

        self.query = nn.Parameter(torch.empty(num_queries, token_dim))
        nn.init.normal_(self.query, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.post_ln = nn.LayerNorm(token_dim)

        if dtype is not None:
            self.query.data = self.query.data.to(dtype)
            self.post_ln.weight.data = self.post_ln.weight.data.to(dtype)
            self.post_ln.bias.data = self.post_ln.bias.data.to(dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T, D = tokens.shape
        q = self.query.unsqueeze(0).expand(B, -1, -1)

        attn_out, _ = self.attn(q, tokens, tokens, need_weights=False)

        route_vec = attn_out.mean(dim=1)
        route_vec = self.post_ln(route_vec)
        return route_vec


class Router(nn.Module):

    def __init__(self, aggregator: nn.Module, head: nn.Module):
        super().__init__()
        self.aggregator = aggregator
        self.head = head

    def forward(self, tokens: torch.Tensor):
        route_vec = self.aggregator(tokens)
        logits = self.head(route_vec)
        pi = F.softmax(logits, dim=-1)
        return pi, logits

class PromptLearner(nn.Module):

    def __init__(
        self,
        cfg,
        classnames,
        clip_model,
        num_experts: int = 4,
        rank: int = 8,
        router_num_queries: int = 1,
        router_num_heads: int = 4,
        router_dropout: float = 0.1,
    ):
        super().__init__()

        self.num_experts = int(num_experts)
        self.rank = int(rank)
        self._router_num_queries = int(router_num_queries)
        self._router_num_heads = int(router_num_heads)
        self._router_dropout = float(router_dropout)

        self.n_ctx = cfg.TRAINER.FEDSEPT.N_CTX
        self.n_cls = len(classnames)

        ctx_dim = clip_model.ln_final.weight.shape[0]
        init_dtype = clip_model.ln_final.weight.dtype
        self.ctx_dim = int(ctx_dim)

        A = torch.empty(self.num_experts, self.n_ctx, self.rank, dtype=init_dtype)
        nn.init.normal_(A, std=0.02)
        self.experts_A = nn.Parameter(A.reshape(self.num_experts * self.n_ctx, self.rank))
        self._expert_width = self.rank

        public_basis = torch.empty(self.rank, self.ctx_dim, dtype=init_dtype)
        nn.init.normal_(public_basis, std=0.02)
        self.register_buffer("B0", public_basis)

        R = torch.zeros(self.n_ctx, self.ctx_dim, dtype=init_dtype)
        self.local_R = nn.Parameter(R)


        self._router: Optional[nn.Module] = None
        self._router_token_dim: Optional[int] = None
        self._use_router = (self.num_experts > 1)

        image_feat_dim = getattr(clip_model.visual, "output_dim", None)
        if image_feat_dim is None:
            image_feat_dim = clip_model.text_projection.shape[0]
        self._image_feat_dim_guess = int(image_feat_dim)


        from clip import clip
        from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
        _tokenizer = _Tokenizer()

        prompt_prefix = " ".join(["X"] * self.n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            tok_emb = clip_model.token_embedding(tokenized_prompts)

        self.register_buffer("token_prefix", tok_emb[:, :1, :])
        self.register_buffer("token_suffix", tok_emb[:, 1 + self.n_ctx:, :])
        self.register_buffer("embedding", tok_emb)
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = cfg.TRAINER.FEDSEPT.CLASS_TOKEN_POSITION

    def _get_expert_coeffs_all(self) -> torch.Tensor:
        return self.experts_A.view(self.num_experts, self.n_ctx, self._expert_width)

    def build_ctx_from_A(self, A: torch.Tensor) -> torch.Tensor:
        return A @ self.B0 + self.local_R

    def _build_router(self, token_dim: int, dtype: torch.dtype, device: torch.device) -> nn.Module:
        agg = QueryCrossAttnAggregator(
            token_dim=token_dim,
            num_queries=self._router_num_queries,
            num_heads=self._router_num_heads,
            dropout=self._router_dropout,
            dtype=None,
        )

        head = LinearGateHead(
            in_dim=token_dim,
            num_experts=self.num_experts,
            dtype=None,
        )

        router = Router(aggregator=agg, head=head)
        router.to(device=device, dtype=dtype)
        return router

    def _ensure_router(self, token_dim: int, dtype: torch.dtype, device: torch.device) -> Optional[nn.Module]:
        if not self._use_router:
            return None
        if (self._router is None) or (self._router_token_dim != int(token_dim)):
            self._router = self._build_router(token_dim=int(token_dim), dtype=dtype, device=device)
            self._router_token_dim = int(token_dim)
            self.add_module("router", self._router)
        else:
            self._router.to(device=device, dtype=dtype)
        return self._router

    def compute_gate(self, image_features: torch.Tensor):
        if not self._use_router:
            B = image_features.shape[0]
            device, dtype = image_features.device, image_features.dtype
            pi = torch.ones(B, 1, device=device, dtype=dtype)
            logits = torch.zeros(B, 1, device=device, dtype=dtype)
            return pi, logits

        tokens = image_features.unsqueeze(1)

        token_dim = tokens.shape[-1]
        router = self._ensure_router(token_dim=token_dim, dtype=tokens.dtype, device=tokens.device)
        return router(tokens=tokens)

    def construct_prompts(self, ctx_embeddings: torch.Tensor) -> torch.Tensor:
        if ctx_embeddings.dim() == 2:
            ctx_embeddings = ctx_embeddings.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix.to(dtype=ctx_embeddings.dtype)
        suffix = self.token_suffix.to(dtype=ctx_embeddings.dtype)

        if self.class_token_position == "end":
            return torch.cat([prefix, ctx_embeddings, suffix], dim=1)

        if self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            all_prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i_half1 = ctx_embeddings[i:i+1, :half_n_ctx, :]
                ctx_i_half2 = ctx_embeddings[i:i+1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                all_prompts.append(prompt)
            return torch.cat(all_prompts, dim=0)

        if self.class_token_position == "front":
            all_prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i = ctx_embeddings[i:i+1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                all_prompts.append(prompt)
            return torch.cat(all_prompts, dim=0)

        return torch.cat([prefix, ctx_embeddings, suffix], dim=1)

class CustomCLIP(nn.Module):
    def __init__(
        self,
        cfg,
        classnames,
        clip_model,
        num_experts: int = 4,
        rank: int = 8,
        router_num_queries: int = 1,
        router_num_heads: int = 4,
        router_dropout: float = 0.1,
    ):
        super().__init__()
        self.prompt_learner = PromptLearner(
            cfg,
            classnames,
            clip_model,
            num_experts=num_experts,
            rank=rank,
            router_num_queries=router_num_queries,
            router_num_heads=router_num_heads,
            router_dropout=router_dropout,
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

    def _encode_image_with_tokens(self, image: torch.Tensor):

        if hasattr(self.image_encoder, "forward"):
            try:
                out = self.image_encoder(image, return_tokens=True)
                if isinstance(out, (tuple, list)) and len(out) == 2:
                    return out[0], out[1]
            except TypeError:
                pass


        if hasattr(self.image_encoder, "encode_image_with_tokens"):
            return self.image_encoder.encode_image_with_tokens(image)


        if hasattr(self.image_encoder, "transformer") and hasattr(self.image_encoder, "ln_post"):
            tokens_cache = {}

            def hook_fn(module, input, output):
                if isinstance(input, (tuple, list)) and len(input) > 0:
                    tokens_cache["tokens"] = input[0].permute(1, 0, 2)

            handle = self.image_encoder.ln_post.register_forward_hook(hook_fn)
            try:
                feat = self.image_encoder(image)
                if "tokens" in tokens_cache:
                    tokens = tokens_cache["tokens"]
                    handle.remove()
                    return feat, tokens
            except Exception:
                handle.remove()
                pass

        raise RuntimeError(
            "Cannot obtain patch tokens from CLIP visual. "
            "Please modify the visual backbone to return token sequence (patch tokens). "
            "E.g., add return_tokens flag in visual forward, or implement encode_image_with_tokens method."
        )

    def forward(
        self,
        image,
        client_idx=None,
    ):
        del client_idx
        img_dtype = next(self.image_encoder.parameters()).dtype
        image = image.to(dtype=img_dtype)

        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)


        gate_inp = image_features.detach()

        pl = self.prompt_learner
        tokenized_prompts = self.tokenized_prompts.to(image_features.device)
        logit_scale = self.logit_scale.exp()

        pi, _ = pl.compute_gate(gate_inp)
        experts_A = pl._get_expert_coeffs_all()
        experts_ctx_list = [pl.build_ctx_from_A(experts_A[m]) for m in range(experts_A.size(0))]
        experts_ctx = torch.stack(experts_ctx_list, dim=0)

        tf_all = []
        for m in range(pl.num_experts):
            ctx_m = experts_ctx[m]
            prompts_m = pl.construct_prompts(ctx_m)
            tf_m = pl.text_features_from_prompts(self.text_encoder, prompts_m, tokenized_prompts)
            tf_all.append(tf_m)
        tf_all = torch.stack(tf_all, dim=0)

        logits_all = logit_scale * torch.einsum("bd,mcd->bmc", image_features, tf_all)
        logits = torch.einsum("bm,bmc->bc", pi, logits_all)
        return logits



def _text_features_from_prompts(text_encoder: TextEncoder, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
    tf = text_encoder(prompts, tokenized_prompts, return_tokens=False)
    tf = tf / tf.norm(dim=-1, keepdim=True)
    return tf



PromptLearner.text_features_from_prompts = staticmethod(_text_features_from_prompts)


class FEDSEPT(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.FEDSEPT.PREC in ["fp16", "fp32", "amp"]

    def _expert_diversity_reg(self, experts_A: Optional[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:

        if experts_A is None or experts_A.dim() != 3 or experts_A.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        K = experts_A.size(0)
        if K <= 1:
            return experts_A.sum() * 0.0

        X = experts_A.reshape(K, -1)
        X = F.normalize(X, dim=-1, eps=eps)
        sim = X @ X.t()
        mask = 1.0 - torch.eye(K, device=X.device, dtype=X.dtype)
        return (sim.pow(2) * mask).sum() / mask.sum().clamp_min(eps)

    def _dp_param_iter(self):
        pl = self.model.prompt_learner
        if getattr(pl, "experts_A", None) is not None and pl.experts_A.requires_grad:
            yield pl.experts_A

    def _compute_private_updates(self, per_sample_losses):
        return noise_backend.compute_private_gradients(
            trainer=self,
            per_sample_losses=per_sample_losses,
            params=self._dp_param_iter(),
        )


    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        num_experts = int(getattr(self.args, "fedsept_num_experts", 4))
        rank = int(getattr(self.args, "fedsept_rank", 8))
        router_num_queries = 1
        router_num_heads = int(getattr(self.args, "fedsept_router_num_heads", 4))

        logger.info("Loading CLIP...")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()
 
        if num_experts == 1:
            logger.info("[FEDSEPT] single-expert mode detected: router/gating is disabled by construction.")

        self.model = CustomCLIP(
            cfg,
            classnames,
            clip_model,
            num_experts=num_experts,
            rank=rank,
            router_num_queries=router_num_queries,
            router_num_heads=router_num_heads,
            router_dropout=0.1,
        )

        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        pl = self.model.prompt_learner
        if pl._use_router:
            dummy_dim = pl._image_feat_dim_guess
            dummy_tokens = torch.zeros(1, 1, dummy_dim, device=self.device, dtype=torch.float32)
            pl._ensure_router(token_dim=dummy_dim, dtype=dummy_tokens.dtype, device=dummy_tokens.device)

            router_params = list(pl.router.parameters()) if hasattr(pl, "router") and pl.router is not None else []
            if router_params:
                base_lr = cfg.OPTIM.LR
                router_lr_factor = 1.0
                in_optim = any(
                    any(p is q for p in group.get("params", []))
                    for group in self.optim.param_groups
                    for q in router_params
                )
                if not in_optim:
                    self.optim.add_param_group({"params": router_params, "lr": base_lr * router_lr_factor, "weight_decay": 0.0})
                logger.info(f"[FEDSEPT] router in optimizer: True, router_lr_factor={router_lr_factor:.4g}")
        else:
            logger.info("[FEDSEPT] router disabled (M=1)")

        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if (cfg.TRAINER.FEDSEPT.PREC == "amp") else None

        self.global_epoch = 0 
        self.lambda_div = float(getattr(self.args, "fedsept_lambda_div", 10.0))

        self.dp_mode = str(getattr(self.args, "dp_mode", "local")).lower()
        self.dp_enable = self.dp_mode == "local"
        self.dp_clip = float(getattr(self.args, "dp_clip", 1.0))
        self.dp_sigma = float(getattr(self.args, "dp_sigma", 0.0))

        self._ldp_sigma_by_client = {}

        logger.info(
            "[FEDSEPT] M=%d, rank=%d, lambda_div=%g, "
            "privacy_backend=%s",
            num_experts,
            rank,
            self.lambda_div,
            self.dp_enable,
        )


    def train(self, idx: int = -1, global_epoch: int = 0, is_fed: bool = False, **kwargs):
        self.global_epoch = int(global_epoch)

        if self.dp_enable:
            noise_backend.prepare_client_privacy(
                self,
                client_idx=idx,
                logger=logger,
            )

        return super().train(
            idx=idx,
            global_epoch=global_epoch,
            is_fed=is_fed,
            **kwargs,
        )


    def forward_backward(self, batch_idx, batch, idx: int = -1, **kwargs):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.FEDSEPT.PREC
        use_amp = (prec == "amp")

        def _compute_losses(): 
            logits = self.model(image)
            ce_losses = F.cross_entropy(
                logits,
                label,
                reduction="none",
            )
            ce = ce_losses.mean()
            pl = self.model.prompt_learner
            M = int(getattr(pl, "num_experts", 1) or 1)

            E_all = pl._get_expert_coeffs_all()
            if M > 1:
                reg_div = self._expert_diversity_reg(E_all)
            else:
                reg_div = ce * 0.0

            per_sample_losses = (
                ce_losses
                + self.lambda_div * reg_div
            )
            return logits, per_sample_losses

        if use_amp:
            with autocast():
                logits, per_sample_losses = _compute_losses()
        else:
            logits, per_sample_losses = _compute_losses()

        loss = per_sample_losses.mean()
        private_updates = (
            self._compute_private_updates(per_sample_losses)
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
            "acc": compute_accuracy(logits, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def parse_batch_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return image, label
