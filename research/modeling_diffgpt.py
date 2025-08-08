import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .config import DiffGPTConfig


def precompute_rotary_emb(dim: int, max_positions: int) -> torch.Tensor:
    thetas = [10000 ** (-2 * (i - 1) / dim) for i in range(1, dim // 2 + 1)]
    cos_vals = []
    sin_vals = []
    for t in range(max_positions):
        cos_row = [math.cos(t * theta) for theta in thetas]
        sin_row = [math.sin(t * theta) for theta in thetas]
        cos_vals.append(cos_row)
        sin_vals.append(sin_row)
    cos_tensor = torch.tensor(cos_vals)
    sin_tensor = torch.tensor(sin_vals)
    rope_cache = torch.stack([cos_tensor, sin_tensor], dim=-1)
    return rope_cache


def apply_rotary_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    B, nh, T, ns = x.size()
    rope_cache_truncated = rope_cache[:T]
    assert ns % 2 == 0, "Attention head size must be even for RoPE"
    x = x.view(B, nh, T, ns // 2, 2)
    x_complex = torch.view_as_complex(x)
    rope_complex = torch.view_as_complex(rope_cache_truncated)
    rope_complex = rope_complex.unsqueeze(0).unsqueeze(0)
    rotated_x_complex = x_complex * rope_complex
    rotated_x = torch.view_as_real(rotated_x_complex).view(B, nh, T, ns)
    return rotated_x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: DiffGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
        )
        self.n_head = config.n_head
        self.rope = config.rope
        if self.rope:
            assert (config.n_embd // config.n_head) % 2 == 0
            rope_cache = precompute_rotary_emb(config.n_embd // config.n_head, config.block_size)
            self.register_buffer("rope_cache", rope_cache)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.rope:
            q = apply_rotary_emb(q, self.rope_cache)
            k = apply_rotary_emb(k, self.rope_cache)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        bad_val = torch.finfo(att.dtype).min if att.dtype.is_floating_point else -1e9
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, bad_val)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config: DiffGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DiffGPTForCausalLM(PreTrainedModel):
    config_class = DiffGPTConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: DiffGPTConfig):
        super().__init__(config)
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        if not config.rope:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.start_emb = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self._gradient_checkpointing = False
        self._gc_use_reentrant = True
        self.post_init()

    def _set_gradient_checkpointing(self, module, value: bool = False):
        self._gradient_checkpointing = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if isinstance(gradient_checkpointing_kwargs, dict):
            self._gc_use_reentrant = bool(gradient_checkpointing_kwargs.get("use_reentrant", True))
        return super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self._gc_use_reentrant = True
        return super().gradient_checkpointing_disable()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.tok_emb

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.tok_emb = new_embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_head: nn.Module):
        self.lm_head = new_head

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        assert input_ids is not None, "input_ids required"
        bsz, seqlen = input_ids.size()
        if seqlen > self.config.block_size:
            raise ValueError(f"Sequence length {seqlen} exceeds block size {self.config.block_size}")

        token_embeddings = self.tok_emb(input_ids)  # (B, T, D)
        original_token_embeddings = None

        if self.config.research:
            original_token_embeddings = token_embeddings
            token_embeddings_shifted = torch.roll(token_embeddings, shifts=1, dims=1)
            token_embeddings_shifted[:, 0, :] = self.start_emb
            diff_token_embeddings = token_embeddings - token_embeddings_shifted
            # Collect statistics for visualization callbacks (no_grad and detached)
            with torch.no_grad():
                diff_l2 = diff_token_embeddings.norm(dim=-1)  # (B, T)
                diff_l2_flat = diff_l2.detach().float().view(-1)
                # Limit histogram sample size to avoid large tensor logging
                max_hist_elems = 8192
                hist_sample = diff_l2_flat[: min(max_hist_elems, diff_l2_flat.numel())].cpu()
                self._last_diff_stats = {
                    "diff_mean_l2": diff_l2_flat.mean().item(),
                    "diff_std_l2": diff_l2_flat.std(unbiased=False).item() if diff_l2_flat.numel() > 1 else 0.0,
                    "diff_max_l2": diff_l2_flat.max().item(),
                    "diff_min_l2": diff_l2_flat.min().item(),
                    "diff_l2_hist_sample": hist_sample,
                }
            token_embeddings = diff_token_embeddings

        if self.config.rope:
            x = token_embeddings
        else:
            position_embeddings = self.pos_emb[:, :seqlen, :]
            x = token_embeddings + position_embeddings

        x = self.drop(x)
        if self._gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.blocks:
                x = checkpoint(block, x, use_reentrant=self._gc_use_reentrant)
        else:
            for block in self.blocks:
                x = block(x)
        x = self.ln_f(x)

        if self.config.research and original_token_embeddings is not None:
            x = x + original_token_embeddings

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift labels to predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        return CausalLMOutputWithPast(loss=loss, logits=logits)