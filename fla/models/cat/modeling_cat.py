# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Contributed by: Jatin Prakash (bicycleman15)
# Controllably Efficient Language Models (https://arxiv.org/abs/2511.05313)

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.models.cat.configuration_cat import CATConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm, RotaryEmbedding
from fla.modules import GatedMLP as CATMLP
from fla.modules.rotary import rotary_embedding

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

# Adaptive CAT cycles power-of-two chunk sizes, each with a different FlexAttention
# sequence length / block mask. Raise Dynamo limits so those shapes stay compiled
# instead of falling back to dense math attention after the default recompile cap.
_CAT_DYNAMO_CACHE_LIMIT = 64
_create_block_mask_compiled = torch.compile(create_block_mask)
_flex_attention_compiled = torch.compile(flex_attention, dynamic=False, mode="default")

logger = logging.get_logger(__name__)


def _cat_dynamo_config_patch():
    config = {
        'cache_size_limit': max(
            getattr(torch._dynamo.config, 'cache_size_limit', 0),
            _CAT_DYNAMO_CACHE_LIMIT,
        ),
    }
    if hasattr(torch._dynamo.config, 'recompile_limit'):
        config['recompile_limit'] = max(
            getattr(torch._dynamo.config, 'recompile_limit', 0),
            _CAT_DYNAMO_CACHE_LIMIT,
        )
    return torch._dynamo.config.patch(config)


@torch.compiler.disable(recursive=False)
def create_block_mask_compiled(
    mask_mod,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
) -> BlockMask:
    with _cat_dynamo_config_patch():
        return _create_block_mask_compiled(mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN)


@torch.compiler.disable(recursive=False)
def flex_attention_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: BlockMask | None = None,
) -> torch.Tensor:
    with _cat_dynamo_config_patch():
        return _flex_attention_compiled(q, k, v, block_mask=block_mask)


def _ceil_div(x: int, y: int) -> int:
    return -(-x // y)


def get_cat_mask_mod(block_size: int):
    """
    Create the interleaved CAT attention mask.

    The decoder sequence is `[fx, sep, tokens...]` per chunk, followed by a final
    `[fx, sep]` pair. Queries attend causally within their local block and to prior
    compressed chunk representations at `kv_idx % block_size == 0`.
    """
    def cat_mask(b, h, q_idx, kv_idx):
        within_block = (q_idx // block_size) == (kv_idx // block_size)
        compressed_token = (kv_idx % block_size) == 0
        causal = q_idx >= kv_idx
        return (compressed_token | within_block) & causal
    return cat_mask


def get_block_diagonal_mask_mod(block_size: int):
    """Create a block-diagonal mask for independent chunk compression."""
    def block_diagonal(b, h, q_idx, kv_idx):
        return (q_idx // block_size) == (kv_idx // block_size)
    return block_diagonal


class CATCompressorBlock(GradientCheckpointingLayer):
    """Transformer block for the CAT compressor."""

    def __init__(self, config: CATConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.compressor_hidden_size
        self.num_heads = config.compressor_num_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(self.hidden_size, eps=config.norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(self.hidden_size, eps=config.norm_eps)
        self.mlp = CATMLP(
            hidden_size=self.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        q = rearrange(self.q_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(self.v_proj(hidden_states), 'b t (h d) -> b h t d', h=self.num_heads)

        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        q = rotary_embedding(q, cos, sin, interleaved=False)
        k = rotary_embedding(k, cos, sin, interleaved=False)

        q = rearrange(q, 'b t h d -> b h t d')
        k = rearrange(k, 'b t h d -> b h t d')
        hidden_states = flex_attention_compiled(q, k, v, block_mask=block_mask)
        hidden_states = rearrange(hidden_states, 'b h t d -> b t (h d)')
        hidden_states = self.o_proj(hidden_states)

        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class CATCompressor(nn.Module):
    """Compress chunks of tokens into fixed-size CAT representations."""

    def __init__(self, config: CATConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.compressor_hidden_size
        self.max_chunk_size = config.max_chunk_size

        self.wte = nn.Embedding(config.vocab_size, self.hidden_size)
        self.pos_tokens = nn.Embedding(config.max_position_embeddings, self.hidden_size)
        self.adaptive_tokens = nn.Embedding(config.max_chunk_size + 1, self.hidden_size)
        self.layers = nn.ModuleList([CATCompressorBlock(config, layer_idx=i) for i in range(config.compressor_num_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(self.hidden_size, eps=config.norm_eps)
        self.proj_fx = nn.Linear(self.hidden_size * config.max_chunk_size, config.dim_fx, bias=False)
        self.rotary = RotaryEmbedding(dim=self.hidden_size // config.compressor_num_heads, base=config.rope_theta)

    def compress_batched(
        self,
        input_ids_chunked: torch.LongTensor,
        chunk_size: int,
    ) -> torch.Tensor:
        batch_size, num_chunks, input_chunk_size = input_ids_chunked.shape
        if input_chunk_size != chunk_size:
            raise ValueError(f"`chunk_size` ({chunk_size}) must match input chunks ({input_chunk_size}).")
        if chunk_size > self.max_chunk_size:
            raise ValueError(f"`chunk_size` ({chunk_size}) cannot exceed `max_chunk_size` ({self.max_chunk_size}).")

        device = input_ids_chunked.device
        block_len = 2 + chunk_size

        token_embeds = self.wte(input_ids_chunked)
        dtype = token_embeds.dtype

        chunk_size_idx = torch.tensor([chunk_size], device=device, dtype=torch.long)
        adaptive_token = self.adaptive_tokens(chunk_size_idx)
        adaptive_token = repeat(adaptive_token, '1 d -> b k 1 d', b=batch_size, k=num_chunks)

        chunk_indices = torch.arange(num_chunks, device=device, dtype=torch.long)
        pos_tokens = self.pos_tokens(chunk_indices)
        pos_tokens = repeat(pos_tokens, 'k d -> b k 1 d', b=batch_size)

        hidden_states = torch.cat([adaptive_token, pos_tokens, token_embeds], dim=2)
        total_len = num_chunks * block_len
        hidden_states = rearrange(hidden_states, 'b k l d -> b (k l) d')

        block_mask = create_block_mask_compiled(
            get_block_diagonal_mask_mod(block_len),
            B=None,
            H=None,
            Q_LEN=total_len,
            KV_LEN=total_len,
        )

        max_block_len = 2 + self.max_chunk_size
        self.rotary._update_cos_sin_cache(max_block_len, device=device, dtype=dtype)
        position_ids = torch.arange(total_len, device=device) % block_len
        cos = self.rotary._cos_cached[position_ids]
        sin = self.rotary._sin_cached[position_ids]

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, block_mask=block_mask)
        hidden_states = self.norm(hidden_states)

        hidden_states = rearrange(hidden_states, 'b (k l) d -> b k l d', k=num_chunks, l=block_len)
        hidden_states = hidden_states[:, :, 2:, :]
        hidden_states = rearrange(hidden_states, 'b k l d -> b k (l d)')

        if chunk_size == self.max_chunk_size:
            return self.proj_fx(hidden_states)

        target_in_features = chunk_size * self.hidden_size
        weight = F.interpolate(
            self.proj_fx.weight.unsqueeze(0).unsqueeze(0),
            size=(self.config.dim_fx, target_in_features),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        return F.linear(hidden_states, weight, bias=None)


class CATDecoderAttention(nn.Module):
    """Decoder self-attention with the CAT structural FlexAttention mask."""

    def __init__(self, config: CATConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, dtype=torch.float32)
            self.k_norm = RMSNorm(self.head_dim, dtype=torch.float32)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = rearrange(self.q_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_kv_heads)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        q = rotary_embedding(q, cos, sin, interleaved=False)
        k = rotary_embedding(k, cos, sin, interleaved=False)

        if self.num_kv_groups > 1:
            k = repeat(k, 'b t h d -> b t (h g) d', g=self.num_kv_groups)
            v = repeat(v, 'b t h d -> b t (h g) d', g=self.num_kv_groups)

        q = rearrange(q, 'b t h d -> b h t d')
        k = rearrange(k, 'b t h d -> b h t d')
        v = rearrange(v, 'b t h d -> b h t d')

        hidden_states = flex_attention_compiled(q, k, v, block_mask=block_mask)
        hidden_states = rearrange(hidden_states, 'b h t d -> b t (h d)')
        return self.o_proj(hidden_states.reshape(batch_size, seq_len, self.hidden_size))


class CATBlock(GradientCheckpointingLayer):
    """Transformer block for the CAT decoder."""

    def __init__(self, config: CATConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = CATDecoderAttention(config, layer_idx)
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = CATMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_mask: BlockMask,
        **kwargs: Unpack[Any],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(hidden_states, cos=cos, sin=sin, block_mask=block_mask)

        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)

        hidden_states = self.mlp(hidden_states, **kwargs)
        return residual + hidden_states


class CATPreTrainedModel(PreTrainedModel):
    """Base class for CAT models."""

    config_class = CATConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['CATBlock', 'CATCompressorBlock']
    _supports_cache_class = False

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if rescale_prenorm_residual:
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class CATModel(CATPreTrainedModel):
    """CAT backbone using adaptive chunk sizes and the official interleaved mask."""

    def __init__(self, config: CATConfig):
        super().__init__(config)

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.compressor = CATCompressor(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.separator = nn.Embedding(1, config.hidden_size)
        self.adaptive_token = nn.Embedding(config.max_chunk_size + 1, config.hidden_size)
        self.layers = nn.ModuleList([CATBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.rotary = RotaryEmbedding(dim=config.hidden_size // config.num_heads, base=config.rope_theta)
        self.gradient_checkpointing = False
        self._train_chunk_sizes = self._build_train_chunk_sizes()
        self.register_buffer('_train_chunk_size_step', torch.zeros((), dtype=torch.long), persistent=True)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def _build_train_chunk_sizes(self) -> tuple[int, ...]:
        if self.config.train_chunk_size_schedule != "power_of_2":
            return (self.config.max_chunk_size,)

        chunk_sizes = []
        chunk_size = self.config.min_train_chunk_size
        while chunk_size <= self.config.max_chunk_size:
            chunk_sizes.append(chunk_size)
            chunk_size *= 2
        return tuple(chunk_sizes)

    def _next_train_chunk_size(self) -> int:
        chunk_sizes = self._train_chunk_sizes
        step = int(self._train_chunk_size_step.item())
        chunk_size = chunk_sizes[step % len(chunk_sizes)]
        self._train_chunk_size_step.add_(1)
        return chunk_size

    def _resolve_chunk_size(self, chunk_size: int | None) -> int:
        if chunk_size is not None:
            return chunk_size
        if self.training and self.config.train_chunk_size_schedule == "power_of_2":
            return self._next_train_chunk_size()
        return self.config.max_chunk_size

    def _get_padded_seq_len(self, seq_len: int, chunk_size: int) -> int:
        max_seq_len = self.config.max_position_embeddings
        if seq_len > max_seq_len:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds `max_position_embeddings` ({max_seq_len}).")

        padded_seq_len = _ceil_div(seq_len, self.config.pad_to_multiple_of) * self.config.pad_to_multiple_of
        padded_seq_len = max(padded_seq_len, chunk_size)
        padded_seq_len = min(padded_seq_len, max_seq_len)

        if padded_seq_len % chunk_size == 0:
            return padded_seq_len

        chunk_aligned = _ceil_div(padded_seq_len, chunk_size) * chunk_size
        if chunk_aligned <= max_seq_len:
            return chunk_aligned

        chunk_aligned = (max_seq_len // chunk_size) * chunk_size
        if chunk_aligned < seq_len:
            raise ValueError(
                f"Cannot pad sequence length {seq_len} to a multiple of chunk_size {chunk_size} within "
                f"`max_position_embeddings` ({max_seq_len})."
            )
        return chunk_aligned

    def _validate_attention_mask(self, attention_mask: torch.Tensor | None, seq_len: int) -> None:
        if attention_mask is None:
            return
        if attention_mask.shape[-1] != seq_len:
            raise ValueError(
                f"`attention_mask` length ({attention_mask.shape[-1]}) must match input length ({seq_len}) for CAT."
            )
        if not bool(attention_mask.to(dtype=torch.bool).all()):
            raise ValueError("CAT only supports `attention_mask=None` or all-ones masks; padded batches are not supported.")

    def _build_cat_positions(
        self,
        num_chunks: int,
        chunk_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        block_len = 2 + chunk_size
        positions = torch.arange(block_len, device=device).repeat(num_chunks)
        return torch.cat([positions, torch.arange(2, device=device)])

    def _build_cat_rope_cache(
        self,
        num_chunks: int,
        chunk_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_pos = 4 + self.config.max_chunk_size
        self.rotary._update_cos_sin_cache(max_pos, device=device, dtype=dtype)
        positions = self._build_cat_positions(num_chunks, chunk_size, device)
        return self.rotary._cos_cached[positions], self.rotary._sin_cached[positions]

    def _rearrange_hidden_states(
        self,
        hidden_states: torch.Tensor,
        num_chunks: int,
        chunk_size: int,
        original_seq_len: int,
    ) -> torch.Tensor:
        block_len = 2 + chunk_size
        last_pred = hidden_states[:, -1:, :]
        hidden_states_chunks = rearrange(
            hidden_states[:, :-2, :],
            'b (k l) d -> b k l d',
            k=num_chunks,
            l=block_len,
        )
        first_chunk_preds = hidden_states_chunks[:, 0, 2:-1, :]
        middle_chunk_preds = hidden_states_chunks[:, 1:, 1:-1, :]
        middle_chunk_preds = rearrange(middle_chunk_preds, 'b k l d -> b (k l) d')
        hidden_states = torch.cat([first_chunk_preds, middle_chunk_preds, last_pred], dim=1)
        return hidden_states[:, :original_seq_len, :]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        chunk_size: int | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | BaseModelOutputWithPast:
        if output_attentions:
            logger.warning_once("`CATModel` does not support `output_attentions`; setting it to `False`.")
        if use_cache:
            logger.warning_once("CAT does not support KV cache; setting `use_cache=False`.")
        if past_key_values is not None:
            logger.warning_once("CAT does not consume `past_key_values`; ignoring them.")

        kwargs.pop('cache_position', None)
        kwargs.pop('position_ids', None)
        cu_seqlens = kwargs.pop('cu_seqlens', None)
        if cu_seqlens is not None:
            raise ValueError("CAT does not support variable-length `cu_seqlens` inputs yet; use fixed-length batches.")

        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        chunk_size = self._resolve_chunk_size(chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive, got {chunk_size}.")
        if chunk_size > self.config.max_chunk_size:
            raise ValueError(f"`chunk_size` ({chunk_size}) cannot exceed `max_chunk_size` ({self.config.max_chunk_size}).")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both `input_ids` and `inputs_embeds`.")
        if inputs_embeds is not None:
            raise NotImplementedError("`inputs_embeds` are not yet supported for CAT.")
        if input_ids is None:
            raise ValueError("You must specify `input_ids` for CAT.")

        batch_size, original_seq_len = input_ids.shape
        self._validate_attention_mask(attention_mask, original_seq_len)

        seq_len = self._get_padded_seq_len(original_seq_len, chunk_size)
        if seq_len > original_seq_len:
            pad_token = self.padding_idx if self.padding_idx is not None else 0
            input_ids = F.pad(input_ids, (0, seq_len - original_seq_len), value=pad_token)

        num_chunks = seq_len // chunk_size
        input_ids_chunked = input_ids.view(batch_size, num_chunks, chunk_size)
        fx = self.compressor.compress_batched(input_ids_chunked, chunk_size)
        fx_last = fx[:, -1:, :]
        token_embeds = self.embeddings(input_ids_chunked)

        device = input_ids.device
        chunk_size_idx = torch.tensor([chunk_size], device=device, dtype=torch.long)
        adaptive_token = self.adaptive_token(chunk_size_idx)
        adaptive_token = repeat(adaptive_token, '1 d -> b 1 d', b=batch_size)

        sep_idx = torch.zeros(1, device=device, dtype=torch.long)
        sep_token = self.separator(sep_idx)
        sep_tokens = repeat(sep_token, '1 d -> b k 1 d', b=batch_size, k=num_chunks)

        fx_shifted = torch.cat([adaptive_token.unsqueeze(1), fx[:, :-1, :].unsqueeze(2)], dim=1)
        hidden_states = torch.cat([fx_shifted, sep_tokens, token_embeds], dim=2)
        hidden_states = rearrange(hidden_states, 'b k l d -> b (k l) d')
        last_sep = repeat(sep_token, '1 d -> b 1 d', b=batch_size)
        hidden_states = torch.cat([hidden_states, fx_last, last_sep], dim=1)

        cos, sin = self._build_cat_rope_cache(num_chunks, chunk_size, device, hidden_states.dtype)
        block_len = 2 + chunk_size
        block_mask = create_block_mask_compiled(
            get_cat_mask_mod(block_len),
            B=None,
            H=None,
            Q_LEN=hidden_states.shape[1],
            KV_LEN=hidden_states.shape[1],
        )

        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer(hidden_states, cos=cos, sin=sin, block_mask=block_mask, **kwargs)
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            all_hidden_states = tuple(
                self._rearrange_hidden_states(
                    state,
                    num_chunks=num_chunks,
                    chunk_size=chunk_size,
                    original_seq_len=original_seq_len,
                )
                for state in all_hidden_states
            )
        hidden_states = self._rearrange_hidden_states(
            hidden_states,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            original_seq_len=original_seq_len,
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, None] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class CATForCausalLM(CATPreTrainedModel, FLAGenerationMixin):
    """CAT model with a language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: CATConfig):
        super().__init__(config)

        self.model = CATModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        logits_to_keep: int | None = None,
        cache_position: torch.LongTensor | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ):
        model_inputs = {
            'input_ids': input_ids.contiguous() if input_ids is not None else None,
            'inputs_embeds': inputs_embeds if input_ids is None else None,
            'past_key_values': None,
            'use_cache': False,
            'attention_mask': attention_mask,
            'chunk_size': chunk_size,
        }
        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        chunk_size: int | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            chunk_size=chunk_size,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = None if self.config.fuse_linear_cross_entropy and labels is not None else self.lm_head(
            hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:]
        )

        loss = None
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion

            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)

            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
