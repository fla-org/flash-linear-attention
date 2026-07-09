# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Parallax (parameterized local linear attention), contributed by
# Yifei Zuo et al. (https://arxiv.org/abs/2605.29157).

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.parallax import parallel_parallax
from fla.ops.parallax.decode import parallax_decode, parallax_decode_one_step
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)


class Parallax(nn.Module):
    r"""Parallax: parameterized local linear attention.

    A quadratic, causal, softmax-attention-style layer with an extra query-side
    projection ``r`` that injects a first-order correction onto the
    softmax-weighted values (see :func:`fla.ops.parallax.naive_parallax`).
    Rotary position embeddings are applied to ``q``, ``k`` and ``r`` (``r`` is
    rotated with the same ``cos``/``sin`` as ``q``); causality is enforced by the
    kernel mask.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        num_heads (int, Optional):
            The number of query heads. Default: 32.
        num_kv_heads (int, Optional):
            The number of key/value heads, equal to `num_heads` if `None`.
            GQA is applied when `num_heads` is a multiple of `num_kv_heads`. Default: `None`.
        qkv_bias (bool, Optional):
            Whether to use bias in the q/r/k/v projections. Default: `False`.
        qk_norm (bool, Optional):
            Whether to apply per-head RMSNorm to `q`, `r` and `k`. Default: `False`.
        window_size (int, Optional):
            Sliding-window size; `None` for full causal attention. Default: `None`.
        rope_theta (float, Optional):
            The base frequency for the rotary position embedding. Default: 10000.
        max_position_embeddings (int, Optional):
            The maximum sequence length for the rotary cache. Default: `None`.
        layer_idx (int, Optional):
            The index of the layer, used for cache keying. Default: `None`.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int | None = None,
        layer_idx: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        # `r` is a second query-side stream, so `r_proj` mirrors `q_proj`.
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.r_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, dtype=torch.float32)
            self.r_norm = RMSNorm(self.head_dim, dtype=torch.float32)
            self.k_norm = RMSNorm(self.head_dim, dtype=torch.float32)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        r = rearrange(self.r_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, r, k = self.q_norm(q), self.r_norm(r), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens')

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q_len + seqlen_offset
            if attention_mask is not None:
                # account for left-padding when indexing rotary positions
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q_len + seqlen_offset.max().item()
        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        # rope on q, k and r; `r` shares q's positions (same cos/sin), then is rotated alone.
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)
        r, _ = self.rotary(r, r, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            # Decode / cached prefill. `r` is regenerated each step (like `q`), so only
            # `(k, v)` are cached; the new queries attend to the full cached KV.
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            k = rearrange(k_cached, '... (h d) -> ... h d', d=self.head_dim)
            v = rearrange(v_cached, '... (h d) -> ... h d', d=self.head_dim)
            # left-padding: the first `kv_len - valid_len` cached keys are padding.
            cache_start = None
            if attention_mask is not None:
                cache_start = (k.shape[1] - attention_mask.sum(-1)).to(torch.int32)
            # single-token step -> optimized vector decode; chunked prefill -> tile kernel
            decode_fn = parallax_decode_one_step if q_len == 1 else parallax_decode
            o = decode_fn(q, r, k, v, window_size=self.window_size, cache_start=cache_start)
        elif attention_mask is not None:
            # Unpad to a single packed sequence (batch folded into the seq axis).
            q, (r, k, v), indices_q, cu_seqlens, _ = unpad_input(
                q, (r, k, v), attention_mask, q_len, keepdim=True,
            )
            _, cu_seqlens_k = cu_seqlens
            o = parallel_parallax(q, r, k, v, window_size=self.window_size, cu_seqlens=cu_seqlens_k)
            o = pad_input(o.squeeze(0), indices_q, batch_size, q_len)
        else:
            o = parallel_parallax(q, r, k, v, window_size=self.window_size, cu_seqlens=cu_seqlens)

        o = rearrange(o, '... h d -> ... (h d)')
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values
