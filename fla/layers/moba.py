# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import FusedRMSNormGated, RMSNorm, RotaryEmbedding
from fla.ops.moba import moba_attn_varlen
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_moba import flash_moba_varlen_func
except ImportError:
    flash_moba_varlen_func = None

logger = logging.get_logger(__name__)


class MobaAttention(nn.Module):

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
        moba_chunk_size: int = 256,
        moba_topk: int = 4,
        use_output_gate: bool = True,
        use_flash_moba: bool = False,
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
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk
        self.use_output_gate = use_output_gate
        self.use_flash_moba = use_flash_moba

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        if self.use_flash_moba and flash_moba_varlen_func is None:
            raise ImportError(
                "Please install `flash_moba` via `pip install flash-moba` "
                "to use `use_flash_moba=True`."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

        if use_output_gate:
            self.g_proj = nn.Sequential(
                nn.Linear(hidden_size, self.head_dim, bias=False),
                nn.Linear(self.head_dim, self.hidden_size, bias=False)
            )
            self.o_norm = FusedRMSNormGated(self.head_dim, activation='sigmoid', eps=1e-6)
        else:
            logger.info("MobaAttention is NOT using output gate.")
            self.o_norm = RMSNorm(self.head_dim, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens')

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # Handle attention_mask by unpadding

        # Path 1: `attention_mask` is provided (e.g., Ruler tasks)
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

            # Align the padding mask to the actual cached KV span before unpadding.
            if q_len == 1 and attention_mask.shape[-1] != k.shape[1]:
                attention_mask = attention_mask[:, -k.shape[1]:]

            # q, k, v are (B, S, H, D). unpad_input turns them into (Total, H, D)
            q_unpad, (k_unpad, v_unpad), indices_q, cu_seqlens_tuple, max_seq_lens_tuple = unpad_input(
                q, (k, v), attention_mask, q_len)

            if self.use_flash_moba:
                cu_seqlens_q, cu_seqlens_k = cu_seqlens_tuple
                max_seqlen_q, max_seqlen_k = max_seq_lens_tuple
                o_unpad = flash_moba_varlen_func(
                    q=q_unpad,
                    k=k_unpad,
                    v=v_unpad,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    moba_chunk_size=self.moba_chunk_size,
                    moba_topk=self.moba_topk,
                    causal=True
                )
            else:
                if k.shape[1] != q_len:
                    raise NotImplementedError(
                        "MobaAttention cached decoding requires separate Q/KV varlen metadata for the Triton backend."
                    )
                cu_seqlens_moba = cu_seqlens_tuple[0]
                max_seqlen_moba = max_seq_lens_tuple[0]
                o_unpad = moba_attn_varlen(
                    q_unpad,
                    k_unpad,
                    v_unpad,
                    cu_seqlens=cu_seqlens_moba,
                    max_seqlen=max_seqlen_moba,
                    moba_chunk_size=self.moba_chunk_size,
                    moba_topk=self.moba_topk
                )

            # pad_input turns o_unpad (Total, H, D) back into (B, S, H, D)
            o = pad_input(o_unpad, indices_q, batch_size, q_len)

        # Path 2: No `attention_mask` (e.g., wikitext, or data is already unpadded)
        else:
            k_len = k.shape[1]
            q_unbatched = rearrange(q, 'b s h d -> (b s) h d')
            k_unbatched = rearrange(k, 'b s h d -> (b s) h d')
            v_unbatched = rearrange(v, 'b s h d -> (b s) h d')

            if self.use_flash_moba:
                if cu_seqlens is None:
                    cu_seqlens_q = torch.arange(0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=hidden_states.device)
                    cu_seqlens_k = torch.arange(0, (batch_size + 1) * k_len, step=k_len, dtype=torch.int32, device=hidden_states.device)
                elif isinstance(cu_seqlens, (tuple, list)):
                    cu_seqlens_q, cu_seqlens_k = cu_seqlens
                else:
                    cu_seqlens_q = cu_seqlens
                    if k_len == q_len:
                        cu_seqlens_k = cu_seqlens
                    else:
                        cu_seqlens_k = torch.arange(0, (batch_size + 1) * k_len, step=k_len, dtype=torch.int32, device=hidden_states.device)

                o = flash_moba_varlen_func(
                    q=q_unbatched,
                    k=k_unbatched,
                    v=v_unbatched,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=q_len,
                    max_seqlen_k=k_len,
                    moba_chunk_size=self.moba_chunk_size,
                    moba_topk=self.moba_topk,
                    causal=True
                )
            else:
                if k_len != q_len:
                    raise NotImplementedError(
                        "MobaAttention cached decoding requires separate Q/KV varlen metadata for the Triton backend."
                    )
                if cu_seqlens is None:
                    cu_seqlens = torch.arange(0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=hidden_states.device)

                o = moba_attn_varlen(
                    q_unbatched,
                    k_unbatched,
                    v_unbatched,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    moba_chunk_size=self.moba_chunk_size,
                    moba_topk=self.moba_topk
                )

            o = rearrange(o, '(b s) h d -> b s h d', b=batch_size)

        if self.use_output_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        attentions = None
        return o, attentions, past_key_values
