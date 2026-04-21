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
from fla.ops.moba import parallel_moba
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    flash_attn_func = None
    flash_attn_varlen_func = None

try:
    from flash_moba import flash_moba_varlen_func
except ImportError:
    flash_moba_varlen_func = None

logger = logging.get_logger(__name__)


class MoBA(nn.Module):
    """
    The layer implementation for [MoBA: Mixture of Block Attention for Long-Context LLMs]
    (https://arxiv.org/abs/2502.13189).

    MoBA partitions the key/value sequence into fixed-size chunks ("blocks") and, for every query token,
    only attends to a small set of the most relevant blocks instead of the entire history:

    1. Each KV block is summarized by the mean of its keys, producing one representative key per block.
    2. Each query token scores every block via a dot product with these representative keys, and selects
       the `moba_topk` highest-scoring blocks (the block containing the query itself is always included
       to preserve causal locality).
    3. Self-attention is run inside the current block, block-MoBA attention is run over the selected
       blocks, and the two outputs are merged via an online-softmax LSE combination.

    This implementation exposes two backends:
      - Triton / flash-attn path (`use_flash_moba=False`, default): routed through
        `fla.ops.moba.parallel_moba`, which composes `flash_attn_varlen_func` with an online-softmax combine.
      - FlashMoBA CUDA path (`use_flash_moba=True`): routed through
        [`flash_moba_varlen_func`](https://github.com/mit-han-lab/flash-moba), a fused kernel from MIT HAN Lab
        that performs gate computation, top-k selection, and attention in a single pass.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        num_heads (int, Optional):
            The number of query heads. Default: 32.
        num_kv_heads (int, Optional):
            The number of key/value heads for GQA. If None, falls back to MHA. Default: None.
        qkv_bias (bool, Optional):
            Whether to use bias in the Q/K/V projections. Default: `False`.
        qk_norm (bool, Optional):
            Whether to apply RMSNorm to Q and K before attention. Default: `False`.
        window_size (int, Optional):
            Sliding-window size forwarded to the KV cache. Default: None.
        rope_theta (float, Optional):
            The base frequency of RoPE. Default: 10000.
        max_position_embeddings (int, Optional):
            The maximum position used for RoPE scaling. Default: None.
        layer_idx (int, Optional):
            The index of the layer, required for KV cache bookkeeping. Default: None.
        moba_chunk_size (int, Optional):
            The size of each KV block. Tail blocks are handled via the chunk-masking
            logic in `prepare_moba_chunks`. Default: 256.
        moba_topk (int, Optional):
            The number of blocks each query attends to, including the local block. Default: 4.
        use_output_gate (bool, Optional):
            Whether to apply a sigmoid-gated RMSNorm on the attention output. Default: `False`.
        use_flash_moba (bool, Optional):
            Whether to use the fused FlashMoBA CUDA kernel. Requires `pip install flash-moba`.
            Default: `False`.
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
        moba_chunk_size: int = 256,
        moba_topk: int = 4,
        use_output_gate: bool = False,
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
                "Please install FlashMoBA via `pip install flash-moba` first "
                "(see https://github.com/mit-han-lab/flash-moba)."
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

        # During decoding, fall back to dense full attention as prescribed by
        # the MoBA paper (https://arxiv.org/abs/2502.13189, Sec. 3.3), which
        # defines a seamless switch from block-sparse to full attention for the
        # generation phase.
        is_decoding = k.shape[1] != q_len

        # Path 1: padding mask provided AND caller did not pre-compute cu_seqlens.
        # When both are present, `cu_seqlens` wins because it carries strictly
        # more packing info than a 2D padding mask (see #842).
        if attention_mask is not None and cu_seqlens is None:
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
            cu_seqlens_q, cu_seqlens_k = cu_seqlens_tuple
            max_seqlen_q, max_seqlen_k = max_seq_lens_tuple

            if is_decoding:
                o_unpad = flash_attn_varlen_func(
                    q_unpad, k_unpad, v_unpad,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    causal=True,
                )
            elif self.use_flash_moba:
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
                    causal=True,
                )
            else:
                o_unpad = parallel_moba(
                    q_unpad.unsqueeze(0),
                    k_unpad.unsqueeze(0),
                    v_unpad.unsqueeze(0),
                    cu_seqlens=cu_seqlens_q,
                    max_seqlen=max_seqlen_q,
                    chunk_size=self.moba_chunk_size,
                    topk=self.moba_topk,
                ).squeeze(0)

            # pad_input turns o_unpad (Total, H, D) back into (B, S, H, D)
            o = pad_input(o_unpad, indices_q, batch_size, q_len)

        # Path 2: no padding mask, or caller already provided cu_seqlens.
        elif is_decoding:
            # Dense causal attention over the cached KV (q_len typically 1).
            o = flash_attn_func(q, k, v, causal=True)
        else:
            # Pack the batch along the token axis ([1, B*T, H, D]) so that
            # cu_seqlens can describe sample boundaries — the fla varlen
            # convention requires batch size 1 with packed inputs.
            q_packed = rearrange(q, 'b s h d -> 1 (b s) h d').contiguous()
            k_packed = rearrange(k, 'b s h d -> 1 (b s) h d').contiguous()
            v_packed = rearrange(v, 'b s h d -> 1 (b s) h d').contiguous()

            offsets = torch.arange(batch_size + 1, dtype=torch.int32, device=hidden_states.device)
            if isinstance(cu_seqlens, (tuple, list)):
                cu_seqlens_q, _ = cu_seqlens
            else:
                cu_seqlens_q = offsets * q_len if cu_seqlens is None else cu_seqlens

            if self.use_flash_moba:
                o = flash_moba_varlen_func(
                    q=q_packed.squeeze(0),
                    k=k_packed.squeeze(0),
                    v=v_packed.squeeze(0),
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_q,
                    max_seqlen_q=q_len,
                    max_seqlen_k=q_len,
                    moba_chunk_size=self.moba_chunk_size,
                    moba_topk=self.moba_topk,
                    causal=True,
                )
            else:
                o = parallel_moba(
                    q_packed,
                    k_packed,
                    v_packed,
                    cu_seqlens=cu_seqlens_q,
                    max_seqlen=max_seqlen,
                    chunk_size=self.moba_chunk_size,
                    topk=self.moba_topk,
                ).squeeze(0)

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
