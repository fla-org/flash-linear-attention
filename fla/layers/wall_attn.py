# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Wall attention, contributed by Tilde Research (Timor Averbuch, Dhruv Pai).

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import GroupNorm
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_global_cumsum
from fla.ops.wall_attn import build_wall_kv_cache, parallel_wall_attn, parallel_wall_attn_decode

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)

# Chunk size for the pre-rescaled Wall decode cache (per-chunk log-space anchors).
WALL_DECODE_CHUNK = 64


class WallAttention(nn.Module):
    r"""Wall attention: full attention with a learned per-channel multiplicative decay.

    With ``P = cumsum(g)`` in :math:`\log_2` space, the logit for pair :math:`(i, j)` is
    :math:`\mathrm{scale}\sum_n q_{in} k_{jn}\,\exp_2(P_{in} - P_{jn})`. The per-channel
    log-decay ``g`` is produced by ``g_proj`` followed by ``logsigmoid`` (so ``g <= 0``
    and the prefix is monotone). An optional FoX-style per-head scalar gate can be added.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        use_output_gate: bool = False,
        use_scalar_gate: bool = False,
        gate_init_bias: float = 8.0,
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
        self.use_output_gate = use_output_gate
        self.use_scalar_gate = use_scalar_gate
        self.gate_init_bias = gate_init_bias
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        # Per-channel (per Q-head per key channel) log-decay gate.
        self.g_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # Tag so the model's `_init_weights` seeds the bias to `gate_init_bias`
        # (logsigmoid(bias) ~ 0 => near-vanilla attention at init) instead of zero.
        self.g_proj._is_wall_gate_proj = True
        self.g_proj._wall_gate_init_bias = gate_init_bias
        if use_scalar_gate:
            self.gs_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
            self.gs_proj._is_wall_gate_proj = True
            self.gs_proj._wall_gate_init_bias = gate_init_bias

        if use_output_gate:
            self.o_gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = GroupNorm(
                num_groups=self.num_heads,
                hidden_size=self.hidden_size,
                is_rms_norm=True,
            )
            self.k_norm = GroupNorm(
                num_groups=self.num_kv_heads,
                hidden_size=self.kv_dim,
                is_rms_norm=True,
            )

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

        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        # Cast to fp32 then logsigmoid for log-domain accumulation precision (g <= 0).
        g = F.logsigmoid(self.g_proj(hidden_states).float())
        gs = F.logsigmoid(self.gs_proj(hidden_states).float()) if self.use_scalar_gate else None
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
        g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)

        cu_seqlens = kwargs.get('cu_seqlens')

        if past_key_values is not None:
            assert cu_seqlens is None, "cu_seqlens should not be provided when past_key_values is not None"
            cached = (k, v, g, gs) if self.use_scalar_gate else (k, v, g)
            state = past_key_values.update(
                attn_state=cached,
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )
            cached = state['attn_state']
            if self.use_scalar_gate:
                k, v, g, gs = cached
            else:
                k, v, g = cached

        kv_len = k.shape[1]
        if past_key_values is not None and q_len < kv_len:
            # Incremental decode: queries are the suffix; attend against the full cache
            # via the pre-rescaled Wall decode kernel.
            o = self._decode(q, k, v, g, gs)
        elif attention_mask is not None:
            states = (k, v, g, gs) if self.use_scalar_gate else (k, v, g)
            q, states, indices_q, cu_seqlens, max_seq_lens = unpad_input(q, states, attention_mask, q_len, keepdim=True)
            if self.use_scalar_gate:
                k, v, g, gs = states
            else:
                k, v, g = states
            _, cu_seqlens_k = cu_seqlens
            cu_seqlens = cu_seqlens_k
            o = parallel_wall_attn(q, k, v, g, g_scalar=gs, window_size=self.window_size, cu_seqlens=cu_seqlens)
            o = pad_input(o.squeeze(0), indices_q, batch_size, q_len)
        else:
            o = parallel_wall_attn(q, k, v, g, g_scalar=gs, window_size=self.window_size, cu_seqlens=cu_seqlens)

        o = rearrange(o, '... h d -> ... (h d)')
        if self.use_output_gate:
            o = self.o_gate_proj(hidden_states).sigmoid() * o
        o = self.o_proj(o)
        return o, None, past_key_values

    def _decode(self, q, k, v, g, gs):
        scale = self.head_dim ** -0.5
        # P carries the base-2 conversion (RCP_LN2), matching the training forward.
        p = chunk_global_cumsum(g, scale=RCP_LN2)
        k_tilde, r_cache = build_wall_kv_cache(k, p, WALL_DECODE_CHUNK)
        p_curr = p[:, -q.shape[1]:].contiguous()
        gs_cumsum = chunk_global_cumsum(gs, scale=RCP_LN2) if gs is not None else None
        return parallel_wall_attn_decode(
            q, v, p_curr, k_tilde, r_cache,
            sink_bias=None,
            scale=scale,
            cache_chunk_size=WALL_DECODE_CHUNK,
            g_scalar_cumsum=gs_cumsum,
        )
