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
import torch.nn.functional as F
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import get_layer_cache, get_unpad_data, index_first_axis, pad_input, unpad_input, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, RotaryEmbedding
from fla.modules.activations import swiglu
from fla.modules.rotary import rotary_embedding
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.attn.parallel import parallel_attn
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)


class YOCORotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        scale_base: float | None = None,
        interleaved: bool = False,
        pos_idx_in_fp32: bool = True,
        device: torch.device | None = None,
        rope_inv_freq: str = 'fla',
    ):
        self.rope_inv_freq = rope_inv_freq
        super().__init__(
            dim=dim,
            base=base,
            scale_base=scale_base,
            interleaved=interleaved,
            pos_idx_in_fp32=pos_idx_in_fp32,
            device=device,
        )

    def _compute_inv_freq(self, device=None):
        if self.rope_inv_freq == 'fla':
            return super()._compute_inv_freq(device=device)
        if self.rope_inv_freq == 'yoco':
            return 1.0 / (
                self.base ** torch.linspace(0, 1, self.dim // 2, device=device, dtype=torch.float32)
            )
        raise ValueError(f"Unsupported YOCO rope_inv_freq: {self.rope_inv_freq}")

    def forward_states(
        self,
        states: torch.Tensor,
        seqlen_offset: int | torch.Tensor = 0,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=states.device, dtype=states.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(states.shape[1] + seqlen_offset, device=states.device, dtype=states.dtype)
        return rotary_embedding(
            states,
            self._cos_cached,
            self._sin_cached,
            interleaved=self.interleaved,
            seqlen_offsets=seqlen_offset,
            cu_seqlens=cu_seqlens,
        )


class YOCOGatedRetention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 2048,
        num_heads: int = 32,
        rope_theta: float = 10000.,
        rope_inv_freq: str = 'fla',
        max_position_embeddings: int | None = None,
        gate_logit_normalizer: float = 16.,
        norm_eps: float = 1e-6,
        fuse_norm: bool = True,
        layer_idx: int = None,
    ):
        super().__init__()

        if mode not in {'chunk', 'fused_recurrent'}:
            raise ValueError(f"Unsupported GatedRetention mode: {mode}")
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        self.gate_logit_normalizer = gate_logit_normalizer
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.g_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gk_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        if fuse_norm:
            self.o_norm = FusedRMSNormGated(self.head_dim, elementwise_affine=False, eps=norm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.o_norm = RMSNorm(self.head_dim, elementwise_affine=False, eps=norm_eps, dtype=torch.float32)
            self.fuse_norm_and_gate = False

        self.rotary = YOCORotaryEmbedding(
            dim=self.head_dim,
            base=rope_theta,
            interleaved=True,
            rope_inv_freq=rope_inv_freq,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        del output_attentions

        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if past_key_values is not None and q_len == 1 else self.mode
        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get('cu_seqlens')
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, 'b s ... -> (b s) ...'), indices).unsqueeze(0)

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        g = self.g_proj(hidden_states)
        gk = F.logsigmoid(self.gk_proj(hidden_states)) / self.gate_logit_normalizer

        seqlen_offset, max_seqlen = 0, q.shape[1]
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None and seqlen_offset > 0:
                seqlen_offset = prepare_lens_from_mask(attention_mask) - q_len
                max_seqlen = q.shape[1] + seqlen_offset.max().item()

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        q = self.rotary.forward_states(q, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)
        k = self.rotary.forward_states(k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state = chunk_simple_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_simple_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Unsupported GatedRetention mode: {mode}")

        update_layer_cache(self, past_key_values, recurrent_state=recurrent_state, offset=q_len)

        if self.fuse_norm_and_gate:
            o = self.o_norm(o, rearrange(g, '... (h d) -> ... h d', d=self.head_dim))
            o = rearrange(o, '... h d -> ... (h d)')
        else:
            o = rearrange(self.o_norm(o), '... h d -> ... (h d)')
            o = swiglu(g, o)

        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        attentions = None
        return o, attentions, past_key_values


class YOCOSharedKVBuilder(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        rope_theta: float | None = 10000.,
        rope_inv_freq: str = 'fla',
        max_position_embeddings: int | None = None,
        norm_eps: float = 1e-6,
        fuse_norm: bool = True,
        layer_idx: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = self.num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.kv_norm = (RMSNorm if fuse_norm else nn.RMSNorm)(self.hidden_size, eps=norm_eps)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        # Official YOCO applies interleaved rotary embeddings in the cross-decoder KV path.
        self.rotary = YOCORotaryEmbedding(
            dim=self.head_dim,
            base=self.rope_theta,
            interleaved=True,
            rope_inv_freq=rope_inv_freq,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, Cache | None]:
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        hidden_states = self.kv_norm(hidden_states)
        _, q_len, _ = hidden_states.size()
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        cu_seqlens = kwargs.get('cu_seqlens')

        seqlen_offset, max_seqlen = 0, q_len
        if use_cache and past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = k.shape[1] + seqlen_offset

        if attention_mask is not None and (past_key_values is None or use_cache):
            # Left-padded prefills should use the same effective RoPE positions as the unpadded sequence.
            seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
            max_seqlen = k.shape[1] + seqlen_offset.max().item()

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        k = self.rotary.forward_states(k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if use_cache and past_key_values is not None:
            cache_state = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
            )
            attn_state = cache_state.get('attn_state') if cache_state is not None else None
            if attn_state is not None:
                k_cached, v_cached = attn_state
                k = rearrange(k_cached, 'b s (h d) -> b s h d', h=self.num_kv_heads, d=self.head_dim).contiguous()
                v = rearrange(v_cached, 'b s (h d) -> b s h d', h=self.num_kv_heads, d=self.head_dim).contiguous()

        return k, v, past_key_values


class YOCOCrossAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        rope_inv_freq: str = 'fla',
        max_position_embeddings: int | None = None,
        layer_idx: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = self.num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            # YOCO only applies the extra per-head norm on queries here.
            # The cross-decoder KV path already normalizes its source hidden states
            # once in `YOCOSharedKVBuilder.kv_norm` before projecting shared K/V.
            self.q_norm = RMSNorm(self.head_dim)

        # Official YOCO applies interleaved rotary embeddings in cross attention.
        self.rotary = YOCORotaryEmbedding(
            dim=self.head_dim,
            base=self.rope_theta,
            interleaved=True,
            rope_inv_freq=rope_inv_freq,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        shared_k: torch.Tensor,
        shared_v: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()
        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)

        cu_seqlens = kwargs.get('cu_seqlens')
        rotary_cu_seqlens = cu_seqlens
        seqlen_offset = shared_k.shape[1] - q_len
        max_seqlen = shared_k.shape[1]
        if attention_mask is not None:
            seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
            max_seqlen = max(max_seqlen, q.shape[1] + max(seqlen_offset))
        elif cu_seqlens is not None and q.shape[0] == 1 and q.shape[1] == cu_seqlens.numel() - 1:
            # Packed one-step decoding keeps one query token per sequence, while
            # shared_k/shared_v remain concatenated full-prefix KV states. RoPE
            # therefore needs a per-sequence offset and a query-local cu_seqlens.
            seqlen_offset = cu_seqlens[1:] - cu_seqlens[:-1] - 1
            rotary_cu_seqlens = torch.arange(0, q.shape[1] + 1, dtype=torch.int32, device=q.device)
        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        q = self.rotary.forward_states(
            q,
            seqlen_offset=seqlen_offset,
            max_seqlen=max_seqlen,
            cu_seqlens=rotary_cu_seqlens,
        )
        should_transform_output = False
        pad_indices = None

        if attention_mask is not None:
            q, (shared_k, shared_v), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                q,
                (shared_k, shared_v),
                attention_mask,
                q_len,
                keepdim=True,
            )
            _, cu_seqlens = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            is_decoding = max_seqlen_q != max_seqlen_k
            pad_indices = indices_q
        else:
            is_decoding = q.shape[1] != shared_k.shape[1]

        if is_decoding:
            if attention_mask is not None:
                assert max_seqlen_q == 1, "only support q_len == 1 for decoding"
            elif cu_seqlens is not None:
                assert q.shape[1] == cu_seqlens.numel() - 1, "expected one query per sequence for varlen decoding"
            else:
                assert q.shape[1] == 1, "only support q_len == 1 for decoding"
                cu_seqlens = torch.arange(
                    0,
                    (batch_size + 1) * shared_k.shape[1],
                    shared_k.shape[1],
                    dtype=torch.int32,
                    device=q.device,
                )
                q = rearrange(q, 'b t h d -> t b h d').contiguous()
                shared_k = rearrange(shared_k, 'b t h d -> 1 (b t) h d').contiguous()
                shared_v = rearrange(shared_v, 'b t h d -> 1 (b t) h d').contiguous()
                should_transform_output = True

        if is_decoding:
            o = attn_decoding_one_step(q, shared_k, shared_v, cu_seqlens=cu_seqlens)
        else:
            o = parallel_attn(q, shared_k, shared_v, window_size=self.window_size, cu_seqlens=cu_seqlens)
        if should_transform_output:
            o = rearrange(o, 't b h d -> b t h d')
        if pad_indices is not None:
            o = pad_input(o.squeeze(0), pad_indices, batch_size, q_len)
        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)
        attentions = None
        return o, attentions
