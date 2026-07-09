# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Portions of this file are adapted from Raven:
#   Copyright (c) 2023-2025 Arshia Afzal and Aviv Bick

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_layer_cache, get_unpad_data, index_first_axis, pad_input, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, RotaryEmbedding
from fla.modules.activations import ACT2FN
from fla.modules.feature_map import ReLUFeatureMap, SwishFeatureMap, T2RFeatureMap
from fla.modules.layernorm import rms_norm_linear
from fla.ops.gsa import chunk_gsa, fused_recurrent_gsa
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def _max_offset(seqlen_offset: int | torch.Tensor) -> int:
    if isinstance(seqlen_offset, int):
        return seqlen_offset
    return int(seqlen_offset.max().item())


class Raven(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 1.,
        expand_v: float = 1.,
        num_heads: int = 4,
        num_kv_heads: int | None = None,
        num_slots: int | None = None,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 8,
        feature_map: str = 'swish',
        use_output_gate: bool = False,
        gate_fn: str = 'swish',
        layer_idx: int | None = None,
        scale: float | None = 1.,
        decay_type: str = 'Mamba2',
        topk: int = 32,
        bias_rmm: bool = False,
        add_gumbel_noise: bool = True,
        router_score: str = 'sigmoid',
        router_type: str = 'lin',
        use_rope: bool = False,
        rope_theta: float = 10000.,
        max_position_embeddings: int | None = None,
        use_short_conv: bool = False,
        fuse_norm: bool = True,
        **kwargs,
    ) -> Raven:
        super().__init__()

        if mode not in ('chunk', 'fused_recurrent'):
            raise ValueError(f"Unsupported mode `{mode}`.")
        if decay_type not in ('Mamba2', 'GLA'):
            raise ValueError(f"Unsupported decay type `{decay_type}`.")
        if router_score not in ('sigmoid', 'softmax'):
            raise ValueError(f"Unsupported router score `{router_score}`.")
        if router_type not in ('lin', 'mlp'):
            raise ValueError(f"Unsupported router type `{router_type}`.")
        if use_short_conv:
            raise NotImplementedError("Raven does not use short convolutional memory.")
        if gate_fn not in ACT2FN:
            raise ValueError(f"Unsupported gate function `{gate_fn}`.")

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_heads < 1:
            raise ValueError(f"`num_heads` must be a positive integer, got {self.num_heads}.")
        if self.num_kv_heads < 1:
            raise ValueError(f"`num_kv_heads` must be a positive integer, got {self.num_kv_heads}.")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_kv_heads`, got {self.num_heads} and {self.num_kv_heads}.",
            )

        self.num_kv_groups = self.num_heads // self.num_kv_heads
        key_dim = hidden_size * expand_k
        value_dim = hidden_size * expand_v
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        if not math.isclose(key_dim, self.key_dim):
            raise ValueError(f"`hidden_size * expand_k` must be an integer, got {key_dim}.")
        if not math.isclose(value_dim, self.value_dim):
            raise ValueError(f"`hidden_size * expand_v` must be an integer, got {value_dim}.")
        if self.key_dim % self.num_heads != 0:
            raise ValueError(
                f"`hidden_size * expand_k` must be divisible by `num_heads`, "
                f"got key_dim={self.key_dim} and num_heads={self.num_heads}.",
            )
        if self.value_dim % self.num_heads != 0:
            raise ValueError(
                f"`hidden_size * expand_v` must be divisible by `num_heads`, "
                f"got value_dim={self.value_dim} and num_heads={self.num_heads}.",
            )
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        default_num_slots = num_slots is None
        if num_slots is None:
            num_slots = self.head_k_dim
        if default_num_slots and topk > num_slots:
            topk = num_slots
        if topk < 1 or topk > num_slots:
            raise ValueError(f"`topk` must be in [1, num_slots], got topk={topk}, num_slots={num_slots}.")

        self.num_slots = num_slots
        self.topk = topk
        self.decay_type = decay_type
        self.use_output_gate = use_output_gate
        self.use_rope = use_rope
        self.gate_logit_normalizer = gate_logit_normalizer
        self.gate_fn = ACT2FN[gate_fn]
        self.fuse_norm = fuse_norm
        self.scale = scale
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.bias_rmm = bias_rmm
        self.add_gumbel_noise = add_gumbel_noise
        self.router_score = router_score
        self.router_type = router_type
        self.layer_idx = layer_idx

        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class.",
            )

        if feature_map == 'swish':
            self.feature_map = SwishFeatureMap()
        elif feature_map == 'relu':
            self.feature_map = ReLUFeatureMap()
        elif feature_map == 't2r':
            self.feature_map = T2RFeatureMap(self.head_k_dim, self.head_k_dim)
        else:
            raise NotImplementedError(f"Feature map `{feature_map}` is not supported now.")

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim_per_group, bias=False)

        if self.decay_type == 'Mamba2':
            self.a_proj = nn.Linear(self.hidden_size, self.num_kv_heads, bias=False)
            A = torch.empty(self.num_kv_heads, dtype=torch.float32).uniform_(1e-4, 16)
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True

            dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
            dt = torch.exp(
                torch.rand(self.num_kv_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min),
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True
        else:
            self.f_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.num_slots, bias=False)

        if self.bias_rmm:
            self.r_bias = nn.Parameter(torch.zeros(self.num_kv_heads, self.num_slots, dtype=torch.float32))

        if self.router_type == 'lin':
            self.r_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.num_slots, bias=False)
        else:
            self.r_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.num_kv_heads * self.num_slots, bias=False),
            )

        if self.use_rope:
            self.rotary = RotaryEmbedding(dim=self.head_k_dim, base=self.rope_theta)

        norm_cls = RMSNorm if self.fuse_norm else nn.RMSNorm
        self.q_norm = norm_cls(self.head_k_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
        self.k_norm = norm_cls(self.head_k_dim, elementwise_affine=elementwise_affine, eps=norm_eps)

        if self.use_output_gate:
            self.o_gate_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            if self.fuse_norm and gate_fn in ('swish', 'silu', 'sigmoid'):
                self.o_norm = FusedRMSNormGated(
                    self.head_v_dim,
                    elementwise_affine=elementwise_affine,
                    eps=norm_eps,
                    activation=gate_fn,
                )
                self.fuse_norm_and_gate = True
            else:
                self.o_norm = norm_cls(self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
                self.fuse_norm_and_gate = False
        else:
            self.g_norm = norm_cls(self.value_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = get_layer_cache(self, past_key_values)
        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q_len + _max_offset(seqlen_offset)

        cu_seqlens = kwargs.get('cu_seqlens')
        indices = None
        if cu_seqlens is None and attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            # Shift RoPE positions by each sequence's left padding only when decoding on
            # top of cached tokens. During prefill the unpadded tokens are already packed
            # contiguously per `cu_seqlens`, so positions start at 0; adding `lens - mask_len`
            # there would push them negative.
            if _max_offset(seqlen_offset) > 0:
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q_len + _max_offset(seqlen_offset)

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_k_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
        router = rearrange(self.r_proj(hidden_states), '... (h m) -> ... h m', m=self.num_slots)

        if self.decay_type == 'Mamba2':
            f = (-self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)).unsqueeze(-1)
        else:
            f = rearrange(self.f_proj(hidden_states), '... (h m) -> ... h m', m=self.num_slots)
            f = F.logsigmoid(f) / self.gate_logit_normalizer

        q, k = self.feature_map(q), self.feature_map(k)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rope:
            if self.max_position_embeddings is not None:
                max_seqlen = max(max_seqlen, self.max_position_embeddings)
            q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        v = self.gate_fn(v)

        # Training-time Gumbel routing is RNG-sensitive. Activation checkpointing
        # must preserve RNG state, otherwise backward recomputes different routes.
        if self.add_gumbel_noise and self.training:
            router = router - torch.empty_like(router).exponential_().log()
        orig_scores = torch.sigmoid(router) if self.router_score == 'sigmoid' else torch.softmax(router, dim=-1)
        scores = orig_scores + self.r_bias.float() if self.bias_rmm else orig_scores

        route_idx = scores.topk(self.topk, dim=-1).indices
        topk_weights = torch.gather(orig_scores, dim=-1, index=route_idx)
        if self.router_score == 'sigmoid':
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        s_multihot = torch.zeros_like(router).scatter_(-1, route_idx, topk_weights.to(router.dtype))
        f = (f * s_multihot).to(q.dtype)
        s = (1 - f.exp()).to(q.dtype)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if self.num_kv_groups > 1:
            k, v, f, s = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_kv_groups), (k, v, f, s))

        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gsa(
                q=q,
                k=k,
                v=v,
                s=s,
                g=f,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                scale=self.scale,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gsa(
                q=q,
                k=k,
                v=v,
                s=s,
                g=f,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                scale=self.scale,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=None,
            offset=q_len,
        )

        if self.use_output_gate:
            gate_out = rearrange(self.o_gate_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.gate_fn(o)
            if self.fuse_norm_and_gate:
                o = self.o_norm(o, gate_out)
            else:
                o = self.o_norm(o) * self.gate_fn(gate_out)
            o = rearrange(o, '... h d -> ... (h d)')
            o = self.o_proj(o)
        else:
            o = rearrange(o, '... h d -> ... (h d)')
            o = self.gate_fn(o)
            if self.fuse_norm:
                o = rms_norm_linear(
                    o,
                    self.g_norm.weight,
                    self.g_norm.bias,
                    self.o_proj.weight,
                    self.o_proj.bias,
                    eps=self.g_norm.eps,
                )
            else:
                o = self.o_proj(self.g_norm(o))

        if indices is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
