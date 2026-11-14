# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from fla.layers.utils import pad_input, unpad_input
from fla.modules import FusedRMSNormGated, RMSNorm, RotaryEmbedding
from fla.ops.phba import chunk_phba
from fla.ops.phba.backends.cublaslt import _Linear, exact_backend
from fla.ops.phba.plans import _resolve_length_plan


class PHBALinear(nn.Linear):
    """PHBA projection with an optional byte-validated H20 weight-gradient path."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eligible = (
            not torch.compiler.is_compiling()
            and type(x) in (torch.Tensor, nn.Parameter)
            and type(self.weight) in (torch.Tensor, nn.Parameter)
            and self.bias is None
            and self.training
            and torch.is_grad_enabled()
            and x.is_cuda
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16
            and x.is_contiguous()
            and self.weight.is_contiguous()
            and x.shape[-1] == 2048
            and x.numel() == 8192 * 2048
            and self.out_features in (2048, 4272)
        )
        backend = exact_backend(x.device) if eligible else None
        if backend is None:
            return F.linear(x, self.weight, self.bias)
        return _Linear.apply(backend.autocast.cached_bf16(x), backend.autocast.cached_bf16(self.weight))


class PHBA(nn.Module):
    """Causal block attention augmented by gated prefix states."""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        max_position_embeddings: int | None = None,
        rope_theta: float = 10000.0,
        layer_idx: int | None = None,
        moba_chunk_size: int = 128,
        moba_topk: int = 8,
        use_output_gate: bool = True,
        use_phba: bool = True,
        gsa_num_slots: int = 128,
        gate_logit_normalizer: float = 8.0,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        if not use_phba:
            raise ValueError("PHBA requires use_phba=True")
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads or num_kv_heads not in (None, num_heads):
            raise ValueError("PHBA requires equal query/key/value head counts dividing hidden_size")
        if window_size is not None:
            raise ValueError("PHBA uses causal blocks rather than an additional sliding window")
        if not 16 <= hidden_size // num_heads <= 128 or (hidden_size // num_heads) % 8:
            raise ValueError("PHBA head dimensions must be multiples of 8 from 16 through 128")
        if type(elementwise_affine) is not bool:
            raise ValueError("PHBA elementwise_affine must be bool")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx
        self.max_position_embeddings = max_position_embeddings
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk
        self.gsa_num_slots = gsa_num_slots
        self.gate_logit_normalizer = gate_logit_normalizer
        self.use_output_gate = use_output_gate
        self.q_proj = PHBALinear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = PHBALinear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = PHBALinear(hidden_size, hidden_size, bias=qkv_bias)
        self.o_proj = PHBALinear(hidden_size, hidden_size, bias=False)
        self.gsa_gate_proj = PHBALinear(hidden_size, num_heads * gsa_num_slots, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=elementwise_affine) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=elementwise_affine) if qk_norm else nn.Identity()
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=rope_theta)
        if use_output_gate:
            self.g_proj = nn.Sequential(
                PHBALinear(hidden_size, self.head_dim, bias=False),
                PHBALinear(self.head_dim, hidden_size, bias=False),
            )
            self.o_norm = FusedRMSNormGated(
                self.head_dim,
                activation="sigmoid",
                eps=1e-06,
                elementwise_affine=elementwise_affine,
            )
        else:
            self.o_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        if use_cache or past_key_values is not None:
            raise ValueError("The deterministic PHBA path supports cache-free training and prefill; set use_cache=False")
        if output_attentions:
            raise ValueError("PHBA does not materialize attention weights")
        batch, tokens, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")
        plan = kwargs.get("phba_length_plan")
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != (batch, tokens):
                raise ValueError("PHBA attention_mask must be a [B,T] padding mask")
            if cu_seqlens is not None:
                raise ValueError("PHBA padding masks and packed cu_seqlens are mutually exclusive")
        else:
            plan = _resolve_length_plan(batch, tokens, hidden_states.device, cu_seqlens, plan)
            if (
                cu_seqlens is not None
                and not plan._matches_source(cu_seqlens)
                or cu_seqlens is None
                and batch == 1
                and len(plan.lengths) > 1
            ):
                cu_seqlens = plan.cu_seqlens
        q = rearrange(self.q_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q_rope, k_rope = self.rotary(
            q=q,
            k=k,
            seqlen_offset=0,
            max_seqlen=max(tokens, self.max_position_embeddings or tokens),
            cu_seqlens=cu_seqlens,
        )
        indices = None
        if attention_mask is not None:
            q_rope, other, indices, cumulative, _ = unpad_input(
                q=q_rope,
                states=(k_rope, v, q, k, hidden_states),
                attention_mask=attention_mask,
                q_len=tokens,
            )
            k_rope, v, q, k, flat_hidden = other
            cu_seqlens = cumulative[0]
            plan = _resolve_length_plan(1, q.shape[0], q.device, cu_seqlens, plan)
        else:
            q_rope = rearrange(q_rope, "b t ... -> (b t) ...")
            k_rope = rearrange(k_rope, "b t ... -> (b t) ...")
            v = rearrange(v, "b t ... -> (b t) ...")
            q = rearrange(q, "b t ... -> (b t) ...")
            k = rearrange(k, "b t ... -> (b t) ...")
            flat_hidden = rearrange(hidden_states, "b t d -> (b t) d")
        gate = rearrange(self.gsa_gate_proj(flat_hidden), "t (h m) -> t h m", h=self.num_heads, m=self.gsa_num_slots)
        output, _ = chunk_phba(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            q_rope=q_rope.unsqueeze(0),
            k_rope=k_rope.unsqueeze(0),
            gate=gate.unsqueeze(0),
            chunk_size=self.moba_chunk_size,
            topk=self.moba_topk,
            gate_logit_normalizer=self.gate_logit_normalizer,
            length_plan=plan,
        )
        if indices is not None:
            output = pad_input(output.squeeze(0), indices, batch, tokens)
        else:
            output = output.reshape(batch, tokens, self.num_heads, self.head_dim)
        if self.use_output_gate:
            gate = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim)
            output = self.o_norm(output, gate)
        else:
            output = self.o_norm(output)
        return self.o_proj(rearrange(output, "... h d -> ... (h d)")), None, None


PhbaAttention = PHBA
