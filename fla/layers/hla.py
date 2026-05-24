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
from einops import rearrange, repeat

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import RMSNorm
from fla.ops.hla import recurrent_hla

if TYPE_CHECKING:
    from fla.models.utils import Cache


class HigherOrderLinearAttention(nn.Module):
    """
    Second-order Higher-order Linear Attention (HLA).

    This is a reference FLA layer for the masked second-order operator from
    "Higher-order Linear Attention" (arXiv:2510.27258). It exposes a standard
    FLA recurrent-layer interface and uses the exact streaming identity with
    state tuple ``(S, C, m, G, h)``. The implementation is intentionally kept as
    a PyTorch baseline; chunk/Triton kernels can replace ``recurrent_hla`` while
    preserving this API.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        expand_v: float = 1.0,
        head_dim: int = 64,
        num_heads: int = 8,
        num_v_heads: int | None = None,
        mode: str = "recurrent",
        normalize: bool = False,
        eps: float = 1e-6,
        ridge: float = 0.0,
        output_norm: str = "rmsnorm",
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        layer_idx: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if mode not in {"recurrent", "fused_recurrent"}:
            raise ValueError(f"Unsupported mode `{mode}`.")
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.num_heads = num_heads
        self.num_v_heads = num_heads if num_v_heads is None else num_v_heads
        self.mode = mode
        self.normalize = normalize
        self.eps = eps
        self.ridge = ridge
        self.layer_idx = layer_idx

        if hidden_size <= 0:
            raise ValueError("`hidden_size` must be positive.")
        if head_dim <= 0:
            raise ValueError("`head_dim` must be positive.")
        if num_heads <= 0 or self.num_v_heads <= 0:
            raise ValueError("`num_heads` and `num_v_heads` must be positive.")
        if self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"`num_v_heads` must be divisible by `num_heads`, got {self.num_v_heads} and {self.num_heads}."
            )
        if self.head_v_dim <= 0:
            raise ValueError("`head_dim * expand_v` must be positive.")

        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if output_norm == "rmsnorm":
            self.norm = RMSNorm(
                hidden_size=self.head_v_dim,
                elementwise_affine=elementwise_affine,
                eps=norm_eps,
                dtype=torch.float32,
            )
        elif output_norm == "identity":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported output_norm `{output_norm}`.")
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("HigherOrderLinearAttention expects a 2D padding attention_mask.")

        q = rearrange(self.q_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
        if attention_mask is not None:
            v = v.mul(attention_mask[:, -v.shape[1]:, None, None])

        if self.num_v_heads != self.num_heads:
            groups = self.num_v_heads // self.num_heads
            q = repeat(q, "b t h d -> b t (h g) d", g=groups)
            k = repeat(k, "b t h d -> b t (h g) d", g=groups)

        last_state = get_layer_cache(self, past_key_values)
        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        o, recurrent_state = recurrent_hla(
            q=q,
            k=k,
            v=v,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            normalize=self.normalize,
            eps=self.eps,
            ridge=self.ridge,
        )
        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            offset=q.shape[1],
        )
        o = self.norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o, None, past_key_values
