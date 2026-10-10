# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.gated_oja_rule2 import chunk_gated_oja_rule2, fused_recurrent_gated_oja_rule2
from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class GatedSlotAttention2(nn.Module):
    r"""
    Gated Slot Attention 2 (GSA2) layer.

    GSA2 stacks two gated delta-rule memories. The first one is an Oja2 memory `S1` of shape `[K, M]` that reads the
    context through `M` slots addressed by a learned slot code `w_t`, the second one is a GDN-2 memory `S2` of shape
    `[D, V]` that is queried with the activated slot readout:

        S1_t = S1_{t-1} Diag(exp(g1_t)) (I - (b1_t * w_t) w_t^T) + (c1_t * k_t) w_t^T
        z_t  = act(S1_t^T q_t)
        S2_t = (I - d_t (b2_t * d_t)^T) Diag(exp(g2_t)) S2_{t-1} + d_t (c2_t * v_t)^T
        o_t  = S2_t^T z_t

    where `*` is the elementwise product, `d_t` is the slot code seen from the second level, and `act` is `silu` or
    `softmax` over the slot axis. `level2_axis` selects the key axis of `S2`: `head` projects the slot readout back to
    `head_dim`, so `D = head_dim`, while `slot` keeps it as is, so `D = num_slots`. The two coincide when
    `num_slots == head_dim`.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
        num_heads (int, Optional):
            The number of QK heads. Default: 16.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA (Grouped Value Attention) is applied if `num_v_heads` > `num_heads`, where the second level is
            broadcast to the value-head dimension. Default: `None`.
        num_slots (int, Optional):
            The number of slots `M`, one of 64, 128 and 256. Equal to `head_dim` if `None`. Default: `None`.
        level2_axis (str, Optional):
            The key axis of the second-level memory, `head` or `slot`. Default: `head`.
        w_rank (int, Optional):
            The rank of the low-rank slot code projection, equal to `head_dim` if `None`. Default: `None`.
        use_slot_proj (bool, Optional):
            Whether to project the slot readout back to `head_dim` when `level2_axis="head"` and
            `num_slots != head_dim`. Default: `True`.
        w_scale (float, Optional):
            The scale applied to the slot code before normalization. Default: 1.0.
        w_clip (float, Optional):
            The absolute clipping value of the normalized slot code. Default: `None`.
        use_w_l2norm (bool, Optional):
            Whether to L2-normalize the slot code. The slot code is then kept in fp32. Default: `True`.
        w_l2norm_eps (float, Optional):
            The epsilon added to the norm of the slot code. Default: 1e-6.
        mode (str, Optional):
            Which kernel to use, `chunk` or `fused_recurrent`.
            The layer falls back to `fused_recurrent` for short inference sequences (`q_len <= 64`) when
            `num_slots <= 128`. Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions on q, k and v. Default: `True`.
        use_w_conv (bool, Optional):
            Whether to use a short convolution on the slot code. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolutions. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolutions. Default: `False`.
        oja_out_act (str, Optional):
            The activation between the two levels, `silu`, `softmax` or `none`. Default: `silu`.
        oja_scale (float, Optional):
            The scale of the first level, `1 / sqrt(head_dim)` if `None`. Default: `None`.
        gdn_scale (float, Optional):
            The scale of the second level, `1 / sqrt(D)` if `None`. Default: `None`.
        use_oja_q_l2norm (bool, Optional):
            Whether to L2-normalize q in the first level. Default: `True`.
        use_oja_k_l2norm (bool, Optional):
            Whether to L2-normalize k in the first level. Default: `True`.
        use_gdn_qk_l2norm_in_kernel (bool, Optional):
            Whether to L2-normalize q and k in the second level. Default: `True`.
        oja_lower_bound (float, Optional):
            The lower bound of the first-level log-decay, which is `-oja_lower_bound * sigmoid(...)`. Default: 5.0.
        layer_idx (int, Optional):
            The index of the layer. Default: `None`.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        num_slots: int | None = None,
        level2_axis: str = "head",
        w_rank: int | None = None,
        use_slot_proj: bool = True,
        w_scale: float = 1.0,
        w_clip: float | None = None,
        use_w_l2norm: bool = True,
        w_l2norm_eps: float = 1e-6,
        mode: str = "chunk",
        use_short_conv: bool = True,
        use_w_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        oja_out_act: str = "silu",
        oja_scale: float | None = None,
        gdn_scale: float | None = None,
        use_oja_q_l2norm: bool = True,
        use_oja_k_l2norm: bool = True,
        use_gdn_qk_l2norm_in_kernel: bool = True,
        oja_lower_bound: float = 5.0,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.use_w_conv = use_w_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.layer_idx = layer_idx
        self.level2_axis = level2_axis
        self.use_slot_proj = use_slot_proj
        self.w_scale = w_scale
        self.w_clip = w_clip
        self.use_w_l2norm = use_w_l2norm
        self.w_l2norm_eps = w_l2norm_eps

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)

        if num_slots is None:
            num_slots = self.head_k_dim
        self.num_slots = num_slots
        self.level2_dim = self.head_k_dim if self.level2_axis == "head" else self.num_slots

        if w_rank is None:
            w_rank = self.head_k_dim
        self.w_rank = w_rank

        self.oja_out_act = oja_out_act
        self.oja_scale = oja_scale
        self.gdn_scale = gdn_scale
        self.use_oja_q_l2norm = use_oja_q_l2norm
        self.use_oja_k_l2norm = use_oja_k_l2norm
        self.use_gdn_qk_l2norm_in_kernel = use_gdn_qk_l2norm_in_kernel
        self.oja_lower_bound = oja_lower_bound

        if self.level2_axis not in ("head", "slot"):
            raise ValueError(f"level2_axis must be 'head' or 'slot', got {self.level2_axis!r}.")
        if self.oja_out_act not in ("silu", "swish", "softmax", "none", "identity"):
            raise ValueError(f"Unsupported oja_out_act `{self.oja_out_act}`.")
        if self.num_slots not in (64, 128, 256):
            raise ValueError(f"num_slots must be 64, 128, or 256, got {self.num_slots}.")
        if not math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}.")
        if self.num_v_heads < self.num_heads:
            raise ValueError(f"num_v_heads={self.num_v_heads} must be >= num_heads={self.num_heads}.")
        if self.num_v_heads % self.num_heads != 0:
            raise ValueError(f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.")
        if self.level2_axis == "head" and self.num_slots != self.head_k_dim and not self.use_slot_proj:
            raise ValueError("head-axis level 2 requires slot projections when num_slots differs from head_dim.")
        assert mode in ("chunk", "fused_recurrent"), f"Unsupported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        # slot code, shared by both levels
        self.w_down = nn.Linear(hidden_size, self.w_rank, bias=False)
        self.w_up = nn.Linear(self.w_rank, self.num_heads * self.num_slots, bias=False)
        if use_w_conv:
            self.w_conv1d = ShortConvolution(
                hidden_size=self.num_heads * self.num_slots,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        # slot -> head_dim projections that bridge the two levels on the head axis
        if self.level2_axis == "head" and self.num_slots != self.head_k_dim and self.use_slot_proj:
            self.gdn_q_proj = nn.Linear(self.num_slots, self.head_k_dim, bias=False)
            self.gdn_k_proj = nn.Linear(self.num_slots, self.head_k_dim, bias=False)
        else:
            self.gdn_q_proj = None
            self.gdn_k_proj = None

        # level 1: per-slot decay, per-slot erase gate and per-channel write gate
        self.oja_gv_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_dim, bias=False),
            nn.Linear(self.head_dim, self.num_heads * self.num_slots, bias=False),
        )
        self.oja_b_proj = nn.Linear(hidden_size, self.num_heads * self.num_slots, bias=False)
        self.oja_c_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.A1_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A1_log._no_weight_decay = True
        self.dt1_bias = nn.Parameter(torch.zeros(self.num_heads * self.num_slots, dtype=torch.float32))
        self.dt1_bias._no_weight_decay = True

        # level 2: channel-wise decay, erase gate on the key axis and write gate on the value axis
        self.gdn_g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.num_v_heads * self.level2_dim, bias=False),
        )
        self.gdn_b_proj = nn.Linear(hidden_size, self.num_v_heads * self.level2_dim, bias=False)
        self.gdn_w_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.A2_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A2_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.num_v_heads * self.level2_dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt2_bias = nn.Parameter(inv_dt)
        self.dt2_bias._no_weight_decay = True

        # output path: sigmoid-gated RMSNorm + projection
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _compute_oja_decay(self, x: torch.Tensor) -> torch.Tensor:
        # bounded log-decay in [-oja_lower_bound, 0], evaluated in fp32 regardless of the autocast context
        A1 = self.A1_log.float().exp()
        dt1_bias = self.dt1_bias.float().view(self.num_heads, self.num_slots)
        return -self.oja_lower_bound * torch.sigmoid(A1.unsqueeze(-1) * (x.float() + dt1_bias))

    def _apply_oja_out_act(self, o: torch.Tensor) -> torch.Tensor:
        if self.oja_out_act in ("silu", "swish"):
            return F.silu(o)
        if self.oja_out_act == "softmax":
            return F.softmax(o.float(), dim=-1).to(dtype=o.dtype)
        return o

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
        # the fused recurrent Oja2 kernel keeps the whole [K, M] state in registers, which only fits M <= 128
        if torch.is_grad_enabled():
            mode = "chunk"
        elif q_len <= 64 and not self.training and self.num_slots <= 128:
            mode = "fused_recurrent"
        else:
            mode = self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get("cu_seqlens")
        hidden_states, indices, cu_seqlens = unpad_hidden_states(hidden_states, cu_seqlens, attention_mask, q_len)

        conv_state_q, conv_state_k, conv_state_v, conv_state_w = None, None, None, None
        if last_state is not None and last_state["conv_state"] is not None:
            conv_state_q, conv_state_k, conv_state_v, conv_state_w = last_state["conv_state"]

        if self.use_short_conv:
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        w = self.w_up(self.w_down(hidden_states))
        if self.use_w_conv:
            w, conv_state_w = self.w_conv1d(x=w, cache=conv_state_w, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        if self.w_scale != 1.0:
            w = w * self.w_scale

        gv = self.oja_gv_proj(hidden_states)
        b1 = self.oja_b_proj(hidden_states)
        c1 = self.oja_c_proj(hidden_states)
        g2 = self.gdn_g_proj(hidden_states)
        b2 = self.gdn_b_proj(hidden_states)
        w2 = self.gdn_w_proj(hidden_states)

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        v, w2 = (rearrange(x, "... (h d) -> ... h d", d=self.head_v_dim) for x in (v, w2))
        w, gv, b1 = (rearrange(x, "... (h d) -> ... h d", d=self.num_slots) for x in (w, gv, b1))
        c1 = rearrange(c1, "... (h d) -> ... h d", d=self.head_k_dim)
        g2, b2 = (rearrange(x, "... (h d) -> ... h d", d=self.level2_dim) for x in (g2, b2))

        # the slot code stays in fp32 so that its normalization and the first-level state are not rounded
        if self.use_w_l2norm:
            w = w.float()
            w = w / (w.norm(p=2, dim=-1, keepdim=True) + self.w_l2norm_eps)
        if self.w_clip is not None:
            w = w.clamp(min=-self.w_clip, max=self.w_clip)

        gv = self._compute_oja_decay(gv)

        state_oja, state_gdn = None, None
        if last_state is not None and last_state["recurrent_state"] is not None:
            state_oja, state_gdn = last_state["recurrent_state"]

        if mode == "chunk":
            o, state_oja = chunk_gated_oja_rule2(
                q=q,
                k=k,
                v=w,
                gv=gv,
                b=b1.sigmoid(),
                c=c1.sigmoid(),
                scale=self.oja_scale,
                initial_state=state_oja,
                output_final_state=use_cache,
                use_q_l2norm=self.use_oja_q_l2norm,
                use_k_l2norm=self.use_oja_k_l2norm,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=kwargs.get("cu_seqlens_cpu"),
            )
        else:
            o, state_oja = fused_recurrent_gated_oja_rule2(
                q=q,
                k=k,
                v=w,
                gv=gv,
                b=b1.sigmoid(),
                c=c1.sigmoid(),
                scale=self.oja_scale,
                initial_state=state_oja,
                output_final_state=use_cache,
                use_q_l2norm=self.use_oja_q_l2norm,
                use_k_l2norm=self.use_oja_k_l2norm,
                cu_seqlens=cu_seqlens,
            )

        # the slot readout becomes the query of the second level, the slot code its key
        q2 = self._apply_oja_out_act(o)
        k2 = w
        if self.gdn_q_proj is not None:
            q2 = self.gdn_q_proj(q2)
            k2 = self.gdn_k_proj(k2.to(dtype=self.gdn_k_proj.weight.dtype))
        k2 = k2.to(dtype=q2.dtype)

        if self.num_v_heads > self.num_heads:
            q2, k2 = (
                repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads)
                for x in (q2, k2)
            )

        if mode == "chunk":
            o, state_gdn = chunk_gdn2(
                q=q2,
                k=k2,
                v=v,
                g=g2,
                b=b2.sigmoid(),
                w=w2.sigmoid(),
                scale=self.gdn_scale,
                initial_state=state_gdn,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=self.use_gdn_qk_l2norm_in_kernel,
                use_gate_in_kernel=True,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=kwargs.get("cu_seqlens_cpu"),
                A_log=self.A2_log,
                dt_bias=self.dt2_bias,
            )
        else:
            o, state_gdn = fused_recurrent_gdn2(
                q=q2,
                k=k2,
                v=v,
                g=g2,
                b=b2.sigmoid(),
                w=w2.sigmoid(),
                scale=self.gdn_scale,
                initial_state=state_gdn,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=self.use_gdn_qk_l2norm_in_kernel,
                use_gate_in_kernel=True,
                cu_seqlens=cu_seqlens,
                A_log=self.A2_log,
                dt_bias=self.dt2_bias,
            )

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=(state_oja, state_gdn) if use_cache else None,
            conv_state=(conv_state_q, conv_state_k, conv_state_v, conv_state_w) if use_cache else None,
            offset=q_len,
        )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        o = repad_hidden_states(o, indices, batch_size, q_len)

        return o, None, past_key_values
