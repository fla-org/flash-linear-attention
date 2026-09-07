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
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.utils.graph import (
    host_chunk_statistics,
    is_graph_capable_device,
    normalize_graph_mode,
    route_graph_execution,
    static_chunk_capacity,
)
from fla.ops.utils.index import prepare_chunk_indices_static
from fla.utils import IS_NPU, IS_NVIDIA

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class KimiDeltaAttention(nn.Module):
    """
    Kimi Delta Attention (KDA) layer implementation.

    Reference: `Kimi Linear: An Expressive, Efficient Attention Architecture <https://arxiv.org/abs/2510.26692>`_

    KDA extends the delta rule with per-key-dim gating: the forget gate ``g`` has shape ``[B, T, H, K]``
    (vector per head), compared to GDN's scalar per-head gate ``[B, T, H]``.
    The gate is computed as ``g = -exp(A_log) * softplus(f_proj(x) + dt_bias)`` where
    ``A_log`` has shape ``[H]`` and ``dt_bias`` has shape ``[H * K]``.

    Each layer contains around ``hidden_size * key_dim * 2 + hidden_size * value_dim * 3`` parameters
    (q/k projections + v/g/o projections), plus the low-rank f_proj bottleneck.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA (Grouped Value Attention) is applied if `num_v_heads` > `num_heads`,
            where `num_v_heads` must be divisible by `num_heads`. Default: `None`.
        mode (str, Optional):
            Which KDA kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference:
            `Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues <https://arxiv.org/abs/2411.12537>`_
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        safe_gate (bool, Optional):
            Whether the kernel can assume the gate values (in log space) are in ``[lower_bound, 0)``
            and use M=16 TensorCore acceleration. Requires ``lower_bound`` to be set.
            See :func:`chunk_kda` for details. Default: ``False``.
        lower_bound (float, Optional):
            Lower bound for the forget gate in log space. Clamps the gate output to ``[lower_bound, 0)``.
            With ``-5``, the per-step decay ``exp(g) ≈ 0.0067`` at minimum — negligible impact on quality.
            See :func:`chunk_kda` for details. Default: ``None``.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> KimiDeltaAttention:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.safe_gate = safe_gate
        self.lower_bound = lower_bound
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

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

        # Gate dim = HV * K: per value-head, per key-dim gating.
        self.gate_dim = int(self.num_v_heads * self.head_k_dim)
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        # A_log and dt_bias are per value-head for native GVA support.
        if safe_gate:
            self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        else:
            self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.gate_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        graph_mode = kwargs.pop("graph_mode", None)
        use_graph = kwargs.pop("use_graph", False)
        graph_t_max = kwargs.pop("graph_t_max", None)
        graph_n_max = kwargs.pop("graph_n_max", None)
        graph_nt_max = kwargs.pop("graph_nt_max", None)
        graph_actual_tokens = kwargs.pop("graph_actual_tokens", None)
        graph_actual_sequences = kwargs.pop("graph_actual_sequences", None)
        graph_actual_nt = kwargs.pop("graph_actual_nt", None)
        min_graph_utilization = kwargs.pop("min_graph_utilization", 0.75)
        graph_chunk_size = kwargs.pop("chunk_size", 64)
        graph_chunk_indices = kwargs.pop("chunk_indices", None)
        graph_chunk_offsets = kwargs.pop("chunk_offsets", None)
        normalized_graph_mode = normalize_graph_mode(graph_mode, use_graph)

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")
        cu_seqlens_cpu = kwargs.get("cu_seqlens_cpu")
        last_state = get_layer_cache(self, past_key_values)

        graph_requested = normalized_graph_mode != "eager"
        graph_force = normalized_graph_mode == "force_graph"
        graph_selected = False
        if graph_requested and IS_NPU:
            if graph_force:
                raise NotImplementedError("Layer graph mode is not implemented for the Ascend backend yet.")
            graph_requested = False
        if graph_requested:
            if not (IS_NVIDIA or IS_NPU) and graph_force:
                raise RuntimeError("Graph mode requires the CUDA or Ascend NPU backend.")
            if cu_seqlens is None and attention_mask is not None:
                if graph_force:
                    raise ValueError("Graph mode requires prepacked inputs when using an attention mask.")
            elif batch_size != 1:
                if graph_force:
                    raise ValueError("CUDA Graph mode requires a flattened batch with batch size 1.")
            elif graph_t_max is None:
                graph_t_max = q_len

            if batch_size == 1 and graph_t_max is not None and (cu_seqlens is not None or attention_mask is None):
                input_sequences = 1 if cu_seqlens is None else len(cu_seqlens) - 1
                if graph_n_max is None:
                    graph_n_max = input_sequences
                if graph_nt_max is None:
                    graph_nt_max = static_chunk_capacity(graph_t_max, graph_n_max, graph_chunk_size)

                actual_tokens, actual_sequences, actual_nt = (
                    graph_actual_tokens,
                    graph_actual_sequences,
                    graph_actual_nt,
                )
                if cu_seqlens is None:
                    actual_tokens, actual_sequences, actual_nt = q_len, 1, (q_len + graph_chunk_size - 1) // graph_chunk_size
                elif cu_seqlens_cpu is not None and any(value is None for value in (actual_tokens, actual_sequences, actual_nt)):
                    actual_tokens, actual_sequences, actual_nt = host_chunk_statistics(cu_seqlens_cpu, graph_chunk_size)
                elif cu_seqlens.device.type == 'cpu' and any(
                    value is None for value in (actual_tokens, actual_sequences, actual_nt)
                ):
                    actual_tokens, actual_sequences, actual_nt = host_chunk_statistics(cu_seqlens, graph_chunk_size)

                decision = route_graph_execution(
                    normalized_graph_mode,
                    actual_tokens=actual_tokens,
                    actual_sequences=actual_sequences,
                    actual_nt=actual_nt,
                    t_max=graph_t_max,
                    n_max=graph_n_max,
                    nt_max=graph_nt_max,
                    min_graph_utilization=min_graph_utilization,
                    chunk_size=graph_chunk_size,
                    input_tokens=q_len,
                    input_sequences=input_sequences,
                )
                graph_selected = decision.selected_path == "graph"
                if graph_selected:
                    if not is_graph_capable_device(hidden_states.device if cu_seqlens is None else cu_seqlens.device):
                        if graph_force:
                            raise ValueError("Graph mode requires device-resident `cu_seqlens`.")
                        graph_selected = False
                    if q_len != graph_t_max:
                        if graph_force:
                            raise ValueError(f"graph input shape must use graph_t_max={graph_t_max}, got q_len={q_len}")
                        graph_selected = False
                    if past_key_values is not None or last_state is not None or use_cache:
                        if graph_force:
                            raise ValueError("Graph layer mode does not support cache or `use_cache=True`.")
                        graph_selected = False
                    if self.use_short_conv and any(
                        conv.backend != "triton" for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
                    ):
                        if graph_force:
                            raise ValueError("Graph layer mode requires a graph-capable short convolution backend.")
                        graph_selected = False

        if graph_selected and cu_seqlens is not None:
            if graph_mode == "force_graph" and (graph_chunk_indices is None or graph_chunk_offsets is None):
                raise ValueError(
                    "graph_mode='force_graph' requires caller-provided fixed `chunk_indices` and `chunk_offsets`"
                )
            if graph_chunk_indices is None:
                graph_chunk_indices, generated_offsets = prepare_chunk_indices_static(
                    cu_seqlens,
                    graph_chunk_size,
                    graph_nt_max,
                )
                if graph_chunk_offsets is None:
                    graph_chunk_offsets = generated_offsets
            elif graph_chunk_offsets is None:
                _, graph_chunk_offsets = prepare_chunk_indices_static(
                    cu_seqlens,
                    graph_chunk_size,
                    graph_nt_max,
                )
        elif normalized_graph_mode == "auto" or graph_mode == "eager":
            graph_chunk_indices = None
            graph_chunk_offsets = None

        if torch.is_grad_enabled():
            mode = "chunk"
        elif q_len <= 64 and not self.training:
            mode = "fused_recurrent"
        else:
            mode = self.mode
        if graph_selected:
            mode = "chunk"
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."
        hidden_states, indices, cu_seqlens = unpad_hidden_states(hidden_states, cu_seqlens, attention_mask, q_len)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                chunk_indices=graph_chunk_indices,
                chunk_size=graph_chunk_size,
                use_graph=graph_selected,
                graph_nt_max=graph_nt_max,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                chunk_indices=graph_chunk_indices,
                chunk_size=graph_chunk_size,
                use_graph=graph_selected,
                graph_nt_max=graph_nt_max,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                chunk_indices=graph_chunk_indices,
                chunk_size=graph_chunk_size,
                use_graph=graph_selected,
                graph_nt_max=graph_nt_max,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        g = self.f_proj(hidden_states)
        beta = self.b_proj(hidden_states)

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        # g and v are at value-head dimension (HV); q/k are at qk-head dimension (H).
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=self.allow_neg_eigval,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                state_v_first=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=graph_chunk_indices,
                chunk_offsets=graph_chunk_offsets,
                use_graph=graph_selected,
                graph_t_max=graph_t_max,
                graph_n_max=graph_n_max,
                graph_nt_max=graph_nt_max,
                chunk_size=graph_chunk_size,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=self.allow_neg_eigval,
                lower_bound=self.lower_bound,
                state_v_first=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        o = repad_hidden_states(o, indices, batch_size, q_len)

        return o, None, past_key_values
