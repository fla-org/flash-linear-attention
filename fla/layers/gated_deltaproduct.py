from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.ops.delta_rule import chunk_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


def interleave_multiple_sequences(*sequences):
    """
    Interleave multiple sequences together.
    For example, with sequences [A1, A2], [B1, B2], [C1, C2],
    returns [A1, B1, C1, A2, B2, C2]
    """
    if isinstance(sequences[0], (list, tuple)):
        sequences = sequences[0]

    if len(sequences) == 1:
        return sequences[0]

    # All sequences should have the same shape
    assert all(s.shape == sequences[0].shape for s in sequences)

    # Get the original shape
    batch_size, seq_len, *rest = sequences[0].shape

    # Stack sequences along a new dimension
    stacked = torch.stack(sequences, dim=2)

    # Reshape to interleave
    reshaped = stacked.view(batch_size, seq_len * len(sequences), *rest)

    return reshaped


class GatedDeltaProduct(nn.Module):
    """
    Generalized version of GatedDoubleDeltaNet that supports arbitrary number of householder transformations.
    """

    def __init__(
            self,
            hidden_size: int = 2048,
            expand_v: float = 2,
            head_dim: int = 256,
            num_heads: int = 6,
            num_householder: int = 2,  # New parameter for number of householder transformations
            mode: str = "chunk",
            use_gate: bool = True,
            use_forget_gate: bool = True,  # when true Gated DeltaProduct, when false DeltaProduct
            use_short_conv: bool = True,
            conv_size: int = 4,
            conv_bias: bool = False,
            layer_idx: int | None = None,
            norm_eps: float = 1e-5,
            allow_neg_eigval: bool = False,  # when true (Gated) DeltaProduct [-1, 1], when false (Gated) DeltaProduct [0, 1]
            **kwargs,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_householder = num_householder
        self.allow_neg_eigval = allow_neg_eigval
        self.use_forget_gate = use_forget_gate
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = int(self.key_dim * self.expand_v)
        self.head_qk_dim = head_dim
        self.head_v_dim = int(head_dim * self.expand_v)
        self.layer_idx = layer_idx
        self.silu = nn.SiLU()
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."
        # Create multiple projection layers for each householder transformation
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)

        self.k_projs = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.key_dim, bias=False)
                for _ in range(num_householder)
            ]
        )
        self.v_projs = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.value_dim, bias=False)
                for _ in range(num_householder)
            ]
        )
        self.b_projs = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.num_heads, bias=False)
                for _ in range(num_householder)
            ]
        )
        if use_short_conv:
            self.q_conv1ds = nn.ModuleList(
                [
                    ShortConvolution(
                        hidden_size=self.key_dim,
                        kernel_size=conv_size,
                        activation="silu",
                    )
                    for _ in range(num_householder)
                ]
            )
            self.k_conv1ds = nn.ModuleList(
                [
                    ShortConvolution(
                        hidden_size=self.key_dim,
                        kernel_size=conv_size,
                        activation="silu",
                    )
                    for _ in range(num_householder)
                ]
            )
            self.v_conv1ds = nn.ModuleList(
                [
                    ShortConvolution(
                        hidden_size=self.value_dim,
                        kernel_size=conv_size,
                        activation="silu",
                    )
                    for _ in range(num_householder)
                ]
            )

        if self.use_forget_gate:
            self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
            A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
            A_log = torch.log(A)
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True

            # Initialize dt parameters
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            dt = torch.exp(
                torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.k_id = torch.nn.Identity()
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Cache] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            **kwargs: Unpack[Dict],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding)."
            )

        mode = (
            "chunk"  # 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        )
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # Process each householder transformation
        ks, vs, betas = [], [], []
        conv_states = []

        for i in range(self.num_householder):
            if self.use_short_conv:
                conv_state_q, conv_state_k, conv_state_v = None, None, None
                if last_state is not None:
                    conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"][
                        i
                    ]
                conv_mask = (
                    attention_mask[:, -hidden_states.shape[1]:]
                    if attention_mask is not None
                    else None
                )

                k, conv_state_k = self.k_conv1ds[i](
                    x=self.k_projs[i](hidden_states),
                    mask=conv_mask,
                    cache=conv_state_k,
                    output_final_state=use_cache,
                )
                v, conv_state_v = self.v_conv1ds[i](
                    x=self.v_projs[i](hidden_states),
                    mask=conv_mask,
                    cache=conv_state_v,
                    output_final_state=use_cache,
                )
                conv_states.append((conv_state_q, conv_state_k, conv_state_v))
            else:
                k = self.silu(self.k_projs[i](hidden_states))
                v = self.silu(self.v_projs[i](hidden_states))

            ks.append(k)
            vs.append(v)

            beta = self.b_projs[i](
                hidden_states
            ).sigmoid()  # bs, sequence_length, num_heads
            if attention_mask is not None:
                beta = beta.mul(attention_mask[:, -hidden_states.shape[1]:, None])
            if self.allow_neg_eigval:
                beta = beta * 2
            betas.append(beta)

        if self.use_short_conv:
            q, conv_state_q = self.q_conv1ds[0](
                x=self.q_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_q,
                output_final_state=use_cache,
            )
        else:
            q = self.silu(self.q_proj(hidden_states))
        q = interleave_multiple_sequences(
            [torch.zeros_like(q)] * (self.num_householder - 1) + [q]
        )
        # Interleave all sequences
        k = interleave_multiple_sequences(ks)
        v = interleave_multiple_sequences(vs)
        beta = interleave_multiple_sequences(betas)

        q, k, v = (
            rearrange(x, "b t (h d) -> b t h d", h=self.num_heads) for x in (q, k, v)
        )

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        offsets = kwargs.get("offsets")

        if mode == "chunk":
            if self.use_forget_gate:
                g = -self.A_log.float().exp() * F.softplus(
                    self.a_proj(hidden_states).float() + self.dt_bias
                )
                if attention_mask is not None:
                    g = g.mul(attention_mask[:, -g.shape[-2]:, None])

                # Interleave g with zeros for non-first transformations
                g = interleave_multiple_sequences(
                    [g] + [torch.zeros_like(g)] * (self.num_householder - 1)
                )

                o, recurrent_state = chunk_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    cu_seqlens=offsets,
                    use_qk_l2norm_in_kernel=True
                )
            else:
                o, recurrent_state = chunk_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    cu_seqlens=offsets,
                    use_qk_l2norm_in_kernel=True
                )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # Take every nth element for n householder transformations
        o = o[:, self.num_householder - 1:: self.num_householder, :]

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=conv_states if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[2],
            )

        if self.use_gate:
            g = rearrange(
                self.g_proj(hidden_states),
                "... (h d) -> ... h d",
                h=self.num_heads,
            )
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        return o, None, past_key_values
