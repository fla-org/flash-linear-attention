# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for QuasarAttention

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.quasar import chunk_quasar, fused_recurrent_quasar
from fla.ops.quasar.gate import fused_quasar_gate

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies Rotary Position Embedding to the query and key tensors."""
    # cos, sin: [1, 1, seq_len, rotary_dim]
    # q, k: [batch_size, seq_len, n_heads, head_dim]
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos = cos.transpose(1, 2) # [1, seq_len, 1, rotary_dim]
    sin = sin.transpose(1, 2) # [1, seq_len, 1, rotary_dim]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class QuasarAttention(nn.Module):
    """
    QuasarAttention layer implementation.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        mode (str, Optional):
            Which QuasarAttention kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 128,
        num_heads: int = 16,
        mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> QuasarAttention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.key_dim = int(self.num_heads * self.head_dim)
        self.value_dim = int(self.num_heads * self.head_dim)
        self.layer_idx = layer_idx

        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # KDA matching: Use SiLU on q, k, v for better learning if not using short conv
        # (Short conv already has its own activation)
        self.q_act = nn.SiLU()
        self.k_act = nn.SiLU()
        self.v_act = nn.SiLU()

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

        # Data-dependent Beta (Adaptive Decay)
        # Instead of a static per-head parameter, we use a linear projection 
        # to allow the model to learn contextual importance (read/write sharpness).
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # Learnable state decay (like KDA/Mamba A matrix)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True

        # KIMI matches: separate f_proj for kernel and g_proj for final output gating
        self.f_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_dim, bias=False),
            nn.Linear(self.head_dim, self.value_dim, bias=True),
        )


        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid", eps=norm_eps)
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
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # Force chunk mode to avoid fused_recurrent BT conflict
        mode = "chunk"
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = None
        recurrent_state = None
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        
        if past_key_values is not None and self.layer_idx is not None:
            if hasattr(past_key_values, "recurrent_states") and self.layer_idx in past_key_values.recurrent_states:
                recurrent_state = past_key_values.recurrent_states[self.layer_idx]
            if hasattr(past_key_values, "conv_states") and self.layer_idx in past_key_values.conv_states:
                conv_state_q, conv_state_k, conv_state_v = past_key_values.conv_states[self.layer_idx]
            else:
                try:
                    # Standard list/tuple cache (FLA style fallback)
                    if len(past_key_values) > self.layer_idx:
                        last_state = past_key_values[self.layer_idx]
                        if isinstance(last_state, dict):
                            recurrent_state = last_state.get("recurrent_state", None)
                            convs = last_state.get("conv_state", None)
                            if convs is not None:
                                conv_state_q, conv_state_k, conv_state_v = convs
                except TypeError:
                    pass

        cu_seqlens = kwargs.get("cu_seqlens")
        if attention_mask is not None:
            # Optimization: Skip unpadding if all tokens are valid (common in packed distillation)
            if attention_mask.all():
                indices, cu_seqlens = None, None
            else:
                indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
                hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        else:
            indices = None

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
            q = self.q_act(self.q_proj(hidden_states))
            k = self.k_act(self.k_proj(hidden_states))
            v = self.v_act(self.v_proj(hidden_states))

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        if torch.isnan(q).any(): pass # print("!!! NAN IN Q (before RoPE)", flush=True)
        if torch.isnan(v).any(): pass # print("!!! NAN IN V", flush=True)

        # Apply RoPE if provided
        cos = kwargs.get("cos")
        sin = kwargs.get("sin")
        if cos is not None and sin is not None:
            if attention_mask is not None:
                # Unpad cos/sin using the same indices
                # cos/sin shape is [1, 1, seq_len, head_dim] or [batch_size, seq_len, head_dim]
                if cos.shape[0] == 1 and cos.shape[1] == 1:
                    # Broadcastable/Shared RoPE [1, 1, seq_len, head_dim]
                    # We need to expand to [batch_size, seq_len, head_dim] before unpadding
                    cos_expanded = cos.squeeze(1).expand(batch_size, -1, -1)
                    sin_expanded = sin.squeeze(1).expand(batch_size, -1, -1)
                    cos = index_first_axis(rearrange(cos_expanded, "b s d -> (b s) d"), indices).unsqueeze(0).unsqueeze(1)
                    sin = index_first_axis(rearrange(sin_expanded, "b s d -> (b s) d"), indices).unsqueeze(0).unsqueeze(1)
                else:
                    # Already [batch_size, 1, seq_len, head_dim] or [batch_size, seq_len, head_dim]
                    if cos.dim() == 4:
                        cos = cos.squeeze(1)
                        sin = sin.squeeze(1)
                    cos = index_first_axis(rearrange(cos, "b s d -> (b s) d"), indices).unsqueeze(0).unsqueeze(1)
                    sin = index_first_axis(rearrange(sin, "b s d -> (b s) d"), indices).unsqueeze(0).unsqueeze(1)
            
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # QK Normalization AFTER RoPE — ensures kernel receives unit-norm vectors
        # regardless of any precision drift introduced by the rotation
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Adaptive Beta: Sigmoid(b_proj(x)) is bounded to (0, 1) to prevent explosions.
        beta = self.b_proj(hidden_states).sigmoid()

        if mode == "chunk":
            o, recurrent_state = chunk_quasar(
                q=q,
                k=k,
                v=v,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == "fused_recurrent":
            # Use f_proj for kernel gate in fused mode
            f_gate = self.f_proj(hidden_states)
            f_gate = rearrange(f_gate, "... (h d) -> ... h d", d=self.head_dim)
            o, recurrent_state = fused_recurrent_quasar(
                q=q,
                k=k,
                v=v,
                g=f_gate,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            if hasattr(past_key_values, "update_quasar_state"):
                past_key_values.update_quasar_state(
                    self.layer_idx, 
                    recurrent_state, 
                    (conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None
                )
            else:
                with contextlib.suppress(TypeError):
                    past_key_values.update(
                        recurrent_state=recurrent_state,
                        conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                        layer_idx=self.layer_idx,
                        offset=q_len,
                    )

        # Final output gating using g_proj
        # Handle flattened inputs (unpadded) from FSDP/Flash-Linear-Attention
        if hidden_states.dim() == 2:
            # (N, D) -> (N, H, D/H)
            g = self.g_proj(hidden_states)
            g = rearrange(g, "n (h d) -> n h d", d=self.head_dim)
            o = self.o_norm(o, g)
            o = rearrange(o, "n h d -> n (h d)")
        else:
            # (B, S, D) -> (B, S, H, D/H)
            g = self.g_proj(hidden_states)
            g = rearrange(g, "b s (h d) -> b s h d", d=self.head_dim)
            o = self.o_norm(o, g)
            o = rearrange(o, "b s h d -> b s (h d)")
        
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        # LFM2 expects 2 return values (hidden_states, _)
        return o, None
