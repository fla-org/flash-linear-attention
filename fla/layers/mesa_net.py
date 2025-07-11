# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2_norm
from fla.ops.mesa_net import chunk_mesa_net, mesa_net_decoding_one_step

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class MesaNet(nn.Module):
    """
    The layer implementaion for [MesaNet: Sequence Modeling by Locally Optimal Test-Time Training].  # noqa

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 1.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        mode (str, Optional):
            Which MesaNet kernel to use.
            Currently available: `chunk`.
            Default: `chunk`.
        use_output_gate (bool, Optional):
            Whether to use output gate. Default: `False`.
        conv_size (int):
            The kernel size of the short convolution. Default: 4.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
        lambda_lower_bound (float):
            The lower bound for the lambda parameter. Default: 0.25.
        max_cg_step_training (int):
            The maximum number of CG steps for training. Default: 30.
        max_cg_step_decoding (int):
            The maximum number of CG steps for decoding. Default: 30.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        head_dim: int = 128,
        mode: str = 'chunk',
        use_output_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        lambda_lower_bound: float = 0.25,
        max_cg_step_training: int = 30,
        max_cg_step_decoding: int = 30,
        **kwargs
    ) -> MesaNet:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.use_output_gate = use_output_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.key_dim
        self.head_k_dim = self.head_dim
        self.head_v_dim = self.head_dim
        self.layer_idx = layer_idx
        self.lambda_lower_bound = lambda_lower_bound
        self.max_cg_step_training = max_cg_step_training
        self.max_cg_step_decoding = max_cg_step_decoding

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=True)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=True)

        lambda_initial_value = 1.0
        init_lamb_value = torch.log(torch.exp(torch.tensor(lambda_initial_value - lambda_lower_bound)) - 1.0)
        init_lamb_params = torch.empty(self.key_dim, dtype=torch.float32).fill_(init_lamb_value)

        self.lambda_params = nn.Parameter(init_lamb_params)
        self.lambda_params._no_weight_decay = True

        self.conv_size = conv_size
        self.q_conv1d = ShortConvolution(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=self.conv_bias,
            activation='silu',
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=self.conv_bias,
            activation='silu',
        )
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        conv_state_q, conv_state_k = None, None
        if last_state is not None:
            conv_state_q, conv_state_k = last_state['conv_state']
        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        v = self.v_proj(hidden_states)

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        beta = self.b_proj(hidden_states).float().sigmoid()
        g = F.logsigmoid(self.a_proj(hidden_states).float())
        lamb = F.softplus(self.lambda_params.float()) + self.lambda_lower_bound
        lamb = lamb.reshape(self.num_heads, -1)

        last_h_kk, last_h_kv = last_state['recurrent_state'] if last_state is not None else (None, None)

        # prefilling or training
        # Note that QK will be normalized inside the kernel to avoid saving the activations, thereby reducing the memory usage.
        if last_state is None:
            o, h_kk, h_kv = chunk_mesa_net(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                lamb=lamb,
                output_final_state=use_cache,
                max_CG_iteration=self.max_cg_step_training,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        # decoding
        else:
            q = l2_norm(q)
            k = l2_norm(k)
            o, h_kk, h_kv = mesa_net_decoding_one_step(
                q=q.squeeze(0),
                k=k.squeeze(0),
                v=v.squeeze(0),
                g=g.squeeze(0),
                beta=beta.squeeze(0),
                lamb=lamb,
                prev_h_kk=last_h_kk,
                prev_h_kv=last_h_kv,
                max_CG_iteration=self.max_cg_step_decoding
            )
            o = o.unsqueeze(0).to(q)

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=(h_kk, h_kv),
                conv_state=(conv_state_q, conv_state_k),
                layer_idx=self.layer_idx,
                offset=q_len
            )
        if self.use_output_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o, None, past_key_values
