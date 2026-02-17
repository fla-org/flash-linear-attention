from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.modules.activations import swiglu
from fla.ops.linoss import fused_recurrent_linoss

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class LinOSSAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'fused_recurrent',
        hidden_size: int = 1024,
        ssm_size: int | None = None,
        expand_ratio: float | None = 1.,
        discretization: str = 'IM',
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
    ) -> LinOSSAttention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_ratio = expand_ratio
        self.input_dim = int(hidden_size * expand_ratio)
        self.ssm_size = ssm_size if ssm_size is not None else self.input_dim
        self.discretization = discretization

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.layer_idx = layer_idx

        assert mode in ['fused_recurrent'], f"Not supported mode `{mode}`."
        assert discretization in ['IM', 'IMEX'], f"Not supported discretization `{discretization}`."

        P = self.ssm_size
        H = self.input_dim

        self.i_proj = nn.Linear(hidden_size, H, bias=False)

        if use_short_conv:
            self.i_conv1d = ShortConvolution(
                hidden_size=H,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )

        self.A_diag = nn.Parameter(torch.empty(P))
        self.B_param = nn.Parameter(torch.empty(P, H, 2))
        self.C_param = nn.Parameter(torch.empty(H, P, 2))
        self.D = nn.Parameter(torch.empty(H))
        self.dt = nn.Parameter(torch.empty(P))

        self.g_proj = nn.Linear(hidden_size, H, bias=False)
        self.g_norm = FusedRMSNormGated(
            hidden_size=H,
            elementwise_affine=elementwise_affine,
            eps=norm_eps,
        )
        self.o_proj = nn.Linear(H, hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        P = self.ssm_size
        H = self.input_dim
        nn.init.uniform_(self.A_diag, 0, 1)
        std_b = 1.0 / math.sqrt(H)
        nn.init.uniform_(self.B_param, -std_b, std_b)
        std_c = 1.0 / math.sqrt(P)
        nn.init.uniform_(self.C_param, -std_c, std_c)
        nn.init.normal_(self.D, std=1.0)
        nn.init.uniform_(self.dt, 0, 1)

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

        mode = self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if self.use_short_conv:
            conv_state = None
            if last_state is not None:
                conv_state = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            i, conv_state = self.i_conv1d(
                x=self.i_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state,
                output_final_state=use_cache,
            )
        else:
            conv_state = None
            i = self.i_proj(hidden_states)

        if attention_mask is not None:
            i = i.mul(attention_mask[:, -i.shape[-2]:, None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        B_re = self.B_param[..., 0].contiguous()
        B_im = self.B_param[..., 1].contiguous()
        C_re = self.C_param[..., 0].contiguous()
        C_im = self.C_param[..., 1].contiguous()

        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_linoss(
                x=i,
                B_re=B_re,
                B_im=B_im,
                C_re=C_re,
                C_im=C_im,
                a_diag=self.A_diag,
                dt=self.dt,
                d_skip=self.D,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                discretization=self.discretization,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=conv_state if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=i.shape[2] if len(i.shape) > 2 else 1,
            )

        o = self.g_norm(o, self.g_proj(hidden_states))
        o = self.o_proj(o)

        return o, None, past_key_values

    def state_size(self, **kwargs) -> int:
        state_size = 2 * self.ssm_size
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size