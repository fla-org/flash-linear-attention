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
import torch.nn.functional as F

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import ShortConvolution
from fla.ops.hgrn import chunk_hgrn, fused_recurrent_hgrn

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache

# bounds the derivative of `sqrt(1 - a^2)` as `a -> 1`, which would otherwise overflow in bf16
MAX_SQRT_GRADIENT = 1000.0


class SqrtBoundDerivative(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        return grad_output / torch.sqrt(torch.clamp(4.0 * x, min=MAX_SQRT_GRADIENT ** -2))


class BlockDiagonalLinear(nn.Module):
    """Affine map with a block-diagonal weight of `num_blocks` square blocks."""

    def __init__(self, width: int, num_blocks: int):
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.block_width = width // num_blocks
        self.weight = nn.Parameter(torch.empty(num_blocks, self.block_width, self.block_width))
        self.bias = nn.Parameter(torch.empty(num_blocks, self.block_width))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(self.block_width))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unflatten(-1, (self.num_blocks, self.block_width))
        return (torch.einsum('...hi,hij->...hj', x, self.weight) + self.bias).flatten(-2)


class RGLRU(nn.Module):
    """
    The recurrent block of Griffin and Hawk (https://arxiv.org/abs/2402.19427), built around the Real-Gated Linear
    Recurrent Unit (RG-LRU).

    The input is projected to two branches: a GeLU gate, and a short convolution followed by the RG-LRU
    `h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * x_t)` with `a_t = a^(c * r_t)` and a learned per-channel decay
    `a = exp(-softplus(a_param))`, where the recurrence gate `r_t` and the input gate `i_t` are block-diagonal projections
    of `x_t`. The product of both branches is
    projected back to the hidden size.

    Args:
        mode (str, Optional):
            Which kernel to use. Available options: `chunk` and `fused_recurrent`. Default: `chunk`.
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        lru_width (int, Optional):
            The width of the recurrence. Default: `None`, which means `hidden_size`.
        num_heads (int, Optional):
            The number of blocks of the block-diagonal gate projections. Default: 8.
        conv_size (int, Optional):
            The kernel size of the short convolution. Default: 4.
        c (float, Optional):
            The constant scaling the recurrence gate in the exponent of `a_t`. Default: 8.0.
        min_rad (float, Optional):
            The lower bound of the decay `a` at initialization. Default: 0.9.
        max_rad (float, Optional):
            The upper bound of the decay `a` at initialization. Default: 0.999.
        layer_idx (int, Optional):
            The index of the layer. Default: `None`.
    """

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        lru_width: int | None = None,
        num_heads: int = 8,
        conv_size: int = 4,
        c: float = 8.0,
        min_rad: float = 0.9,
        max_rad: float = 0.999,
        layer_idx: int | None = None,
    ):
        super().__init__()

        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        self.mode = mode
        self.hidden_size = hidden_size
        self.lru_width = lru_width if lru_width is not None else hidden_size
        if self.lru_width % num_heads != 0:
            raise ValueError(f"lru_width ({self.lru_width}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.conv_size = conv_size
        self.c = c
        self.min_rad = min_rad
        self.max_rad = max_rad
        self.layer_idx = layer_idx

        self.linear_y = nn.Linear(hidden_size, self.lru_width)
        self.linear_x = nn.Linear(hidden_size, self.lru_width)
        self.conv1d = ShortConvolution(self.lru_width, conv_size, bias=True, activation=None)
        self.input_gate = BlockDiagonalLinear(self.lru_width, num_heads)
        self.a_gate = BlockDiagonalLinear(self.lru_width, num_heads)
        self.a_param = nn.Parameter(torch.empty(self.lru_width))
        self.linear_out = nn.Linear(self.lru_width, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        # the decay `a` is distributed uniformly in area within the ring [min_rad, max_rad]
        with torch.no_grad():
            log_a = self.a_param.uniform_(self.min_rad ** 2 + 1e-8, self.max_rad ** 2 + 1e-8).log_().mul_(0.5)
            self.a_param.copy_(torch.log(torch.expm1(-log_a)))

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

        cu_seqlens = kwargs.get('cu_seqlens')
        # launching the triton kernel for just one token will actually be slower
        if cu_seqlens is not None or (not self.training and hidden_states.shape[1] <= 64):
            mode = 'fused_recurrent'
        else:
            mode = self.mode

        last_state = get_layer_cache(self, past_key_values)
        batch_size, q_len, _ = hidden_states.shape
        mask = attention_mask[:, -q_len:] if attention_mask is not None else None

        y = F.gelu(self.linear_y(hidden_states), approximate='tanh')
        x, conv_state = self.conv1d(
            x=self.linear_x(hidden_states),
            mask=mask,
            cache=last_state['conv_state'] if last_state is not None else None,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        # the first token of each sequence starts from a zero state and skips the input normalization
        if last_state is not None:
            reset = torch.zeros(batch_size, q_len, dtype=torch.bool, device=x.device)
        elif cu_seqlens is not None:
            reset = torch.zeros(batch_size, q_len, dtype=torch.bool, device=x.device)
            reset[:, cu_seqlens[:-1]] = True
        elif mask is not None:
            reset = mask.bool() & (mask.cumsum(-1) == 1)
        else:
            reset = torch.zeros(batch_size, q_len, dtype=torch.bool, device=x.device)
            reset[:, 0] = True

        log_a = -self.c * torch.sigmoid(self.a_gate(x).float()) * F.softplus(self.a_param.float())
        multiplier = SqrtBoundDerivative.apply(-torch.expm1(2 * log_a))
        multiplier = torch.where(reset[..., None], torch.ones_like(multiplier), multiplier)
        x = x * torch.sigmoid(self.input_gate(x)) * multiplier.to(x.dtype)
        if mask is not None:
            x = x.mul(mask[..., None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state = chunk_hgrn(
                x=x,
                g=log_a.to(x.dtype),
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        else:
            o, recurrent_state = fused_recurrent_hgrn(
                x=x,
                g=log_a.to(x.dtype),
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=conv_state,
            offset=q_len,
        )

        o = self.linear_out(o * y)
        return o, None, past_key_values

    def state_size(self, **kwargs) -> int:
        return self.lru_width + self.conv1d.state_size
