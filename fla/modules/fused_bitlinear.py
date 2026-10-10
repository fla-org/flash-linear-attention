# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# BitLinear with fused normalization and quantized linear projection.
# [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
# [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528)
#
# adapted from https://github.com/ridgerchu/matmulfreellm/

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules.norm import RMSNorm
from fla.modules.norm.layernorm_quant import LayerNormLinearQuantFunction, weight_quant


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Quantize activations to 8 bits per token and return dequantized values."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def layer_norm_linear_quant(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_bias: torch.Tensor | None,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return LayerNormLinearQuantFunction.apply(
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
    )


def rms_norm_linear_quant(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_bias: torch.Tensor | None,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return layer_norm_linear_quant(
        x=x,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        linear_weight=linear_weight,
        linear_bias=linear_bias,
        residual=residual,
        eps=eps,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=True,
    )


def bit_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_bias: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply RMS normalization and a linear projection with quantized activations and weights."""
    return layer_norm_linear_quant(
        x=x,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        linear_weight=weight,
        linear_bias=bias,
        is_rms_norm=True,
    )


class BitLinear(nn.Linear):
    """
    RMS-normalized linear layer with 8-bit activations and ternary weights.

    Quantization uses a straight-through estimator during training.
    Efficient deployment requires specialized kernels.

    Args:
        in_features (int):
            Size of each input sample.
        out_features (int):
            Size of each output sample.
        bias (bool, Optional):
            Whether to allocate an additive bias. Default: `False`.
        norm_eps (float, Optional):
            Epsilon for RMS normalization. Default: 1e-8.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, norm_eps: float = 1e-8) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.norm = RMSNorm(hidden_size=in_features, eps=norm_eps, dtype=torch.float32)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().extra_repr()}, norm_eps={self.norm.eps})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        x_norm = self.norm(x)

        # straight-through gradients bypass quantization
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(input=x_quant, weight=w_quant)

        return y


class FusedBitLinear(BitLinear):
    """BitLinear with fused RMS normalization and activation quantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm_linear_quant(
            x=x,
            norm_weight=self.norm.weight,
            norm_bias=self.norm.bias,
            linear_weight=self.weight,
            linear_bias=self.bias,
            is_rms_norm=True,
        )
