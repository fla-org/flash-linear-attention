# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch
import torch.nn as nn

from fla.modules.norm.fused_norm_gate.ops import (
    layer_norm_gated,
    layer_norm_swish_gate_linear,
    rms_norm_gated,
    rms_norm_swish_gate_linear,
)


class FusedLayerNormGated(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        activation: str = "swish",
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation

        if self.activation not in ["swish", "silu", "sigmoid"]:
            raise ValueError(f"Unsupported activation: {self.activation}")

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += f", activation={self.activation}"
        s += ")"
        return s

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return layer_norm_gated(
            x=x,
            g=g,
            weight=self.weight,
            bias=self.bias,
            activation=self.activation,
            residual=residual,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            eps=self.eps,
        )


class FusedRMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        activation: str = "swish",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation

        if self.activation not in ["swish", "silu", "sigmoid"]:
            raise ValueError(f"Unsupported activation: {self.activation}")

        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += f", activation={self.activation}"
        s += ")"
        return s

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return rms_norm_gated(
            x=x,
            g=g,
            weight=self.weight,
            bias=self.bias,
            activation=self.activation,
            residual=residual,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            eps=self.eps,
        )


class FusedLayerNormSwishGate(FusedLayerNormGated):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            elementwise_affine=elementwise_affine,
            bias=bias,
            eps=eps,
            device=device,
            dtype=dtype,
        )


class FusedRMSNormSwishGate(FusedRMSNormGated):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            elementwise_affine=elementwise_affine,
            eps=eps,
            device=device,
            dtype=dtype,
        )


class FusedLayerNormGatedLinear(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return layer_norm_swish_gate_linear(
            x=x,
            g=g,
            norm_weight=self.weight,
            norm_bias=self.bias,
            linear_weight=weight,
            linear_bias=bias,
            residual=residual,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            eps=self.eps,
        )


class FusedLayerNormSwishGateLinear(FusedLayerNormGatedLinear):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            elementwise_affine=elementwise_affine,
            eps=eps,
            device=device,
            dtype=dtype,
        )


class FusedRMSNormGatedLinear(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return rms_norm_swish_gate_linear(
            x=x,
            g=g,
            norm_weight=self.weight,
            norm_bias=self.bias,
            linear_weight=weight,
            linear_bias=bias,
            residual=residual,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            eps=self.eps,
        )


class FusedRMSNormSwishGateLinear(FusedRMSNormGatedLinear):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            elementwise_affine=elementwise_affine,
            eps=eps,
            device=device,
            dtype=dtype,
        )
