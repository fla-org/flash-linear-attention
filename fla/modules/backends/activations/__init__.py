# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend registration for activations."""

import torch

from fla.backends import BackendRegistry, BaseBackend


class TritonAscendActivationsBackend(BaseBackend):
    backend_type = "triton_ascend"
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def sigmoid_fwd(self, x: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
        from fla.modules.backends.activations.triton_ascend import sigmoid_fwd
        return sigmoid_fwd(x=x, output_contiguous=output_contiguous)

    def sigmoid_bwd(self, x: torch.Tensor, dy: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
        from fla.modules.backends.activations.triton_ascend import sigmoid_bwd
        return sigmoid_bwd(x=x, dy=dy, output_contiguous=output_contiguous)

    def logsigmoid_fwd(self, x: torch.Tensor, temperature: float = 1.0, output_contiguous: bool = False) -> torch.Tensor:
        from fla.modules.backends.activations.triton_ascend import logsigmoid_fwd
        return logsigmoid_fwd(x=x, temperature=temperature, output_contiguous=output_contiguous)

    def logsigmoid_bwd(
        self,
        x: torch.Tensor,
        dy: torch.Tensor,
        temperature: float = 1.0,
        output_contiguous: bool = False,
    ) -> torch.Tensor:
        from fla.modules.backends.activations.triton_ascend import logsigmoid_bwd
        return logsigmoid_bwd(x=x, dy=dy, temperature=temperature, output_contiguous=output_contiguous)

    def swish_fwd(self, x: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
        from fla.modules.backends.activations.triton_ascend import swish_fwd
        return swish_fwd(x=x, output_contiguous=output_contiguous)

    def swish_bwd(self, x: torch.Tensor, dy: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
        from fla.modules.backends.activations.triton_ascend import swish_bwd
        return swish_bwd(x=x, dy=dy, output_contiguous=output_contiguous)

    def swiglu_fwd(self, x: torch.Tensor, y: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
        from fla.modules.backends.activations.triton_ascend import swiglu_fwd
        return swiglu_fwd(x=x, y=y, output_contiguous=output_contiguous)

    def swiglu_fwdbwd(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        g: torch.Tensor,
        use_weight: bool = False,
        output_contiguous: bool = False,
    ):
        from fla.modules.backends.activations.triton_ascend import swiglu_fwdbwd
        return swiglu_fwdbwd(x=x, y=y, g=g, use_weight=use_weight, output_contiguous=output_contiguous)

    def swiglu_linear(self, x, y, weight, bias):
        from fla.modules.backends.activations.triton_ascend import swiglu_linear
        return swiglu_linear(x=x, y=y, weight=weight, bias=bias)

    def powglu_fwd(self, x: torch.Tensor, y: torch.Tensor, power: float = 3.0, output_contiguous: bool = False) -> torch.Tensor:
        from fla.modules.backends.activations.triton_ascend import powglu_fwd
        return powglu_fwd(x=x, y=y, power=power, output_contiguous=output_contiguous)

    def powglu_fwdbwd(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        g: torch.Tensor,
        power: float = 3.0,
        use_weight: bool = False,
        output_contiguous: bool = False,
    ):
        from fla.modules.backends.activations.triton_ascend import powglu_fwdbwd
        return powglu_fwdbwd(x=x, y=y, g=g, power=power, use_weight=use_weight, output_contiguous=output_contiguous)

    def powglu_linear(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        power: float = 3.0,
    ) -> torch.Tensor:
        from fla.modules.backends.activations.triton_ascend import powglu_linear
        return powglu_linear(x=x, y=y, weight=weight, bias=bias, power=power)


activations_registry = BackendRegistry("modules.activations")
activations_registry.register(TritonAscendActivationsBackend())
