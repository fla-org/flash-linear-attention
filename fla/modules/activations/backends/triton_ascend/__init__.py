# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import BaseBackend, register_backend


@register_backend('modules.activations')
class TritonAscendBackend(BaseBackend):
    """Ascend NPU backend for activations."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def sigmoid_fwd(self, x, output_contiguous=False):
        from fla.modules.activations.backends.triton_ascend.ops import sigmoid_fwd_npu
        return sigmoid_fwd_npu(x, output_contiguous=output_contiguous)

    def sigmoid_bwd(self, x, dy, output_contiguous=False):
        from fla.modules.activations.backends.triton_ascend.ops import sigmoid_bwd_npu
        return sigmoid_bwd_npu(x, dy, output_contiguous=output_contiguous)

    def logsigmoid_fwd(self, x, temperature=1., output_contiguous=False):
        from fla.modules.activations.backends.triton_ascend.ops import logsigmoid_fwd_npu
        return logsigmoid_fwd_npu(x, temperature=temperature, output_contiguous=output_contiguous)

    def logsigmoid_bwd(self, x, dy, temperature=1., output_contiguous=False):
        from fla.modules.activations.backends.triton_ascend.ops import logsigmoid_bwd_npu
        return logsigmoid_bwd_npu(x, dy, temperature=temperature, output_contiguous=output_contiguous)

    def swish_fwd(self, x, output_contiguous=False):
        from fla.modules.activations.backends.triton_ascend.ops import swish_fwd_npu
        return swish_fwd_npu(x, output_contiguous=output_contiguous)

    def swish_bwd(self, x, dy, output_contiguous=False):
        from fla.modules.activations.backends.triton_ascend.ops import swish_bwd_npu
        return swish_bwd_npu(x, dy, output_contiguous=output_contiguous)

    def swiglu_fwd(self, x, y, output_contiguous=False):
        from fla.modules.activations.backends.triton_ascend.ops import swiglu_fwd_npu
        return swiglu_fwd_npu(x, y, output_contiguous=output_contiguous)

    def swiglu_fwdbwd(self, x, y, g, use_weight=False, output_contiguous=False):
        from fla.modules.activations.backends.triton_ascend.ops import swiglu_fwdbwd_npu
        return swiglu_fwdbwd_npu(x, y, g, use_weight=use_weight, output_contiguous=output_contiguous)

    def swiglu_linear(self, x, y, weight, bias):
        from fla.modules.activations.backends.triton_ascend.ops import swiglu_linear_npu
        return swiglu_linear_npu(x, y, weight, bias)

    def powglu_fwd(self, x, y, power=3.0, output_contiguous=False):
        from fla.modules.activations.backends.triton_ascend.ops import powglu_fwd_npu
        return powglu_fwd_npu(x, y, power=power, output_contiguous=output_contiguous)

    def powglu_fwdbwd(self, x, y, g, power=3.0, use_weight=False, output_contiguous=False):
        from fla.modules.activations.backends.triton_ascend.ops import powglu_fwdbwd_npu
        return powglu_fwdbwd_npu(
            x, y, g, power=power, use_weight=use_weight, output_contiguous=output_contiguous,
        )

    def powglu_linear(self, x, y, weight, bias, power=3.0):
        from fla.modules.activations.backends.triton_ascend.ops import powglu_linear_npu
        return powglu_linear_npu(x, y, weight, bias, power)
