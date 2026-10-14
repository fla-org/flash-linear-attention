# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# For a list of contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for gated Oja rule ops."""

from __future__ import annotations

from fla.backends import BaseBackend, register


@register('gated_oja_rule')
class TritonAscendGatedOjaRuleBackend(BaseBackend):
    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def fused_recurrent_oja_fwd(self, *args, **kwargs):
        from fla.ops.gated_oja_rule.backends.triton_ascend.fused_recurrent import fused_recurrent_oja_fwd_npu
        return fused_recurrent_oja_fwd_npu(*args, **kwargs)

    def chunk_oja_fwd_o_verifier(self, *args, **kwargs):
        return True, None

    def chunk_oja_fwd_o(self, *args, **kwargs):
        from fla.ops.gated_oja_rule.backends.triton_ascend.chunk_o import chunk_oja_fwd_o_npu
        return chunk_oja_fwd_o_npu(*args, **kwargs)

    def chunk_oja_bwd_dqk_verifier(self, *args, **kwargs):
        return True, None

    def chunk_oja_bwd_dqk(self, *args, **kwargs):
        from fla.ops.gated_oja_rule.backends.triton_ascend.chunk_o import chunk_oja_bwd_dqk_npu
        return chunk_oja_bwd_dqk_npu(*args, **kwargs)

    def chunk_oja_bwd_dv_o_verifier(self, *args, **kwargs):
        return True, None

    def chunk_oja_bwd_dv_o(self, *args, **kwargs):
        from fla.ops.gated_oja_rule.backends.triton_ascend.chunk_o import chunk_oja_bwd_dv_o_npu
        return chunk_oja_bwd_dv_o_npu(*args, **kwargs)
