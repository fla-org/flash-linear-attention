# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for common chunk ops."""

from __future__ import annotations

from fla.ops.backends import BaseBackend


class TritonAscendCommonBackend(BaseBackend):
    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_scaled_dot_kkt_fwd_verifier(self, *args, **kwargs):
        return True, None

    def chunk_scaled_dot_kkt_fwd(self, *args, **kwargs):
        from fla.ops.common.backends.triton_ascend.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_npu
        return chunk_scaled_dot_kkt_fwd_npu(*args, **kwargs)

    def chunk_gated_delta_rule_fwd_h_verifier(self, *args, **kwargs):
        return True, None

    def chunk_gated_delta_rule_fwd_h(self, *args, **kwargs):
        from fla.ops.common.backends.triton_ascend.chunk_delta_h import chunk_gated_delta_rule_fwd_h_npu
        return chunk_gated_delta_rule_fwd_h_npu(*args, **kwargs)

    def chunk_fwd_o_verifier(self, *args, **kwargs):
        return True, None

    def chunk_fwd_o(self, *args, **kwargs):
        from fla.ops.common.backends.triton_ascend.chunk_o import chunk_fwd_o_npu
        return chunk_fwd_o_npu(*args, **kwargs)

    def chunk_bwd_dv_local_verifier(self, *args, **kwargs):
        return True, None

    def chunk_bwd_dv_local(self, *args, **kwargs):
        from fla.ops.common.backends.triton_ascend.chunk_o import chunk_bwd_dv_local_npu
        return chunk_bwd_dv_local_npu(*args, **kwargs)

    def chunk_bwd_dqkwg_verifier(self, *args, **kwargs):
        return True, None

    def chunk_bwd_dqkwg(self, *args, **kwargs):
        from fla.ops.common.backends.triton_ascend.chunk_o import chunk_bwd_dqkwg_npu
        return chunk_bwd_dqkwg_npu(*args, **kwargs)

    def chunk_gated_delta_rule_bwd_dhu_verifier(self, *args, **kwargs):
        return True, None

    def chunk_gated_delta_rule_bwd_dhu(self, *args, **kwargs):
        from fla.ops.common.backends.triton_ascend.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu_npu
        return chunk_gated_delta_rule_bwd_dhu_npu(*args, **kwargs)

    def fused_beta_sigmoid_fwd_verifier(self, *args, **kwargs):
        return True, None

    def fused_beta_sigmoid_fwd(self, x, scale=1.0):
        from fla.ops.common.backends.triton_ascend.gate import fused_beta_sigmoid_fwd_npu
        return fused_beta_sigmoid_fwd_npu(x, scale)

    def fused_beta_sigmoid_bwd_verifier(self, *args, **kwargs):
        return True, None

    def fused_beta_sigmoid_bwd(self, x, dy, scale=1.0):
        from fla.ops.common.backends.triton_ascend.gate import fused_beta_sigmoid_bwd_npu
        return fused_beta_sigmoid_bwd_npu(x, dy, scale)
