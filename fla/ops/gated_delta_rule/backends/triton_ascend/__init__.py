# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for GDN gated_delta_rule ops."""

from __future__ import annotations

from fla.ops.backends import BaseBackend


class TritonAscendGDNBackend(BaseBackend):
    """Ascend NPU backend for GDN gate and WY-representation kernels."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def gdn_gate_fwd_verifier(self, *args, **kwargs):
        return True, None

    def gdn_gate_fwd(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.triton_ascend.gate import gdn_gate_fwd_npu
        return gdn_gate_fwd_npu(*args, **kwargs)

    def gdn_gate_chunk_cumsum_verifier(self, *args, **kwargs):
        return True, None

    def gdn_gate_chunk_cumsum(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.triton_ascend.gate import gdn_gate_chunk_cumsum_npu
        return gdn_gate_chunk_cumsum_npu(*args, **kwargs)

    def gdn_gate_bwd_verifier(self, *args, **kwargs):
        return True, None

    def gdn_gate_bwd(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.triton_ascend.gate import gdn_gate_bwd_npu
        return gdn_gate_bwd_npu(*args, **kwargs)

    def recompute_w_u_fwd_verifier(self, *args, **kwargs):
        return True, None

    def recompute_w_u_fwd(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.triton_ascend.wy_fast import recompute_w_u_fwd_npu
        return recompute_w_u_fwd_npu(*args, **kwargs)

    def prepare_wy_repr_bwd_verifier(self, *args, **kwargs):
        return True, None

    def prepare_wy_repr_bwd(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.triton_ascend.wy_fast import prepare_wy_repr_bwd_npu
        return prepare_wy_repr_bwd_npu(*args, **kwargs)

    def chunk_gated_delta_rule_fwd_intra_verifier(self, *args, **kwargs):
        return True, None

    def chunk_gated_delta_rule_fwd_intra(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.triton_ascend.chunk_fwd import chunk_gated_delta_rule_fwd_intra_npu
        return chunk_gated_delta_rule_fwd_intra_npu(*args, **kwargs)
