# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for common chunk ops."""

from __future__ import annotations

import torch

from fla.ops.backends import BaseBackend


def _verify_npu_kv(k: torch.Tensor, v: torch.Tensor, *extra: torch.Tensor) -> tuple[bool, str | None]:
    from fla.utils import npu_verify_kv

    return npu_verify_kv(k, v, extra=extra)


class TritonAscendCommonBackend(BaseBackend):
    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_scaled_dot_kkt_fwd_verifier(
        self,
        k,
        g=None,
        beta=None,
        cu_seqlens=None,
        chunk_size=64,
        output_dtype=torch.float32,
        chunk_indices=None,
    ) -> tuple[bool, str | None]:
        extra = (beta,) if g is None else (g, beta)
        return _verify_npu_kv(k, k, *extra)

    def chunk_scaled_dot_kkt_fwd(self, *args, **kwargs):
        from fla.ops.common.backends.triton_ascend.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_npu
        return chunk_scaled_dot_kkt_fwd_npu(*args, **kwargs)

    def chunk_gated_delta_rule_fwd_h_verifier(self, k, w, u, g=None, gk=None, **kwargs):
        extra = tuple(t for t in (w, u, g, gk) if t is not None)
        return _verify_npu_kv(k, u, *extra)

    def chunk_gated_delta_rule_fwd_h(self, *args, **kwargs):
        from fla.ops.common.backends.triton_ascend.chunk_delta_h import chunk_gated_delta_rule_fwd_h_npu
        return chunk_gated_delta_rule_fwd_h_npu(*args, **kwargs)

    def chunk_fwd_h_verifier(self, k, v, *args, **kwargs):
        K, V = k.shape[-1], v.shape[-1]
        if K > 512 or V > 512:
            return False, f'NPU chunk_fwd_h supports K,V<=512, got K={K}, V={V}'
        return _verify_npu_kv(k, v)

    def chunk_fwd_h(self, *args, **kwargs):
        from fla.ops.common.backends.triton_ascend.chunk_h import chunk_fwd_h_npu
        return chunk_fwd_h_npu(*args, **kwargs)

    def chunk_bwd_dh_verifier(self, q, k, v, *args, **kwargs):
        K, V = k.shape[-1], v.shape[-1]
        if K > 512 or V > 512:
            return False, f'NPU chunk_bwd_dh supports K,V<=512, got K={K}, V={V}'
        return _verify_npu_kv(k, v, q)

    def chunk_bwd_dh(self, *args, **kwargs):
        from fla.ops.common.backends.triton_ascend.chunk_h import chunk_bwd_dh_npu
        return chunk_bwd_dh_npu(*args, **kwargs)

    def chunk_fwd_o_verifier(self, q, k, v, h, g=None, g_gamma=None, **kwargs):
        extra = tuple(t for t in (q, h, g, g_gamma) if t is not None)
        return _verify_npu_kv(k, v, *extra)

    def chunk_fwd_o(self, *args, **kwargs):
        from fla.ops.common.backends.triton_ascend.chunk_o import chunk_fwd_o_npu
        return chunk_fwd_o_npu(*args, **kwargs)

    def chunk_bwd_dv_local_verifier(self, q, k, do, g=None, g_gamma=None, **kwargs):
        extra = tuple(t for t in (q, do, g, g_gamma) if t is not None)
        return _verify_npu_kv(k, do, *extra)

    def chunk_bwd_dv_local(self, *args, **kwargs):
        from fla.ops.common.backends.triton_ascend.chunk_o import chunk_bwd_dv_local_npu
        return chunk_bwd_dv_local_npu(*args, **kwargs)

    def chunk_bwd_dqkwg_verifier(self, q, k, v, do, h, dh, w=None, g=None, g_gamma=None, dv=None, **kwargs):
        extra = tuple(t for t in (q, do, h, dh, w, g, g_gamma, dv) if t is not None)
        return _verify_npu_kv(k, v, *extra)

    def chunk_bwd_dqkwg(self, *args, **kwargs):
        from fla.ops.common.backends.triton_ascend.chunk_o import chunk_bwd_dqkwg_npu
        return chunk_bwd_dqkwg_npu(*args, **kwargs)

    def chunk_gated_delta_rule_bwd_dhu_verifier(
        self, q, k, w, do, dv, g=None, gk=None, h0=None, dht=None, **kwargs,
    ):
        extra = tuple(t for t in (q, w, do, dv, g, gk, h0, dht) if t is not None)
        return _verify_npu_kv(k, do, *extra)

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
