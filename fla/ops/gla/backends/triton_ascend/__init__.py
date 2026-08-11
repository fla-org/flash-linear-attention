# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for GLA ops."""

from __future__ import annotations

from fla.ops.backends import BaseBackend

_MAX_KV = 512


def _verify_kv(K: int, V: int | None = None) -> tuple[bool, str | None]:
    if K > _MAX_KV:
        return False, f'NPU GLA supports K<={_MAX_KV}, got K={K}'
    if V is not None and V > _MAX_KV:
        return False, f'NPU GLA supports V<={_MAX_KV}, got V={V}'
    return True, None


class TritonAscendGLABackend(BaseBackend):
    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_gla_fwd_intra_gk_verifier(self, q, k, *args, **kwargs):
        return _verify_kv(k.shape[-1])

    def chunk_gla_fwd_intra_gk(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_fwd_intra_gk_npu
        return chunk_gla_fwd_intra_gk_npu(*args, **kwargs)

    def chunk_gla_fwd_o_gk_verifier(self, q, v, *args, **kwargs):
        return _verify_kv(q.shape[-1], v.shape[-1])

    def chunk_gla_fwd_o_gk(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_fwd_o_gk_npu
        return chunk_gla_fwd_o_gk_npu(*args, **kwargs)

    def chunk_gla_bwd_dA_verifier(self, v, do, *args, **kwargs):
        return _verify_kv(v.shape[-1])

    def chunk_gla_bwd_dA(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_bwd_dA_npu
        return chunk_gla_bwd_dA_npu(*args, **kwargs)

    def chunk_gla_bwd_dv_verifier(self, k, g, A, do, dh, *args, **kwargs):
        return _verify_kv(k.shape[-1], do.shape[-1])

    def chunk_gla_bwd_dv(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_bwd_dv_npu
        return chunk_gla_bwd_dv_npu(*args, **kwargs)

    def chunk_gla_bwd_dqk_intra_verifier(self, q, k, *args, **kwargs):
        return _verify_kv(k.shape[-1])

    def chunk_gla_bwd_dqk_intra(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_bwd_dqk_intra_npu
        return chunk_gla_bwd_dqk_intra_npu(*args, **kwargs)

    def chunk_gla_bwd_dqkg_verifier(self, q, k, v, *args, **kwargs):
        return _verify_kv(k.shape[-1], v.shape[-1])

    def chunk_gla_bwd_dqkg(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_bwd_dqkg_npu
        return chunk_gla_bwd_dqkg_npu(*args, **kwargs)
