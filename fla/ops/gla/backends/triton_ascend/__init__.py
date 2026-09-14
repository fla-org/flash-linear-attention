# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for GLA ops."""

from __future__ import annotations

import torch

from fla.ops.backends import BaseBackend

_MAX_KV = 512


def _verify_kv(k: torch.Tensor, v: torch.Tensor | None = None) -> tuple[bool, str | None]:
    from fla.utils import npu_verify_kv, npu_verify_last_dim_tensor

    K = int(k.shape[-1])
    if K > _MAX_KV:
        return False, f'NPU GLA supports K<={_MAX_KV}, got K={K}'
    if v is None:
        return npu_verify_last_dim_tensor(k, label='K')
    V = int(v.shape[-1])
    if V > _MAX_KV:
        return False, f'NPU GLA supports V<={_MAX_KV}, got V={V}'
    return npu_verify_kv(k, v)


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
        return _verify_kv(k)

    def chunk_gla_fwd_intra_gk(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_fwd_intra_gk_npu
        return chunk_gla_fwd_intra_gk_npu(*args, **kwargs)

    def chunk_gla_fwd_o_gk_verifier(self, q, v, *args, **kwargs):
        return _verify_kv(q, v)

    def chunk_gla_fwd_o_gk(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_fwd_o_gk_npu
        return chunk_gla_fwd_o_gk_npu(*args, **kwargs)

    def chunk_gla_bwd_dA_verifier(self, v, do, *args, **kwargs):
        return _verify_kv(v, do)

    def chunk_gla_bwd_dA(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_bwd_dA_npu
        return chunk_gla_bwd_dA_npu(*args, **kwargs)

    def chunk_gla_bwd_dv_verifier(self, k, g, A, do, dh, *args, **kwargs):
        return _verify_kv(k, do)

    def chunk_gla_bwd_dv(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_bwd_dv_npu
        return chunk_gla_bwd_dv_npu(*args, **kwargs)

    def chunk_gla_bwd_dqk_intra_verifier(self, q, k, *args, **kwargs):
        return _verify_kv(k)

    def chunk_gla_bwd_dqk_intra(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_bwd_dqk_intra_npu
        return chunk_gla_bwd_dqk_intra_npu(*args, **kwargs)

    def chunk_gla_bwd_dqkg_verifier(self, q, k, v, *args, **kwargs):
        return _verify_kv(k, v)

    def chunk_gla_bwd_dqkg(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_bwd_dqkg_npu
        return chunk_gla_bwd_dqkg_npu(*args, **kwargs)
