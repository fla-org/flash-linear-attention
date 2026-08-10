# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for GLA ops."""

from __future__ import annotations

from fla.ops.backends import BaseBackend

# Flip per-op after numerical smoke passes. Unready ops fall back to CUDA kernels.
_READY = {
    'chunk_gla_fwd_intra_gk': True,
    'chunk_gla_fwd_o_gk': True,
    'chunk_gla_bwd_dA': True,
    'chunk_gla_bwd_dv': True,
    'chunk_gla_bwd_dqk_intra': True,
    'chunk_gla_bwd_dqkg': True,
}


class TritonAscendGLABackend(BaseBackend):
    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def _gate(self, name: str):
        if not _READY.get(name, False):
            return False, f'{name} NPU path not ready'
        return True, None

    def chunk_gla_fwd_intra_gk_verifier(self, *args, **kwargs):
        return self._gate('chunk_gla_fwd_intra_gk')

    def chunk_gla_fwd_intra_gk(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_fwd_intra_gk_npu
        return chunk_gla_fwd_intra_gk_npu(*args, **kwargs)

    def chunk_gla_fwd_o_gk_verifier(self, *args, **kwargs):
        return self._gate('chunk_gla_fwd_o_gk')

    def chunk_gla_fwd_o_gk(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_fwd_o_gk_npu
        return chunk_gla_fwd_o_gk_npu(*args, **kwargs)

    def chunk_gla_bwd_dA_verifier(self, *args, **kwargs):
        return self._gate('chunk_gla_bwd_dA')

    def chunk_gla_bwd_dA(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_bwd_dA_npu
        return chunk_gla_bwd_dA_npu(*args, **kwargs)

    def chunk_gla_bwd_dv_verifier(self, *args, **kwargs):
        return self._gate('chunk_gla_bwd_dv')

    def chunk_gla_bwd_dv(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_bwd_dv_npu
        return chunk_gla_bwd_dv_npu(*args, **kwargs)

    def chunk_gla_bwd_dqk_intra_verifier(self, *args, **kwargs):
        return self._gate('chunk_gla_bwd_dqk_intra')

    def chunk_gla_bwd_dqk_intra(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_bwd_dqk_intra_npu
        return chunk_gla_bwd_dqk_intra_npu(*args, **kwargs)

    def chunk_gla_bwd_dqkg_verifier(self, *args, **kwargs):
        return self._gate('chunk_gla_bwd_dqkg')

    def chunk_gla_bwd_dqkg(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk import chunk_gla_bwd_dqkg_npu
        return chunk_gla_bwd_dqkg_npu(*args, **kwargs)
