# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for KDA ops."""

from __future__ import annotations

from fla.ops.backends import BaseBackend

_SUPPORTED_INTRA_CHUNK_SIZES = (32, 64)
_SUB_CHUNK = 16


def _chunk_size_from(args, kwargs, default: int = 64) -> int:
    if 'chunk_size' in kwargs and kwargs['chunk_size'] is not None:
        return int(kwargs['chunk_size'])
    return default


def _verify_intra_chunk_size(args, kwargs):
    chunk_size = _chunk_size_from(args, kwargs)
    if chunk_size not in _SUPPORTED_INTRA_CHUNK_SIZES:
        return False, f'KDA Ascend intra only supports chunk_size in {_SUPPORTED_INTRA_CHUNK_SIZES}, got {chunk_size}'
    return True, None


def _verify_subchunk_aligned(args, kwargs):
    chunk_size = _chunk_size_from(args, kwargs)
    if chunk_size % _SUB_CHUNK != 0:
        return False, f'KDA Ascend bwd requires chunk_size % {_SUB_CHUNK} == 0, got {chunk_size}'
    return True, None


class TritonAscendKDABackend(BaseBackend):
    """Ascend NPU backend for KDA gate, intra, WY, and backward kernels."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_kda_fwd_intra_verifier(self, *args, **kwargs):
        return _verify_intra_chunk_size(args, kwargs)

    def chunk_kda_fwd_intra(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_intra import chunk_kda_fwd_intra_npu
        return chunk_kda_fwd_intra_npu(*args, **kwargs)

    def chunk_kda_fwd_intra_token_parallel_verifier(self, *args, **kwargs):
        return _verify_intra_chunk_size(args, kwargs)

    def chunk_kda_fwd_intra_token_parallel(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_intra_token_parallel import (
            chunk_kda_fwd_intra_token_parallel_npu,
        )
        return chunk_kda_fwd_intra_token_parallel_npu(*args, **kwargs)

    def recompute_w_u_fwd_verifier(self, *args, **kwargs):
        return True, None

    def recompute_w_u_fwd(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.wy_fast import recompute_w_u_fwd_kda_npu
        return recompute_w_u_fwd_kda_npu(*args, **kwargs)

    def chunk_kda_bwd_intra_verifier(self, *args, **kwargs):
        return _verify_intra_chunk_size(args, kwargs)

    def chunk_kda_bwd_intra(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_intra import chunk_kda_bwd_intra_npu
        return chunk_kda_bwd_intra_npu(*args, **kwargs)

    def chunk_kda_bwd_wy_dqkg_fused_verifier(self, *args, **kwargs):
        return _verify_subchunk_aligned(args, kwargs)

    def chunk_kda_bwd_wy_dqkg_fused(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused_npu
        return chunk_kda_bwd_wy_dqkg_fused_npu(*args, **kwargs)

    def chunk_kda_bwd_dAv_verifier(self, *args, **kwargs):
        return True, None

    def chunk_kda_bwd_dAv(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_bwd import chunk_kda_bwd_dAv_npu
        return chunk_kda_bwd_dAv_npu(*args, **kwargs)

    def kda_gate_fwd_verifier(self, *args, **kwargs):
        return True, None

    def kda_gate_fwd(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.gate import kda_gate_fwd_npu
        return kda_gate_fwd_npu(*args, **kwargs)

    def kda_gate_bwd_verifier(self, *args, **kwargs):
        return True, None

    def kda_gate_bwd(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.gate import kda_gate_bwd_npu
        return kda_gate_bwd_npu(*args, **kwargs)

    def kda_gate_chunk_cumsum_verifier(self, *args, **kwargs):
        return True, None

    def kda_gate_chunk_cumsum(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.gate import kda_gate_chunk_cumsum_npu
        return kda_gate_chunk_cumsum_npu(*args, **kwargs)

    def fused_kda_gate_verifier(self, *args, **kwargs):
        return True, None

    def fused_kda_gate(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.gate import fused_kda_gate_npu
        return fused_kda_gate_npu(*args, **kwargs)
