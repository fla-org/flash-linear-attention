# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for KDA ops."""

from __future__ import annotations

import torch
import triton

from fla.ops.backends import BaseBackend

_SUPPORTED_INTRA_CHUNK_SIZES = (32, 64)
_SUB_CHUNK = 16


def _verify_intra_chunk_size(chunk_size: int = 64):
    if chunk_size not in _SUPPORTED_INTRA_CHUNK_SIZES:
        return False, f'KDA Ascend intra only supports chunk_size in {_SUPPORTED_INTRA_CHUNK_SIZES}, got {chunk_size}'
    return True, None


def _verify_subchunk_aligned(chunk_size: int = 64):
    if chunk_size % _SUB_CHUNK != 0:
        return False, f'KDA Ascend bwd requires chunk_size % {_SUB_CHUNK} == 0, got {chunk_size}'
    return True, None


def _verify_bwd_intra_k(k: torch.Tensor, safe_gate: bool = False):
    # beyond the single-block UB envelope dispatch falls back to the mainline kernel
    K = int(k.shape[-1])
    limit = 512 if safe_gate else 256
    if triton.next_power_of_2(K) > limit:
        return False, f'KDA Ascend bwd_intra only supports K <= {limit} with safe_gate={safe_gate}, got K={K}'
    return True, None


def _verify_kv(k, v=None, extra=()):
    from fla.utils import npu_verify_kv

    extra = tuple(t for t in extra if t is not None)
    # Some KDA ops (e.g. chunk_kda_bwd_intra) have k/g but no v.
    if v is None:
        return _verify_k_tensor(k, extra=extra)
    return npu_verify_kv(k, v, extra=extra)


def _verify_k_tensor(t: torch.Tensor, *, extra=()) -> tuple[bool, str | None]:
    from fla.utils import npu_verify_last_dim_tensor

    ok, reason = npu_verify_last_dim_tensor(t, label='K')
    if not ok:
        return ok, reason
    for tensor in extra:
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return False, f'unsupported dtype for NPU KDA kernels: {tensor.dtype}'
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

    def chunk_kda_fwd_intra_verifier(self, q, k, v, gk=None, chunk_size=64, **kwargs):
        ok, reason = _verify_intra_chunk_size(chunk_size)
        if not ok:
            return ok, reason
        extra = tuple(t for t in (q, gk) if t is not None)
        return _verify_kv(k, v, extra=extra)

    def chunk_kda_fwd_intra(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_intra import chunk_kda_fwd_intra_npu
        return chunk_kda_fwd_intra_npu(*args, **kwargs)

    def chunk_kda_fwd_intra_token_parallel_verifier(self, q, k, gk, chunk_size=64, **kwargs):
        # Signature is (q, k, gk, ...); there is no v tensor.
        ok, reason = _verify_intra_chunk_size(chunk_size)
        if not ok:
            return ok, reason
        extra = tuple(t for t in (q, gk) if t is not None)
        return _verify_k_tensor(k, extra=extra)

    def chunk_kda_fwd_intra_token_parallel(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_intra_token_parallel import (
            chunk_kda_fwd_intra_token_parallel_npu,
        )
        return chunk_kda_fwd_intra_token_parallel_npu(*args, **kwargs)

    def recompute_w_u_fwd_verifier(self, k, v, beta, A, gk=None, q=None, **kwargs):
        extra = tuple(t for t in (beta, A, gk, q) if t is not None)
        return _verify_kv(k, v, extra=extra)

    def recompute_w_u_fwd(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.wy_fast import recompute_w_u_fwd_kda_npu
        return recompute_w_u_fwd_kda_npu(*args, **kwargs)

    def chunk_kda_bwd_intra_verifier(self, q, k, g, chunk_size=64, safe_gate=False, **kwargs):
        ok, reason = _verify_intra_chunk_size(chunk_size)
        if not ok:
            return ok, reason
        ok, reason = _verify_bwd_intra_k(k, safe_gate=safe_gate)
        if not ok:
            return ok, reason
        extra = tuple(t for t in (q, g) if t is not None)
        return _verify_k_tensor(k, extra=extra)

    def chunk_kda_bwd_intra(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_intra import chunk_kda_bwd_intra_npu
        return chunk_kda_bwd_intra_npu(*args, **kwargs)

    def chunk_kda_bwd_wy_dqkg_fused_verifier(self, q, k, v, chunk_size=64, **kwargs):
        ok, reason = _verify_subchunk_aligned(chunk_size)
        if not ok:
            return ok, reason
        return _verify_kv(k, v, extra=(q,) if q is not None else ())

    def chunk_kda_bwd_wy_dqkg_fused(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused_npu
        return chunk_kda_bwd_wy_dqkg_fused_npu(*args, **kwargs)

    def chunk_kda_bwd_dAv_verifier(self, q, k, v, do=None, **kwargs):
        extra = tuple(t for t in (q, do) if t is not None)
        return _verify_kv(k, v, extra=extra)

    def chunk_kda_bwd_dAv(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_bwd import chunk_kda_bwd_dAv_npu
        return chunk_kda_bwd_dAv_npu(*args, **kwargs)

    def kda_gate_fwd_verifier(self, g, **kwargs):
        return _verify_k_tensor(g)

    def kda_gate_fwd(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.gate import kda_gate_fwd_npu
        return kda_gate_fwd_npu(*args, **kwargs)

    def kda_gate_bwd_verifier(self, g, A_log=None, dt_bias=None, dyg=None, **kwargs):
        extra = (dyg,) if dyg is not None else ()
        return _verify_k_tensor(g, extra=extra)

    def kda_gate_bwd(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.gate import kda_gate_bwd_npu
        return kda_gate_bwd_npu(*args, **kwargs)

    def kda_gate_chunk_cumsum_verifier(self, g, **kwargs):
        return _verify_k_tensor(g)

    def kda_gate_chunk_cumsum(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.gate import kda_gate_chunk_cumsum_npu
        return kda_gate_chunk_cumsum_npu(*args, **kwargs)

    def fused_kda_gate_verifier(self, g, **kwargs):
        return _verify_k_tensor(g)

    def fused_kda_gate(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.gate import fused_kda_gate_npu
        return fused_kda_gate_npu(*args, **kwargs)

    def fused_recurrent_kda_fwd_verifier(self, q, k, v, **kwargs):
        return _verify_kv(k, v, extra=(q,) if q is not None else ())

    def fused_recurrent_kda_fwd(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.fused_recurrent import fused_recurrent_kda_fwd_npu

        return fused_recurrent_kda_fwd_npu(*args, **kwargs)
