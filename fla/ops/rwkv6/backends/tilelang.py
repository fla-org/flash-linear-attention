# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang implementation of RWKV6 intra-chunk forward A construction."""

import tilelang
import tilelang.language as T
import torch

from fla.backends import TileLangBackend, register
from fla.ops.rwkv6.chunk import chunk_rwkv6_fwd_intra


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def _build_rwkv6_fwd_intra_kernel(
    B,
    H,
    K,
    BT,
    BC,
    BK,
    dtype_str,
    num_warps=1,
):
    dtype_map = {'float16': T.float16, 'bfloat16': T.bfloat16, 'float32': T.float32}
    dtype = dtype_map[dtype_str]
    accum_dtype = T.float32
    NC = tilelang.cdiv(BT, BC)
    threads = num_warps * 32

    _B, _H, _K, _BT, _BC, _BK, _NC = B, H, K, BT, BC, BK, NC
    _dtype = dtype
    _threads = threads

    T_d = T.dynamic("T")

    qk_s = (_B, T_d, _H, _K)
    A_s = (_B, T_d, _H, _BT)
    u_s = (_H, _K)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_s, _dtype),
        k: T.Tensor(qk_s, _dtype),
        gi: T.Tensor(qk_s, accum_dtype),
        ge: T.Tensor(qk_s, accum_dtype),
        u: T.Tensor(u_s, _dtype),
        A: T.Tensor(A_s, accum_dtype),
        scale: accum_dtype,
    ):
        with T.Kernel(T.ceildiv(T_d, _BT), _NC * _NC, _B * _H, threads=_threads) as (i_t, i_c, i_bh):
            i_b = i_bh // _H
            i_h = i_bh % _H
            i_i = i_c // _NC
            i_j = i_c % _NC
            t_s = i_t * _BT
            q_s = t_s + i_i * _BC
            k_s = t_s + i_j * _BC

            q_frag = T.alloc_fragment((_BC, _BK), accum_dtype)
            k_shared = T.alloc_shared((_BC, _BK), accum_dtype)
            q_diag_frag = T.alloc_fragment((_BC, _BK), accum_dtype)
            ku_diag_shared = T.alloc_shared((_BC, _BK), accum_dtype)
            acc = T.alloc_fragment((_BC, _BC), accum_dtype)
            diag_acc = T.alloc_fragment((_BC, _BC), accum_dtype)
            diag_shared = T.alloc_shared((_BC, _BC), accum_dtype)
            gn_frag = T.alloc_fragment((_K,), accum_dtype)
            T.clear(acc)
            T.clear(diag_acc)

            if i_i < i_j:
                for i, j in T.Parallel(_BC, _BC):
                    A[i_b, q_s + i, i_h, i_j * _BC + j] = 0.0
            else:
                # Block-local reference for the decay exponents. gi/ge are sequence-global
                # cumulative log2 decays, so raw exp2(ge) / exp2(-gi) overflow or underflow
                # fp32 on long sequences. Any finite center cancels in the product and keeps
                # both factors bounded by the intra-chunk decay range.
                gn_pos = T.if_then_else(t_s + i_i * _BC > 0, t_s + i_i * _BC - 1, 0)
                for k_i in T.Parallel(_K):
                    gn_frag[k_i] = gi[i_b, gn_pos, i_h, k_i]
                for k_blk in T.Pipelined(T.ceildiv(_K, _BK), num_stages=2):
                    for i, k_i in T.Parallel(_BC, _BK):
                        k_idx = k_blk * _BK + k_i
                        q_frag[i, k_i] = (
                            T.cast(q[i_b, q_s + i, i_h, k_idx], accum_dtype) *
                            T.exp2(ge[i_b, q_s + i, i_h, k_idx] - gn_frag[k_idx]) * scale
                        )
                    for j, k_i in T.Parallel(_BC, _BK):
                        k_idx = k_blk * _BK + k_i
                        k_shared[j, k_i] = (
                            T.cast(k[i_b, k_s + j, i_h, k_idx], accum_dtype) *
                            T.exp2(gn_frag[k_idx] - gi[i_b, k_s + j, i_h, k_idx])
                        )
                    T.gemm(q_frag, k_shared, acc, transpose_B=True)

                    if i_i == i_j:
                        for i, k_i in T.Parallel(_BC, _BK):
                            k_idx = k_blk * _BK + k_i
                            q_diag_frag[i, k_i] = T.cast(q[i_b, q_s + i, i_h, k_idx], accum_dtype)
                            ku_diag_shared[i, k_i] = (
                                T.cast(k[i_b, q_s + i, i_h, k_idx], accum_dtype) *
                                T.cast(u[i_h, k_idx], accum_dtype)
                            )
                        T.gemm(q_diag_frag, ku_diag_shared, diag_acc, transpose_B=True)

                if i_i == i_j:
                    T.copy(diag_acc, diag_shared)

                for i, j in T.Parallel(_BC, _BC):
                    lower_value = T.if_then_else(i_i > i_j, acc[i, j], 0.0)
                    diag_lower = T.if_then_else(i > j, acc[i, j], 0.0)
                    diag_value = T.if_then_else(i == j, diag_shared[i, i] * scale, diag_lower)
                    A[i_b, q_s + i, i_h, i_j * _BC + j] = T.if_then_else(i_i == i_j, diag_value, lower_value)

    return kernel


def chunk_rwkv6_fwd_intra_verifier(
    q: torch.Tensor,
    k: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    u: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[bool, str | None]:
    if cu_seqlens is not None or chunk_indices is not None:
        return False, "TileLang RWKV6 intra backend supports dense inputs only"
    if chunk_size != 64:
        return False, f"TileLang RWKV6 intra backend requires chunk_size=64, got {chunk_size}"
    if q.dtype != torch.bfloat16:
        return False, f"TileLang RWKV6 intra backend supports benchmark dtype bfloat16 only, got {q.dtype}"
    if k.dtype != q.dtype or u.dtype != q.dtype:
        return False, f"TileLang RWKV6 intra backend requires q/k/u dtype match, got {q.dtype}/{k.dtype}/{u.dtype}"
    if gi.dtype != torch.float32 or ge.dtype != torch.float32:
        return False, f"TileLang RWKV6 intra backend requires fp32 gi/ge, got {gi.dtype}/{ge.dtype}"
    if not q.is_cuda:
        return False, "TileLang RWKV6 intra backend requires CUDA tensors"
    if q.shape != k.shape or q.shape != gi.shape or q.shape != ge.shape:
        return False, "TileLang RWKV6 intra backend requires q, k, gi, and ge to have identical shapes"
    if q.ndim != 4:
        return False, f"TileLang RWKV6 intra backend expects q with rank 4, got {q.ndim}"
    B, T, H, K = q.shape
    if u.shape != (H, K):
        return False, f"TileLang RWKV6 intra backend requires u shape {(H, K)}, got {tuple(u.shape)}"
    if T % chunk_size != 0:
        return False, f"TileLang RWKV6 intra backend requires T divisible by {chunk_size}, got T={T}"
    if K != 64:
        return False, f"TileLang RWKV6 intra backend currently supports the D=64 benchmark bucket only, got K={K}"
    if B <= 0 or H <= 0:
        return False, f"TileLang RWKV6 intra backend requires positive B/H, got B={B}, H={H}"
    return True, None


@register(
    chunk_rwkv6_fwd_intra,
    backend=TileLangBackend,
    verifier=chunk_rwkv6_fwd_intra_verifier,
    default_enable=False,
)
def chunk_rwkv6_fwd_intra_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    u: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T_seq, H, K = q.shape
    BT = chunk_size
    BC = 16
    BK = 32
    dtype_str = {torch.bfloat16: 'bfloat16'}[q.dtype]

    A = torch.empty(B, T_seq, H, BT, dtype=torch.float32, device=q.device)

    kernel = _build_rwkv6_fwd_intra_kernel(
        B,
        H,
        K,
        BT,
        BC,
        BK,
        dtype_str,
        num_warps=1,
    )
    kernel(q, k, gi, ge, u, A, scale)
    return A
