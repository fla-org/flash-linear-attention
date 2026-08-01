# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""DPLR WY representation forward.

Two stages:
1. `_prepare_wy_repr_fwd_kernel`: invert the strictly lower-triangular A_ab
   in-place per chunk via the iterative formula M_{k+1} = M_k + e_i (e_i^T M_k)
   for i = 1..BT-1, then add identity. Yields fp32 A_ab_inv, matching FLA's
   public dtype boundary.
2. `_wu_fwd_kernel`: compute `b_Aak = A_ab_inv @ A_ak` (causal masks applied),
   then `w = A_ab_inv @ ag` and `u = b_Aak @ v`.

Adapted from FLA's prepare_wy_repr_fwd_kernel_chunk32 + wu_fwd_kernel.
"""

import tilelang
import tilelang.language as T
import torch

from .schedules import device_cc
from .utils import ChunkLayout, build_rect_chunk_layout, build_varlen_chunk_layout

_WY_INV_CONFIGS = [
    {"threads": 32},
]


def _wu_fwd_config(BT: int, device: torch.device) -> dict[str, int]:
    # 256 threads + bulk copies pay at BT=64 on cc90 and cc120 alike
    # (kernel-level 1.7-2.1x over the scalar path on both); they measure
    # flat at BT=32 on cc120, so the gate stays BT>=64.
    if BT <= 16:
        return {"threads": 32, "bulk_copy": False}
    if device_cc(device) in (90, 120) and BT >= 64:
        return {"threads": 256, "bulk_copy": True}
    return {"threads": 128, "bulk_copy": False}


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
                  tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
                  },
)
def _prepare_wy_repr_fwd_kernel(H, BT, in_dtype, threads: int = 32):
    acc_dtype = "float32"
    n_tokens, n_seq_plus_one, n_chunks = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks"
    )

    @T.prim_func
    def prepare_wy_repr_fwd_tl(
        A_ab: T.Tensor((n_tokens, H, BT), "float16"),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_indices: T.Tensor((n_chunks, 2), "int32"),
        A_ab_inv: T.Tensor((n_tokens, H, BT), acc_dtype),
    ):
        with T.Kernel(n_chunks, H, threads=threads) as (i_c, i_h):
            i_n = chunk_indices[i_c, 0]
            i_t = chunk_indices[i_c, 1]
            safe_i_n = T.max(i_n, 0)
            seq_bos = cu_seqlens[safe_i_n]
            seq_eos = cu_seqlens[safe_i_n + 1]
            bos_raw = seq_bos + i_t * BT
            eos_raw = T.min(bos_raw + BT, seq_eos)
            is_valid_chunk = i_n >= 0
            bos = T.if_then_else(is_valid_chunk, bos_raw, T.int32(0))
            eos = T.if_then_else(is_valid_chunk, eos_raw, T.int32(0))

            M = T.alloc_shared((BT, BT), acc_dtype)
            v = T.alloc_shared((BT,), acc_dtype)
            v_new = T.alloc_fragment((BT,), acc_dtype)
            T.clear(v_new)

            # Load A_ab, mask to strict lower triangular.
            for r, c in T.Parallel(BT, BT):
                t = bos + r
                if (r > c) and (t < eos):
                    M[r, c] = A_ab[t, i_h, c]
                else:
                    M[r, c] = 0.0

            # Iterative inversion: for i in 1..BT-1. The j-reduction runs
            # serially so each c-lane accumulator stays private to one thread.
            for i in T.serial(BT - 1):
                row_i = i + 1
                for c in T.Parallel(BT):
                    v[c] = M[row_i, c]

                for j in T.serial(BT):
                    for c in T.Parallel(BT):
                        if c < j:
                            v_new[c] = v_new[c] + v[j] * M[j, c]

                for c in T.Parallel(BT):
                    if c < row_i:
                        M[row_i, c] = v[c] + v_new[c]
                    v_new[c] = 0.0

            # Add identity to diagonal
            for r, c in T.Parallel(BT, BT):
                if r == c:
                    M[r, c] = M[r, c] + 1.0

            # Store
            for r, c in T.Parallel(BT, BT):
                t = bos + r
                if t < eos:
                    A_ab_inv[t, i_h, c] = M[r, c]

    return prepare_wy_repr_fwd_tl


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
                  tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
                  },
)
def _prepare_wy_repr_fwd_kernel64(H, in_dtype, threads: int = 32):
    acc_dtype = "float32"
    BT = 64
    BC = 32
    n_tokens, n_seq_plus_one, n_chunks = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks"
    )

    @T.prim_func
    def prepare_wy_repr_fwd64_tl(
        A_ab: T.Tensor((n_tokens, H, BT), "float16"),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_indices: T.Tensor((n_chunks, 2), "int32"),
        A_ab_inv: T.Tensor((n_tokens, H, BT), acc_dtype),
    ):
        with T.Kernel(n_chunks, H, threads=threads) as (i_c, i_h):
            i_n = chunk_indices[i_c, 0]
            i_t = chunk_indices[i_c, 1]
            safe_i_n = T.max(i_n, 0)
            seq_bos = cu_seqlens[safe_i_n]
            seq_eos = cu_seqlens[safe_i_n + 1]
            bos_raw = seq_bos + i_t * BT
            eos_raw = T.min(bos_raw + BT, seq_eos)
            is_valid_chunk = i_n >= 0
            bos = T.if_then_else(is_valid_chunk, bos_raw, T.int32(0))
            eos = T.if_then_else(is_valid_chunk, eos_raw, T.int32(0))

            A1 = T.alloc_shared((BC, BC), acc_dtype)
            A2 = T.alloc_shared((BC, BC), acc_dtype)
            A3 = T.alloc_shared((BC, BC), acc_dtype)
            tmp = T.alloc_fragment((BC, BC), acc_dtype)
            tmp_shared = T.alloc_shared((BC, BC), acc_dtype)
            A3_out = T.alloc_fragment((BC, BC), acc_dtype)
            v = T.alloc_shared((BC,), acc_dtype)
            v_new = T.alloc_fragment((BC,), acc_dtype)
            T.clear(v_new)

            # FLA's chunk64 path inverts the two 32x32 diagonal blocks
            # independently, then forms the bottom-left block with two GEMMs.
            for r, c in T.Parallel(BC, BC):
                t1 = bos + r
                t2 = bos + BC + r
                if (r > c) and (t1 < eos):
                    A1[r, c] = A_ab[t1, i_h, c]
                else:
                    A1[r, c] = 0.0
                if (r > c) and (t2 < eos):
                    A2[r, c] = A_ab[t2, i_h, BC + c]
                else:
                    A2[r, c] = 0.0
                if t2 < eos:
                    A3[r, c] = A_ab[t2, i_h, c]
                else:
                    A3[r, c] = 0.0

            for i in T.serial(BC - 1):
                row_i = i + 1
                for c in T.Parallel(BC):
                    v[c] = A1[row_i, c]
                for j in T.serial(BC):
                    for c in T.Parallel(BC):
                        if c < j:
                            v_new[c] = v_new[c] + v[j] * A1[j, c]
                for c in T.Parallel(BC):
                    if c < row_i:
                        A1[row_i, c] = v[c] + v_new[c]
                    v_new[c] = 0.0

            T.clear(v_new)
            for i in T.serial(BC - 1):
                row_i = i + 1
                for c in T.Parallel(BC):
                    v[c] = A2[row_i, c]
                for j in T.serial(BC):
                    for c in T.Parallel(BC):
                        if c < j:
                            v_new[c] = v_new[c] + v[j] * A2[j, c]
                for c in T.Parallel(BC):
                    if c < row_i:
                        A2[row_i, c] = v[c] + v_new[c]
                    v_new[c] = 0.0

            for r, c in T.Parallel(BC, BC):
                if r == c:
                    A1[r, c] = A1[r, c] + 1.0
                    A2[r, c] = A2[r, c] + 1.0

            T.gemm(A2, A3, tmp, clear_accum=True)
            T.copy(tmp, tmp_shared)
            T.gemm(tmp_shared, A1, A3_out, clear_accum=True)

            for r, c in T.Parallel(BC, BC):
                t1 = bos + r
                t2 = bos + BC + r
                if t1 < eos:
                    A_ab_inv[t1, i_h, c] = A1[r, c]
                    A_ab_inv[t1, i_h, BC + c] = 0.0
                if t2 < eos:
                    A_ab_inv[t2, i_h, c] = A3_out[r, c]
                    A_ab_inv[t2, i_h, BC + c] = A2[r, c]

    return prepare_wy_repr_fwd64_tl


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
                  tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
                  },
)
def _wu_fwd_kernel(H, K, V, BT, in_dtype, threads: int = 32, bulk_copy: bool = False):
    acc_dtype = "float32"
    n_tokens, n_seq_plus_one, n_chunks = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks"
    )

    @T.prim_func
    def wu_fwd_tl(
        ag: T.Tensor((n_tokens, H, K), in_dtype),
        v: T.Tensor((n_tokens, H, V), in_dtype),
        A_ab_inv: T.Tensor((n_tokens, H, BT), acc_dtype),
        A_ak: T.Tensor((n_tokens, H, BT), "float16"),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_indices: T.Tensor((n_chunks, 2), "int32"),
        w: T.Tensor((n_tokens, H, K), in_dtype),
        u: T.Tensor((n_tokens, H, V), in_dtype),
    ):
        with T.Kernel(n_chunks, H, threads=threads) as (i_c, i_h):
            i_n = chunk_indices[i_c, 0]
            i_t = chunk_indices[i_c, 1]
            safe_i_n = T.max(i_n, 0)
            seq_bos = cu_seqlens[safe_i_n]
            seq_eos = cu_seqlens[safe_i_n + 1]
            bos_raw = seq_bos + i_t * BT
            eos_raw = T.min(bos_raw + BT, seq_eos)
            is_valid_chunk = i_n >= 0
            bos = T.if_then_else(is_valid_chunk, bos_raw, T.int32(0))
            eos = T.if_then_else(is_valid_chunk, eos_raw, T.int32(0))

            # FLA keeps A_ab_inv @ A_ak in fp32/tf32, then downcasts the
            # processed Aak before the u = Aak @ v GEMM. Keep separate
            # fp32 and input-dtype views so w remains aligned with FLA's
            # bf16 A_ab_inv @ ag path.
            A_inv_acc_shared = T.alloc_shared((BT, BT), acc_dtype)
            A_ak_acc_shared = T.alloc_shared((BT, BT), acc_dtype)
            A_inv_shared = T.alloc_shared((BT, BT), in_dtype)
            Aak_processed_frag = T.alloc_fragment((BT, BT), acc_dtype)
            Aak_processed_shared = T.alloc_shared((BT, BT), in_dtype)
            ag_shared = T.alloc_shared((BT, K), in_dtype)
            v_shared = T.alloc_shared((BT, V), in_dtype)
            w_frag = T.alloc_fragment((BT, K), acc_dtype)
            u_frag = T.alloc_fragment((BT, V), acc_dtype)

            # Load A_ab_inv (inclusive lower-tri after diagonal-add) and A_ak.
            # Interior chunks bulk-copy and mask in shared; boundary chunks
            # keep the scalar predicated path (same gating as chunk_A_bwd).
            full_tile = (is_valid_chunk and (bos + BT <= eos)) if bulk_copy else False
            if full_tile:
                T.copy(A_ab_inv[bos: bos + BT, i_h, 0:BT], A_inv_acc_shared)
                T.copy(A_ab_inv[bos: bos + BT, i_h, 0:BT], A_inv_shared)
                T.copy(A_ak[bos: bos + BT, i_h, 0:BT], A_ak_acc_shared)
                for r, c in T.Parallel(BT, BT):
                    if r < c:
                        A_inv_acc_shared[r, c] = 0.0
                        A_inv_shared[r, c] = T.Cast(in_dtype, 0.0)
                    if r <= c:
                        A_ak_acc_shared[r, c] = 0.0
            else:
                for r, c in T.Parallel(BT, BT):
                    t = bos + r
                    if (t < eos) and (r >= c):
                        A_inv_acc_shared[r, c] = A_ab_inv[t, i_h, c]
                        A_inv_shared[r, c] = T.Cast(in_dtype, A_ab_inv[t, i_h, c])
                    else:
                        A_inv_acc_shared[r, c] = 0.0
                        A_inv_shared[r, c] = T.Cast(in_dtype, 0.0)
                    if (t < eos) and (r > c):
                        A_ak_acc_shared[r, c] = A_ak[t, i_h, c]
                    else:
                        A_ak_acc_shared[r, c] = 0.0

            T.gemm(A_inv_acc_shared, A_ak_acc_shared, Aak_processed_frag, clear_accum=True)
            for r, c in T.Parallel(BT, BT):
                Aak_processed_shared[r, c] = T.Cast(in_dtype, Aak_processed_frag[r, c])

            if full_tile:
                T.copy(ag[bos: bos + BT, i_h, 0:K], ag_shared)
            else:
                for r, c in T.Parallel(BT, K):
                    t = bos + r
                    if t < eos:
                        ag_shared[r, c] = ag[t, i_h, c]
                    else:
                        ag_shared[r, c] = T.Cast(in_dtype, 0.0)
            T.gemm(A_inv_shared, ag_shared, w_frag, clear_accum=True)
            if full_tile:
                for r, c in T.Parallel(BT, K):
                    w[bos + r, i_h, c] = T.Cast(in_dtype, w_frag[r, c])
            else:
                for r, c in T.Parallel(BT, K):
                    t = bos + r
                    if t < eos:
                        w[t, i_h, c] = T.Cast(in_dtype, w_frag[r, c])

            if full_tile:
                T.copy(v[bos: bos + BT, i_h, 0:V], v_shared)
            else:
                for r, c in T.Parallel(BT, V):
                    t = bos + r
                    if t < eos:
                        v_shared[r, c] = v[t, i_h, c]
                    else:
                        v_shared[r, c] = T.Cast(in_dtype, 0.0)
            T.gemm(Aak_processed_shared, v_shared, u_frag, clear_accum=True)
            if full_tile:
                for r, c in T.Parallel(BT, V):
                    u[bos + r, i_h, c] = T.Cast(in_dtype, u_frag[r, c])
            else:
                for r, c in T.Parallel(BT, V):
                    t = bos + r
                    if t < eos:
                        u[t, i_h, c] = T.Cast(in_dtype, u_frag[r, c])

    return wu_fwd_tl


def prepare_wy_repr_fwd(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 16,
    chunk_layout: ChunkLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T_, H, K = ag.shape
    V = v.shape[-1]
    BT = chunk_size
    is_varlen = cu_seqlens is not None

    if is_varlen:
        assert B == 1
        layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, BT, T_)
    else:
        layout = build_rect_chunk_layout(B, T_, BT, ag.device)
    N_tokens = B * T_
    in_dtype = str(ag.dtype).split(".")[-1]

    A_ab_f = A_ab.reshape(N_tokens, H, BT).contiguous()
    A_ak_f = A_ak.reshape(N_tokens, H, BT).contiguous()

    inv_threads = _WY_INV_CONFIGS[0]["threads"]
    if BT == 64:
        inv_kernel = _prepare_wy_repr_fwd_kernel64(
            H, in_dtype, threads=inv_threads,
        )
    else:
        inv_kernel = _prepare_wy_repr_fwd_kernel(
            H, BT, in_dtype, threads=inv_threads,
        )
    A_ab_inv_f = torch.empty((N_tokens, H, BT), dtype=torch.float32, device=ag.device)
    inv_kernel(A_ab_f, layout.cu_seqlens, layout.chunk_indices, A_ab_inv_f)

    ag_f = ag.reshape(N_tokens, H, K).contiguous()
    v_f = v.reshape(N_tokens, H, V).contiguous()

    wu_kernel = _wu_fwd_kernel(
        H, K, V, BT, in_dtype,
        **_wu_fwd_config(BT, ag.device),
    )
    w_f = torch.empty((N_tokens, H, K), dtype=ag.dtype, device=ag.device)
    u_f = torch.empty((N_tokens, H, V), dtype=v.dtype, device=v.device)
    wu_kernel(ag_f, v_f, A_ab_inv_f, A_ak_f, layout.cu_seqlens, layout.chunk_indices, w_f, u_f)

    w = w_f.view(B, T_, H, K)
    u = u_f.view(B, T_, H, V)
    A_ab_inv = A_ab_inv_f.view(B, T_, H, BT)
    return w, u, A_ab_inv
