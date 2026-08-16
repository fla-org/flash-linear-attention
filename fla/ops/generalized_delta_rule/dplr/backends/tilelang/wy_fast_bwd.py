# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""DPLR WY-representation backward.

Forward (from wy_fast_fwd.py):
    A_ak_processed = A_ab_inv @ A_ak              (strict-lower-tri output)
    w = A_ab_inv @ ag
    u = A_ak_processed @ v

Backward (this kernel):
    1.  dA_ak_processed = du @ v^T              (strict-lower-tri)
        dv          = dv0 + A_ak_processed^T @ du
    2.  dA_ak       = A_ab_inv^T @ dA_ak_processed     (strict-lower-tri)
    3.  dA_ab_inv   = dA_ak_processed @ A_ak^T
    4.  dag         = A_ab_inv^T @ dw
        dA_ab_inv += dw @ ag^T
    5.  dA_ab       = strict_lower(A_ab_inv^T @ inclusive_lower(dA_ab_inv) @ A_ab_inv^T)
        (matrix-inverse sensitivity identity)
"""

import tilelang
import tilelang.language as T
import torch

from .schedules import device_cc
from .utils import ChunkLayout, build_rect_chunk_layout, build_varlen_chunk_layout


def _wy_bwd_config(BT: int, device: torch.device) -> dict[str, int]:
    # 256 threads + bulk copies pay at BT=64 on cc90 and cc120 alike
    # (kernel-level 1.5x over the scalar path on both); cc90 also wins at
    # BT=32 with 128 threads, while cc120 measures flat there, so the
    # cc120 gate stays BT>=64.
    if BT <= 16:
        return {"threads": 32, "bulk_copy": False}
    cc = device_cc(device)
    if cc == 90:
        return {"threads": 128 if BT < 64 else 256, "bulk_copy": True}
    if cc == 120 and BT >= 64:
        return {"threads": 256, "bulk_copy": True}
    return {"threads": 128, "bulk_copy": False}


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
                  tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
                  },
)
def _wy_fast_bwd_kernel(
    H, K, V, BT, in_dtype,
    STORE_DV: bool = True,
    threads: int = 128,
    bulk_copy: bool = False,
):
    acc_dtype = "float32"
    n_tokens, n_seq_plus_one, n_chunks = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks"
    )

    @T.prim_func
    def wy_fast_bwd_tl(
        A_ab_inv: T.Tensor((n_tokens, H, BT), acc_dtype),
        A_ak: T.Tensor((n_tokens, H, BT), "float16"),
        ag: T.Tensor((n_tokens, H, K), in_dtype),
        v: T.Tensor((n_tokens, H, V), in_dtype),
        dw: T.Tensor((n_tokens, H, K), in_dtype),
        du: T.Tensor((n_tokens, H, V), in_dtype),
        dv0: T.Tensor((n_tokens, H, V), in_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_indices: T.Tensor((n_chunks, 2), "int32"),
        dA_ab: T.Tensor((n_tokens, H, BT), in_dtype),
        dA_ak: T.Tensor((n_tokens, H, BT), in_dtype),
        dv: T.Tensor((n_tokens, H, V), in_dtype),
        dag: T.Tensor((n_tokens, H, K), in_dtype),
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

            # A matrices in in_dtype for GEMM operands (cast from fp32 input).
            A_ab_inv_shared = T.alloc_shared((BT, BT), in_dtype)
            A_ak_shared = T.alloc_shared((BT, BT), in_dtype)
            A_tmp_shared = T.alloc_shared((BT, BT), in_dtype)
            dA_tmp_frag = T.alloc_fragment((BT, BT), acc_dtype)
            dA_tmp_shared = T.alloc_shared((BT, BT), in_dtype)
            dA_ak_frag = T.alloc_fragment((BT, BT), acc_dtype)
            dA_ab_inv_frag = T.alloc_fragment((BT, BT), acc_dtype)
            dA_ab_inv_shared = T.alloc_shared((BT, BT), in_dtype)
            dA_ab_frag = T.alloc_fragment((BT, BT), acc_dtype)
            tmp_frag = T.alloc_fragment((BT, BT), acc_dtype)
            tmp_shared = T.alloc_shared((BT, BT), in_dtype)
            A_tmp_frag = T.alloc_fragment((BT, BT), acc_dtype)

            v_shared = T.alloc_shared((BT, V), in_dtype)
            du_shared = T.alloc_shared((BT, V), in_dtype)
            if STORE_DV:
                dv0_shared = T.alloc_shared((BT, V), in_dtype)
                dv_frag = T.alloc_fragment((BT, V), acc_dtype)
                dv_shared = T.alloc_shared((BT, V), in_dtype)
            ag_shared = T.alloc_shared((BT, K), in_dtype)
            dw_shared = T.alloc_shared((BT, K), in_dtype)
            dag_frag = T.alloc_fragment((BT, K), acc_dtype)
            dag_shared = T.alloc_shared((BT, K), in_dtype)

            # Load A_ab_inv with inclusive-lower mask, A_ak with strict-lower
            # mask.  Interior chunks bulk-copy and mask in shared; boundary
            # chunks keep the scalar predicated path (see chunk_A_bwd).
            full_tile = (is_valid_chunk and (bos + BT <= eos)) if bulk_copy else False
            if full_tile:
                T.copy(A_ab_inv[bos: bos + BT, i_h, 0:BT], A_ab_inv_shared)
                T.copy(A_ak[bos: bos + BT, i_h, 0:BT], A_ak_shared)
                for r, c in T.Parallel(BT, BT):
                    if r < c:
                        A_ab_inv_shared[r, c] = T.Cast(in_dtype, 0.0)
                    if r <= c:
                        A_ak_shared[r, c] = T.Cast(in_dtype, 0.0)
            else:
                for r, c in T.Parallel(BT, BT):
                    t = bos + r
                    if (t < eos) and (r >= c):
                        A_ab_inv_shared[r, c] = T.Cast(in_dtype, A_ab_inv[t, i_h, c])
                    else:
                        A_ab_inv_shared[r, c] = T.Cast(in_dtype, 0.0)
                    if (t < eos) and (r > c):
                        A_ak_shared[r, c] = T.Cast(in_dtype, A_ak[t, i_h, c])
                    else:
                        A_ak_shared[r, c] = T.Cast(in_dtype, 0.0)

            # A_tmp = A_ab_inv @ A_ak (strict-lower)
            T.gemm(A_ab_inv_shared, A_ak_shared, A_tmp_frag, clear_accum=True)
            T.copy(A_tmp_frag, A_tmp_shared)

            # Load v, du, dv0; compute dA_tmp = du @ v^T and dv = dv0 + A_tmp^T @ du
            if full_tile:
                T.copy(v[bos: bos + BT, i_h, 0:V], v_shared)
                T.copy(du[bos: bos + BT, i_h, 0:V], du_shared)
                if STORE_DV:
                    T.copy(dv0[bos: bos + BT, i_h, 0:V], dv0_shared)
            else:
                for r, c in T.Parallel(BT, V):
                    t = bos + r
                    if t < eos:
                        v_shared[r, c] = v[t, i_h, c]
                        du_shared[r, c] = du[t, i_h, c]
                        if STORE_DV:
                            dv0_shared[r, c] = dv0[t, i_h, c]
                    else:
                        v_shared[r, c] = T.Cast(in_dtype, 0.0)
                        du_shared[r, c] = T.Cast(in_dtype, 0.0)
                        if STORE_DV:
                            dv0_shared[r, c] = T.Cast(in_dtype, 0.0)

            T.gemm(du_shared, v_shared, dA_tmp_frag, transpose_B=True, clear_accum=True)
            if STORE_DV:
                # dv = dv0 + A_tmp^T @ du
                T.gemm(A_tmp_shared, du_shared, dv_frag, transpose_A=True, clear_accum=True)
                for r, c in T.Parallel(BT, V):
                    dv_frag[r, c] = dv_frag[r, c] + T.Cast(acc_dtype, dv0_shared[r, c])
                T.copy(dv_frag, dv_shared)
                if full_tile:
                    for r, c in T.Parallel(BT, V):
                        dv[bos + r, i_h, c] = dv_shared[r, c]
                else:
                    for r, c in T.Parallel(BT, V):
                        t = bos + r
                        if t < eos:
                            dv[t, i_h, c] = dv_shared[r, c]

            # dA_tmp = strict_lower(dA_tmp)
            for r, c in T.Parallel(BT, BT):
                if r > c:
                    dA_tmp_frag[r, c] = dA_tmp_frag[r, c]
                else:
                    dA_tmp_frag[r, c] = 0.0
            T.copy(dA_tmp_frag, dA_tmp_shared)

            # dA_ak = A_ab_inv^T @ dA_tmp (strict-lower)
            T.gemm(A_ab_inv_shared, dA_tmp_shared, dA_ak_frag, transpose_A=True, clear_accum=True)
            if full_tile:
                for r, c in T.Parallel(BT, BT):
                    if r > c:
                        dA_ak[bos + r, i_h, c] = T.Cast(in_dtype, dA_ak_frag[r, c])
                    else:
                        dA_ak[bos + r, i_h, c] = T.Cast(in_dtype, 0.0)
            else:
                for r, c in T.Parallel(BT, BT):
                    t = bos + r
                    if t < eos:
                        if r > c:
                            dA_ak[t, i_h, c] = T.Cast(in_dtype, dA_ak_frag[r, c])
                        else:
                            dA_ak[t, i_h, c] = T.Cast(in_dtype, 0.0)

            # dA_ab_inv = dA_tmp @ A_ak^T
            T.gemm(dA_tmp_shared, A_ak_shared, dA_ab_inv_frag, transpose_B=True, clear_accum=True)

            # Load ag, dw; compute dA_ab_inv += dw @ ag^T; dag = A_ab_inv^T @ dw
            if full_tile:
                T.copy(ag[bos: bos + BT, i_h, 0:K], ag_shared)
                T.copy(dw[bos: bos + BT, i_h, 0:K], dw_shared)
            else:
                for r, c in T.Parallel(BT, K):
                    t = bos + r
                    if t < eos:
                        ag_shared[r, c] = ag[t, i_h, c]
                        dw_shared[r, c] = dw[t, i_h, c]
                    else:
                        ag_shared[r, c] = T.Cast(in_dtype, 0.0)
                        dw_shared[r, c] = T.Cast(in_dtype, 0.0)
            T.gemm(dw_shared, ag_shared, dA_ab_inv_frag, transpose_B=True)
            T.gemm(A_ab_inv_shared, dw_shared, dag_frag, transpose_A=True, clear_accum=True)
            T.copy(dag_frag, dag_shared)
            if full_tile:
                for r, c in T.Parallel(BT, K):
                    dag[bos + r, i_h, c] = dag_shared[r, c]
            else:
                for r, c in T.Parallel(BT, K):
                    t = bos + r
                    if t < eos:
                        dag[t, i_h, c] = dag_shared[r, c]

            # dA_ab = strict_lower(A_ab_inv^T @ inclusive_lower(dA_ab_inv) @ A_ab_inv^T)
            for r, c in T.Parallel(BT, BT):
                if r >= c:
                    dA_ab_inv_shared[r, c] = T.Cast(in_dtype, dA_ab_inv_frag[r, c])
                else:
                    dA_ab_inv_shared[r, c] = T.Cast(in_dtype, 0.0)

            # tmp = A_ab_inv^T @ dA_ab_inv
            T.gemm(A_ab_inv_shared, dA_ab_inv_shared, tmp_frag, transpose_A=True, clear_accum=True)
            T.copy(tmp_frag, tmp_shared)
            # dA_ab = tmp @ A_ab_inv^T
            T.gemm(tmp_shared, A_ab_inv_shared, dA_ab_frag, transpose_B=True, clear_accum=True)
            if full_tile:
                for r, c in T.Parallel(BT, BT):
                    if r > c:
                        dA_ab[bos + r, i_h, c] = T.Cast(in_dtype, dA_ab_frag[r, c])
                    else:
                        dA_ab[bos + r, i_h, c] = T.Cast(in_dtype, 0.0)
            else:
                for r, c in T.Parallel(BT, BT):
                    t = bos + r
                    if t < eos:
                        if r > c:
                            dA_ab[t, i_h, c] = T.Cast(in_dtype, dA_ab_frag[r, c])
                        else:
                            dA_ab[t, i_h, c] = T.Cast(in_dtype, 0.0)

    return wy_fast_bwd_tl


def chunk_dplr_bwd_wy_into(
    A_ab_inv: torch.Tensor,
    A_ak: torch.Tensor,
    v: torch.Tensor,
    ag: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dv0: torch.Tensor,
    dA_ab_out: torch.Tensor,
    dA_ak_out: torch.Tensor,
    dv_out: torch.Tensor,
    dag_out: torch.Tensor,
    store_dv: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 16,
    chunk_layout: ChunkLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mutable-output variant used by recompute backward workspace reuse."""
    for out in (dA_ab_out, dA_ak_out, dv_out, dag_out):
        assert out.is_contiguous(), "chunk_dplr_bwd_wy_into requires contiguous outputs"
    B, T_, H, K = dw.shape
    V = du.shape[-1]
    BT = chunk_size
    is_varlen = cu_seqlens is not None

    if is_varlen:
        assert B == 1
        layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, BT, T_)
    else:
        layout = build_rect_chunk_layout(B, T_, BT, dw.device)
    N_tokens = B * T_
    in_dtype = str(dw.dtype).split(".")[-1]

    A_ab_inv_f = A_ab_inv.reshape(N_tokens, H, BT).contiguous()
    A_ak_f = A_ak.reshape(N_tokens, H, BT).contiguous()
    ag_f = ag.reshape(N_tokens, H, K).contiguous()
    v_f = v.reshape(N_tokens, H, V).contiguous()
    dw_f = dw.reshape(N_tokens, H, K).contiguous()
    du_f = du.reshape(N_tokens, H, V).contiguous()
    dv0_f = dv0.reshape(N_tokens, H, V).contiguous()
    dA_ab_f = dA_ab_out.reshape(N_tokens, H, BT).contiguous()
    dA_ak_f = dA_ak_out.reshape(N_tokens, H, BT).contiguous()
    dv_f = dv_out.reshape(N_tokens, H, V).contiguous()
    dag_f = dag_out.reshape(N_tokens, H, K).contiguous()

    kernel = _wy_fast_bwd_kernel(
        H, K, V, BT, in_dtype,
        STORE_DV=bool(store_dv),
        **_wy_bwd_config(BT, dw.device),
    )
    kernel(
        A_ab_inv_f, A_ak_f, ag_f, v_f, dw_f, du_f, dv0_f,
        layout.cu_seqlens, layout.chunk_indices,
        dA_ab_f, dA_ak_f, dv_f, dag_f,
    )
    return dA_ab_out, dA_ak_out, dv_out, dag_out
