# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import tilelang
import tilelang.language as T
import torch
import triton

from fla.ops.utils import prepare_chunk_indices
from fla.utils import check_shared_mem


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def _build_chunk_bwd_dqkwg_kernel(
    B,
    H,
    HV,
    K,
    V,
    BT,
    BK,
    BV,
    NK,
    hD1,
    hD2,
    dtype_str,
    USE_G,
    USE_DW,
    STATE_V_FIRST,
    IS_VARLEN=False,
    num_warps=4,
):
    dtype_map = {'float16': T.float16, 'bfloat16': T.bfloat16, 'float32': T.float32}
    _dtype = dtype_map[dtype_str]
    NV = tilelang.cdiv(V, BV)
    threads = num_warps * 32
    tile_hD1, tile_hD2 = (BV, BK) if STATE_V_FIRST else (BK, BV)

    G = HV // H

    T_d, NT_d, total_h_d, Ncu_d = T.dynamic("T, NT, total_h, Ncu")

    # q/k use H heads; per-value-head gradients use HV
    qk_s = (B, T_d, H, K)
    dqk_s = (B, T_d, HV, K)
    v_s = (B, T_d, HV, V)
    h_s = (total_h_d, hD1, hD2)
    g_s = (B, T_d, HV)
    dg_s = (NK, B, T_d, HV)

    @T.macro
    def kernel_body(q, k, v, g, h, do, dh, dq, dk, dw, dv, dg, scale, i_b, i_h, i_k, t_s, T_seq, i_t_local, h_idx, k_off):
        i_hqk = i_h // G
        b_dq = T.alloc_fragment((BT, BK), T.float32)
        b_dk = T.alloc_fragment((BT, BK), T.float32)
        b_ds = T.alloc_fragment((BT, BT), T.float32)
        T.clear(b_dq)
        T.clear(b_dk)
        T.clear(b_ds)

        if USE_DW:
            b_dw = T.alloc_fragment((BT, BK), T.float32)
            T.clear(b_dw)

        s_v = T.alloc_shared((BT, BV), _dtype)
        s_do = T.alloc_shared((BT, BV), _dtype)
        s_h = T.alloc_shared((tile_hD1, tile_hD2), _dtype)
        s_dh = T.alloc_shared((tile_hD1, tile_hD2), _dtype)

        if USE_G:
            s_dg_last_acc = T.alloc_shared((1,), T.float32)
            for _i in T.Parallel(1):
                s_dg_last_acc[0] = 0.0
            T.sync_threads()

        for i_v_py in T.Pipelined(NV, num_stages=2):
            v_off_c = i_v_py * BV

            T.copy(v[i_b, t_s:t_s + BT, i_h, v_off_c:v_off_c + BV], s_v)
            T.copy(do[i_b, t_s:t_s + BT, i_h, v_off_c:v_off_c + BV], s_do)

            if STATE_V_FIRST:
                T.copy(h[h_idx, v_off_c:v_off_c + BV, k_off:k_off + BK], s_h)
                T.copy(dh[h_idx, v_off_c:v_off_c + BV, k_off:k_off + BK], s_dh)
            else:
                T.copy(h[h_idx, k_off:k_off + BK, v_off_c:v_off_c + BV], s_h)
                T.copy(dh[h_idx, k_off:k_off + BK, v_off_c:v_off_c + BV], s_dh)

            T.gemm(s_do, s_v, b_ds, transpose_B=True)

            # reduce h·dh before the GEMMs that keep the next prefetch from overwriting s_h/s_dh
            if USE_G:
                f_hdh = T.alloc_fragment((tile_hD1, tile_hD2), T.float32)
                for _i, _j in T.Parallel(tile_hD1, tile_hD2):
                    f_hdh[_i, _j] = T.cast(s_h[_i, _j], T.float32) * T.cast(s_dh[_i, _j], T.float32)
                f_hdh_row = T.alloc_fragment((tile_hD1,), T.float32)
                T.reduce_sum(f_hdh, f_hdh_row, dim=1)
                f_hdh_scalar = T.alloc_fragment((1,), T.float32)
                T.reduce_sum(f_hdh_row, f_hdh_scalar, dim=0)
                s_dg_last_acc[0] = s_dg_last_acc[0] + f_hdh_scalar[0]

            if STATE_V_FIRST:
                T.gemm(s_do, s_h, b_dq)
                T.gemm(s_v, s_dh, b_dk)
            else:
                T.gemm(s_do, s_h, b_dq, transpose_B=True)
                T.gemm(s_v, s_dh, b_dk, transpose_B=True)

            if USE_DW:
                s_dv = T.alloc_shared((BT, BV), _dtype)
                T.copy(dv[i_b, t_s:t_s + BT, i_h, v_off_c:v_off_c + BV], s_dv)
                if STATE_V_FIRST:
                    T.gemm(s_dv, s_h, b_dw)
                else:
                    T.gemm(s_dv, s_h, b_dw, transpose_B=True)

        if USE_DW:
            s_dw_out = T.alloc_shared((BT, BK), _dtype)
            for _i, _j in T.Parallel(BT, BK):
                s_dw_out[_i, _j] = T.cast(-b_dw[_i, _j], _dtype)
            T.sync_threads()
            for _i, _j in T.Parallel(BT, BK):
                if (i_t_local * BT + _i) < T_seq:
                    dw[i_b, t_s + _i, i_h, k_off + _j] = s_dw_out[_i, _j]

        s_q = T.alloc_shared((BT, BK), _dtype)
        s_k = T.alloc_shared((BT, BK), _dtype)
        T.copy(q[i_b, t_s:t_s + BT, i_hqk, k_off:k_off + BK], s_q)
        T.copy(k[i_b, t_s:t_s + BT, i_hqk, k_off:k_off + BK], s_k)

        if USE_G:
            # shared memory lets all threads read the last valid gate
            s_g = T.alloc_shared((BT,), T.float32)
            T.copy(g[i_b, t_s:t_s + BT, i_h], s_g, disable_tma=True)

            last_pos = T.max(0, T.min(BT, T_seq - i_t_local * BT) - 1)
            g_last = s_g[last_pos]
            b_dg_last = T.alloc_var(T.float32)
            b_dg_last = s_dg_last_acc[0] * T.exp2(g_last)

            for _i, _j in T.Parallel(BT, BK):
                b_dq[_i, _j] = b_dq[_i, _j] * T.exp2(s_g[_i]) * scale

            for _i, _j in T.Parallel(BT, BK):
                m_t = (i_t_local * BT + _i) < T_seq
                b_dk[_i, _j] = T.if_then_else(m_t, b_dk[_i, _j] * T.exp2(-s_g[_i] + g_last), 0.0)

            # accumulate gated dk*k before the intra-chunk GEMMs update dk
            f_prod2 = T.alloc_fragment((BT, BK), T.float32)
            for _i, _j in T.Parallel(BT, BK):
                f_prod2[_i, _j] = b_dk[_i, _j] * s_k[_i, _j]
            f_dg2 = T.alloc_fragment((BT,), T.float32)
            T.reduce_sum(f_prod2, f_dg2, dim=1)
            f_dkk_scalar = T.alloc_fragment((1,), T.float32)
            T.reduce_sum(f_dg2, f_dkk_scalar, dim=0)
            b_dg_last = b_dg_last + f_dkk_scalar[0]

            for _i, _j in T.Parallel(BT, BT):
                causal = (_i >= _j) & ((i_t_local * BT + _i) < T_seq) & ((i_t_local * BT + _j) < T_seq)
                b_ds[_i, _j] = T.if_then_else(causal, b_ds[_i, _j] * T.exp2(s_g[_i] - s_g[_j]) * scale, 0.0)

            s_ds = T.alloc_shared((BT, BT), _dtype)
            f_ds = T.alloc_fragment((BT, BT), _dtype)
            for _i, _j in T.Parallel(BT, BT):
                f_ds[_i, _j] = T.cast(b_ds[_i, _j], _dtype)
            T.copy(f_ds, s_ds)

            T.gemm(s_ds, s_k, b_dq)
            T.gemm(s_ds, s_q, b_dk, transpose_A=True)

            # dg uses the fully updated dq/dk
            f_prod1 = T.alloc_fragment((BT, BK), T.float32)
            for _i, _j in T.Parallel(BT, BK):
                f_prod1[_i, _j] = b_dq[_i, _j] * s_q[_i, _j]
                f_prod2[_i, _j] = b_dk[_i, _j] * s_k[_i, _j]
            f_dg1 = T.alloc_fragment((BT,), T.float32)
            T.reduce_sum(f_prod1, f_dg1, dim=1)
            T.reduce_sum(f_prod2, f_dg2, dim=1)
            f_dg_diff = T.alloc_fragment((BT,), T.float32)
            for _i in T.Parallel(BT):
                f_dg_diff[_i] = f_dg1[_i] - f_dg2[_i]
            s_dg = T.alloc_shared((BT,), T.float32)
            T.copy(f_dg_diff, s_dg)

            # shared staging converts MMA fragments to an indexable layout
            f_out = T.alloc_fragment((BT, BK), _dtype)
            s_out = T.alloc_shared((BT, BK), _dtype)
            for _i, _j in T.Parallel(BT, BK):
                f_out[_i, _j] = T.cast(b_dq[_i, _j], _dtype)
            T.copy(f_out, s_out)
            T.sync_threads()
            for _i, _j in T.Parallel(BT, BK):
                if (i_t_local * BT + _i) < T_seq:
                    dq[i_b, t_s + _i, i_h, k_off + _j] = s_out[_i, _j]
            for _i, _j in T.Parallel(BT, BK):
                f_out[_i, _j] = T.cast(b_dk[_i, _j], _dtype)
            T.copy(f_out, s_out)
            T.sync_threads()
            for _i, _j in T.Parallel(BT, BK):
                if (i_t_local * BT + _i) < T_seq:
                    dk[i_b, t_s + _i, i_h, k_off + _j] = s_out[_i, _j]

            for _i in T.Parallel(BT):
                if (i_t_local * BT + _i) < T_seq:
                    val = T.if_then_else(_i == last_pos, s_dg[_i] + b_dg_last, s_dg[_i])
                    dg[i_k, i_b, t_s + _i, i_h] = val

    if IS_VARLEN:
        @T.prim_func
        def kernel(
            q: T.Tensor(qk_s, _dtype),
            k: T.Tensor(qk_s, _dtype),
            v: T.Tensor(v_s, _dtype),
            g: T.Tensor(g_s, T.float32),
            h: T.Tensor(h_s, _dtype),
            do: T.Tensor(v_s, _dtype),
            dh: T.Tensor(h_s, _dtype),
            dq: T.Tensor(dqk_s, _dtype),
            dk: T.Tensor(dqk_s, _dtype),
            dw: T.Tensor(dqk_s, _dtype),
            dv: T.Tensor(v_s, _dtype),
            dg: T.Tensor(dg_s, T.float32),
            cu_seqlens: T.Tensor((Ncu_d,), T.int32),
            chunk_indices: T.Tensor((NT_d, 2), T.int32),
            scale: T.float32,
        ):
            with T.Kernel(NK, NT_d, HV, threads=threads) as (i_k, i_t, i_h):
                i_n = chunk_indices[i_t, 0]
                i_t_local = chunk_indices[i_t, 1]
                bos = cu_seqlens[i_n]
                T_seq = cu_seqlens[i_n + 1] - bos
                h_idx = i_t * HV + i_h
                t_s = bos + i_t_local * BT
                kernel_body(
                    q,
                    k,
                    v,
                    g,
                    h,
                    do,
                    dh,
                    dq,
                    dk,
                    dw,
                    dv,
                    dg,
                    scale,
                    0,
                    i_h,
                    i_k,
                    t_s,
                    T_seq,
                    i_t_local,
                    h_idx,
                    i_k * BK,
                )
    else:
        @T.prim_func
        def kernel(
            q: T.Tensor(qk_s, _dtype),
            k: T.Tensor(qk_s, _dtype),
            v: T.Tensor(v_s, _dtype),
            g: T.Tensor(g_s, T.float32),
            h: T.Tensor(h_s, _dtype),
            do: T.Tensor(v_s, _dtype),
            dh: T.Tensor(h_s, _dtype),
            dq: T.Tensor(dqk_s, _dtype),
            dk: T.Tensor(dqk_s, _dtype),
            dw: T.Tensor(dqk_s, _dtype),
            dv: T.Tensor(v_s, _dtype),
            dg: T.Tensor(dg_s, T.float32),
            scale: T.float32,
        ):
            with T.Kernel(NK, T.ceildiv(T_d, BT), B * HV, threads=threads) as (i_k, i_t, i_bh):
                i_b = i_bh // HV
                i_h = i_bh % HV
                NT_local = T.ceildiv(T_d, BT)
                h_idx = (i_b * NT_local + i_t) * HV + i_h
                t_s = i_t * BT
                kernel_body(q, k, v, g, h, do, dh, dq, dk, dw, dv, dg, scale, i_b, i_h, i_k, t_s, T_d, i_t, h_idx, i_k * BK)

    return kernel


def chunk_bwd_dqkwg_tilelang(
    q,
    k,
    v,
    do,
    h,
    dh,
    w=None,
    g=None,
    g_gamma=None,
    dv=None,
    scale=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
):
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT)
    IS_VARLEN = cu_seqlens is not None

    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)
    if scale is None:
        scale = K ** -0.5

    USE_G = g is not None
    USE_DW = w is not None

    # dq/dk are computed per value head and reduced to query/key heads after the kernel
    dq = torch.empty(B, T, HV, K, dtype=q.dtype, device=q.device)
    dk = torch.empty(B, T, HV, K, dtype=k.dtype, device=k.device)
    dw_out = torch.empty_like(w) if USE_DW else None
    dg = torch.zeros(NK, B, T, HV, dtype=torch.float32, device=q.device) if USE_G else None

    h_flat = h.reshape(-1, h.shape[-2], h.shape[-1])
    dh_flat = dh.reshape(-1, dh.shape[-2], dh.shape[-1])
    hD1, hD2 = h_flat.shape[-2], h_flat.shape[-1]
    dtype_str = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}[q.dtype]

    # small head dimensions need two warps for TileLang GEMM partitioning
    num_warps = 4 if min(K, V) >= 64 else 2
    kernel = _build_chunk_bwd_dqkwg_kernel(
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NK=NK,
        hD1=hD1,
        hD2=hD2,
        dtype_str=dtype_str,
        USE_G=USE_G,
        USE_DW=USE_DW,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=IS_VARLEN,
        num_warps=num_warps,
    )

    # optional inputs need shape-matching buffers for the kernel signature
    g_kern = g if USE_G else q.new_empty(B, T, HV)
    dw_kern = dw_out if USE_DW else q.new_empty(B, T, HV, K)
    dv_kern = dv if USE_DW else q.new_empty(B, T, HV, V)
    dg_kern = dg if USE_G else q.new_empty(NK, B, T, HV, dtype=torch.float32)

    if IS_VARLEN:
        kernel(
            q,
            k,
            v,
            g_kern,
            h_flat,
            do,
            dh_flat,
            dq,
            dk,
            dw_kern,
            dv_kern,
            dg_kern,
            cu_seqlens.int(),
            chunk_indices.int(),
            scale,
        )
    else:
        kernel(q, k, v, g_kern, h_flat, do, dh_flat, dq, dk, dw_kern, dv_kern, dg_kern, scale)

    if dg is not None:
        dg = dg.sum(0)
    if H != HV:
        G = HV // H
        dq = dq.view(B, T, H, G, K).sum(3)
        dk = dk.view(B, T, H, G, K).sum(3)
    return dq, dk, dw_out, dg
