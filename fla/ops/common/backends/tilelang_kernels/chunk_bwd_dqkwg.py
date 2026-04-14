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
def _build_kernel(
    B, H, K, V, BT, BK, BV, NK,
    hD1, hD2,
    dtype_str,
    USE_G, USE_DW, USE_EXP2, TRANSPOSE_STATE, IS_VARLEN=False,
    num_warps=4,
):
    dtype_map = {'float16': T.float16, 'bfloat16': T.bfloat16, 'float32': T.float32}
    _dtype = dtype_map[dtype_str]
    NV = tilelang.cdiv(V, BV)
    threads = num_warps * 32
    tile_hD1, tile_hD2 = (BV, BK) if TRANSPOSE_STATE else (BK, BV)

    # Rebind to underscore-prefixed locals so the kernel body (a closure below)
    # stays identical to the previous nested layout. tilelang caches _build_kernel
    # by the outer (B, H, K, V, BT, BK, BV, NK, hD1, hD2, dtype_str, ...) tuple.
    _B, _H, _K, _V = B, H, K, V
    _BT, _BK, _BV, _NK = BT, BK, BV, NK
    _NV = NV
    _hD1, _hD2, _thD1, _thD2 = hD1, hD2, tile_hD1, tile_hD2
    _threads = threads
    _USE_G, _USE_DW, _USE_EXP2 = USE_G, USE_DW, USE_EXP2
    _TS, _VAR = TRANSPOSE_STATE, IS_VARLEN

    # T, NT, total_h are dynamic (vary with sequence length, no recompilation).
    # B, H are compile-time (stable across batches, enables fast integer division).
    T_d, NT_d, total_h_d, Ncu_d = T.dynamic("T, NT, total_h, Ncu")

    # 4D tensor shapes using dynamic T + compile-time B, H, K, V.
    qk_s = (_B, T_d, _H, _K)
    v_s = (_B, T_d, _H, _V)
    h_s = (total_h_d, _hD1, _hD2)
    g_s = (_B, T_d, _H)
    dg_s = (_NK, _B, T_d, _H)

    @T.macro
    def kernel_body(q, k, v, g, h, do, dh, dq, dk, dw, dv, dg, scale,
                    i_b, i_h, i_k, t_s, T_seq, i_t_local, h_idx, k_off):
        # -- accumulators --
        b_dq = T.alloc_fragment((_BT, _BK), T.float32)
        b_dk = T.alloc_fragment((_BT, _BK), T.float32)
        b_ds = T.alloc_fragment((_BT, _BT), T.float32)
        T.clear(b_dq)
        T.clear(b_dk)
        T.clear(b_ds)

        if _USE_DW:
            b_dw = T.alloc_fragment((_BT, _BK), T.float32)
            T.clear(b_dw)

        # -- shared tiles --
        s_v = T.alloc_shared((_BT, _BV), _dtype)
        s_do = T.alloc_shared((_BT, _BV), _dtype)
        s_h = T.alloc_shared((_thD1, _thD2), _dtype)
        s_dh = T.alloc_shared((_thD1, _thD2), _dtype)

        # dg_last accumulator (shared scalar, accumulated across V-loop)
        if _USE_G:
            s_dg_last_acc = T.alloc_shared((1,), T.float32)
            for _i in T.Parallel(1):
                s_dg_last_acc[0] = 0.0
            T.sync_threads()
            s_hdh = T.alloc_shared((_thD1, _thD2), T.float32)

        # ========== V-loop ==========
        for i_v_py in T.Pipelined(_NV, num_stages=2):
            v_off_c = i_v_py * _BV

            T.copy(v[i_b, t_s:t_s + _BT, i_h, v_off_c:v_off_c + _BV], s_v)
            T.copy(do[i_b, t_s:t_s + _BT, i_h, v_off_c:v_off_c + _BV], s_do)

            if _TS:
                T.copy(h[h_idx, v_off_c:v_off_c + _BV, k_off:k_off + _BK], s_h)
                T.copy(dh[h_idx, v_off_c:v_off_c + _BV, k_off:k_off + _BK], s_dh)
            else:
                T.copy(h[h_idx, k_off:k_off + _BK, v_off_c:v_off_c + _BV], s_h)
                T.copy(dh[h_idx, k_off:k_off + _BK, v_off_c:v_off_c + _BV], s_dh)

            T.gemm(s_do, s_v, b_ds, transpose_B=True)

            # h·dh reduction must precede the gemms that consume s_h/s_dh:
            # the downstream gemms are what the pipeline recognizes as consumers,
            # so they act as the barrier against the next iter's prefetch.
            if _USE_G:
                for _i, _j in T.Parallel(_thD1, _thD2):
                    s_hdh[_i, _j] = T.cast(s_h[_i, _j], T.float32) * T.cast(s_dh[_i, _j], T.float32)
                T.sync_threads()
                f_hdh = T.alloc_fragment((_thD1, _thD2), T.float32)
                T.copy(s_hdh, f_hdh)
                f_hdh_row = T.alloc_fragment((_thD1,), T.float32)
                T.reduce_sum(f_hdh, f_hdh_row, dim=1)
                s_hdh_row = T.alloc_shared((_thD1,), T.float32)
                T.copy(f_hdh_row, s_hdh_row)
                T.sync_threads()
                s_hdh_scalar = T.alloc_shared((1,), T.float32)
                for _i in T.Parallel(1):
                    acc = T.alloc_local((1,), T.float32)
                    acc[0] = 0.0
                    for _j in T.serial(_thD1):
                        acc[0] = acc[0] + s_hdh_row[_j]
                    s_hdh_scalar[0] = acc[0]
                T.sync_threads()
                s_dg_last_acc[0] = s_dg_last_acc[0] + s_hdh_scalar[0]
                T.sync_threads()

            if _TS:
                T.gemm(s_do, s_h, b_dq)
                T.gemm(s_v, s_dh, b_dk)
            else:
                T.gemm(s_do, s_h, b_dq, transpose_B=True)
                T.gemm(s_v, s_dh, b_dk, transpose_B=True)

            if _USE_DW:
                s_dv = T.alloc_shared((_BT, _BV), _dtype)
                T.copy(dv[i_b, t_s:t_s + _BT, i_h, v_off_c:v_off_c + _BV], s_dv)
                if _TS:
                    T.gemm(s_dv, s_h, b_dw)
                else:
                    T.gemm(s_dv, s_h, b_dw, transpose_B=True)

        # ========== store dw (negated, with varlen boundary mask) ==========
        if _USE_DW:
            s_dw_out = T.alloc_shared((_BT, _BK), _dtype)
            for _i, _j in T.Parallel(_BT, _BK):
                s_dw_out[_i, _j] = T.cast(-b_dw[_i, _j], _dtype)
            T.sync_threads()
            for _i, _j in T.Parallel(_BT, _BK):
                if (i_t_local * _BT + _i) < T_seq:
                    dw[i_b, t_s + _i, i_h, k_off + _j] = s_dw_out[_i, _j]

        # dg_last is now in s_dg_last_acc[0] (shared memory, visible to all threads)

        # ========== load q, k ==========
        s_q = T.alloc_shared((_BT, _BK), _dtype)
        s_k = T.alloc_shared((_BT, _BK), _dtype)
        T.copy(q[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK], s_q)
        T.copy(k[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK], s_k)

        # ========== USE_G path ==========
        if _USE_G:
            # dg_last_acc already reduced to `red` scalar above

            # g in shared memory (all threads can cross-read any element)
            s_g = T.alloc_shared((_BT,), T.float32)
            T.copy(g[i_b, t_s:t_s + _BT, i_h], s_g, disable_tma=True)

            last_pos = T.max(0, T.min(_BT, T_seq - i_t_local * _BT) - 1)
            g_last = s_g[last_pos]
            b_dg_last = T.alloc_var(T.float32)
            b_dg_last = s_dg_last_acc[0] * (T.exp2(g_last) if _USE_EXP2 else T.exp(g_last))

            # b_dq *= exp(g) * scale  (inline, no extra fragment)
            for _i, _j in T.Parallel(_BT, _BK):
                b_dq[_i, _j] = b_dq[_i, _j] * (T.exp2(s_g[_i]) if _USE_EXP2 else T.exp(s_g[_i])) * scale

            # Gate b_dk with m_t mask: zero out OOB positions
            for _i, _j in T.Parallel(_BT, _BK):
                m_t = (i_t_local * _BT + _i) < T_seq
                b_dk[_i, _j] = T.if_then_else(
                    m_t,
                    b_dk[_i, _j] * (T.exp2(-s_g[_i] + g_last) if _USE_EXP2 else T.exp(-s_g[_i] + g_last)),
                    0.0)

            # dq*q and dk*k reductions: compute products directly in fragments
            # via fragment×shared (safe per pattern 174). Avoids the shared
            # round-trip through s_A1/s_A2.
            f_prod1 = T.alloc_fragment((_BT, _BK), T.float32)
            f_prod2 = T.alloc_fragment((_BT, _BK), T.float32)
            for _i, _j in T.Parallel(_BT, _BK):
                f_prod1[_i, _j] = b_dq[_i, _j] * s_q[_i, _j]
                f_prod2[_i, _j] = b_dk[_i, _j] * s_k[_i, _j]
            f_dg1 = T.alloc_fragment((_BT,), T.float32)
            f_dg2 = T.alloc_fragment((_BT,), T.float32)
            T.reduce_sum(f_prod1, f_dg1, dim=1)
            T.reduce_sum(f_prod2, f_dg2, dim=1)
            # f_dg_diff = dq*q - dk*k in fragment (both from T.reduce_sum, same layout)
            f_dg_diff = T.alloc_fragment((_BT,), T.float32)
            for _i in T.Parallel(_BT):
                f_dg_diff[_i] = f_dg1[_i] - f_dg2[_i]
            s_dg = T.alloc_shared((_BT,), T.float32)
            s_tmp1d = T.alloc_shared((_BT,), T.float32)
            T.copy(f_dg_diff, s_dg)
            T.copy(f_dg2, s_tmp1d)  # kept for scalar sum below
            T.sync_threads()
            # b_dg_last += sum(dk*k) via serial sum on shared
            s_scalar = T.alloc_shared((1,), T.float32)
            for _i in T.Parallel(1):
                acc_all = T.alloc_local((1,), T.float32)
                acc_all[0] = 0.0
                for _j in T.serial(_BT):
                    acc_all[0] = acc_all[0] + s_tmp1d[_j]
                s_scalar[0] = acc_all[0]
            T.sync_threads()
            b_dg_last = b_dg_last + s_scalar[0]

            # b_ds = where(causal, b_ds * exp(g_i - g_j), 0) * scale
            for _i, _j in T.Parallel(_BT, _BT):
                causal = (_i >= _j) & ((i_t_local * _BT + _i) < T_seq) & ((i_t_local * _BT + _j) < T_seq)
                b_ds[_i, _j] = T.if_then_else(
                    causal,
                    b_ds[_i, _j] * (T.exp2(s_g[_i] - s_g[_j]) if _USE_EXP2 else T.exp(s_g[_i] - s_g[_j])) * scale,
                    0.0)

            # ds2 = ds * (q @ k^T);  dg += row_sum(ds2) - col_sum(ds2)
            b_qk = T.alloc_fragment((_BT, _BT), T.float32)
            T.clear(b_qk)
            T.gemm(s_q, s_k, b_qk, transpose_B=True)
            # Write both to shared, multiply to get ds2
            s_ds_f32 = T.alloc_shared((_BT, _BT), T.float32)
            s_qk_f32 = T.alloc_shared((_BT, _BT), T.float32)
            T.copy(b_ds, s_ds_f32)
            T.copy(b_qk, s_qk_f32)
            T.sync_threads()
            for _i, _j in T.Parallel(_BT, _BT):
                s_ds_f32[_i, _j] = s_ds_f32[_i, _j] * s_qk_f32[_i, _j]
            T.sync_threads()
            # row_sum(ds2) via T.reduce_sum
            f_ds2 = T.alloc_fragment((_BT, _BT), T.float32)
            T.copy(s_ds_f32, f_ds2)
            f_row_sum = T.alloc_fragment((_BT,), T.float32)
            T.reduce_sum(f_ds2, f_row_sum, dim=1)
            s_row = T.alloc_shared((_BT,), T.float32)
            T.copy(f_row_sum, s_row)
            T.sync_threads()
            for _i in T.Parallel(_BT):
                s_dg[_i] = s_dg[_i] + s_row[_i]
            T.sync_threads()
            # col_sum(ds2): transpose in shared, then T.reduce_sum
            # Reuse s_qk_f32 for the transposed ds2
            for _i, _j in T.Parallel(_BT, _BT):
                s_qk_f32[_i, _j] = s_ds_f32[_j, _i]
            T.sync_threads()
            f_ds2t = T.alloc_fragment((_BT, _BT), T.float32)
            T.copy(s_qk_f32, f_ds2t)
            f_col_sum = T.alloc_fragment((_BT,), T.float32)
            T.reduce_sum(f_ds2t, f_col_sum, dim=1)
            s_col = T.alloc_shared((_BT,), T.float32)
            T.copy(f_col_sum, s_col)
            T.sync_threads()
            for _i in T.Parallel(_BT):
                s_dg[_i] = s_dg[_i] - s_col[_i]
            T.sync_threads()

            # cast ds for final gemms
            s_ds = T.alloc_shared((_BT, _BT), _dtype)
            f_ds = T.alloc_fragment((_BT, _BT), _dtype)
            for _i, _j in T.Parallel(_BT, _BT):
                f_ds[_i, _j] = T.cast(b_ds[_i, _j], _dtype)
            T.copy(f_ds, s_ds)

            T.gemm(s_ds, s_k, b_dq)               # dq += ds @ k
            T.gemm(s_ds, s_q, b_dk, transpose_A=True)  # dk += ds^T @ q

            # store dq, dk via shared (fragment has MMA layout, can't index directly)
            f_out = T.alloc_fragment((_BT, _BK), _dtype)
            s_out = T.alloc_shared((_BT, _BK), _dtype)
            for _i, _j in T.Parallel(_BT, _BK):
                f_out[_i, _j] = T.cast(b_dq[_i, _j], _dtype)
            T.copy(f_out, s_out)
            T.sync_threads()
            for _i, _j in T.Parallel(_BT, _BK):
                if (i_t_local * _BT + _i) < T_seq:
                    dq[i_b, t_s + _i, i_h, k_off + _j] = s_out[_i, _j]
            for _i, _j in T.Parallel(_BT, _BK):
                f_out[_i, _j] = T.cast(b_dk[_i, _j], _dtype)
            T.copy(f_out, s_out)
            T.sync_threads()
            for _i, _j in T.Parallel(_BT, _BK):
                if (i_t_local * _BT + _i) < T_seq:
                    dk[i_b, t_s + _i, i_h, k_off + _j] = s_out[_i, _j]

            # store dg: merge dg_last into last valid position
            for _i in T.Parallel(_BT):
                if (i_t_local * _BT + _i) < T_seq:
                    val = T.if_then_else(_i == last_pos, s_dg[_i] + b_dg_last, s_dg[_i])
                    dg[i_k, i_b, t_s + _i, i_h] = val

    if _VAR:
        @T.prim_func
        def kernel(
            q: T.Tensor(qk_s, _dtype), k: T.Tensor(qk_s, _dtype),
            v: T.Tensor(v_s, _dtype), g: T.Tensor(g_s, T.float32),
            h: T.Tensor(h_s, _dtype), do: T.Tensor(v_s, _dtype),
            dh: T.Tensor(h_s, _dtype), dq: T.Tensor(qk_s, _dtype),
            dk: T.Tensor(qk_s, _dtype), dw: T.Tensor(qk_s, _dtype),
            dv: T.Tensor(v_s, _dtype), dg: T.Tensor(dg_s, T.float32),
            cu_seqlens: T.Tensor((Ncu_d,), T.int32),
            chunk_indices: T.Tensor((NT_d, 2), T.int32),
            scale: T.float32,
        ):
            with T.Kernel(_NK, NT_d, _H, threads=_threads) as (i_k, i_t, i_h):
                i_n = chunk_indices[i_t, 0]
                i_t_local = chunk_indices[i_t, 1]
                bos = cu_seqlens[i_n]
                T_seq = cu_seqlens[i_n + 1] - bos
                h_idx = i_t * _H + i_h
                t_s = bos + i_t_local * _BT
                kernel_body(q, k, v, g, h, do, dh, dq, dk, dw, dv, dg,
                            scale, 0, i_h, i_k, t_s, T_seq, i_t_local,
                            h_idx, i_k * _BK)
    else:
        @T.prim_func
        def kernel(
            q: T.Tensor(qk_s, _dtype), k: T.Tensor(qk_s, _dtype),
            v: T.Tensor(v_s, _dtype), g: T.Tensor(g_s, T.float32),
            h: T.Tensor(h_s, _dtype), do: T.Tensor(v_s, _dtype),
            dh: T.Tensor(h_s, _dtype), dq: T.Tensor(qk_s, _dtype),
            dk: T.Tensor(qk_s, _dtype), dw: T.Tensor(qk_s, _dtype),
            dv: T.Tensor(v_s, _dtype), dg: T.Tensor(dg_s, T.float32),
            scale: T.float32,
        ):
            with T.Kernel(_NK, T.ceildiv(T_d, _BT), _B * _H, threads=_threads) as (i_k, i_t, i_bh):
                i_b = i_bh // _H
                i_h = i_bh % _H
                NT_local = T.ceildiv(T_d, _BT)
                h_idx = (i_b * NT_local + i_t) * _H + i_h
                t_s = i_t * _BT
                kernel_body(q, k, v, g, h, do, dh, dq, dk, dw, dv, dg,
                            scale, i_b, i_h, i_k, t_s, T_d, i_t,
                            h_idx, i_k * _BK)

    return kernel


def chunk_bwd_dqkwg_tilelang(
    q, k, v, do, h, dh,
    w=None, g=None, g_gamma=None, dv=None,
    scale=None, cu_seqlens=None, chunk_size=64,
    chunk_indices=None, use_exp2=False, transpose_state_layout=False,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    IS_VARLEN = cu_seqlens is not None

    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)
    if scale is None:
        scale = K ** -0.5

    USE_G = g is not None
    USE_DW = w is not None

    # Outputs — kernel writes directly
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dw_out = torch.empty_like(w) if USE_DW else None
    dg = torch.zeros(NK, B, T, H, dtype=torch.float32, device=q.device) if USE_G else None

    h_flat = h.reshape(-1, h.shape[-2], h.shape[-1])
    dh_flat = dh.reshape(-1, dh.shape[-2], dh.shape[-1])
    hD1, hD2 = h_flat.shape[-2], h_flat.shape[-1]
    dtype_str = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}[q.dtype]

    # Cache key: B, H, tile sizes, flags. T is dynamic (no recompilation for different seq lengths).
    kernel = _build_kernel(
        B, H, K, V, BT, BK, BV, NK,
        hD1, hD2, dtype_str,
        USE_G, USE_DW, use_exp2, transpose_state_layout, IS_VARLEN,
    )

    # Unused optional params still need shape-matching tensors for TileLang
    g_kern = g if USE_G else q.new_empty(B, T, H)
    dw_kern = dw_out if USE_DW else q.new_empty(B, T, H, K)
    dv_kern = dv if USE_DW else q.new_empty(B, T, H, V)
    dg_kern = dg if USE_G else q.new_empty(NK, B, T, H, dtype=torch.float32)

    if IS_VARLEN:
        kernel(q, k, v, g_kern, h_flat, do, dh_flat, dq, dk, dw_kern, dv_kern, dg_kern,
               cu_seqlens.int(), chunk_indices.int(), scale)
    else:
        kernel(q, k, v, g_kern, h_flat, do, dh_flat, dq, dk, dw_kern, dv_kern, dg_kern, scale)

    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw_out, dg
