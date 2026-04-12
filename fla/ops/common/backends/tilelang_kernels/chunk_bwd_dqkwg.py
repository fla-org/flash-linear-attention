# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from functools import lru_cache

import tilelang
import tilelang.language as T
import torch
import triton

from fla.ops.utils import prepare_chunk_indices
from fla.utils import check_shared_mem


@lru_cache(maxsize=64)
def _build_kernel(
    B, T_val, H, K, V, NT, BT, BK, BV, NK,
    total_h, hD1, hD2,
    dtype_str,
    USE_G, USE_DW, TRANSPOSE_STATE,
    num_warps=4,
):
    dtype_map = {'float16': T.float16, 'bfloat16': T.bfloat16, 'float32': T.float32}
    dtype = dtype_map[dtype_str]
    NV = tilelang.cdiv(V, BV)
    threads = num_warps * 32
    tile_hD1, tile_hD2 = (BV, BK) if TRANSPOSE_STATE else (BK, BV)

    # 4D tensor shapes: (B, T, H, D) — matches original layout, no flattening.
    # TileLang's LegalizeSafeMemoryAccess pass auto-handles OOB reads (zero-fill)
    # and OOB writes (skip) so no padding is needed.
    qk_s = (B, T_val, H, K)
    v_s = (B, T_val, H, V)
    h_s = (total_h, hD1, hD2)
    g_s = (B, T_val, H)
    dg_s = (NK, B, T_val, H)

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def make_kernel(
        _B=B, _T=T_val, _H=H, _K=K, _V=V,
        _NT=NT, _BT=BT, _BK=BK, _BV=BV, _NK=NK,
        _NV=NV, _hD1=hD1, _hD2=hD2, _thD1=tile_hD1, _thD2=tile_hD2,
        _dtype=dtype, _threads=threads,
        _USE_G=USE_G, _USE_DW=USE_DW, _TS=TRANSPOSE_STATE,
    ):
        @T.prim_func
        def kernel(
            q:  T.Tensor(qk_s, _dtype),
            k:  T.Tensor(qk_s, _dtype),
            v:  T.Tensor(v_s, _dtype),
            g:  T.Tensor(g_s, T.float32),
            h:  T.Tensor(h_s, _dtype),
            do: T.Tensor(v_s, _dtype),
            dh: T.Tensor(h_s, _dtype),
            dq: T.Tensor(qk_s, _dtype),
            dk: T.Tensor(qk_s, _dtype),
            dw: T.Tensor(qk_s, _dtype),
            dv: T.Tensor(v_s, _dtype),
            dg: T.Tensor(dg_s, T.float32),
            scale: T.float32,
        ):
            with T.Kernel(_NK, _NT, _B * _H, threads=_threads) as (i_k, i_t, i_bh):
                i_b = i_bh // _H
                i_h = i_bh % _H
                i_tg = i_b * _NT + i_t
                h_idx = i_tg * _H + i_h
                t_s = i_t * _BT          # start position in T dim
                k_off = i_k * _BK

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

                    # dg_last: h*dh sum merged into V-loop (avoids re-reading h/dh)
                    # Use separate s_hdh buffer (not s_h) to avoid write conflict in pipeline
                    if _USE_G:
                        s_hdh = T.alloc_shared((_thD1, _thD2), _dtype)
                        for _i, _j in T.Parallel(_thD1, _thD2):
                            s_hdh[_i, _j] = s_h[_i, _j] * s_dh[_i, _j]
                        T.sync_threads()
                        f_hdh = T.alloc_fragment((_thD1, _thD2), T.float32)
                        T.copy(s_hdh, f_hdh)
                        f_hdh_row = T.alloc_fragment((_thD1,), T.float32)
                        T.reduce_sum(f_hdh, f_hdh_row, dim=1)
                        red_hdh = T.alloc_reducer((1,), T.float32, op="sum", replication="all")
                        T.fill(red_hdh, 0.0)
                        for _i in T.Parallel(_thD1):
                            red_hdh[0] = red_hdh[0] + f_hdh_row[_i]
                        T.finalize_reducer(red_hdh)
                        s_dg_last_acc[0] = s_dg_last_acc[0] + red_hdh[0]
                        T.sync_threads()

                # ========== store dw (negated) ==========
                if _USE_DW:
                    f_dw = T.alloc_fragment((_BT, _BK), _dtype)
                    for _i, _j in T.Parallel(_BT, _BK):
                        f_dw[_i, _j] = T.cast(-b_dw[_i, _j], _dtype)
                    T.copy(f_dw, dw[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK])

                # dg_last reducer result
                if _USE_G:
                    red = T.alloc_reducer((1,), T.float32, op="sum", replication="all")
                    T.fill(red, 0.0)
                    for _i in T.Parallel(1):
                        red[0] = red[0] + s_dg_last_acc[0]
                    T.finalize_reducer(red)

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

                    last_pos = T.min(_BT, _T - i_t * _BT) - 1
                    g_last = s_g[last_pos]
                    b_dg_last = T.alloc_var(T.float32)
                    b_dg_last = red[0] * T.exp(g_last)

                    # b_dq *= exp(g) * scale  (inline, no extra fragment)
                    for _i, _j in T.Parallel(_BT, _BK):
                        b_dq[_i, _j] = b_dq[_i, _j] * T.exp(s_g[_i]) * scale

                    # Gate b_dk before dg reductions (doesn't depend on dq result)
                    for _i, _j in T.Parallel(_BT, _BK):
                        b_dk[_i, _j] = b_dk[_i, _j] * T.exp(-s_g[_i] + g_last)

                    # Batched dg reductions: write both b_dq and b_dk to shared
                    # in one round to reduce syncthreads count.
                    s_A1 = T.alloc_shared((_BT, _BK), T.float32)
                    s_A2 = T.alloc_shared((_BT, _BK), T.float32)
                    T.copy(b_dq, s_A1)
                    T.copy(b_dk, s_A2)
                    T.sync_threads()
                    # Multiply both with q/k in shared
                    for _i, _j in T.Parallel(_BT, _BK):
                        s_A1[_i, _j] = s_A1[_i, _j] * s_q[_i, _j]
                        s_A2[_i, _j] = s_A2[_i, _j] * s_k[_i, _j]
                    T.sync_threads()
                    # Reduce both: dq*q → dg_add, dk*k → dg_sub
                    f_prod1 = T.alloc_fragment((_BT, _BK), T.float32)
                    f_prod2 = T.alloc_fragment((_BT, _BK), T.float32)
                    T.copy(s_A1, f_prod1)
                    T.copy(s_A2, f_prod2)
                    f_dg1 = T.alloc_fragment((_BT,), T.float32)
                    f_dg2 = T.alloc_fragment((_BT,), T.float32)
                    T.reduce_sum(f_prod1, f_dg1, dim=1)
                    T.reduce_sum(f_prod2, f_dg2, dim=1)
                    # Write both to shared
                    s_dg = T.alloc_shared((_BT,), T.float32)
                    s_tmp1d = T.alloc_shared((_BT,), T.float32)
                    T.copy(f_dg1, s_dg)
                    T.copy(f_dg2, s_tmp1d)
                    T.sync_threads()
                    # s_dg = dq*q - dk*k
                    for _i in T.Parallel(_BT):
                        s_dg[_i] = s_dg[_i] - s_tmp1d[_i]
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
                        causal = (_i >= _j) & ((i_t * _BT + _i) < _T) & ((i_t * _BT + _j) < _T)
                        b_ds[_i, _j] = T.if_then_else(
                            causal,
                            b_ds[_i, _j] * T.exp(s_g[_i] - s_g[_j]) * scale,
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

                    # store dq, dk
                    f_out = T.alloc_fragment((_BT, _BK), _dtype)
                    for _i, _j in T.Parallel(_BT, _BK):
                        f_out[_i, _j] = T.cast(b_dq[_i, _j], _dtype)
                    T.copy(f_out, dq[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK])
                    for _i, _j in T.Parallel(_BT, _BK):
                        f_out[_i, _j] = T.cast(b_dk[_i, _j], _dtype)
                    T.copy(f_out, dk[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK])

                    # store dg: merge dg_last into last valid position
                    for _i in T.Parallel(_BT):
                        val = T.if_then_else(_i == last_pos, s_dg[_i] + b_dg_last, s_dg[_i])
                        dg[i_k, i_b, t_s + _i, i_h] = val

                # NOTE: USE_G_GAMMA and no-gating branches removed.
                # The TileLang backend verifier rejects those cases.
                # Keeping dead branches here causes TileLang to see extra
                # fragment allocations and fail layout inference.

        return kernel

    return make_kernel()


def chunk_bwd_dqkwg_tilelang(
    q, k, v, do, h, dh,
    w=None, g=None, g_gamma=None, dv=None,
    scale=None, cu_seqlens=None, chunk_size=64,
    chunk_indices=None, transpose_state_layout=False,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)

    if scale is None:
        scale = K ** -0.5

    USE_G = g is not None
    USE_DW = w is not None

    # Outputs — kernel writes directly, no intermediate buffers
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dw_out = torch.empty_like(w) if USE_DW else None
    dg = torch.zeros(NK, B, T, H, dtype=torch.float32, device=q.device) if USE_G else None

    # h/dh: flatten batch*chunk dims only (no padding needed)
    h_flat = h.reshape(-1, h.shape[-2], h.shape[-1])
    dh_flat = dh.reshape(-1, dh.shape[-2], dh.shape[-1])
    total_h = h_flat.shape[0]
    hD1, hD2 = h_flat.shape[-2], h_flat.shape[-1]

    dtype_str = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}[q.dtype]

    kernel = _build_kernel(
        B, T, H, K, V, NT, BT, BK, BV, NK,
        total_h, hD1, hD2, dtype_str,
        USE_G, USE_DW, transpose_state_layout,
    )

    # Kernel signature: q, k, v, g, h, do, dh, dq, dk, dw, dv, dg, scale
    # TileLang's LegalizeSafeMemoryAccess handles OOB reads (zero-fill)
    # and OOB writes (skip), so no padding needed.
    kernel(
        q, k, v,
        g if USE_G else q.new_zeros(1, 1, 1),
        h_flat, do, dh_flat,
        dq, dk,
        dw_out if USE_DW else q.new_zeros(1, 1, 1, 1),
        dv if USE_DW else q.new_zeros(1, 1, 1, 1),
        dg if USE_G else q.new_zeros(1, 1, 1, 1),
        scale,
    )

    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw_out, dg
