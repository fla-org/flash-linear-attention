# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang implementation of chunk_bwd_kernel_dqkwg.

1:1 translation of the Triton kernel in fla/ops/common/chunk_o.py.
All tensors are flattened to (B*T, H, D) to match Triton pointer arithmetic.
"""

# NOTE: do NOT use `from __future__ import annotations` —
# TileLang resolves @T.prim_func annotations eagerly.

from functools import lru_cache

import tilelang
import tilelang.language as T
import torch
import triton

from fla.ops.utils import prepare_chunk_indices
from fla.utils import check_shared_mem


@lru_cache(maxsize=64)
def _build_kernel(
    B, T_val, BT_total, H, K, V, NT, BT, BK, BV, NK,
    total_h,
    dtype_str,
    USE_G, USE_G_GAMMA, USE_DW, TRANSPOSE_STATE, IS_VARLEN,
    num_warps=4,
):
    dtype_map = {'float16': T.float16, 'bfloat16': T.bfloat16, 'float32': T.float32}
    dtype = dtype_map[dtype_str]
    NV = tilelang.cdiv(V, BV)
    threads = num_warps * 32

    hD1, hD2 = (V, K) if TRANSPOSE_STATE else (K, V)
    # Tile dimensions for h/dh shared buffers (BK×BV or BV×BK)
    tile_hD1, tile_hD2 = (BV, BK) if TRANSPOSE_STATE else (BK, BV)

    # All q/k/v/do/dv/dq/dk/dw are (BT_total, H, K_or_V) — flattened B*T
    qk_s = (BT_total, H, K)
    v_s = (BT_total, H, V)
    h_s = (total_h, hD1, hD2)
    # Only declare shapes for tensors that the kernel actually uses.
    # Unused tensor params cause TileLang layout inference to fail.
    g_s = (BT_total, H)
    dv_s = v_s
    dw_s = qk_s
    dg_s = (NK, BT_total, H)

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    })
    def make_kernel(
        _B=B, _T=T_val, _H=H, _K=K, _V=V,
        _NT=NT, _BT=BT, _BK=BK, _BV=BV, _NK=NK,
        _NV=NV, _hD1=hD1, _hD2=hD2, _thD1=tile_hD1, _thD2=tile_hD2,
        _dtype=dtype, _threads=threads,
        _USE_G=USE_G, _USE_G_GAMMA=USE_G_GAMMA,
        _USE_DW=USE_DW, _TS=TRANSPOSE_STATE, _VAR=IS_VARLEN,
    ):
        # Only declare tensor params that the kernel uses.
        # Extra unused params cause TileLang layout inference failure.
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
            dw: T.Tensor(dw_s, _dtype),
            dv: T.Tensor(dv_s, _dtype),
            dg: T.Tensor(dg_s, T.float32),
            scale: T.float32,
            T_orig: T.int32,
        ):
            with T.Kernel(_NK, _NT, _B * _H, threads=_threads) as (i_k, i_t, i_bh):
                i_b = i_bh // _H
                i_h = i_bh % _H

                # -- index computation --
                # All tensor shapes use _T (= T_pad, block-aligned).
                # T_orig (runtime) is only for boundary checks.
                if _VAR:
                    i_tg = i_t
                    i_n = chunk_indices[i_t, 0]  # noqa: F821
                    i_t_local = chunk_indices[i_t, 1]  # noqa: F821
                    bos = cu_seqlens[i_n]  # noqa: F821
                    T_seq = cu_seqlens[i_n + 1] - bos  # noqa: F821
                else:
                    i_tg = i_b * _NT + i_t
                    i_t_local = i_t
                    bos = i_b * _T
                    T_seq = T_orig

                h_idx = i_tg * _H + i_h   # index into flattened h
                t_off = bos + i_t_local * _BT   # row offset in (B*T, H, D) tensors
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

                # ========== V-loop: Python-unrolled (NV is compile-time) ==========
                for i_v_py in range(NV):
                    v_off_c = i_v_py * _BV

                    T.copy(v[t_off:t_off + _BT, i_h, v_off_c:v_off_c + _BV], s_v)
                    T.copy(do[t_off:t_off + _BT, i_h, v_off_c:v_off_c + _BV], s_do)

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
                        T.copy(dv[t_off:t_off + _BT, i_h, v_off_c:v_off_c + _BV], s_dv)
                        if _TS:
                            T.gemm(s_dv, s_h, b_dw)
                        else:
                            T.gemm(s_dv, s_h, b_dw, transpose_B=True)

                    # dg_last: h*dh sum merged into V-loop (avoids re-reading h/dh)
                    if _USE_G:
                        for _i, _j in T.Parallel(_thD1, _thD2):
                            s_h[_i, _j] = s_h[_i, _j] * s_dh[_i, _j]
                        T.sync_threads()
                        # reduce h*dh to scalar via fragment reduce
                        f_hdh = T.alloc_fragment((_thD1, _thD2), T.float32)
                        T.copy(s_h, f_hdh)
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
                    T.copy(f_dw, dw[t_off:t_off + _BT, i_h, k_off:k_off + _BK])

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
                T.copy(q[t_off:t_off + _BT, i_h, k_off:k_off + _BK], s_q)
                T.copy(k[t_off:t_off + _BT, i_h, k_off:k_off + _BK], s_k)

                f_q = T.alloc_fragment((_BT, _BK), T.float32)
                f_k = T.alloc_fragment((_BT, _BK), T.float32)
                T.copy(s_q, f_q)
                T.copy(s_k, f_k)

                # ========== USE_G path ==========
                if _USE_G:
                    # dg_last_acc already reduced to `red` scalar above

                    # g in shared memory (all threads can cross-read any element)
                    s_g = T.alloc_shared((_BT,), T.float32)
                    T.copy(g[t_off:t_off + _BT, i_h], s_g)

                    last_pos = T.min(_BT, T_seq - i_t_local * _BT) - 1
                    g_last = s_g[last_pos]
                    b_dg_last = T.alloc_var(T.float32)
                    b_dg_last = red[0] * T.exp(g_last)

                    # b_dq *= exp(g) * scale  (inline, no extra fragment)
                    for _i, _j in T.Parallel(_BT, _BK):
                        b_dq[_i, _j] = b_dq[_i, _j] * T.exp(s_g[_i]) * scale

                    # dg reductions: use shared memory s_dg instead of fragment b_dg.
                    # A 1D fragment b_dg[BT] compiles to b_dg[1] per thread (one element).
                    # T.Parallel(BT) with serial inner body gets serialized, so
                    # b_dg[_i] always writes to b_dg[0], losing per-row values.
                    # Shared memory is correctly indexed by _i in all cases.
                    s_dg = T.alloc_shared((_BT,), T.float32)
                    s_A = T.alloc_shared((_BT, _BK), T.float32)

                    # s_dg = sum(b_dq * q, axis=1)
                    # Use T.reduce_sum for parallel warp-level reduction:
                    # frag→shared, shared*=shared, shared→frag, reduce_sum, frag→shared
                    T.copy(b_dq, s_A)
                    T.sync_threads()
                    for _i, _j in T.Parallel(_BT, _BK):
                        s_A[_i, _j] = s_A[_i, _j] * s_q[_i, _j]
                    T.sync_threads()
                    f_prod = T.alloc_fragment((_BT, _BK), T.float32)
                    T.copy(s_A, f_prod)
                    f_dg_row = T.alloc_fragment((_BT,), T.float32)
                    T.reduce_sum(f_prod, f_dg_row, dim=1)
                    T.copy(f_dg_row, s_dg)
                    T.sync_threads()

                    # b_dk *= exp(-g + g_last) — elementwise gating
                    for _i, _j in T.Parallel(_BT, _BK):
                        b_dk[_i, _j] = b_dk[_i, _j] * T.exp(-s_g[_i] + g_last)

                    # s_dg -= sum(k * dk, axis=1)  AND  b_dg_last += sum(dk * k)
                    T.copy(b_dk, s_A)
                    T.sync_threads()
                    for _i, _j in T.Parallel(_BT, _BK):
                        s_A[_i, _j] = s_A[_i, _j] * s_k[_i, _j]
                    T.sync_threads()
                    T.copy(s_A, f_prod)
                    T.reduce_sum(f_prod, f_dg_row, dim=1)  # reuse f_dg_row
                    # Write row sums to shared for dg subtraction + dg_last
                    s_tmp1d = T.alloc_shared((_BT,), T.float32)
                    T.copy(f_dg_row, s_tmp1d)
                    T.sync_threads()
                    for _i in T.Parallel(_BT):
                        s_dg[_i] = s_dg[_i] - s_tmp1d[_i]
                    T.sync_threads()
                    # b_dg_last += sum(dk * k) via serial sum over shared
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
                    T.copy(f_out, dq[t_off:t_off + _BT, i_h, k_off:k_off + _BK])
                    for _i, _j in T.Parallel(_BT, _BK):
                        f_out[_i, _j] = T.cast(b_dk[_i, _j], _dtype)
                    T.copy(f_out, dk[t_off:t_off + _BT, i_h, k_off:k_off + _BK])

                    # store dg: merge dg_last into last valid position
                    for _i in T.Parallel(_BT):
                        val = T.if_then_else(_i == last_pos, s_dg[_i] + b_dg_last, s_dg[_i])
                        dg[i_k, t_off + _i, i_h] = val

                # NOTE: USE_G_GAMMA and no-gating branches removed.
                # The TileLang backend verifier rejects those cases.
                # Keeping dead branches here causes TileLang to see extra
                # fragment allocations and fail layout inference.

        return kernel

    return make_kernel()


# ---------------------------------------------------------------------------
# Python wrapper – same signature as chunk_bwd_dqkwg in chunk_o.py
# ---------------------------------------------------------------------------

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
    USE_G_GAMMA = g_gamma is not None
    USE_DW = w is not None
    IS_VARLEN = cu_seqlens is not None

    # allocate outputs
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if USE_G else None
    dw_out = torch.empty_like(w) if USE_DW else None

    # TileLang T.gemm requires block-aligned K/V dims (Triton uses boundary_check).
    # Pad K and V to multiples of BK/BV with zeros.
    K_pad = NK * BK
    V_pad = triton.cdiv(V, BV) * BV
    pad_k = K_pad - K
    pad_v = V_pad - V

    def pad_dim(x, pad, dim=-1):
        if pad == 0:
            return x
        dim = dim % x.ndim
        pads = [0] * (2 * x.ndim)
        pads[2 * (x.ndim - 1 - dim) + 1] = pad
        return torch.nn.functional.pad(x, pads)

    # flatten (B, T) -> (B*T), pad T to NT*BT for block alignment
    T_pad = NT * BT
    BT_total = B * T_pad
    pad_t = T_pad - T
    # reshape each batch to (T_pad, H, D) with zero-padding, then flatten

    def reshape_pad_flat(x, D, pad_d):
        # x: (B, T, H, D) → (B, T_pad, H, D_pad) → (BT_total, H, D_pad)
        xr = x.reshape(B, T, H, D)
        if pad_t > 0:
            xr = torch.nn.functional.pad(xr, (0, 0, 0, 0, 0, pad_t))  # pad T dim
        xr = xr.reshape(BT_total, H, D)
        return pad_dim(xr, pad_d) if pad_d > 0 else xr

    q_flat = reshape_pad_flat(q, K, pad_k)
    k_flat = reshape_pad_flat(k, K, pad_k)
    v_flat = reshape_pad_flat(v, V, pad_v)
    do_flat = reshape_pad_flat(do, V, pad_v)

    # outputs: allocate padded, will slice back
    dq_flat = torch.zeros(BT_total, H, K_pad, dtype=q.dtype, device=q.device)
    dk_flat = torch.zeros(BT_total, H, K_pad, dtype=k.dtype, device=k.device)
    dw_flat = torch.zeros(BT_total, H, K_pad, dtype=k.dtype, device=k.device) if USE_DW else None
    dv_flat = reshape_pad_flat(dv, V, pad_v) if USE_DW else None

    # h/dh: (B, NT, H, K, V) or (V, K) -> flatten first 3 dims, pad last 2
    h_flat = h.reshape(-1, h.shape[-2], h.shape[-1])
    dh_flat = dh.reshape(-1, dh.shape[-2], dh.shape[-1])
    if transpose_state_layout:
        h_flat = pad_dim(pad_dim(h_flat, pad_k, -1), pad_v, -2)
        dh_flat = pad_dim(pad_dim(dh_flat, pad_k, -1), pad_v, -2)
    else:
        h_flat = pad_dim(pad_dim(h_flat, pad_v, -1), pad_k, -2)
        dh_flat = pad_dim(pad_dim(dh_flat, pad_v, -1), pad_k, -2)
    total_h = h_flat.shape[0]

    if USE_G:
        g_r = g.reshape(B, T, H)
        if pad_t > 0:
            g_r = torch.nn.functional.pad(g_r, (0, 0, 0, pad_t))
        g_flat = g_r.reshape(BT_total, H)
        dg_flat = torch.zeros(NK, BT_total, H, dtype=torch.float32, device=g.device)
    else:
        g_flat = None
        dg_flat = None

    # dummies for optional args
    dummy1 = q.new_zeros(1)
    dummy_ci = torch.zeros(1, 2, dtype=torch.int32, device=q.device)
    cu_i32 = cu_seqlens.to(torch.int32) if IS_VARLEN else dummy1.to(torch.int32)
    ci_i32 = chunk_indices.to(torch.int32) if IS_VARLEN else dummy_ci

    dtype_str = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}[q.dtype]

    kernel = _build_kernel(
        B, T_pad, BT_total, H, K_pad, V_pad, NT, BT, BK, BV, NK, total_h,
        dtype_str, USE_G, USE_G_GAMMA, USE_DW,
        transpose_state_layout, IS_VARLEN,
    )

    # Only pass tensors that match the kernel signature
    # (unused params break TileLang layout inference)
    args = [q_flat, k_flat, v_flat]
    if USE_G:
        args.append(g_flat)
    args.extend([h_flat, do_flat, dh_flat, dq_flat, dk_flat])
    if USE_DW:
        args.extend([dw_flat, dv_flat])
    if USE_G:
        args.append(dg_flat)
    if IS_VARLEN:
        args.extend([cu_i32, ci_i32])
    args.append(scale)
    args.append(T)  # T_orig for boundary checks
    kernel(*args)

    # slice padded outputs back to original (B, T, H, K)
    def unpad_flat(flat, D_orig):
        return flat.reshape(B, T_pad, H, -1)[:, :T, :, :D_orig].reshape(B, T, H, D_orig)

    dq.copy_(unpad_flat(dq_flat, K))
    dk.copy_(unpad_flat(dk_flat, K))
    if dw_out is not None:
        dw_out.copy_(unpad_flat(dw_flat, K))

    if dg is not None:
        dg = dg_flat.sum(0).reshape(B, T_pad, H)[:, :T, :].contiguous()
    return dq, dk, dw_out, dg
