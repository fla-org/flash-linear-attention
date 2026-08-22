# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang implementation of chunk_kda_bwd_wy_dqkg_fused.

Ported from the Triton kernel in fla/ops/kda/chunk_bwd.py.
Key structural differences from the common chunk_bwd_dqkwg kernel:
  - K-loop is INTERNAL (no NK grid dimension): grid is (NT, B*HV)
  - Gate g is 2D: (B, T, HV, K), not (B, T, HV)
  - Extra inputs: v_new, beta, A (attention matrix)
  - Extra outputs: dv2, db, dA
  - dgk is per-K accumulator (BK,), not scalar
  - dA post-processing: causal mask -> dA@A -> A@(dA@A) -> negate
  - exp2 is always used (no exp/exp2 option)
"""


import tilelang
import tilelang.language as T
import torch

from fla.utils import check_shared_mem


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _tile_extent(dim: int, const_tiling: int) -> int:
    return min(max(_next_power_of_2(dim), 16), const_tiling)


_DTYPE_NAMES = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}


def _check_cuda_contiguous_supported(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"KDA TileLang backend requires {name} to be a CUDA tensor")
    if tensor.dtype not in _DTYPE_NAMES:
        raise ValueError(f"KDA TileLang backend does not support {name} dtype {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"KDA TileLang backend requires {name} to be contiguous")


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def _build_kda_bwd_kernel(
    B, H, HV, K, V, BT, BK, BV,
    hD1, hD2,
    dtype_str, beta_dtype_str,
    STATE_V_FIRST,
    num_warps=4,
):
    dtype_map = {'float16': T.float16, 'bfloat16': T.bfloat16, 'float32': T.float32}
    dtype = dtype_map[dtype_str]
    beta_dtype = dtype_map[beta_dtype_str]
    NK = tilelang.cdiv(K, BK)
    NV = tilelang.cdiv(V, BV)
    threads = num_warps * 32
    tile_hD1, tile_hD2 = (BV, BK) if STATE_V_FIRST else (BK, BV)

    # rebind to underscore-prefixed locals for closure capture
    _B, _H, _HV, _K, _V = B, H, HV, K, V
    _G = HV // H
    _BT, _BK, _BV, _NK, _NV = BT, BK, BV, NK, NV
    _hD1, _hD2, _thD1, _thD2 = hD1, hD2, tile_hD1, tile_hD2
    _dtype = dtype
    _beta_dtype = beta_dtype
    _threads = threads
    _TS = STATE_V_FIRST

    T_d, total_h_d = T.dynamic("T, total_h")

    qk_s = (_B, T_d, _H, _K)
    dqk_s = (_B, T_d, _HV, _K)
    v_s = (_B, T_d, _HV, _V)
    g_s = (_B, T_d, _HV, _K)       # 2D gate: per-K dimension
    beta_s = (_B, T_d, _HV)
    A_s = (_B, T_d, _HV, _BT)      # attention matrix: (B, T, HV, BT)
    h_s = (total_h_d, _hD1, _hD2)
    db_s = (_B, T_d, _HV)
    dA_s = (_B, T_d, _HV, _BT)

    @T.macro
    def kernel_body(q, k, v, v_new, g, beta, A, h, do, dh, dq, dk, dv, dv2, dg, db, dA, scale,
                    i_b, i_hv, t_s, T_seq, i_t_local, h_idx):
        i_h = i_hv // _G
        # load beta and A for this chunk
        s_beta = T.alloc_shared((_BT,), T.float32)
        T.copy(beta[i_b, t_s:t_s + _BT, i_hv], s_beta, disable_tma=True)
        # zero boundary positions outside the active sequence span
        for _i in T.Parallel(_BT):
            if (i_t_local * _BT + _i) >= T_seq:
                s_beta[_i] = 0.0

        # load A as Triton does, transposed: b_A[bt, t] = A[i_b, t_s+t, i_hv, bt]
        # load, transpose, and zero boundary rows/cols outside the active sequence span
        s_A = T.alloc_shared((_BT, _BT), _dtype)
        T.copy(A[i_b, t_s:t_s + _BT, i_hv, 0:_BT], s_A)
        f_A_tmp = T.alloc_fragment((_BT, _BT), _dtype)
        for _i, _j in T.Parallel(_BT, _BT):
            valid = ((i_t_local * _BT + _i) < T_seq) & ((i_t_local * _BT + _j) < T_seq)
            f_A_tmp[_i, _j] = T.if_then_else(valid, s_A[_j, _i], T.cast(0, _dtype))
        T.copy(f_A_tmp, s_A)

        # initialize dA and db accumulators
        b_dA = T.alloc_fragment((_BT, _BT), T.float32)
        T.clear(b_dA)
        b_db = T.alloc_fragment((_BT,), T.float32)
        T.clear(b_db)

        # run K-independent V work: dA += dv@v^T, dv2 = A@dv*beta, db += sum((A@dv)*v)
        s_dv_pre = T.alloc_shared((_BT, _BV), _dtype)
        s_v_orig = T.alloc_shared((_BT, _BV), _dtype)
        b_dvb = T.alloc_fragment((_BT, _BV), T.float32)
        s_dv2 = T.alloc_shared((_BT, _BV), _dtype)
        f_dvv = T.alloc_fragment((_BT, _BV), T.float32)
        f_dvv_row = T.alloc_fragment((_BT,), T.float32)

        for i_v in T.serial(_NV):
            v_off = i_v * _BV

            T.copy(dv[i_b, t_s:t_s + _BT, i_hv, v_off:v_off + _BV], s_dv_pre)
            T.copy(v[i_b, t_s:t_s + _BT, i_hv, v_off:v_off + _BV], s_v_orig)

            # b_dA += dv @ v^T (BT, BT)
            T.gemm(s_dv_pre, s_v_orig, b_dA, transpose_B=True)

            # b_dvb = A @ dv (BT, BV)
            T.clear(b_dvb)
            T.gemm(s_A, s_dv_pre, b_dvb)

            # store dv2 = dvb * beta[:, None]
            for _i, _j in T.Parallel(_BT, _BV):
                s_dv2[_i, _j] = T.cast(b_dvb[_i, _j] * s_beta[_i], _dtype)
            for _i, _j in T.Parallel(_BT, _BV):
                if (i_t_local * _BT + _i) < T_seq:
                    dv2[i_b, t_s + _i, i_hv, v_off + _j] = s_dv2[_i, _j]

            # db += sum(dvb * v, dim=1)
            for _i, _j in T.Parallel(_BT, _BV):
                f_dvv[_i, _j] = b_dvb[_i, _j] * T.cast(s_v_orig[_i, _j], T.float32)
            T.reduce_sum(f_dvv, f_dvv_row, dim=1)
            for _i in T.Parallel(_BT):
                b_db[_i] = b_db[_i] + f_dvv_row[_i]

        # run the K loop
        for i_k in T.serial(_NK):
            k_off = i_k * _BK

            # load k and g for this K-block, zeroing boundary rows outside the active sequence span
            s_k = T.alloc_shared((_BT, _BK), _dtype)
            T.copy(k[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK], s_k)
            for _i, _j in T.Parallel(_BT, _BK):
                if (i_t_local * _BT + _i) >= T_seq:
                    s_k[_i, _j] = T.cast(0, _dtype)

            s_g = T.alloc_shared((_BT, _BK), T.float32)
            T.copy(g[i_b, t_s:t_s + _BT, i_hv, k_off:k_off + _BK], s_g)
            for _i, _j in T.Parallel(_BT, _BK):
                if (i_t_local * _BT + _i) >= T_seq:
                    s_g[_i, _j] = 0.0

            # g at last valid position: (BK,)
            last_pos = T.max(0, T.min(_BT, T_seq - i_t_local * _BT) - 1)
            s_gn = T.alloc_shared((_BK,), T.float32)
            for _j in T.Parallel(_BK):
                s_gn[_j] = s_g[last_pos, _j]

            # per-K accumulators
            b_dq = T.alloc_fragment((_BT, _BK), T.float32)
            b_dk = T.alloc_fragment((_BT, _BK), T.float32)
            b_dw = T.alloc_fragment((_BT, _BK), T.float32)
            T.clear(b_dq)
            T.clear(b_dk)
            T.clear(b_dw)

            # dgk: per-K accumulator for sum(h*dh, axis=0) -- shape (BK,)
            # use shared memory to avoid 1D fragment/reducer issues
            s_dgk = T.alloc_shared((_BK,), T.float32)
            for _j in T.Parallel(_BK):
                s_dgk[_j] = 0.0

            # run the V loop for dq, dk, dw, and dgk
            s_v_new = T.alloc_shared((_BT, _BV), _dtype)
            s_do = T.alloc_shared((_BT, _BV), _dtype)
            s_h = T.alloc_shared((_thD1, _thD2), _dtype)
            s_dh = T.alloc_shared((_thD1, _thD2), _dtype)
            s_dv_tile = T.alloc_shared((_BT, _BV), _dtype)
            f_hdh = T.alloc_fragment((_thD1, _thD2), T.float32)
            f_hdh_k_nots = T.alloc_fragment((_thD1,), T.float32)
            f_hdh_t = T.alloc_fragment((_thD2, _thD1), T.float32)
            f_hdh_k_ts = T.alloc_fragment((_thD2,), T.float32)

            for i_v in T.serial(_NV):
                v_off = i_v * _BV

                T.copy(v_new[i_b, t_s:t_s + _BT, i_hv, v_off:v_off + _BV], s_v_new)
                T.copy(do[i_b, t_s:t_s + _BT, i_hv, v_off:v_off + _BV], s_do)

                if _TS:
                    T.copy(h[h_idx, v_off:v_off + _BV, k_off:k_off + _BK], s_h)
                    T.copy(dh[h_idx, v_off:v_off + _BV, k_off:k_off + _BK], s_dh)
                else:
                    T.copy(h[h_idx, k_off:k_off + _BK, v_off:v_off + _BV], s_h)
                    T.copy(dh[h_idx, k_off:k_off + _BK, v_off:v_off + _BV], s_dh)

                # consume s_h and s_dh before another allocation can reuse their memory
                # b_dq += do @ h, b_dk += v_new @ dh
                if _TS:
                    T.gemm(s_do, s_h, b_dq)
                    T.gemm(s_v_new, s_dh, b_dk)
                else:
                    T.gemm(s_do, s_h, b_dq, transpose_B=True)
                    T.gemm(s_v_new, s_dh, b_dk, transpose_B=True)

                # b_dw += dv @ h (dv feeds into dw → cross-row dA/dkgb, needs OOB mask)
                T.copy(dv[i_b, t_s:t_s + _BT, i_hv, v_off:v_off + _BV], s_dv_tile)

                if _TS:
                    T.gemm(s_dv_tile, s_h, b_dw)
                else:
                    T.gemm(s_dv_tile, s_h, b_dw, transpose_B=True)

                # dgk += sum(h * dh, axis=0) -- per-column sum -> (BK,)
                for _i, _j in T.Parallel(_thD1, _thD2):
                    f_hdh[_i, _j] = T.cast(s_h[_i, _j], T.float32) * T.cast(s_dh[_i, _j], T.float32)

                if not _TS:
                    T.reduce_sum(f_hdh, f_hdh_k_nots, dim=1)
                    for _j in T.Parallel(_BK):
                        s_dgk[_j] = s_dgk[_j] + f_hdh_k_nots[_j]

                else:
                    for _i, _j in T.Parallel(_thD2, _thD1):
                        f_hdh_t[_i, _j] = f_hdh[_j, _i]

                    T.reduce_sum(f_hdh_t, f_hdh_k_ts, dim=1)
                    for _j in T.Parallel(_BK):
                        s_dgk[_j] = s_dgk[_j] + f_hdh_k_ts[_j]

            # apply gate factors to dq and dk
            # b_dq = b_dq * exp2(g) * scale
            s_dq = T.alloc_shared((_BT, _BK), T.float32)
            T.copy(b_dq, s_dq)
            for _i, _j in T.Parallel(_BT, _BK):
                s_dq[_i, _j] = s_dq[_i, _j] * T.exp2(s_g[_i, _j]) * scale

            # b_dk = b_dk * where(m_t, exp2(gn - g), 0)
            s_dk = T.alloc_shared((_BT, _BK), T.float32)
            T.copy(b_dk, s_dk)
            for _i, _j in T.Parallel(_BT, _BK):
                m_t = (i_t_local * _BT + _i) < T_seq
                s_dk[_i, _j] = T.if_then_else(
                    m_t,
                    s_dk[_i, _j] * T.exp2(s_gn[_j] - s_g[_i, _j]),
                    0.0)

            # compute kg = k * exp2(g) and scale dgk by exp2(gn)
            s_kg = T.alloc_shared((_BT, _BK), _dtype)
            for _i, _j in T.Parallel(_BT, _BK):
                s_kg[_i, _j] = T.cast(T.cast(s_k[_i, _j], T.float32) * T.exp2(s_g[_i, _j]), _dtype)

            for _j in T.Parallel(_BK):
                s_dgk[_j] = s_dgk[_j] * T.exp2(s_gn[_j])

            # negate dw
            s_dw = T.alloc_shared((_BT, _BK), _dtype)
            T.copy(b_dw, s_dw)
            for _i, _j in T.Parallel(_BT, _BK):
                s_dw[_i, _j] = T.cast(-T.cast(s_dw[_i, _j], T.float32), _dtype)

            # accumulate dA += dw @ kg^T
            T.gemm(s_dw, s_kg, b_dA, transpose_B=True)

            # compute dkgb = A @ dw
            b_dkgb = T.alloc_fragment((_BT, _BK), T.float32)
            T.clear(b_dkgb)
            T.gemm(s_A, s_dw, b_dkgb)

            # accumulate db += sum(dkgb * kg, dim=1)
            f_dkgb_kg = T.alloc_fragment((_BT, _BK), T.float32)
            for _i, _j in T.Parallel(_BT, _BK):
                f_dkgb_kg[_i, _j] = b_dkgb[_i, _j] * T.cast(s_kg[_i, _j], T.float32)
            f_db_row = T.alloc_fragment((_BT,), T.float32)
            T.reduce_sum(f_dkgb_kg, f_db_row, dim=1)
            for _i in T.Parallel(_BT):
                b_db[_i] = b_db[_i] + f_db_row[_i]

            # accumulate dgk += col_sum(k * dk)
            f_kdk = T.alloc_fragment((_BT, _BK), T.float32)
            for _i, _j in T.Parallel(_BT, _BK):
                f_kdk[_i, _j] = T.cast(s_k[_i, _j], T.float32) * s_dk[_i, _j]
            # column-sum kdk into (BK,) via transpose then row-reduce
            f_kdk_t = T.alloc_fragment((_BK, _BT), T.float32)
            for _i, _j in T.Parallel(_BK, _BT):
                f_kdk_t[_i, _j] = f_kdk[_j, _i]
            f_kdk_col = T.alloc_fragment((_BK,), T.float32)
            T.reduce_sum(f_kdk_t, f_kdk_col, dim=1)
            s_kdk_col = T.alloc_shared((_BK,), T.float32)
            T.copy(f_kdk_col, s_kdk_col)
            for _j in T.Parallel(_BK):
                s_dgk[_j] = s_dgk[_j] + s_kdk_col[_j]

            # compute dg = q*dq - kdk + m_last*dgk + kg*dkgb*beta
            s_dg_out = T.alloc_shared((_BT, _BK), T.float32)
            m_last_pos = T.max(0, T.min(_BT, T_seq - i_t_local * _BT) - 1)
            for _i, _j in T.Parallel(_BT, _BK):
                q_dq = T.cast(q[i_b, t_s + _i, i_h, k_off + _j], T.float32) * s_dq[_i, _j]
                last_term = T.if_then_else(_i == m_last_pos, s_dgk[_j], 0.0)
                beta_term = T.cast(s_kg[_i, _j], T.float32) * b_dkgb[_i, _j] * s_beta[_i]
                s_dg_out[_i, _j] = q_dq - f_kdk[_i, _j] + last_term + beta_term

            # accumulate dk += dkgb * gb where gb = exp2(g) * beta
            for _i, _j in T.Parallel(_BT, _BK):
                s_dk[_i, _j] = s_dk[_i, _j] + b_dkgb[_i, _j] * T.exp2(s_g[_i, _j]) * s_beta[_i]

            # store dq, dk, and dg for this K-block
            for _i, _j in T.Parallel(_BT, _BK):
                if (i_t_local * _BT + _i) < T_seq:
                    dq[i_b, t_s + _i, i_hv, k_off + _j] = s_dq[_i, _j]
            for _i, _j in T.Parallel(_BT, _BK):
                if (i_t_local * _BT + _i) < T_seq:
                    dk[i_b, t_s + _i, i_hv, k_off + _j] = s_dk[_i, _j]
            for _i, _j in T.Parallel(_BT, _BK):
                if (i_t_local * _BT + _i) < T_seq:
                    dg[i_b, t_s + _i, i_hv, k_off + _j] = s_dg_out[_i, _j]

        # post-process dA
        # m_A = (i > j) & valid
        # b_dA = where(m_A, b_dA * beta[j], 0)
        # b_dA = b_dA @ A
        # b_dA = A @ b_dA
        # b_dA = where(m_A, -b_dA, 0)

        s_dA = T.alloc_shared((_BT, _BT), T.float32)
        T.copy(b_dA, s_dA)

        # apply causal mask and beta weighting
        for _i, _j in T.Parallel(_BT, _BT):
            m_valid = ((i_t_local * _BT + _i) < T_seq) & ((i_t_local * _BT + _j) < T_seq)
            m_upper = (_i > _j) & m_valid
            s_dA[_i, _j] = T.if_then_else(m_upper, s_dA[_i, _j] * s_beta[_j], 0.0)

        # dA = dA @ A
        s_dA_dtype = T.alloc_shared((_BT, _BT), _dtype)
        for _i, _j in T.Parallel(_BT, _BT):
            s_dA_dtype[_i, _j] = T.cast(s_dA[_i, _j], _dtype)

        b_dA2 = T.alloc_fragment((_BT, _BT), T.float32)
        T.clear(b_dA2)
        T.gemm(s_dA_dtype, s_A, b_dA2)

        # dA = A @ dA
        # reuse s_dA_dtype for the second cast
        for _i, _j in T.Parallel(_BT, _BT):
            s_dA_dtype[_i, _j] = T.cast(b_dA2[_i, _j], _dtype)

        b_dA3 = T.alloc_fragment((_BT, _BT), T.float32)
        T.clear(b_dA3)
        T.gemm(s_A, s_dA_dtype, b_dA3)

        # apply mask and negate, reusing s_dA
        for _i, _j in T.Parallel(_BT, _BT):
            m_valid = ((i_t_local * _BT + _i) < T_seq) & ((i_t_local * _BT + _j) < T_seq)
            m_upper = (_i > _j) & m_valid
            s_dA[_i, _j] = T.if_then_else(m_upper, -b_dA3[_i, _j], 0.0)

        for _i, _j in T.Parallel(_BT, _BT):
            if (i_t_local * _BT + _i) < T_seq:
                dA[i_b, t_s + _i, i_hv, _j] = s_dA[_i, _j]

        # store db
        for _i in T.Parallel(_BT):
            if (i_t_local * _BT + _i) < T_seq:
                db[i_b, t_s + _i, i_hv] = b_db[_i]

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_s, _dtype), k: T.Tensor(qk_s, _dtype),
        v: T.Tensor(v_s, _dtype), v_new: T.Tensor(v_s, _dtype),
        g: T.Tensor(g_s, T.float32), beta: T.Tensor(beta_s, _beta_dtype),
        A: T.Tensor(A_s, _dtype),
        h: T.Tensor(h_s, _dtype), do: T.Tensor(v_s, _dtype),
        dh: T.Tensor(h_s, _dtype),
        dq: T.Tensor(dqk_s, T.float32), dk: T.Tensor(dqk_s, T.float32),
        dv: T.Tensor(v_s, _dtype), dv2: T.Tensor(v_s, _dtype),
        dg: T.Tensor(g_s, T.float32), db: T.Tensor(db_s, T.float32),
        dA: T.Tensor(dA_s, T.float32),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(T_d, _BT), _B * _HV, threads=_threads) as (i_t, i_bh):
            i_b = i_bh // _HV
            i_hv = i_bh % _HV
            NT_local = T.ceildiv(T_d, _BT)
            h_idx = (i_b * NT_local + i_t) * _HV + i_hv
            t_s = i_t * _BT
            kernel_body(q, k, v, v_new, g, beta, A, h, do, dh,
                        dq, dk, dv, dv2, dg, db, dA,
                        scale, i_b, i_hv, t_s, T_d, i_t, h_idx)

    return kernel


def chunk_kda_bwd_wy_dqkg_fused_tilelang(
    q, k, v, v_new, g, beta, A, h, do, dh, dv,
    scale=None, cu_seqlens=None, chunk_size=64,
    chunk_indices=None, state_v_first=False,
):
    data_tensors = {
        "q": q,
        "k": k,
        "v": v,
        "v_new": v_new,
        "A": A,
        "h": h,
        "do": do,
        "dh": dh,
        "dv": dv,
    }
    aux_tensors = {"g": g, "beta": beta}
    for name, tensor in {**data_tensors, **aux_tensors}.items():
        _check_cuda_contiguous_supported(name, tensor)
    if g.dtype != torch.float32:
        raise ValueError(f"KDA TileLang backend requires g dtype torch.float32, got {g.dtype}")
    for name, tensor in data_tensors.items():
        if tensor.dtype != q.dtype:
            raise ValueError(f"KDA TileLang backend requires {name} dtype {tensor.dtype} to match q dtype {q.dtype}")

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("KDA TileLang backend requires q, k, and v to be 4D tensors")
    if q.shape != k.shape:
        raise ValueError(f"KDA TileLang backend requires q and k to have the same shape, got {q.shape} vs {k.shape}")
    if chunk_size not in (32, 64):
        raise ValueError(f"KDA TileLang backend supports chunk_size 32 or 64, got {chunk_size}")

    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    if HV % H != 0:
        raise ValueError(f"num_v_heads (HV={HV}) must be divisible by num_qk_heads (H={H}) for KDA TileLang GVA")
    BT = chunk_size
    if cu_seqlens is not None or chunk_indices is not None:
        raise ValueError("KDA TileLang backend currently supports dense fixed-length sequences only")
    if T % BT != 0:
        raise ValueError(f"KDA TileLang backend requires dense sequence length T={T} to be divisible by chunk_size={BT}")
    if v_new.shape != v.shape or do.shape != v.shape or dv.shape != v.shape:
        raise ValueError(
            "KDA TileLang backend requires v, v_new, do, and dv to share shape "
            f"{v.shape}; got v_new={v_new.shape}, do={do.shape}, dv={dv.shape}"
        )
    if g.shape != (B, T, HV, K):
        raise ValueError(f"KDA TileLang backend requires g shape {(B, T, HV, K)}, got {g.shape}")
    if beta.shape != (B, T, HV):
        raise ValueError(f"KDA TileLang backend requires beta shape {(B, T, HV)}, got {beta.shape}")
    if A.shape != (B, T, HV, BT):
        raise ValueError(f"KDA TileLang backend requires A shape {(B, T, HV, BT)}, got {A.shape}")
    state_shape = (V, K) if state_v_first else (K, V)
    expected_state_shape = (B, T // BT, HV, *state_shape)
    if h.shape != expected_state_shape or dh.shape != expected_state_shape:
        raise ValueError(
            f"KDA TileLang backend requires h/dh shape {expected_state_shape}, "
            f"got h={h.shape}, dh={dh.shape}"
        )
    # ensure float32 gate input matches the kernel signature
    g = g.float()

    CONST_TILING = 64 if check_shared_mem(tensor_idx=q.device.index or 0) else 32
    if scale is None:
        scale = K ** -0.5

    # allocate outputs
    dq = torch.empty(B, T, HV, K, dtype=torch.float, device=q.device)
    dk = torch.empty(B, T, HV, K, dtype=torch.float, device=k.device)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)

    h_flat = h.reshape(-1, h.shape[-2], h.shape[-1])
    dh_flat = dh.reshape(-1, dh.shape[-2], dh.shape[-1])
    hD1, hD2 = h_flat.shape[-2], h_flat.shape[-1]
    dtype_str = _DTYPE_NAMES[q.dtype]
    beta_dtype_str = _DTYPE_NAMES[beta.dtype]

    # static tiling config
    BK = _tile_extent(K, CONST_TILING)
    BV = _tile_extent(V, CONST_TILING)
    if K % BK != 0:
        raise ValueError(f"KDA TileLang backend requires K={K} to be divisible by its BK tile {BK}")
    if V % BV != 0:
        raise ValueError(f"KDA TileLang backend requires V={V} to be divisible by its BV tile {BV}")
    num_warps = 8 if K <= 64 else 4

    kernel = _build_kda_bwd_kernel(
        B, H, HV, K, V, BT, BK, BV,
        hD1, hD2, dtype_str, beta_dtype_str,
        state_v_first,
        num_warps=num_warps,
    )

    kernel(q, k, v, v_new, g, beta, A, h_flat, do, dh_flat,
           dq, dk, dv, dv2, dg, db, dA, scale)

    dv = dv2
    return dq, dk, dv, db, dg, dA
