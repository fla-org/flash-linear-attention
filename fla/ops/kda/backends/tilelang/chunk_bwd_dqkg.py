# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang implementation of chunk_kda_bwd_wy_dqkg_fused.

Ported from the Triton kernel in fla/ops/kda/chunk_bwd.py.
Key structural differences from the common chunk_bwd_dqkwg kernel:
  - K-loop is INTERNAL (no NK grid dimension): grid is (NT, B*H) or (NT, H) for varlen
  - Gate g is 2D: (B, T, H, K), not (B, T, H)
  - Extra inputs: v_new, beta, A (attention matrix)
  - Extra outputs: dv2, db, dA
  - dgk is per-K accumulator (BK,), not scalar
  - dA post-processing: causal mask -> dA@A -> A@(dA@A) -> negate
  - exp2 is always used (no exp/exp2 option)
"""


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
def _build_kda_bwd_kernel(
    B, H, K, V, BT, BK, BV,
    hD1, hD2,
    dtype_str,
    TRANSPOSE_STATE, IS_VARLEN=False,
    num_warps=4,
):
    dtype_map = {'float16': T.float16, 'bfloat16': T.bfloat16, 'float32': T.float32}
    dtype = dtype_map[dtype_str]
    NK = tilelang.cdiv(K, BK)
    NV = tilelang.cdiv(V, BV)
    threads = num_warps * 32
    tile_hD1, tile_hD2 = (BV, BK) if TRANSPOSE_STATE else (BK, BV)

    # Rebind to underscore-prefixed locals for closure capture
    _B, _H, _K, _V = B, H, K, V
    _BT, _BK, _BV, _NK, _NV = BT, BK, BV, NK, NV
    _hD1, _hD2, _thD1, _thD2 = hD1, hD2, tile_hD1, tile_hD2
    _dtype = dtype
    _threads = threads
    _TS = TRANSPOSE_STATE
    _VAR = IS_VARLEN

    T_d, NT_d, total_h_d, N_seqs_d = T.dynamic("T, NT, total_h, N_seqs")

    qk_s = (_B, T_d, _H, _K)
    v_s = (_B, T_d, _H, _V)
    g_s = (_B, T_d, _H, _K)       # 2D gate: per-K dimension
    beta_s = (_B, T_d, _H)
    A_s = (_B, T_d, _H, _BT)      # attention matrix: (B, T, H, BT)
    h_s = (total_h_d, _hD1, _hD2)
    db_s = (_B, T_d, _H)
    dA_s = (_B, T_d, _H, _BT)

    @T.macro
    def kernel_body(q, k, v, v_new, g, beta, A, h, do, dh, dq, dk, dv, dv2, dg, db, dA, scale,
                    i_b, i_h, t_s, T_seq, i_t_local, h_idx):
        # ========== 1. Load beta and A for this chunk ==========
        s_beta = T.alloc_shared((_BT,), T.float32)
        T.copy(beta[i_b, t_s:t_s + _BT, i_h], s_beta, disable_tma=True)
        # Zero OOB positions for varlen safety
        for _i in T.Parallel(_BT):
            if (i_t_local * _BT + _i) >= T_seq:
                s_beta[_i] = 0.0

        # A in Triton is loaded transposed: b_A[bt, t] = A[i_b, t_s+t, i_h, bt]
        # Load, transpose, and zero OOB rows/cols for varlen safety
        s_A_raw = T.alloc_shared((_BT, _BT), _dtype)
        T.copy(A[i_b, t_s:t_s + _BT, i_h, 0:_BT], s_A_raw)
        s_A = T.alloc_shared((_BT, _BT), _dtype)
        for _i, _j in T.Parallel(_BT, _BT):
            valid = ((i_t_local * _BT + _i) < T_seq) & ((i_t_local * _BT + _j) < T_seq)
            s_A[_i, _j] = T.if_then_else(valid, s_A_raw[_j, _i], T.cast(0, _dtype))

        # ========== 2. Init dA, db accumulators ==========
        b_dA = T.alloc_fragment((_BT, _BT), T.float32)
        T.clear(b_dA)
        b_db = T.alloc_fragment((_BT,), T.float32)
        T.clear(b_db)

        # ========== 3. Pre-K V-loop (dA_v contribution, dv2, db_v) ==========
        # These computations are K-independent: dA += dv@v^T, dv2 = A@dv*beta, db += sum((A@dv)*v)
        s_dv_pre = T.alloc_shared((_BT, _BV), _dtype)
        s_v_orig = T.alloc_shared((_BT, _BV), _dtype)
        b_dvb = T.alloc_fragment((_BT, _BV), T.float32)
        s_dvb = T.alloc_shared((_BT, _BV), T.float32)
        s_dv2 = T.alloc_shared((_BT, _BV), _dtype)
        s_dvb_v = T.alloc_shared((_BT, _BV), T.float32)
        f_dvv = T.alloc_fragment((_BT, _BV), T.float32)
        f_dvv_row = T.alloc_fragment((_BT,), T.float32)
        s_dvv_row = T.alloc_shared((_BT,), T.float32)
        s_db_tmp = T.alloc_shared((_BT,), T.float32)

        for i_v in T.serial(_NV):
            v_off = i_v * _BV

            T.copy(dv[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_dv_pre)
            T.copy(v[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_v_orig)

            # b_dA += dv @ v^T (BT, BT)
            T.gemm(s_dv_pre, s_v_orig, b_dA, transpose_B=True)

            # b_dvb = A @ dv (BT, BV)
            T.clear(b_dvb)
            T.gemm(s_A, s_dv_pre, b_dvb)

            # Store dv2 = dvb * beta[:, None]
            T.copy(b_dvb, s_dvb)

            for _i, _j in T.Parallel(_BT, _BV):
                s_dv2[_i, _j] = T.cast(s_dvb[_i, _j] * s_beta[_i], _dtype)
            for _i, _j in T.Parallel(_BT, _BV):
                if (i_t_local * _BT + _i) < T_seq:
                    dv2[i_b, t_s + _i, i_h, v_off + _j] = s_dv2[_i, _j]

            # db += sum(dvb * v, dim=1)
            for _i, _j in T.Parallel(_BT, _BV):
                s_dvb_v[_i, _j] = s_dvb[_i, _j] * T.cast(s_v_orig[_i, _j], T.float32)
            T.copy(s_dvb_v, f_dvv)
            T.reduce_sum(f_dvv, f_dvv_row, dim=1)
            # Accumulate into b_db via shared roundtrip
            T.copy(f_dvv_row, s_dvv_row)
            T.copy(b_db, s_db_tmp)
            for _i in T.Parallel(_BT):
                s_db_tmp[_i] = s_db_tmp[_i] + s_dvv_row[_i]
            T.copy(s_db_tmp, b_db)

        # ========== 4. K-loop ==========
        for i_k in T.serial(_NK):
            k_off = i_k * _BK

            # Load k, g for this K-block (zero OOB rows for varlen cross-row safety)
            s_k = T.alloc_shared((_BT, _BK), _dtype)
            T.copy(k[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK], s_k)
            for _i, _j in T.Parallel(_BT, _BK):
                if (i_t_local * _BT + _i) >= T_seq:
                    s_k[_i, _j] = T.cast(0, _dtype)

            s_g = T.alloc_shared((_BT, _BK), T.float32)
            T.copy(g[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK], s_g)
            for _i, _j in T.Parallel(_BT, _BK):
                if (i_t_local * _BT + _i) >= T_seq:
                    s_g[_i, _j] = 0.0

            # g at last valid position: (BK,)
            last_pos = T.max(0, T.min(_BT, T_seq - i_t_local * _BT) - 1)
            s_gn = T.alloc_shared((_BK,), T.float32)
            for _j in T.Parallel(_BK):
                s_gn[_j] = s_g[last_pos, _j]

            # Per-K accumulators
            b_dq = T.alloc_fragment((_BT, _BK), T.float32)
            b_dk = T.alloc_fragment((_BT, _BK), T.float32)
            b_dw = T.alloc_fragment((_BT, _BK), T.float32)
            T.clear(b_dq)
            T.clear(b_dk)
            T.clear(b_dw)

            # dgk: per-K accumulator for sum(h*dh, axis=0) -- shape (BK,)
            # Use shared memory (avoid 1D fragment/reducer issues)
            s_dgk = T.alloc_shared((_BK,), T.float32)
            for _j in T.Parallel(_BK):
                s_dgk[_j] = 0.0

            # ========== 4a. V-loop (dq, dk, dw, dgk) ==========
            s_v_new = T.alloc_shared((_BT, _BV), _dtype)
            s_do = T.alloc_shared((_BT, _BV), _dtype)
            s_h = T.alloc_shared((_thD1, _thD2), _dtype)
            s_dh = T.alloc_shared((_thD1, _thD2), _dtype)
            s_dv_tile = T.alloc_shared((_BT, _BV), _dtype)
            s_hdh = T.alloc_shared((_thD1, _thD2), T.float32)
            f_hdh = T.alloc_fragment((_thD1, _thD2), T.float32)
            f_hdh_k_nots = T.alloc_fragment((_thD1,), T.float32)
            s_hdh_k_nots = T.alloc_shared((_thD1,), T.float32)
            s_hdh_t = T.alloc_shared((_thD2, _thD1), T.float32)
            f_hdh_t = T.alloc_fragment((_thD2, _thD1), T.float32)
            f_hdh_k_ts = T.alloc_fragment((_thD2,), T.float32)
            s_hdh_k_ts = T.alloc_shared((_thD2,), T.float32)

            for i_v in T.serial(_NV):
                v_off = i_v * _BV

                T.copy(v_new[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_v_new)
                T.copy(do[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_do)

                if _TS:
                    T.copy(h[h_idx, v_off:v_off + _BV, k_off:k_off + _BK], s_h)
                    T.copy(dh[h_idx, v_off:v_off + _BV, k_off:k_off + _BK], s_dh)
                else:
                    T.copy(h[h_idx, k_off:k_off + _BK, v_off:v_off + _BV], s_h)
                    T.copy(dh[h_idx, k_off:k_off + _BK, v_off:v_off + _BV], s_dh)

                # GEMMs first (consume s_h/s_dh before any alloc might reuse their memory)
                # b_dq += do @ h, b_dk += v_new @ dh
                if _TS:
                    T.gemm(s_do, s_h, b_dq)
                    T.gemm(s_v_new, s_dh, b_dk)
                else:
                    T.gemm(s_do, s_h, b_dq, transpose_B=True)
                    T.gemm(s_v_new, s_dh, b_dk, transpose_B=True)

                # b_dw += dv @ h (dv feeds into dw → cross-row dA/dkgb, needs OOB mask)
                T.copy(dv[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_dv_tile)

                if _TS:
                    T.gemm(s_dv_tile, s_h, b_dw)
                else:
                    T.gemm(s_dv_tile, s_h, b_dw, transpose_B=True)

                # dgk += sum(h * dh, axis=0) -- per-column sum -> (BK,)
                # Done AFTER GEMMs so s_hdh alloc can't alias s_h/s_dh
                for _i, _j in T.Parallel(_thD1, _thD2):
                    s_hdh[_i, _j] = T.cast(s_h[_i, _j], T.float32) * T.cast(s_dh[_i, _j], T.float32)

                T.copy(s_hdh, f_hdh)
                if not _TS:
                    T.reduce_sum(f_hdh, f_hdh_k_nots, dim=1)
                    T.copy(f_hdh_k_nots, s_hdh_k_nots)

                    for _j in T.Parallel(_BK):
                        s_dgk[_j] = s_dgk[_j] + s_hdh_k_nots[_j]

                else:
                    for _i, _j in T.Parallel(_thD2, _thD1):
                        s_hdh_t[_i, _j] = s_hdh[_j, _i]

                    T.copy(s_hdh_t, f_hdh_t)
                    T.reduce_sum(f_hdh_t, f_hdh_k_ts, dim=1)
                    T.copy(f_hdh_k_ts, s_hdh_k_ts)

                    for _j in T.Parallel(_BK):
                        s_dgk[_j] = s_dgk[_j] + s_hdh_k_ts[_j]

            # ========== 4b. Gate dq, dk ==========
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

            # ========== 4c. kg = k * exp2(g), dgk *= exp2(gn) ==========
            s_kg = T.alloc_shared((_BT, _BK), _dtype)
            for _i, _j in T.Parallel(_BT, _BK):
                s_kg[_i, _j] = T.cast(T.cast(s_k[_i, _j], T.float32) * T.exp2(s_g[_i, _j]), _dtype)

            for _j in T.Parallel(_BK):
                s_dgk[_j] = s_dgk[_j] * T.exp2(s_gn[_j])

            # ========== 4d. dw = -dw ==========
            s_dw = T.alloc_shared((_BT, _BK), _dtype)
            T.copy(b_dw, s_dw)
            for _i, _j in T.Parallel(_BT, _BK):
                s_dw[_i, _j] = T.cast(-T.cast(s_dw[_i, _j], T.float32), _dtype)

            # ========== 4e. dA += dw @ kg^T ==========
            T.gemm(s_dw, s_kg, b_dA, transpose_B=True)

            # ========== 4f. dkgb = A @ dw ==========
            b_dkgb = T.alloc_fragment((_BT, _BK), T.float32)
            T.clear(b_dkgb)
            T.gemm(s_A, s_dw, b_dkgb)

            # ========== 4g. db += sum(dkgb * kg, dim=1) ==========
            s_dkgb = T.alloc_shared((_BT, _BK), T.float32)
            T.copy(b_dkgb, s_dkgb)
            s_dkgb_kg = T.alloc_shared((_BT, _BK), T.float32)
            for _i, _j in T.Parallel(_BT, _BK):
                s_dkgb_kg[_i, _j] = s_dkgb[_i, _j] * T.cast(s_kg[_i, _j], T.float32)
            f_dkgb_kg = T.alloc_fragment((_BT, _BK), T.float32)
            T.copy(s_dkgb_kg, f_dkgb_kg)
            f_db_row = T.alloc_fragment((_BT,), T.float32)
            T.reduce_sum(f_dkgb_kg, f_db_row, dim=1)
            s_db_row = T.alloc_shared((_BT,), T.float32)
            T.copy(f_db_row, s_db_row)
            s_db_cur = T.alloc_shared((_BT,), T.float32)
            T.copy(b_db, s_db_cur)
            for _i in T.Parallel(_BT):
                s_db_cur[_i] = s_db_cur[_i] + s_db_row[_i]
            T.copy(s_db_cur, b_db)

            # ========== 4h. dgk += col_sum(k * dk) ==========
            s_q = T.alloc_shared((_BT, _BK), _dtype)
            T.copy(q[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK], s_q)

            s_kdk = T.alloc_shared((_BT, _BK), T.float32)
            for _i, _j in T.Parallel(_BT, _BK):
                s_kdk[_i, _j] = T.cast(s_k[_i, _j], T.float32) * s_dk[_i, _j]
            # Column sum of kdk -> (BK,) via transpose then row-reduce
            s_kdk_t = T.alloc_shared((_BK, _BT), T.float32)
            for _i, _j in T.Parallel(_BK, _BT):
                s_kdk_t[_i, _j] = s_kdk[_j, _i]
            f_kdk_t = T.alloc_fragment((_BK, _BT), T.float32)
            T.copy(s_kdk_t, f_kdk_t)
            f_kdk_col = T.alloc_fragment((_BK,), T.float32)
            T.reduce_sum(f_kdk_t, f_kdk_col, dim=1)
            s_kdk_col = T.alloc_shared((_BK,), T.float32)
            T.copy(f_kdk_col, s_kdk_col)
            for _j in T.Parallel(_BK):
                s_dgk[_j] = s_dgk[_j] + s_kdk_col[_j]

            # ========== 4i. dg = q*dq - kdk + m_last*dgk + kg*dkgb*beta ==========
            s_dg_out = T.alloc_shared((_BT, _BK), T.float32)
            m_last_pos = T.max(0, T.min(_BT, T_seq - i_t_local * _BT) - 1)
            for _i, _j in T.Parallel(_BT, _BK):
                q_dq = T.cast(s_q[_i, _j], T.float32) * s_dq[_i, _j]
                last_term = T.if_then_else(_i == m_last_pos, s_dgk[_j], 0.0)
                beta_term = T.cast(s_kg[_i, _j], T.float32) * s_dkgb[_i, _j] * s_beta[_i]
                s_dg_out[_i, _j] = q_dq - s_kdk[_i, _j] + last_term + beta_term

            # ========== 4j. dk += dkgb * gb where gb = exp2(g) * beta ==========
            for _i, _j in T.Parallel(_BT, _BK):
                s_dk[_i, _j] = s_dk[_i, _j] + s_dkgb[_i, _j] * T.exp2(s_g[_i, _j]) * s_beta[_i]

            # ========== 4k. Store dq, dk, dg for this K-block ==========
            for _i, _j in T.Parallel(_BT, _BK):
                if (i_t_local * _BT + _i) < T_seq:
                    dq[i_b, t_s + _i, i_h, k_off + _j] = s_dq[_i, _j]
            for _i, _j in T.Parallel(_BT, _BK):
                if (i_t_local * _BT + _i) < T_seq:
                    dk[i_b, t_s + _i, i_h, k_off + _j] = s_dk[_i, _j]
            for _i, _j in T.Parallel(_BT, _BK):
                if (i_t_local * _BT + _i) < T_seq:
                    dg[i_b, t_s + _i, i_h, k_off + _j] = s_dg_out[_i, _j]

        # ========== 5. dA post-processing ==========
        # m_A = (i > j) & valid
        # b_dA = where(m_A, b_dA * beta[j], 0)
        # b_dA = b_dA @ A
        # b_dA = A @ b_dA
        # b_dA = where(m_A, -b_dA, 0)

        s_dA = T.alloc_shared((_BT, _BT), T.float32)
        T.copy(b_dA, s_dA)

        # Apply causal mask and beta weighting
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
        s_dA2 = T.alloc_shared((_BT, _BT), _dtype)
        f_dA2 = T.alloc_fragment((_BT, _BT), _dtype)
        for _i, _j in T.Parallel(_BT, _BT):
            f_dA2[_i, _j] = T.cast(b_dA2[_i, _j], _dtype)
        T.copy(f_dA2, s_dA2)

        b_dA3 = T.alloc_fragment((_BT, _BT), T.float32)
        T.clear(b_dA3)
        T.gemm(s_A, s_dA2, b_dA3)

        # Apply mask and negate
        s_dA_final = T.alloc_shared((_BT, _BT), T.float32)
        T.copy(b_dA3, s_dA_final)

        for _i, _j in T.Parallel(_BT, _BT):
            m_valid = ((i_t_local * _BT + _i) < T_seq) & ((i_t_local * _BT + _j) < T_seq)
            m_upper = (_i > _j) & m_valid
            s_dA_final[_i, _j] = T.if_then_else(m_upper, -s_dA_final[_i, _j], 0.0)

        for _i, _j in T.Parallel(_BT, _BT):
            if (i_t_local * _BT + _i) < T_seq:
                dA[i_b, t_s + _i, i_h, _j] = s_dA_final[_i, _j]

        # ========== 6. Store db ==========
        s_db_final = T.alloc_shared((_BT,), T.float32)
        T.copy(b_db, s_db_final)

        for _i in T.Parallel(_BT):
            if (i_t_local * _BT + _i) < T_seq:
                db[i_b, t_s + _i, i_h] = s_db_final[_i]

    # ========== Kernel entry points ==========
    if _VAR:
        @T.prim_func
        def kernel(
            q: T.Tensor(qk_s, _dtype), k: T.Tensor(qk_s, _dtype),
            v: T.Tensor(v_s, _dtype), v_new: T.Tensor(v_s, _dtype),
            g: T.Tensor(g_s, T.float32), beta: T.Tensor(beta_s, T.float32),
            A: T.Tensor(A_s, _dtype),
            h: T.Tensor(h_s, _dtype), do: T.Tensor(v_s, _dtype),
            dh: T.Tensor(h_s, _dtype),
            dq: T.Tensor(qk_s, T.float32), dk: T.Tensor(qk_s, T.float32),
            dv: T.Tensor(v_s, _dtype), dv2: T.Tensor(v_s, _dtype),
            dg: T.Tensor(g_s, T.float32), db: T.Tensor(db_s, T.float32),
            dA: T.Tensor(dA_s, T.float32),
            cu_seqlens: T.Tensor((N_seqs_d,), T.int32),
            chunk_indices: T.Tensor((NT_d, 2), T.int32),
            scale: T.float32,
        ):
            with T.Kernel(NT_d, _H, threads=_threads) as (i_t, i_h):
                i_n = chunk_indices[i_t, 0]
                i_t_local = chunk_indices[i_t, 1]
                bos = cu_seqlens[i_n]
                T_seq = cu_seqlens[i_n + 1] - bos
                h_idx = i_t * _H + i_h
                t_s = bos + i_t_local * _BT
                kernel_body(q, k, v, v_new, g, beta, A, h, do, dh,
                            dq, dk, dv, dv2, dg, db, dA,
                            scale, 0, i_h, t_s, T_seq, i_t_local, h_idx)
    else:
        @T.prim_func
        def kernel(
            q: T.Tensor(qk_s, _dtype), k: T.Tensor(qk_s, _dtype),
            v: T.Tensor(v_s, _dtype), v_new: T.Tensor(v_s, _dtype),
            g: T.Tensor(g_s, T.float32), beta: T.Tensor(beta_s, T.float32),
            A: T.Tensor(A_s, _dtype),
            h: T.Tensor(h_s, _dtype), do: T.Tensor(v_s, _dtype),
            dh: T.Tensor(h_s, _dtype),
            dq: T.Tensor(qk_s, T.float32), dk: T.Tensor(qk_s, T.float32),
            dv: T.Tensor(v_s, _dtype), dv2: T.Tensor(v_s, _dtype),
            dg: T.Tensor(g_s, T.float32), db: T.Tensor(db_s, T.float32),
            dA: T.Tensor(dA_s, T.float32),
            scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(T_d, _BT), _B * _H, threads=_threads) as (i_t, i_bh):
                i_b = i_bh // _H
                i_h = i_bh % _H
                NT_local = T.ceildiv(T_d, _BT)
                h_idx = (i_b * NT_local + i_t) * _H + i_h
                t_s = i_t * _BT
                kernel_body(q, k, v, v_new, g, beta, A, h, do, dh,
                            dq, dk, dv, dv2, dg, db, dA,
                            scale, i_b, i_h, t_s, T_d, i_t, h_idx)

    return kernel


def chunk_kda_bwd_wy_dqkg_fused_tilelang(
    q, k, v, v_new, g, beta, A, h, do, dh, dv,
    scale=None, cu_seqlens=None, chunk_size=64,
    chunk_indices=None, transpose_state_layout=False,
):
    B, _, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    # Ensure float32 inputs match kernel signature
    g = g.float()
    beta = beta.float()
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    IS_VARLEN = cu_seqlens is not None

    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    if scale is None:
        scale = K ** -0.5

    # Outputs
    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)

    h_flat = h.reshape(-1, h.shape[-2], h.shape[-1])
    dh_flat = dh.reshape(-1, dh.shape[-2], dh.shape[-1])
    hD1, hD2 = h_flat.shape[-2], h_flat.shape[-1]
    dtype_str = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}[q.dtype]

    # Static tiling config
    BK = min(max(triton.next_power_of_2(K), 16), 64)
    BV = min(max(triton.next_power_of_2(V), 16), 64)
    num_warps = 8 if K <= 64 else 4

    kernel = _build_kda_bwd_kernel(
        B, H, K, V, BT, BK, BV,
        hD1, hD2, dtype_str,
        transpose_state_layout, IS_VARLEN,
        num_warps=num_warps,
    )

    if IS_VARLEN:
        kernel(q, k, v, v_new, g, beta, A, h_flat, do, dh_flat,
               dq, dk, dv, dv2, dg, db, dA,
               cu_seqlens.int(), chunk_indices.int(), scale)
    else:
        kernel(q, k, v, v_new, g, beta, A, h_flat, do, dh_flat,
               dq, dk, dv, dv2, dg, db, dA, scale)

    dv = dv2
    return dq, dk, dv, db, dg, dA
