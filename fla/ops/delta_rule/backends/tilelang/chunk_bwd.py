# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang fused delta-rule WY + dq/dk backward experiment.

This ports the op-local Triton fused-WY algebra into a one-program-per-chunk
TileLang kernel. The D128 route removes the separate ``chunk_bwd_dqkwg`` and
``prepare_wy_repr_bwd`` launches for dense ungated delta rule while keeping the
current D256 Triton route as fallback unless explicitly opted in for comparison.
"""

from pathlib import Path

import tilelang
import tilelang.language as T
import torch
import triton


_COMMON_TILELANG_DIR = Path(__file__).parents[3] / "common" / "backends" / "tilelang"
_CUDA126_FP8_E8M0_STUB = _COMMON_TILELANG_DIR / "cuda126_fp8_e8m0_stub.cuh"
_TILELANG_COMPILE_FLAGS = ["-include", str(_CUDA126_FP8_E8M0_STUB)]


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}, compile_flags=_TILELANG_COMPILE_FLAGS)
def _build_delta_rule_wy_dqkw_fused_kernel(
    B,
    H,
    K,
    V,
    BT,
    BK,
    BV,
    hD1,
    hD2,
    dtype_str,
    num_warps=4,
):
    dtype_map = {'float16': T.float16, 'bfloat16': T.bfloat16, 'float32': T.float32}
    _dtype = dtype_map[dtype_str]
    _B, _H, _K, _V = B, H, K, V
    _BT, _BK, _BV = BT, BK, BV
    _NK = tilelang.cdiv(K, BK)
    _NV = tilelang.cdiv(V, BV)
    _hD1, _hD2 = hD1, hD2
    _threads = num_warps * 32

    T_d, total_h_d = T.dynamic("T, total_h")

    qk_s = (_B, T_d, _H, _K)
    v_s = (_B, T_d, _H, _V)
    beta_s = (_B, T_d, _H)
    A_s = (_B, T_d, _H, _BT)
    h_s = (total_h_d, _hD1, _hD2)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_s, _dtype),
        k: T.Tensor(qk_s, _dtype),
        v: T.Tensor(v_s, _dtype),
        v_new: T.Tensor(v_s, _dtype),
        beta: T.Tensor(beta_s, _dtype),
        A: T.Tensor(A_s, _dtype),
        h: T.Tensor(h_s, _dtype),
        do: T.Tensor(v_s, _dtype),
        dh: T.Tensor(h_s, _dtype),
        dv: T.Tensor(v_s, _dtype),
        dq: T.Tensor(qk_s, _dtype),
        dk: T.Tensor(qk_s, _dtype),
        dv2: T.Tensor(v_s, _dtype),
        dbeta: T.Tensor(beta_s, _dtype),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(T_d, _BT), _B * _H, threads=_threads) as (i_t, i_bh):
            i_b = i_bh // _H
            i_h = i_bh % _H
            NT_local = T.ceildiv(T_d, _BT)
            h_idx = (i_b * NT_local + i_t) * _H + i_h
            t_s = i_t * _BT

            s_beta = T.alloc_shared((_BT,), T.float32)
            for _i in T.Parallel(_BT):
                valid = (i_t * _BT + _i) < T_d
                s_beta[_i] = T.if_then_else(
                    valid,
                    T.cast(beta[i_b, t_s + _i, i_h], T.float32),
                    0.0,
                )

            s_A_src = T.alloc_shared((_BT, _BT), _dtype)
            T.copy(A[i_b, t_s:t_s + _BT, i_h, 0:_BT], s_A_src, disable_tma=True)
            s_A = T.alloc_shared((_BT, _BT), _dtype)
            for _i, _j in T.Parallel(_BT, _BT):
                valid = ((i_t * _BT + _i) < T_d) & ((i_t * _BT + _j) < T_d)
                s_A[_i, _j] = T.if_then_else(valid, s_A_src[_j, _i], T.cast(0, _dtype))

            b_ds = T.alloc_fragment((_BT, _BT), T.float32)
            b_dA = T.alloc_fragment((_BT, _BT), T.float32)
            b_dbeta = T.alloc_fragment((_BT,), T.float32)
            T.clear(b_ds)
            T.clear(b_dA)
            T.clear(b_dbeta)

            s_do = T.alloc_shared((_BT, _BV), _dtype)
            s_v_new = T.alloc_shared((_BT, _BV), _dtype)
            s_v = T.alloc_shared((_BT, _BV), _dtype)
            s_v_beta = T.alloc_shared((_BT, _BV), _dtype)
            s_dv = T.alloc_shared((_BT, _BV), _dtype)
            b_dvb = T.alloc_fragment((_BT, _BV), T.float32)
            f_dvv = T.alloc_fragment((_BT, _BV), T.float32)
            f_row_v = T.alloc_fragment((_BT,), T.float32)
            f_dv2 = T.alloc_fragment((_BT, _BV), _dtype)
            s_dv2 = T.alloc_shared((_BT, _BV), _dtype)

            for i_v in T.Pipelined(_NV, num_stages=2):
                v_off = i_v * _BV
                T.copy(do[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_do, disable_tma=True)
                T.copy(v_new[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_v_new, disable_tma=True)
                T.copy(v[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_v, disable_tma=True)
                T.copy(dv[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_dv, disable_tma=True)
                for _i, _j in T.Parallel(_BT, _BV):
                    s_v_beta[_i, _j] = T.cast(T.cast(s_v[_i, _j], T.float32) * s_beta[_i], _dtype)

                T.gemm(s_do, s_v_new, b_ds, transpose_B=True)
                T.gemm(s_dv, s_v_beta, b_dA, transpose_B=True)

                T.clear(b_dvb)
                T.gemm(s_A, s_dv, b_dvb)
                for _i, _j in T.Parallel(_BT, _BV):
                    f_dv2[_i, _j] = T.cast(b_dvb[_i, _j] * s_beta[_i], _dtype)
                    f_dvv[_i, _j] = b_dvb[_i, _j] * T.cast(s_v[_i, _j], T.float32)
                T.copy(f_dv2, s_dv2)
                T.sync_threads()
                for _i, _j in T.Parallel(_BT, _BV):
                    if (i_t * _BT + _i) < T_d:
                        dv2[i_b, t_s + _i, i_h, v_off + _j] = s_dv2[_i, _j]
                T.reduce_sum(f_dvv, f_row_v, dim=1)
                for _i in T.Parallel(_BT):
                    b_dbeta[_i] = b_dbeta[_i] + f_row_v[_i]

            s_ds = T.alloc_shared((_BT, _BT), _dtype)
            f_ds = T.alloc_fragment((_BT, _BT), _dtype)
            for _i, _j in T.Parallel(_BT, _BT):
                valid = ((i_t * _BT + _i) < T_d) & ((i_t * _BT + _j) < T_d)
                causal = (_i >= _j) & valid
                f_ds[_i, _j] = T.if_then_else(causal, T.cast(b_ds[_i, _j], _dtype), T.cast(0, _dtype))
            T.copy(f_ds, s_ds)

            s_q = T.alloc_shared((_BT, _BK), _dtype)
            s_k = T.alloc_shared((_BT, _BK), _dtype)
            s_h = T.alloc_shared((_BK, _BV), _dtype)
            s_dh = T.alloc_shared((_BK, _BV), _dtype)
            s_dw = T.alloc_shared((_BT, _BK), _dtype)
            s_k_beta = T.alloc_shared((_BT, _BK), _dtype)
            s_out = T.alloc_shared((_BT, _BK), _dtype)
            f_out = T.alloc_fragment((_BT, _BK), _dtype)
            f_dkk = T.alloc_fragment((_BT, _BK), T.float32)
            f_row_k = T.alloc_fragment((_BT,), T.float32)

            for i_k in T.serial(_NK):
                k_off = i_k * _BK
                T.copy(q[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK], s_q, disable_tma=True)
                T.copy(k[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK], s_k, disable_tma=True)
                for _i, _j in T.Parallel(_BT, _BK):
                    s_k_beta[_i, _j] = T.cast(T.cast(s_k[_i, _j], T.float32) * s_beta[_i], _dtype)

                b_dq = T.alloc_fragment((_BT, _BK), T.float32)
                b_dk = T.alloc_fragment((_BT, _BK), T.float32)
                b_dk_ds = T.alloc_fragment((_BT, _BK), T.float32)
                b_dw = T.alloc_fragment((_BT, _BK), T.float32)
                b_dk_beta = T.alloc_fragment((_BT, _BK), T.float32)
                T.clear(b_dq)
                T.clear(b_dk)
                T.clear(b_dk_ds)
                T.clear(b_dw)

                for i_v in T.Pipelined(_NV, num_stages=2):
                    v_off = i_v * _BV
                    T.copy(do[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_do, disable_tma=True)
                    T.copy(v_new[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_v_new, disable_tma=True)
                    T.copy(dv[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_dv, disable_tma=True)
                    T.copy(h[h_idx, k_off:k_off + _BK, v_off:v_off + _BV], s_h, disable_tma=True)
                    T.copy(dh[h_idx, k_off:k_off + _BK, v_off:v_off + _BV], s_dh, disable_tma=True)

                    T.gemm(s_do, s_h, b_dq, transpose_B=True)
                    T.gemm(s_v_new, s_dh, b_dk, transpose_B=True)
                    T.gemm(s_dv, s_h, b_dw, transpose_B=True)

                T.gemm(s_ds, s_k, b_dq)
                T.gemm(s_ds, s_q, b_dk_ds, transpose_A=True)

                for _i, _j in T.Parallel(_BT, _BK):
                    b_dq[_i, _j] = b_dq[_i, _j] * scale
                    b_dk[_i, _j] = b_dk[_i, _j] + b_dk_ds[_i, _j] * scale
                    s_dw[_i, _j] = T.cast(-b_dw[_i, _j], _dtype)

                T.gemm(s_dw, s_k_beta, b_dA, transpose_B=True)

                T.clear(b_dk_beta)
                T.gemm(s_A, s_dw, b_dk_beta)
                for _i, _j in T.Parallel(_BT, _BK):
                    b_dk[_i, _j] = b_dk[_i, _j] + b_dk_beta[_i, _j] * s_beta[_i]
                    f_dkk[_i, _j] = b_dk_beta[_i, _j] * T.cast(s_k[_i, _j], T.float32)
                T.reduce_sum(f_dkk, f_row_k, dim=1)
                for _i in T.Parallel(_BT):
                    b_dbeta[_i] = b_dbeta[_i] + f_row_k[_i]

                for _i, _j in T.Parallel(_BT, _BK):
                    f_out[_i, _j] = T.cast(b_dq[_i, _j], _dtype)
                T.copy(f_out, s_out)
                T.sync_threads()
                for _i, _j in T.Parallel(_BT, _BK):
                    if (i_t * _BT + _i) < T_d:
                        dq[i_b, t_s + _i, i_h, k_off + _j] = s_out[_i, _j]
                for _i, _j in T.Parallel(_BT, _BK):
                    f_out[_i, _j] = T.cast(b_dk[_i, _j], _dtype)
                T.copy(f_out, s_out)
                T.sync_threads()
                for _i, _j in T.Parallel(_BT, _BK):
                    if (i_t * _BT + _i) < T_d:
                        dk[i_b, t_s + _i, i_h, k_off + _j] = s_out[_i, _j]

            s_dA = T.alloc_shared((_BT, _BT), T.float32)
            T.copy(b_dA, s_dA)
            for _i, _j in T.Parallel(_BT, _BT):
                valid = ((i_t * _BT + _i) < T_d) & ((i_t * _BT + _j) < T_d)
                strict_lower = (_i > _j) & valid
                s_dA[_i, _j] = T.if_then_else(strict_lower, s_dA[_i, _j], 0.0)

            s_dA_dtype = T.alloc_shared((_BT, _BT), _dtype)
            for _i, _j in T.Parallel(_BT, _BT):
                s_dA_dtype[_i, _j] = T.cast(s_dA[_i, _j], _dtype)

            b_dA2 = T.alloc_fragment((_BT, _BT), T.float32)
            T.clear(b_dA2)
            T.gemm(s_dA_dtype, s_A, b_dA2)
            for _i, _j in T.Parallel(_BT, _BT):
                s_dA_dtype[_i, _j] = T.cast(b_dA2[_i, _j], _dtype)

            b_dA3 = T.alloc_fragment((_BT, _BT), T.float32)
            T.clear(b_dA3)
            T.gemm(s_A, s_dA_dtype, b_dA3)

            s_dA_final = T.alloc_shared((_BT, _BT), _dtype)
            for _i, _j in T.Parallel(_BT, _BT):
                valid = ((i_t * _BT + _i) < T_d) & ((i_t * _BT + _j) < T_d)
                strict_lower = (_i > _j) & valid
                s_dA_final[_i, _j] = T.if_then_else(
                    strict_lower,
                    T.cast(-b_dA3[_i, _j], _dtype),
                    T.cast(0, _dtype),
                )

            s_dk_prev = T.alloc_shared((_BT, _BK), _dtype)
            for i_k in T.serial(_NK):
                k_off = i_k * _BK
                T.copy(k[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK], s_k, disable_tma=True)
                T.copy(dk[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK], s_dk_prev, disable_tma=True)
                for _i, _j in T.Parallel(_BT, _BK):
                    s_k_beta[_i, _j] = T.cast(T.cast(s_k[_i, _j], T.float32) * s_beta[_i], _dtype)

                b_dk_beta = T.alloc_fragment((_BT, _BK), T.float32)
                b_dk_extra = T.alloc_fragment((_BT, _BK), T.float32)
                T.clear(b_dk_beta)
                T.clear(b_dk_extra)
                T.gemm(s_dA_final, s_k, b_dk_beta)
                T.gemm(s_dA_final, s_k_beta, b_dk_extra, transpose_A=True)

                for _i, _j in T.Parallel(_BT, _BK):
                    f_dkk[_i, _j] = b_dk_beta[_i, _j] * T.cast(s_k[_i, _j], T.float32)
                    f_out[_i, _j] = T.cast(
                        T.cast(s_dk_prev[_i, _j], T.float32)
                        + b_dk_extra[_i, _j]
                        + b_dk_beta[_i, _j] * s_beta[_i],
                        _dtype,
                    )
                T.reduce_sum(f_dkk, f_row_k, dim=1)
                for _i in T.Parallel(_BT):
                    b_dbeta[_i] = b_dbeta[_i] + f_row_k[_i]

                T.copy(f_out, s_out)
                T.sync_threads()
                for _i, _j in T.Parallel(_BT, _BK):
                    if (i_t * _BT + _i) < T_d:
                        dk[i_b, t_s + _i, i_h, k_off + _j] = s_out[_i, _j]

            for _i in T.Parallel(_BT):
                if (i_t * _BT + _i) < T_d:
                    dbeta[i_b, t_s + _i, i_h] = T.cast(b_dbeta[_i], _dtype)

    return kernel


def chunk_delta_rule_wy_dqkw_fused_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    V = v.shape[-1]
    if K != V or K not in (128, 256):
        raise ValueError(f"TileLang fused delta WY/DQKW requires K == V in {{128, 256}}, got K={K}, V={V}")
    if scale is None:
        scale = K ** -0.5

    BT = chunk_size
    BK = 64
    BV = 64
    dtype_str = {torch.float16: 'float16', torch.bfloat16: 'bfloat16'}[q.dtype]

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv2 = torch.empty_like(v)
    dbeta = torch.empty_like(beta)

    h_flat = h.reshape(-1, h.shape[-2], h.shape[-1])
    dh_flat = dh.reshape(-1, dh.shape[-2], dh.shape[-1])
    hD1, hD2 = h_flat.shape[-2], h_flat.shape[-1]

    kernel = _build_delta_rule_wy_dqkw_fused_kernel(
        B,
        H,
        K,
        V,
        BT,
        BK,
        BV,
        hD1,
        hD2,
        dtype_str,
        num_warps=4,
    )
    kernel(q, k, v, v_new, beta, A, h_flat, do, dh_flat, dv, dq, dk, dv2, dbeta, scale)
    return dq, dk, dv2, dbeta
