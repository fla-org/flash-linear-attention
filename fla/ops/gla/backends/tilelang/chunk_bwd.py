# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import tilelang
import tilelang.language as T
import torch
import triton


_CUDA126_FP8_E8M0_STUB = (
    Path(__file__).parents[3]
    / "common"
    / "backends"
    / "tilelang"
    / "cuda126_fp8_e8m0_stub.cuh"
)
_TILELANG_COMPILE_FLAGS = ["-include", str(_CUDA126_FP8_E8M0_STUB)]


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}, compile_flags=_TILELANG_COMPILE_FLAGS)
def _build_chunk_gla_fused_bwd_k_tile(
    B,
    H,
    K,
    V,
    BT,
    BK,
    BV,
    NK,
    hD1,
    hD2,
    dtype_str,
    state_dtype_str,
    num_warps=4,
):
    dtype_map = {"float16": T.float16, "bfloat16": T.bfloat16, "float32": T.float32}
    _dtype = dtype_map[dtype_str]
    _state_dtype = dtype_map[state_dtype_str]
    NV = tilelang.cdiv(V, BV)
    threads = num_warps * 32

    _B, _H, _K, _V = B, H, K, V
    _BT, _BK, _BV, _NK, _NV = BT, BK, BV, NK, NV
    _hD1, _hD2 = hD1, hD2
    _threads = threads
    _CAST_STATE_FOR_MMA = state_dtype_str != dtype_str

    T_d, total_h_d = T.dynamic("T, total_h")

    qk_s = (_B, T_d, _H, _K)
    v_s = (_B, T_d, _H, _V)
    g_s = (_B, T_d, _H, _K)
    a_s = (_B, T_d, _H, _BT)
    h_s = (total_h_d, _hD1, _hD2)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_s, _dtype),
        k: T.Tensor(qk_s, _dtype),
        v: T.Tensor(v_s, _dtype),
        g: T.Tensor(g_s, T.float32),
        h: T.Tensor(h_s, _state_dtype),
        do: T.Tensor(v_s, _dtype),
        dh: T.Tensor(h_s, _state_dtype),
        dA: T.Tensor(a_s, T.float32),
        dq: T.Tensor(qk_s, T.float32),
        dk: T.Tensor(qk_s, T.float32),
        dg: T.Tensor(g_s, T.float32),
        scale: T.float32,
    ):
        with T.Kernel(_NK, T.ceildiv(T_d, _BT), _B * _H, threads=_threads) as (i_k, i_t, i_bh):
            i_b = i_bh // _H
            i_h = i_bh % _H
            NT_local = T.ceildiv(T_d, _BT)
            h_idx = (i_b * NT_local + i_t) * _H + i_h
            t_s = i_t * _BT
            k_off = i_k * _BK
            last_pos = T.max(0, T.min(_BT, T_d - i_t * _BT) - 1)

            b_dq_inter = T.alloc_fragment((_BT, _BK), T.float32)
            b_dk_inter = T.alloc_fragment((_BT, _BK), T.float32)
            T.clear(b_dq_inter)
            T.clear(b_dk_inter)

            s_dgk = T.alloc_shared((_BK,), T.float32)
            for _j in T.Parallel(_BK):
                s_dgk[_j] = 0.0

            s_v = T.alloc_shared((_BT, _BV), _dtype)
            s_do = T.alloc_shared((_BT, _BV), _dtype)
            s_h = T.alloc_shared((_BK, _BV), _state_dtype)
            s_dh = T.alloc_shared((_BK, _BV), _state_dtype)
            if _CAST_STATE_FOR_MMA:
                s_h_mma = T.alloc_shared((_BK, _BV), _dtype)
                s_dh_mma = T.alloc_shared((_BK, _BV), _dtype)

            # Inter-chunk contribution: do @ h, v @ dh, and per-K h*dh.
            for i_v_py in T.Pipelined(_NV, num_stages=2):
                v_off = i_v_py * _BV
                T.copy(v[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_v, disable_tma=True)
                T.copy(do[i_b, t_s:t_s + _BT, i_h, v_off:v_off + _BV], s_do, disable_tma=True)
                T.copy(h[h_idx, k_off:k_off + _BK, v_off:v_off + _BV], s_h, disable_tma=True)
                T.copy(dh[h_idx, k_off:k_off + _BK, v_off:v_off + _BV], s_dh, disable_tma=True)

                if _CAST_STATE_FOR_MMA:
                    for _i, _j in T.Parallel(_BK, _BV):
                        s_h_mma[_i, _j] = T.cast(s_h[_i, _j], _dtype)
                        s_dh_mma[_i, _j] = T.cast(s_dh[_i, _j], _dtype)
                    T.gemm(s_do, s_h_mma, b_dq_inter, transpose_B=True)
                    T.gemm(s_v, s_dh_mma, b_dk_inter, transpose_B=True)
                else:
                    T.gemm(s_do, s_h, b_dq_inter, transpose_B=True)
                    T.gemm(s_v, s_dh, b_dk_inter, transpose_B=True)

                f_hdh = T.alloc_fragment((_BK, _BV), T.float32)
                for _i, _j in T.Parallel(_BK, _BV):
                    f_hdh[_i, _j] = T.cast(s_h[_i, _j], T.float32) * T.cast(s_dh[_i, _j], T.float32)
                f_hdh_k = T.alloc_fragment((_BK,), T.float32)
                T.reduce_sum(f_hdh, f_hdh_k, dim=1)
                for _j in T.Parallel(_BK):
                    s_dgk[_j] = s_dgk[_j] + f_hdh_k[_j]

            s_q = T.alloc_shared((_BT, _BK), _dtype)
            s_k = T.alloc_shared((_BT, _BK), _dtype)
            s_g = T.alloc_shared((_BT, _BK), T.float32)
            T.copy(q[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK], s_q, disable_tma=True)
            T.copy(k[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK], s_k, disable_tma=True)
            T.copy(g[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK], s_g, disable_tma=True)

            s_inter_dq = T.alloc_shared((_BT, _BK), T.float32)
            s_inter_dk = T.alloc_shared((_BT, _BK), T.float32)
            for _i, _j in T.Parallel(_BT, _BK):
                b_dq_inter[_i, _j] = b_dq_inter[_i, _j] * T.exp2(s_g[_i, _j]) * scale
                b_dk_inter[_i, _j] = b_dk_inter[_i, _j] * T.exp2(s_g[last_pos, _j] - s_g[_i, _j])
            T.copy(b_dq_inter, s_inter_dq)
            T.copy(b_dk_inter, s_inter_dk)

            for _j in T.Parallel(_BK):
                s_dgk[_j] = s_dgk[_j] * T.exp2(s_g[last_pos, _j])

            f_kdk = T.alloc_fragment((_BT, _BK), T.float32)
            for _i, _j in T.Parallel(_BT, _BK):
                f_kdk[_i, _j] = s_inter_dk[_i, _j] * T.cast(s_k[_i, _j], T.float32)
            f_kdk_t = T.alloc_fragment((_BK, _BT), T.float32)
            for _i, _j in T.Parallel(_BK, _BT):
                f_kdk_t[_i, _j] = f_kdk[_j, _i]
            f_kdk_col = T.alloc_fragment((_BK,), T.float32)
            T.reduce_sum(f_kdk_t, f_kdk_col, dim=1)
            for _j in T.Parallel(_BK):
                s_dgk[_j] = s_dgk[_j] + f_kdk_col[_j]

            # Intra-chunk contribution, fused into the same consumer:
            # dq += exp2(g) * (dA @ (k * exp2(-g)))
            # dk += exp2(-g) * (dA.T @ (q * exp2(g)))
            b_dq = T.alloc_fragment((_BT, _BK), T.float32)
            b_dk = T.alloc_fragment((_BT, _BK), T.float32)
            T.clear(b_dq)
            T.clear(b_dk)
            s_dA = T.alloc_shared((_BT, _BT), T.float32)
            s_kg = T.alloc_shared((_BT, _BK), T.float32)
            s_qg = T.alloc_shared((_BT, _BK), T.float32)
            T.copy(dA[i_b, t_s:t_s + _BT, i_h, 0:_BT], s_dA, disable_tma=True)
            for _i, _j in T.Parallel(_BT, _BK):
                s_kg[_i, _j] = T.cast(s_k[_i, _j], T.float32) * T.exp2(-s_g[_i, _j])
                s_qg[_i, _j] = T.cast(s_q[_i, _j], T.float32) * T.exp2(s_g[_i, _j])
            T.gemm(s_dA, s_kg, b_dq)
            T.gemm(s_dA, s_qg, b_dk, transpose_A=True)

            for _i, _j in T.Parallel(_BT, _BK):
                b_dq[_i, _j] = b_dq[_i, _j] * T.exp2(s_g[_i, _j]) + s_inter_dq[_i, _j]
                b_dk[_i, _j] = b_dk[_i, _j] * T.exp2(-s_g[_i, _j]) + s_inter_dk[_i, _j]

            f_dg = T.alloc_fragment((_BT, _BK), T.float32)
            for _i, _j in T.Parallel(_BT, _BK):
                f_dg[_i, _j] = (
                    T.cast(s_q[_i, _j], T.float32) * b_dq[_i, _j]
                    - T.cast(s_k[_i, _j], T.float32) * b_dk[_i, _j]
                )
            f_dg_t = T.alloc_fragment((_BK, _BT), T.float32)
            for _i, _j in T.Parallel(_BK, _BT):
                f_dg_t[_i, _j] = f_dg[_j, _i]
            f_dg_col = T.alloc_fragment((_BK,), T.float32)
            T.reduce_sum(f_dg_t, f_dg_col, dim=1)

            s_dg_raw = T.alloc_shared((_BT, _BK), T.float32)
            T.copy(f_dg, s_dg_raw)
            T.cumsum(src=f_dg, dim=0)
            for _i, _j in T.Parallel(_BT, _BK):
                f_dg[_i, _j] = s_dg_raw[_i, _j] - f_dg[_i, _j] + f_dg_col[_j] + s_dgk[_j]

            T.copy(b_dq, dq[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK])
            T.copy(b_dk, dk[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK])
            T.copy(f_dg, dg[i_b, t_s:t_s + _BT, i_h, k_off:k_off + _BK])

    return kernel


def chunk_gla_bwd_dqkg_fused_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dA: torch.Tensor,
    scale: float | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = chunk_size
    BK = 64
    BV = 64
    NK = triton.cdiv(K, BK)
    if scale is None:
        scale = K ** -0.5

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    dg = torch.empty_like(g, dtype=torch.float)

    h_flat = h.reshape(-1, h.shape[-2], h.shape[-1])
    dh_flat = dh.reshape(-1, dh.shape[-2], dh.shape[-1])
    hD1, hD2 = h_flat.shape[-2], h_flat.shape[-1]
    dtype_str = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[q.dtype]
    state_dtype_str = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[h.dtype]

    kernel = _build_chunk_gla_fused_bwd_k_tile(
        B,
        H,
        K,
        V,
        BT,
        BK,
        BV,
        NK,
        hD1,
        hD2,
        dtype_str,
        state_dtype_str,
        num_warps=4,
    )
    kernel(q, k, v, g, h_flat, do, dh_flat, dA, dq, dk, dg, scale)
    return dq, dk, dg
