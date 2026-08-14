# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang implementation of dense KDA intra-chunk backward."""

import tilelang
import tilelang.language as T
import torch

_DTYPE_NAMES = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def _build_kda_bwd_intra_kernel(
    B,
    H,
    HV,
    K,
    BT,
    BC,
    BK,
    dtype_str,
    beta_dtype_str,
    num_warps=2,
):
    dtype_map = {'float16': T.float16, 'bfloat16': T.bfloat16, 'float32': T.float32}
    dtype = dtype_map[dtype_str]
    beta_dtype = dtype_map[beta_dtype_str]
    threads = num_warps * 32
    NC = tilelang.cdiv(BT, BC)
    NK = tilelang.cdiv(K, BK)

    _B, _H, _HV, _K = B, H, HV, K
    _G = HV // H
    _BT, _BC, _BK, _NC, _NK = BT, BC, BK, NC, NK
    _dtype = dtype
    _beta_dtype = beta_dtype
    _threads = threads

    T_d = T.dynamic("T")

    qk_s = (_B, T_d, _H, _K)
    hvk_s = (_B, T_d, _HV, _K)
    beta_s = (_B, T_d, _HV)
    dA_s = (_B, T_d, _HV, _BT)
    db2_s = (_NK, _B, T_d, _HV)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_s, _dtype),
        k: T.Tensor(qk_s, _dtype),
        g: T.Tensor(hvk_s, T.float32),
        beta: T.Tensor(beta_s, _beta_dtype),
        dAqk: T.Tensor(dA_s, T.float32),
        dAkk: T.Tensor(dA_s, T.float32),
        dq: T.Tensor(hvk_s, T.float32),
        dq2: T.Tensor(hvk_s, T.float32),
        dk: T.Tensor(hvk_s, T.float32),
        dk2: T.Tensor(hvk_s, T.float32),
        dg: T.Tensor(hvk_s, T.float32),
        dg2: T.Tensor(hvk_s, T.float32),
        db: T.Tensor(db2_s, T.float32),
    ):
        with T.Kernel(_NK * _NC, T.ceildiv(T_d, _BT), _B * _HV, threads=_threads) as (i_kc, i_t, i_bh):
            i_b = i_bh // _HV
            i_hv = i_bh % _HV
            i_h = i_hv // _G
            i_k = i_kc // _NC
            i_i = i_kc % _NC
            t_s = i_t * _BT
            t_i = t_s + i_i * _BC
            k_s = i_k * _BK

            b_g = T.alloc_shared((_BC, _BK), T.float32)
            b_q = T.alloc_shared((_BC, _BK), _dtype)
            b_k = T.alloc_shared((_BC, _BK), _dtype)
            b_beta = T.alloc_shared((_BC,), T.float32)

            T.copy(g[i_b, t_i:t_i + _BC, i_hv, k_s:k_s + _BK], b_g)
            T.copy(q[i_b, t_i:t_i + _BC, i_h, k_s:k_s + _BK], b_q)
            T.copy(k[i_b, t_i:t_i + _BC, i_h, k_s:k_s + _BK], b_k)
            T.copy(beta[i_b, t_i:t_i + _BC, i_hv], b_beta, disable_tma=True)

            b_dq2 = T.alloc_fragment((_BC, _BK), T.float32)
            b_dk2 = T.alloc_fragment((_BC, _BK), T.float32)
            b_dkt = T.alloc_fragment((_BC, _BK), T.float32)
            T.clear(b_dq2)
            T.clear(b_dk2)
            T.clear(b_dkt)

            s_dAqk = T.alloc_shared((_BC, _BC), T.float32)
            s_dAkk = T.alloc_shared((_BC, _BC), T.float32)
            s_kg = T.alloc_shared((_BC, _BK), T.float32)

            if i_i > 0:
                s_gn = T.alloc_shared((_BK,), T.float32)
                T.copy(g[i_b, t_i, i_hv, k_s:k_s + _BK], s_gn, disable_tma=True)
                for i_j in T.serial(0, _NC):
                    if i_j < i_i:
                        t_j = t_s + i_j * _BC
                        s_kj = T.alloc_shared((_BC, _BK), _dtype)
                        s_gj = T.alloc_shared((_BC, _BK), T.float32)
                        T.copy(k[i_b, t_j:t_j + _BC, i_h, k_s:k_s + _BK], s_kj)
                        T.copy(g[i_b, t_j:t_j + _BC, i_hv, k_s:k_s + _BK], s_gj)
                        for ii, kk in T.Parallel(_BC, _BK):
                            s_kg[ii, kk] = T.cast(s_kj[ii, kk], T.float32) * T.exp2(s_gn[kk] - s_gj[ii, kk])
                        T.copy(dAqk[i_b, t_i:t_i + _BC, i_hv, i_j * _BC:i_j * _BC + _BC], s_dAqk)
                        T.copy(dAkk[i_b, t_i:t_i + _BC, i_hv, i_j * _BC:i_j * _BC + _BC], s_dAkk)
                        T.gemm(s_dAqk, s_kg, b_dq2)
                        T.gemm(s_dAkk, s_kg, b_dk2)
                for ii, kk in T.Parallel(_BC, _BK):
                    factor = T.exp2(b_g[ii, kk] - s_gn[kk])
                    b_dq2[ii, kk] = b_dq2[ii, kk] * factor
                    b_dk2[ii, kk] = b_dk2[ii, kk] * factor

            for j in T.serial(0, _BC):
                s_kj = T.alloc_shared((_BK,), T.float32)
                s_gj = T.alloc_shared((_BK,), T.float32)
                s_dAqk_j = T.alloc_shared((_BC,), T.float32)
                s_dAkk_j = T.alloc_shared((_BC,), T.float32)
                T.copy(k[i_b, t_i + j, i_h, k_s:k_s + _BK], s_kj, disable_tma=True)
                T.copy(g[i_b, t_i + j, i_hv, k_s:k_s + _BK], s_gj, disable_tma=True)
                T.copy(dAqk[i_b, t_i:t_i + _BC, i_hv, i_i * _BC + j], s_dAqk_j, disable_tma=True)
                T.copy(dAkk[i_b, t_i:t_i + _BC, i_hv, i_i * _BC + j], s_dAkk_j, disable_tma=True)
                for ii, kk in T.Parallel(_BC, _BK):
                    if ii >= j:
                        factor = T.exp2(b_g[ii, kk] - s_gj[kk]) * T.cast(s_kj[kk], T.float32)
                        b_dq2[ii, kk] = b_dq2[ii, kk] + s_dAqk_j[ii] * factor
                        b_dk2[ii, kk] = b_dk2[ii, kk] + s_dAkk_j[ii] * factor

            f_db_prod = T.alloc_fragment((_BC, _BK), T.float32)
            for ii, kk in T.Parallel(_BC, _BK):
                f_db_prod[ii, kk] = b_dk2[ii, kk] * T.cast(b_k[ii, kk], T.float32)
            b_db = T.alloc_fragment((_BC,), T.float32)
            T.reduce_sum(f_db_prod, b_db, dim=1)
            for ii, kk in T.Parallel(_BC, _BK):
                b_dk2[ii, kk] = b_dk2[ii, kk] * b_beta[ii]

            b_dg2 = T.alloc_fragment((_BC, _BK), T.float32)
            for ii, kk in T.Parallel(_BC, _BK):
                b_dg2[ii, kk] = T.cast(b_q[ii, kk], T.float32) * b_dq2[ii, kk]
                b_dq2[ii, kk] = b_dq2[ii, kk] + dq[i_b, t_i + ii, i_hv, k_s + kk]
                dq2[i_b, t_i + ii, i_hv, k_s + kk] = b_dq2[ii, kk]
            for ii in T.Parallel(_BC):
                db[i_k, i_b, t_i + ii, i_hv] = b_db[ii]

            if i_i < _NC - 1:
                s_gn2 = T.alloc_shared((_BK,), T.float32)
                T.copy(g[i_b, t_i + _BC - 1, i_hv, k_s:k_s + _BK], s_gn2, disable_tma=True)
                for i_j in T.serial(0, _NC):
                    if i_j > i_i:
                        t_j = t_s + i_j * _BC
                        s_qj = T.alloc_shared((_BC, _BK), T.float32)
                        s_kbgj = T.alloc_shared((_BC, _BK), T.float32)
                        s_gj = T.alloc_shared((_BC, _BK), T.float32)
                        s_bj = T.alloc_shared((_BC,), T.float32)
                        T.copy(g[i_b, t_j:t_j + _BC, i_hv, k_s:k_s + _BK], s_gj)
                        T.copy(beta[i_b, t_j:t_j + _BC, i_hv], s_bj, disable_tma=True)
                        for jj, kk in T.Parallel(_BC, _BK):
                            g_factor = T.exp2(s_gj[jj, kk] - s_gn2[kk])
                            s_qj[jj, kk] = T.cast(q[i_b, t_j + jj, i_h, k_s + kk], T.float32) * g_factor
                            s_kbgj[jj, kk] = T.cast(k[i_b, t_j + jj, i_h, k_s + kk], T.float32) * s_bj[jj] * g_factor
                        for ii, jj in T.Parallel(_BC, _BC):
                            s_dAqk[ii, jj] = dAqk[i_b, t_j + jj, i_hv, i_i * _BC + ii]
                            s_dAkk[ii, jj] = dAkk[i_b, t_j + jj, i_hv, i_i * _BC + ii]
                        T.gemm(s_dAqk, s_qj, b_dkt)
                        T.gemm(s_dAkk, s_kbgj, b_dkt)
                for ii, kk in T.Parallel(_BC, _BK):
                    b_dkt[ii, kk] = b_dkt[ii, kk] * T.exp2(s_gn2[kk] - b_g[ii, kk])

            for j in T.serial(0, _BC):
                s_qj = T.alloc_shared((_BK,), T.float32)
                s_kbj = T.alloc_shared((_BK,), T.float32)
                s_gj = T.alloc_shared((_BK,), T.float32)
                s_dAqk_j = T.alloc_shared((_BC,), T.float32)
                s_dAkk_j = T.alloc_shared((_BC,), T.float32)
                beta_j = T.alloc_var(T.float32)
                beta_j = T.cast(beta[i_b, t_i + j, i_hv], T.float32)
                for kk in T.Parallel(_BK):
                    s_qj[kk] = T.cast(q[i_b, t_i + j, i_h, k_s + kk], T.float32)
                    s_kbj[kk] = T.cast(k[i_b, t_i + j, i_h, k_s + kk], T.float32) * beta_j
                    s_gj[kk] = g[i_b, t_i + j, i_hv, k_s + kk]
                for ii in T.Parallel(_BC):
                    s_dAqk_j[ii] = dAqk[i_b, t_i + j, i_hv, i_i * _BC + ii]
                    s_dAkk_j[ii] = dAkk[i_b, t_i + j, i_hv, i_i * _BC + ii]
                for ii, kk in T.Parallel(_BC, _BK):
                    if ii <= j:
                        factor = T.exp2(s_gj[kk] - b_g[ii, kk])
                        b_dkt[ii, kk] = b_dkt[ii, kk] + s_dAqk_j[ii] * s_qj[kk] * factor
                        b_dkt[ii, kk] = b_dkt[ii, kk] + s_dAkk_j[ii] * s_kbj[kk] * factor

            for ii, kk in T.Parallel(_BC, _BK):
                b_dg2[ii, kk] = b_dg2[ii, kk] + (b_dk2[ii, kk] - b_dkt[ii, kk]) * T.cast(b_k[ii, kk], T.float32)
                b_dg2[ii, kk] = b_dg2[ii, kk] + dg[i_b, t_i + ii, i_hv, k_s + kk]
                b_dk2[ii, kk] = b_dk2[ii, kk] + dk[i_b, t_i + ii, i_hv, k_s + kk] + b_dkt[ii, kk]
                dk2[i_b, t_i + ii, i_hv, k_s + kk] = b_dk2[ii, kk]
                dg2[i_b, t_i + ii, i_hv, k_s + kk] = b_dg2[ii, kk]

    return kernel


def chunk_kda_bwd_intra_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
):
    if safe_gate:
        raise ValueError("KDA TileLang bwd_intra currently supports safe_gate=False only")
    if cu_seqlens is not None or chunk_indices is not None:
        raise ValueError("KDA TileLang bwd_intra currently supports dense fixed-length sequences only")
    if chunk_size not in (32, 64):
        raise ValueError(f"KDA TileLang bwd_intra supports chunk_size 32 or 64, got {chunk_size}")
    if q.ndim != 4 or k.ndim != 4 or g.ndim != 4:
        raise ValueError("KDA TileLang bwd_intra requires q, k, and g to be 4D tensors")
    if q.shape != k.shape:
        raise ValueError(f"KDA TileLang bwd_intra requires q and k to share shape, got {q.shape} vs {k.shape}")
    B, T_seq, H, K = q.shape
    HV = g.shape[2]
    if K not in (64, 128):
        raise ValueError(f"KDA TileLang bwd_intra supports K=64 or 128, got {K}")
    if T_seq % chunk_size != 0:
        raise ValueError(f"KDA TileLang bwd_intra requires T={T_seq} to be divisible by chunk_size={chunk_size}")
    if HV % H != 0:
        raise ValueError(f"KDA TileLang bwd_intra requires HV={HV} to be divisible by H={H}")
    if g.shape != (B, T_seq, HV, K):
        raise ValueError(f"KDA TileLang bwd_intra requires g shape {(B, T_seq, HV, K)}, got {g.shape}")
    if beta.shape != (B, T_seq, HV) or db.shape != (B, T_seq, HV):
        raise ValueError(f"KDA TileLang bwd_intra requires beta/db shape {(B, T_seq, HV)}, got {beta.shape}/{db.shape}")
    if dAqk.shape != (B, T_seq, HV, chunk_size) or dAkk.shape != (B, T_seq, HV, chunk_size):
        raise ValueError(
            f"KDA TileLang bwd_intra requires dAqk/dAkk shape {(B, T_seq, HV, chunk_size)}, "
            f"got {dAqk.shape}/{dAkk.shape}"
        )
    if dq.shape != (B, T_seq, HV, K) or dk.shape != (B, T_seq, HV, K) or dg.shape != (B, T_seq, HV, K):
        raise ValueError(f"KDA TileLang bwd_intra requires dq/dk/dg shape {(B, T_seq, HV, K)}")
    for name, tensor in {
        "q": q, "k": k, "g": g, "beta": beta, "dAqk": dAqk, "dAkk": dAkk,
        "dq": dq, "dk": dk, "db": db, "dg": dg,
    }.items():
        if not tensor.is_cuda:
            raise ValueError(f"KDA TileLang bwd_intra requires {name} to be a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"KDA TileLang bwd_intra requires {name} to be contiguous")
    if q.dtype not in _DTYPE_NAMES:
        raise ValueError(f"KDA TileLang bwd_intra does not support q dtype {q.dtype}")
    if k.dtype != q.dtype:
        raise ValueError(f"KDA TileLang bwd_intra requires k dtype {k.dtype} to match q dtype {q.dtype}")
    if beta.dtype not in _DTYPE_NAMES:
        raise ValueError(f"KDA TileLang bwd_intra does not support beta dtype {beta.dtype}")
    for name, tensor in {"g": g, "dAqk": dAqk, "dAkk": dAkk, "dq": dq, "dk": dk, "db": db, "dg": dg}.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"KDA TileLang bwd_intra requires {name} dtype torch.float32, got {tensor.dtype}")

    BC = 16
    BK = 32
    NK = K // BK
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    dg2 = torch.empty_like(dg)
    db2 = torch.empty(NK, B, T_seq, HV, dtype=torch.float32, device=beta.device)
    dtype_str = _DTYPE_NAMES[q.dtype]
    beta_dtype_str = _DTYPE_NAMES[beta.dtype]

    kernel = _build_kda_bwd_intra_kernel(
        B,
        H,
        HV,
        K,
        chunk_size,
        BC,
        BK,
        dtype_str,
        beta_dtype_str,
        num_warps=2,
    )
    kernel(q, k, g, beta, dAqk, dAkk, dq, dq2, dk, dk2, dg, dg2, db2)
    db = db2.sum(0).add_(db)
    return dq2, dk2, db, dg2
