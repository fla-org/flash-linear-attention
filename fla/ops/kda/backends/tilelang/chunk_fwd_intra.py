# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang implementation of dense KDA intra-chunk forward diagonal blocks."""

import tilelang
import tilelang.language as T
import torch

from fla.ops.kda.chunk_intra import chunk_kda_fwd_kernel_inter_solve_fused
from fla.ops.kda.wy_fast import recompute_w_u_fwd
from fla.utils import check_shared_mem

_DTYPE_NAMES = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _tile_extent(dim: int, const_tiling: int) -> int:
    return min(max(_next_power_of_2(dim), 16), const_tiling)


def _check_cuda_contiguous_supported(name: str, tensor: torch.Tensor, dtype_required: torch.dtype | None = None) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"KDA TileLang fwd_intra backend requires {name} to be a CUDA tensor")
    if dtype_required is not None and tensor.dtype != dtype_required:
        raise ValueError(f"KDA TileLang fwd_intra backend requires {name} dtype {dtype_required}, got {tensor.dtype}")
    if dtype_required is None and tensor.dtype not in _DTYPE_NAMES:
        raise ValueError(f"KDA TileLang fwd_intra backend does not support {name} dtype {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"KDA TileLang fwd_intra backend requires {name} to be contiguous")


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def _build_kda_fwd_intra_diag_kernel(
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
    A_s = (_B, T_d, _HV, _BT)
    Akkd_s = (_B, T_d, _HV, _BC)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_s, _dtype),
        k: T.Tensor(qk_s, _dtype),
        g: T.Tensor(hvk_s, T.float32),
        beta: T.Tensor(beta_s, _beta_dtype),
        Aqk: T.Tensor(A_s, _dtype),
        Akkd: T.Tensor(Akkd_s, T.float32),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(T_d, _BT), _NC, _B * _HV, threads=_threads) as (i_t, i_i, i_bh):
            i_b = i_bh // _HV
            i_hv = i_bh % _HV
            i_h = i_hv // _G
            t_s = i_t * _BT
            t_i = t_s + i_i * _BC

            b_Aqk = T.alloc_fragment((_BC, _BC), T.float32)
            b_Akk = T.alloc_fragment((_BC, _BC), T.float32)
            T.clear(b_Aqk)
            T.clear(b_Akk)

            s_qg = T.alloc_shared((_BC, _BK), _dtype)
            s_kg = T.alloc_shared((_BC, _BK), _dtype)
            s_kbg = T.alloc_shared((_BC, _BK), _dtype)
            s_gn = T.alloc_shared((_BK,), T.float32)
            s_beta = T.alloc_shared((_BC,), T.float32)

            T.copy(beta[i_b, t_i:t_i + _BC, i_hv], s_beta, disable_tma=True)

            for i_k in T.serial(_NK):
                k_s = i_k * _BK
                T.copy(g[i_b, t_i + _BC // 2, i_hv, k_s:k_s + _BK], s_gn, disable_tma=True)
                for i, j in T.Parallel(_BC, _BK):
                    gi = g[i_b, t_i + i, i_hv, k_s + j]
                    q_scale = T.exp2(gi - s_gn[j])
                    k_scale = T.exp2(s_gn[j] - gi)
                    q_val = T.cast(q[i_b, t_i + i, i_h, k_s + j], T.float32)
                    k_val = T.cast(k[i_b, t_i + i, i_h, k_s + j], T.float32)
                    s_qg[i, j] = T.cast(q_val * q_scale, _dtype)
                    s_kg[i, j] = T.cast(k_val * k_scale, _dtype)
                    s_kbg[i, j] = T.cast(k_val * T.cast(s_beta[i], T.float32) * q_scale, _dtype)

                T.gemm(s_qg, s_kg, b_Aqk, transpose_B=True)
                T.gemm(s_kbg, s_kg, b_Akk, transpose_B=True)

            s_Akk = T.alloc_shared((_BC, _BC), T.float32)
            s_Ai = T.alloc_shared((_BC, _BC), T.float32)
            for i, j in T.Parallel(_BC, _BC):
                Aqk[i_b, t_i + i, i_hv, i_i * _BC + j] = T.cast(
                    T.if_then_else(i >= j, b_Aqk[i, j] * scale, 0.0),
                    _dtype,
                )
                raw_akk = T.if_then_else(i > j, b_Akk[i, j], 0.0)
                s_Akk[i, j] = raw_akk
                s_Ai[i, j] = -raw_akk

            for row in T.serial(2, _BC):
                for j in T.serial(_BC):
                    acc = T.alloc_var(T.float32)
                    acc = T.if_then_else(j < row, -s_Akk[row, j], 0.0)
                    for p in T.serial(_BC):
                        row_p = T.if_then_else(p < row, -s_Akk[row, p], 0.0)
                        acc += row_p * s_Ai[p, j]
                    s_Ai[row, j] = T.if_then_else(j < row, acc, 0.0)

            for i, j in T.Parallel(_BC, _BC):
                Akkd[i_b, t_i + i, i_hv, j] = T.if_then_else(i == j, 1.0, s_Ai[i, j])

    return kernel


def chunk_kda_fwd_intra_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
):
    if gk is None or beta is None:
        raise ValueError("KDA TileLang fwd_intra backend requires gk and beta")
    if safe_gate:
        raise ValueError("KDA TileLang fwd_intra backend currently supports safe_gate=False only")
    if disable_recompute:
        raise ValueError("KDA TileLang fwd_intra backend currently supports disable_recompute=False only")
    if cu_seqlens is not None or chunk_indices is not None:
        raise ValueError("KDA TileLang fwd_intra backend currently supports dense fixed-length sequences only")
    if chunk_size not in (32, 64):
        raise ValueError(f"KDA TileLang fwd_intra backend supports chunk_size 32 or 64, got {chunk_size}")
    for name, tensor in {"q": q, "k": k, "v": v, "beta": beta}.items():
        _check_cuda_contiguous_supported(name, tensor)
    _check_cuda_contiguous_supported("gk", gk, dtype_required=torch.float32)
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"KDA TileLang fwd_intra backend requires q/k/v to share dtype, got {q.dtype}/{k.dtype}/{v.dtype}")
    if beta.dtype not in _DTYPE_NAMES:
        raise ValueError(f"KDA TileLang fwd_intra backend does not support beta dtype {beta.dtype}")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or gk.ndim != 4:
        raise ValueError("KDA TileLang fwd_intra backend requires q, k, v, and gk to be 4D tensors")
    if q.shape != k.shape:
        raise ValueError(f"KDA TileLang fwd_intra backend requires q and k to share shape, got {q.shape} vs {k.shape}")

    B, T_seq, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    BT = chunk_size
    BC = 16
    if K not in (64, 128):
        raise ValueError(f"KDA TileLang fwd_intra backend supports K=64 or 128, got {K}")
    if T_seq % BT != 0:
        raise ValueError(f"KDA TileLang fwd_intra backend requires T={T_seq} to be divisible by chunk_size={BT}")
    if HV % H != 0:
        raise ValueError(f"KDA TileLang fwd_intra backend requires HV={HV} to be divisible by H={H} for GVA")
    if v.shape != (B, T_seq, HV, V):
        raise ValueError(f"KDA TileLang fwd_intra backend requires v shape {(B, T_seq, HV, V)}, got {v.shape}")
    if gk.shape != (B, T_seq, HV, K):
        raise ValueError(f"KDA TileLang fwd_intra backend requires gk shape {(B, T_seq, HV, K)}, got {gk.shape}")
    if beta.shape != (B, T_seq, HV):
        raise ValueError(f"KDA TileLang fwd_intra backend requires beta shape {(B, T_seq, HV)}, got {beta.shape}")
    if scale is None:
        scale = K ** -0.5

    const_tiling = 64 if check_shared_mem(tensor_idx=q.device.index or 0) else 32
    BK = _tile_extent(K, const_tiling)
    if K % BK != 0:
        raise ValueError(f"KDA TileLang fwd_intra backend requires K={K} to be divisible by its BK tile {BK}")

    NT = triton_cdiv(T_seq, BT)
    NC = triton_cdiv(BT, BC)
    Aqk = torch.empty(B, T_seq, HV, BT, device=q.device, dtype=q.dtype)
    Akk = torch.zeros(B, T_seq, HV, BT, device=q.device, dtype=q.dtype)
    Akkd = torch.empty(B, T_seq, HV, BC, device=q.device, dtype=torch.float32)

    dtype_str = _DTYPE_NAMES[q.dtype]
    beta_dtype_str = _DTYPE_NAMES[beta.dtype]
    kernel = _build_kda_fwd_intra_diag_kernel(
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
    )
    kernel(q, k, gk, beta, Aqk, Akkd, scale)

    grid = (NT, B * HV)
    chunk_kda_fwd_kernel_inter_solve_fused[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        cu_seqlens=None,
        chunk_indices=None,
        T=T_seq,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
        USE_SAFE_GATE=True,
    )
    w, u, qg, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=None,
        gk=gk,
        cu_seqlens=None,
        chunk_indices=None,
    )
    return w, u, qg, kg, Aqk, Akk


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
