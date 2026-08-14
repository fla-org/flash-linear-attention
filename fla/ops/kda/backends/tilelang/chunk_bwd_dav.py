# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang implementation of dense chunk_kda_bwd_dAv."""

import tilelang
import tilelang.language as T
import torch

from fla.utils import check_shared_mem

_DTYPE_NAMES = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _tile_extent(dim: int, const_tiling: int) -> int:
    return min(max(_next_power_of_2(dim), 16), const_tiling)


def _check_cuda_contiguous_supported(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"KDA TileLang dAv backend requires {name} to be a CUDA tensor")
    if tensor.dtype not in _DTYPE_NAMES:
        raise ValueError(f"KDA TileLang dAv backend does not support {name} dtype {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"KDA TileLang dAv backend requires {name} to be contiguous")


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def _build_kda_bwd_dav_kernel(B, H, HV, K, V, BT, BV, dtype_str, num_warps=4):
    dtype_map = {'float16': T.float16, 'bfloat16': T.bfloat16, 'float32': T.float32}
    dtype = dtype_map[dtype_str]
    NV = tilelang.cdiv(V, BV)
    threads = num_warps * 32

    _B, _H, _HV, _K, _V = B, H, HV, K, V
    _BT, _BV, _NV = BT, BV, NV
    _dtype = dtype
    _threads = threads

    T_d = T.dynamic("T")

    qk_s = (_B, T_d, _H, _K)
    v_s = (_B, T_d, _HV, _V)
    A_s = (_B, T_d, _HV, _BT)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_s, _dtype),
        k: T.Tensor(qk_s, _dtype),
        v: T.Tensor(v_s, _dtype),
        A: T.Tensor(A_s, _dtype),
        do: T.Tensor(v_s, _dtype),
        dv: T.Tensor(v_s, _dtype),
        dA: T.Tensor(A_s, T.float32),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(T_d, _BT), _B * _HV, threads=_threads) as (i_t, i_bh):
            i_b = i_bh // _HV
            i_hv = i_bh % _HV
            t_s = i_t * _BT

            s_A_raw = T.alloc_shared((_BT, _BT), _dtype)
            s_A_t = T.alloc_shared((_BT, _BT), _dtype)
            T.copy(A[i_b, t_s:t_s + _BT, i_hv, 0:_BT], s_A_raw)
            for i, j in T.Parallel(_BT, _BT):
                s_A_t[i, j] = T.if_then_else(i <= j, s_A_raw[j, i], T.cast(0, _dtype))

            b_dA = T.alloc_fragment((_BT, _BT), T.float32)
            T.clear(b_dA)
            s_do = T.alloc_shared((_BT, _BV), _dtype)
            s_v = T.alloc_shared((_BT, _BV), _dtype)
            b_dv = T.alloc_fragment((_BT, _BV), T.float32)

            for i_v in T.serial(_NV):
                v_off = i_v * _BV
                T.copy(do[i_b, t_s:t_s + _BT, i_hv, v_off:v_off + _BV], s_do)
                T.copy(v[i_b, t_s:t_s + _BT, i_hv, v_off:v_off + _BV], s_v)

                T.gemm(s_do, s_v, b_dA, transpose_B=True)

                T.clear(b_dv)
                T.gemm(s_A_t, s_do, b_dv)
                for i, j in T.Parallel(_BT, _BV):
                    dv[i_b, t_s + i, i_hv, v_off + j] = T.cast(b_dv[i, j], _dtype)

            for i, j in T.Parallel(_BT, _BT):
                dA[i_b, t_s + i, i_hv, j] = T.if_then_else(i >= j, b_dA[i, j] * scale, 0.0)

    return kernel


def chunk_kda_bwd_dAv_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    A: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if A is None:
        raise ValueError("KDA TileLang dAv backend requires A")
    if cu_seqlens is not None or chunk_indices is not None:
        raise ValueError("KDA TileLang dAv backend currently supports dense fixed-length sequences only")
    if chunk_size not in (32, 64):
        raise ValueError(f"KDA TileLang dAv backend supports chunk_size 32 or 64, got {chunk_size}")
    for name, tensor in {"q": q, "k": k, "v": v, "do": do, "A": A}.items():
        _check_cuda_contiguous_supported(name, tensor)
    if q.dtype != k.dtype:
        raise ValueError(f"KDA TileLang dAv backend requires k dtype {k.dtype} to match q dtype {q.dtype}")
    for name, tensor in {"v": v, "do": do, "A": A}.items():
        if tensor.dtype != q.dtype:
            raise ValueError(f"KDA TileLang dAv backend requires {name} dtype {tensor.dtype} to match q dtype {q.dtype}")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or do.ndim != 4:
        raise ValueError("KDA TileLang dAv backend requires q, k, v, and do to be 4D tensors")
    if q.shape != k.shape:
        raise ValueError(f"KDA TileLang dAv backend requires q and k to share shape, got {q.shape} vs {k.shape}")

    B, T_seq, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    if do.shape != v.shape:
        raise ValueError(f"KDA TileLang dAv backend requires do shape {v.shape}, got {do.shape}")
    if A.shape != (B, T_seq, HV, chunk_size):
        raise ValueError(f"KDA TileLang dAv backend requires A shape {(B, T_seq, HV, chunk_size)}, got {A.shape}")
    if T_seq % chunk_size != 0:
        raise ValueError(
            f"KDA TileLang dAv backend requires T={T_seq} to be divisible by chunk_size={chunk_size}"
        )
    if HV % H != 0:
        raise ValueError(f"KDA TileLang dAv backend requires HV={HV} to be divisible by H={H} for GVA")

    const_tiling = 64 if check_shared_mem(tensor_idx=q.device.index or 0) else 32
    BV = _tile_extent(V, const_tiling)
    if V % BV != 0:
        raise ValueError(f"KDA TileLang dAv backend requires V={V} to be divisible by its BV tile {BV}")
    if scale is None:
        scale = K ** -0.5

    dA = torch.empty(B, T_seq, HV, chunk_size, dtype=torch.float32, device=v.device)
    dv = torch.empty_like(do)
    dtype_str = _DTYPE_NAMES[q.dtype]
    num_warps = 4

    kernel = _build_kda_bwd_dav_kernel(B, H, HV, K, V, chunk_size, BV, dtype_str, num_warps=num_warps)
    kernel(q, k, v, A, do, dv, dA, scale)
    return dA, dv
