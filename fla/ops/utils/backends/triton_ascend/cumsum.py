# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_grid_limited_tile_size,
    compute_row_tile_block_size,
    compute_ub_block_size,
    iter_axis_launch_chunks,
    max_grid_axis_chunks,
)

_NUM_WARPS = 4
# Peak live fp32 tiles: b_s, b_o (and b_z / partial sums for vector/global).
_CUMSUM_SCALAR_MEM_MULT = 3.0
# b_s, b_c, b_z, plus tl.cumsum multi-buffer on [BT, BS] tiles.
_CUMSUM_VECTOR_MEM_MULT = 8.0
_CUMSUM_SAFETY_MARGIN = 0.85
_FALLBACK_BT_GLOBAL = 32
_FALLBACK_BS_LOCAL = 32
_FALLBACK_BS_GLOBAL = 16
_MAX_BT_GLOBAL = 256


def _get_global_scalar_bt(T: int) -> int:
    """UB-safe time chunk size for global scalar cumsum."""
    desired = min(triton.next_power_of_2(T), _MAX_BT_GLOBAL)
    return compute_ub_block_size(
        T,
        _CUMSUM_SCALAR_MEM_MULT,
        safety_margin=_CUMSUM_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BT_GLOBAL,
        desired=desired,
    )


def _get_vector_bs(BT: int, S: int, *, fallback: int, max_block: int | None = None) -> int:
    """UB-safe feature tile size for vector cumsum with fixed time chunk BT."""
    return compute_row_tile_block_size(
        BT,
        S,
        _CUMSUM_VECTOR_MEM_MULT,
        tiling_row=False,
        safety_margin=_CUMSUM_SAFETY_MARGIN,
        dtype_size=4,
        fallback=fallback,
        min_block=1,
        max_block=max_block,
    )


def _get_global_vector_tile_config(T: int, S: int) -> tuple[int, int]:
    """UB-safe (BT, BS) for global vector cumsum."""
    bs_cap = min(_FALLBACK_BS_GLOBAL, triton.next_power_of_2(S))
    BS = _get_vector_bs(
        _FALLBACK_BT_GLOBAL,
        S,
        fallback=_FALLBACK_BS_GLOBAL,
        max_block=bs_cap,
    )
    BT = compute_row_tile_block_size(
        T,
        BS,
        _CUMSUM_VECTOR_MEM_MULT,
        tiling_row=True,
        safety_margin=_CUMSUM_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BT_GLOBAL,
        min_block=1,
        max_block=_FALLBACK_BT_GLOBAL,
    )
    BS = _get_vector_bs(BT, S, fallback=_FALLBACK_BS_GLOBAL, max_block=bs_cap)
    return BT, BS


def _launch_local_cumsum_scalar(
    *,
    g_org,
    g,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    H,
    BT,
    NT,
    head_first,
    reverse,
):
    bh_total = B * H
    kernel_kwargs = dict(
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        num_warps=_NUM_WARPS,
    )
    max_nt = max_grid_axis_chunks(NT, bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        if cu_seqlens is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['chunk_indices'] = chunk_indices
            kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            chunk_local_cumsum_scalar_kernel_npu[(nt_len, bh_len)](**kernel_kwargs)


def _launch_local_cumsum_vector(
    *,
    g_org,
    g,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    H,
    S,
    BT,
    BS,
    NT,
    head_first,
    reverse,
):
    bh_total = B * H
    ns = triton.cdiv(S, BS)
    kernel_kwargs = dict(
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        BS=BS,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        num_warps=_NUM_WARPS,
    )
    max_nt = max_grid_axis_chunks(NT, ns * bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        if cu_seqlens is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['chunk_indices'] = chunk_indices
            kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(bh_total, ns * nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            chunk_local_cumsum_vector_kernel_npu[(ns, nt_len, bh_len)](**kernel_kwargs)


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel_npu(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_t += NT_OFFSET
    i_bh += BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_vector_kernel_npu(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t += NT_OFFSET
    i_bh += BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    else:
        p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    if REVERSE:
        b_o = tl.cumsum(b_s, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_s, axis=0)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_scalar_kernel_npu(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_nh = tl.program_id(0) + BH_OFFSET
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos

    b_z = tl.zeros([], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        else:
            p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0)
        b_ss = tl.sum(b_s, 0)
        if REVERSE:
            b_o = -b_o + b_ss + b_s
        b_o += b_z
        if i_c >= 0:
            b_z += b_ss
        if HAS_SCALE:
            b_o *= scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_vector_kernel_npu(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_s, i_nh = tl.program_id(0), tl.program_id(1) + BH_OFFSET
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos

    b_z = tl.zeros([BS], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_o = tl.make_block_ptr(o + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        else:
            p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        if REVERSE:
            b_c = b_z[None, :] + tl.cumsum(b_s, axis=0, reverse=True)
        else:
            b_c = b_z[None, :] + tl.cumsum(b_s, axis=0)
        if HAS_SCALE:
            b_c *= scale
        tl.store(p_o, b_c.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        b_z += tl.sum(b_s, 0)


def chunk_local_cumsum_scalar_npu(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    _launch_local_cumsum_scalar(
        g_org=g_org,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        NT=NT,
        head_first=head_first,
        reverse=reverse,
    )
    return g


def chunk_local_cumsum_vector_npu(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"

    BS = _get_vector_bs(BT, S, fallback=_FALLBACK_BS_LOCAL)
    BS = compute_grid_limited_tile_size(
        S,
        NT * B * H,
        BS,
        max_grid=ASCEND_MAX_GRID_DIM,
    )
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    _launch_local_cumsum_vector(
        g_org=g_org,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        BS=BS,
        NT=NT,
        head_first=head_first,
        reverse=reverse,
    )
    return g


@input_guard
def chunk_global_cumsum_scalar_npu(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T = s.shape
    else:
        B, T, H = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    BT = _get_global_scalar_bt(T)
    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    bh_total = N * H
    kernel_kwargs = dict(
        s=s,
        o=z,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        num_warps=_NUM_WARPS,
    )
    for bh_off, bh_len in iter_axis_launch_chunks(bh_total, 1, max_grid=ASCEND_MAX_GRID_DIM):
        kernel_kwargs['BH_OFFSET'] = bh_off
        chunk_global_cumsum_scalar_kernel_npu[(bh_len,)](**kernel_kwargs)
    return z


@input_guard
def chunk_global_cumsum_vector_npu(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = s.shape
    else:
        B, T, H, S = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT, BS = _get_global_vector_tile_config(T, S)
    BS = compute_grid_limited_tile_size(
        S,
        N * H,
        BS,
        max_grid=ASCEND_MAX_GRID_DIM,
    )
    ns = triton.cdiv(S, BS)

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    bh_total = N * H
    kernel_kwargs = dict(
        s=s,
        o=z,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        BS=BS,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        num_warps=_NUM_WARPS,
    )
    max_bh = max_grid_axis_chunks(bh_total, ns, max_grid=ASCEND_MAX_GRID_DIM)
    for bh_off in range(0, bh_total, max_bh):
        bh_len = min(max_bh, bh_total - bh_off)
        kernel_kwargs['BH_OFFSET'] = bh_off
        chunk_global_cumsum_vector_kernel_npu[(ns, bh_len)](**kernel_kwargs)
    return z


@input_guard
def chunk_global_cumsum_npu(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert s.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar_npu(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    if len(s.shape) == 4:
        return chunk_global_cumsum_vector_npu(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    raise ValueError(
        f"Unsupported input shape {s.shape}, "
        f"which should be [B, T, H]/[B, T, H, D] if `head_first=False` "
        f"or [B, H, T]/[B, H, T, D] otherwise",
    )


@input_guard
def chunk_local_cumsum_npu(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar_npu(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    if len(g.shape) == 4:
        return chunk_local_cumsum_vector_npu(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    raise ValueError(
        f"Unsupported input shape {g.shape}, "
        f"which should be (B, T, H, D) if `head_first=False` "
        f"or (B, H, T, D) otherwise",
    )
