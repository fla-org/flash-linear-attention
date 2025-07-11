from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN']
)
@triton.jit(do_not_specialize=['T'])
def chunk_comba_cumsum_scalar_fwd_kernel(
    g,
    g0,
    g1,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_g = tl.make_block_ptr(g + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_g0 = tl.make_block_ptr(g0 + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_g1 = tl.make_block_ptr(g1 + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_g = tl.make_block_ptr(g + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_g0 = tl.make_block_ptr(g0 + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_g1 = tl.make_block_ptr(g1 + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_g1 = tl.cumsum(b_g, axis=0)
    b_g0 = b_g1 - b_g
    tl.store(p_g0, b_g0.to(p_g0.dtype.element_ty), boundary_check=(0,))
    tl.store(p_g1, b_g1.to(p_g1.dtype.element_ty), boundary_check=(0,))


def chunk_comba_cumsum_scalar_fwd(
    g: torch.Tensor,
    chunk_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g0, g1 = torch.empty_like(g, dtype=output_dtype or g.dtype), torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    chunk_comba_cumsum_scalar_fwd_kernel[grid](
        g,
        g0,
        g1,
        cu_seqlens,
        chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first
    )
    return g0, g1


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN']
)
@triton.jit(do_not_specialize=['T'])
def chunk_comba_cumsum_scalar_bwd_kernel(
    dg0,
    dgr,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_dg0 = tl.make_block_ptr(dg0 + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_dgr = tl.make_block_ptr(dgr + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_dg0 = tl.make_block_ptr(dg0 + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_dgr = tl.make_block_ptr(dgr + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    """
    b_dg:   1,2,3,4
    b_dg0:  0,1,2,3
    b_temp: 0,1,3,6
    b_dz:   6
    b_dgr:  6,5,3,0
    """
    b_dg0 = tl.load(p_dg0, boundary_check=(0,)).to(tl.float32)
    b_temp = tl.cumsum(b_dg0, axis=0)
    b_dz = tl.sum(b_dg0, axis=0)
    b_dgr = -b_temp + b_dz[None]
    tl.store(p_dgr, b_dgr.to(p_dgr.dtype.element_ty), boundary_check=(0,))


def chunk_comba_cumsum_scalar_bwd(
    dg0: torch.Tensor,
    chunk_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    if head_first:
        B, H, T = dg0.shape
    else:
        B, T, H = dg0.shape
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    dg = torch.empty_like(dg0, dtype=output_dtype or dg0.dtype)
    grid = (NT, B * H)
    chunk_comba_cumsum_scalar_bwd_kernel[grid](
        dg0,
        dg,
        cu_seqlens,
        chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
    )
    return dg
