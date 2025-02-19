# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024, Yu Zhang, Songlin Yang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.utils import contiguous


@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 16}, num_warps=4),
        triton.Config({'BT': 16}, num_warps=8),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def logcumsumexp_fwd_kernel(
    s,
    z,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr
):
    i_bh = tl.program_id(0)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    b_mp = tl.full([S,], float('-inf'), dtype=tl.float32)
    b_zp = tl.zeros([S,], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))

        # [BT, S]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        # [S,]
        b_mc = tl.max(b_s, 0)
        # workaround for compiler bugs
        if i_t > 0:
            b_mc = tl.maximum(b_mp, b_mc)
        b_zp = b_zp * tl.exp(b_mp - b_mc)
        # [BT, S]
        b_s = tl.exp(b_s - b_mc)
        b_z = tl.dot(m_s, b_s, allow_tf32=False) + b_zp
        # [S,]
        b_zc = tl.max(b_z, 0)
        b_mp = b_mc
        b_zp = b_zc
        # [BT, BS]
        # small eps to prevent underflows
        b_z = tl.log(tl.where(b_z != 0, b_z, 1e-20)) + b_mc
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def softmax_fwd_kernel(
    s,
    p,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))

    # [BT, S]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    # [BT]
    b_m = tl.max(b_s, 1)

    # [BT, BS]
    b_s = tl.exp(b_s - b_m[:, None])
    b_z = tl.sum(b_s, 1)
    b_p = tl.where(b_s != 0, b_s / b_z[:, None], 0.)
    tl.store(p_p, b_p.to(p_p.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def softmax_bwd_kernel(
    p,
    dp,
    ds,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    p_dp = tl.make_block_ptr(dp + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    # [BT, BS]
    b_p = tl.load(p_p, boundary_check=(0, 1)).to(tl.float32)
    b_dp = tl.load(p_dp, boundary_check=(0, 1)).to(tl.float32)
    # [BT,]
    b_pp = tl.sum(b_p * b_dp, 1)
    # [BT, BS]
    b_ds = b_p * b_dp - b_p * b_pp[:, None]
    tl.store(p_ds, b_ds.to(p_ds.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 16}, num_warps=4),
        triton.Config({'BT': 16}, num_warps=8),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def chunk_global_reversed_cumsum_vector_kernel(
    s,
    z,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), boundary_check=(0, 1))

        if i_t >= 0:
            b_z += tl.sum(b_s, 0)


@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 16}, num_warps=4),
        triton.Config({'BT': 16}, num_warps=8),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s,
    z,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)
    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), boundary_check=(0, 1))
        if i_t >= 0:
            b_z += tl.sum(b_s, 0)


@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=4),
    ],
    key=[]
)
@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s,
    o,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=4),
    ],
    key=[]
)
@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s,
    o,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0) + b_z[None]
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({'BS': 16}, num_warps=2),
        triton.Config({'BS': 16}, num_warps=4),
        triton.Config({'BS': 16}, num_warps=8),
        triton.Config({'BS': 32}, num_warps=2),
        triton.Config({'BS': 32}, num_warps=4),
        triton.Config({'BS': 32}, num_warps=8),
        triton.Config({'BS': 64}, num_warps=2),
        triton.Config({'BS': 64}, num_warps=4),
        triton.Config({'BS': 64}, num_warps=8),
    ],
    key=['S', 'BT']
)
@triton.jit
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)
    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8)
    ],
    key=['BT']
)
@triton.jit
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum_vector(g, BT):
    B, H, T, S = g.shape
    NT = triton.cdiv(T, BT)
    g_org, g = g, torch.empty_like(g, dtype=torch.float)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    # keep cummulative normalizer in fp32
    # this kernel is equivalent to
    # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)
    chunk_local_cumsum_vector_kernel[grid](
        g_org, g,
        g.stride(1), g.stride(2), g.stride(3),
        T=T, S=S, BT=BT
    )
    return g


def chunk_local_cumsum_scalar(g, BT):
    B, H, T = g.shape
    NT = triton.cdiv(T, BT)
    g_org, g = g, torch.empty_like(g, dtype=torch.float)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        g_org, g,
        T=T, BT=BT
    )
    return g


@contiguous
def chunk_local_cumsum(g, BT):
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(g, BT)
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(g, BT)
    else:
        raise ValueError(f"Unsupported shape {g.shape}. "
                         f"Should be either (batch size, num_heads, seq_len, dim) or (batch_size, num_heads, seq_len)")


@contiguous
def chunk_global_reversed_cumsum_vector(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T, S = s.shape
    BS = 32
    dtype = dtype or s.dtype
    grid = (triton.cdiv(S, BS), B * H)
    z = torch.empty_like(s, dtype=dtype)
    chunk_global_reversed_cumsum_vector_kernel[grid](
        s, z,
        s.stride(1), s.stride(2), s.stride(3),
        T=T, S=S, BS=BS
    )
    return z


@contiguous
def chunk_global_reversed_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T = s.shape
    dtype = dtype or s.dtype
    grid = (B * H,)
    z = torch.empty_like(s, dtype=dtype)
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        s, z,
        T=T
    )
    return z


@contiguous
def chunk_global_cumsum_vector(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T, S = s.shape
    BS = 32
    dtype = dtype or s.dtype
    grid = (triton.cdiv(S, BS), B * H)
    z = torch.empty_like(s, dtype=dtype)
    chunk_global_cumsum_vector_kernel[grid](
        s, z,
        s.stride(1), s.stride(2), s.stride(3),
        T=T, S=S, BS=BS
    )
    return z


@contiguous
def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T = s.shape
    dtype = dtype or s.dtype
    grid = (B * H,)
    z = torch.empty_like(s, dtype=dtype)
    chunk_global_cumsum_scalar_kernel[grid](
        s, z,
        T=T
    )
    return z


@contiguous
def chunk_global_cumsum(s, dtype=None):
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(s, dtype)
    elif len(s.shape) == 4:
        return chunk_global_cumsum_vector(s, dtype)
    else:
        raise ValueError(f"Unsupported shape {s.shape}. "
                         f"Should be either [batch size, num_heads, seq_len] or [batch_size, num_heads, seq_len, dim]")


@contiguous
def chunk_global_reversed_cumsum(s, dtype=None):
    if len(s.shape) == 3:
        return chunk_global_reversed_cumsum_scalar(s, dtype)
    elif len(s.shape) == 4:
        return chunk_global_reversed_cumsum_vector(s, dtype)
    else:
        raise ValueError(f"Unsupported shape {s.shape}. "
                         f"Should be either [batch size, num_heads, seq_len] or [batch_size, num_heads, seq_len, dim]")
