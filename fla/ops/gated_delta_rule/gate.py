# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard


def naive_gdn_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Torch reference implementation for GDN gate computation.

    Computes: ``g = -A_log.exp() * softplus(g + dt_bias)``

    Args:
        g (torch.Tensor):
            Input tensor of shape `[..., HV]`.
        A_log (torch.Tensor):
            Decay parameter tensor with `HV` elements.
        dt_bias (torch.Tensor | None):
            Optional bias tensor added to `g` before activation, shape `[HV]`.

    Returns:
        Output tensor of shape `[..., HV]`.
    """
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.float()
    return (-A_log.float().exp() * F.softplus(g)).to(output_dtype)


@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['H', 'BT', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gdn_gate_chunk_cumsum_scalar_kernel(
    g,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)
    b_A = tl.load(A_log + i_h).to(tl.float32)
    b_gate = -exp(b_A) * softplus(b_g)

    b_o = tl.cumsum(b_gate, axis=0)
    if REVERSE:
        b_z = tl.sum(b_gate, axis=0)
        b_o = -b_o + b_z[None] + b_gate
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['H', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gdn_gate_bwd_kernel(
    g,
    A_log,
    dt_bias,
    dyg,
    dg,
    dA,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dg = tl.make_block_ptr(dg + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dyg = tl.make_block_ptr(dyg + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_dyg = tl.load(p_dyg, boundary_check=(0,)).to(tl.float32)

    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)

    # gate = -exp(A_log) * softplus(g + bias)
    # d(gate)/d(g) = -exp(A_log) * sigmoid(g + bias)   (softplus' = sigmoid)
    # d(gate)/d(A_log) = -exp(A_log) * softplus(g + bias) = gate
    b_neg_expA = -exp(b_A)
    b_yg = b_neg_expA * softplus(b_g)
    b_dg = b_neg_expA * (b_dyg * tl.sigmoid(b_g))
    b_dA = tl.sum(b_dyg * b_yg, 0)

    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
    tl.store(dA + i_t * H + i_h, b_dA)


@input_guard
def gdn_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    B, T, H = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty_like(g, dtype=output_dtype or g.dtype)
    gdn_gate_chunk_cumsum_scalar_kernel[(NT, B * H)](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        REVERSE=False,
    )
    return o


def gdn_gate_bwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    dyg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    H = g.shape[-1]
    T = g.numel() // H
    BT = 32
    NT = triton.cdiv(T, BT)

    dg = torch.empty_like(g, dtype=torch.float32)
    dA = A_log.new_empty(NT, H, dtype=torch.float32)

    gdn_gate_bwd_kernel[(NT, H)](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        dyg=dyg,
        dg=dg,
        dA=dA,
        T=T,
        H=H,
        BT=BT,
    )

    dg = dg.view_as(g).type_as(g)
    dA = dA.sum(0).view_as(A_log).type_as(A_log)
    dbias = dg.view(-1, H).sum(0).to(dt_bias) if dt_bias is not None else None

    return dg, dA, dbias


@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [32, 64, 128]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['H'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gdn_gate_fwd_kernel(
    g,
    A_log,
    dt_bias,
    yg,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_yg = tl.make_block_ptr(yg + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)
    b_yg = -exp(b_A) * softplus(b_g)
    tl.store(p_yg, b_yg.to(p_yg.dtype.element_ty), boundary_check=(0,))


def gdn_gate_fwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H = g.shape[-1]
    T = g.numel() // H

    yg = torch.empty_like(g, dtype=output_dtype)

    def grid(meta):
        return (triton.cdiv(T, meta['BT']), H)

    gdn_gate_fwd_kernel[grid](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        yg=yg,
        T=T,
        H=H,
    )
    return yg


class GDNGateFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        g: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        yg = gdn_gate_fwd(g=g, A_log=A_log, dt_bias=dt_bias, output_dtype=output_dtype)
        ctx.save_for_backward(g, A_log, dt_bias)
        return yg

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dyg: torch.Tensor):
        g, A_log, dt_bias = ctx.saved_tensors
        dg, dA, dbias = gdn_gate_bwd(g=g, A_log=A_log, dt_bias=dt_bias, dyg=dyg)
        return dg, dA, dbias, None


@torch.compiler.disable
def fused_gdn_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Fused GDN gate computation with autograd support.

    Computes: ``g = -A_log.exp() * softplus(g + dt_bias)``

    Args:
        g (torch.Tensor):
            Input tensor of shape `[..., HV]`.
        A_log (torch.Tensor):
            Decay parameter tensor with `HV` elements.
        dt_bias (torch.Tensor | None):
            Optional bias tensor added to `g` before activation, shape `[HV]`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`.

    Returns:
        Output tensor of shape `[..., HV]`.
    """
    return GDNGateFunction.apply(g, A_log, dt_bias, output_dtype)
