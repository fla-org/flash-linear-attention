# -*- coding: utf-8 -*-

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from fla.utils import check_pytorch_version, device, input_guard, use_cuda_graph

logger = logging.getLogger(__name__)

if not check_pytorch_version('2.4'):
    logger.warning('PyTorch < 2.4 detected - computations may be slower due to lack of optimizations')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': block_size}, num_warps=num_warps)
        for block_size in [128, 256, 512, 1024, 2048, 4096, 8192]
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=['hidden_dim'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit
def fused_addcmul_fwd_kernel(
    hidden,
    delta,
    ixr,
    ixw,
    ixk,
    ixv,
    ixa,
    ixg,
    oxr,
    oxw,
    oxk,
    oxv,
    oxa,
    oxg,
    use_xg: tl.constexpr,
    xnumel,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]

    valid_indices = xnumel - xoffset
    xmask = xindex < (xoffset + valid_indices)
    x0 = xindex % hidden_dim
    b_hiddn = tl.load(hidden + xindex, xmask, other=0.)
    b_x = tl.load(delta + xindex, xmask, other=0.)
    b_ixr = tl.load(ixr + x0, eviction_policy='evict_last')
    b_ixw = tl.load(ixw + x0, eviction_policy='evict_last')
    b_ixk = tl.load(ixk + x0, eviction_policy='evict_last')
    b_ixv = tl.load(ixv + x0, eviction_policy='evict_last')
    b_ixa = tl.load(ixa + x0, eviction_policy='evict_last')
    b_oxr = tl.fma(b_x, b_ixr, b_hiddn)
    b_oxw = tl.fma(b_x, b_ixw, b_hiddn)
    b_oxk = tl.fma(b_x, b_ixk, b_hiddn)
    b_oxv = tl.fma(b_x, b_ixv, b_hiddn)
    b_oxa = tl.fma(b_x, b_ixa, b_hiddn)

    tl.store(oxr + xindex, b_oxr.to(oxr.dtype.element_ty), xmask)
    tl.store(oxw + xindex, b_oxw.to(oxw.dtype.element_ty), xmask)
    tl.store(oxk + xindex, b_oxk.to(oxk.dtype.element_ty), xmask)
    tl.store(oxv + xindex, b_oxv.to(oxv.dtype.element_ty), xmask)
    tl.store(oxa + xindex, b_oxa.to(oxa.dtype.element_ty), xmask)

    if use_xg:
        b_ixg = tl.load(ixg + x0, eviction_policy='evict_last')
        b_oxg = tl.fma(b_x, b_ixg, b_hiddn)
        tl.store(oxg + xindex, b_oxg, xmask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': block_size}, num_warps=num_warps)
        for block_size in [128, 256, 512, 1024, 2048, 4096, 8192]
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=['hidden_dim'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit
def addcmul_bwd_kernel1(
    ixr,
    ixw,
    ixk,
    ixv,
    ixa,
    ixg,
    dxr,
    dxw,
    dxk,
    dxv,
    dxa,
    dxg,
    ghidden,
    gx,
    use_xg: tl.constexpr,
    xnumel,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr
):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]

    valid_indices = xnumel - xoffset
    xmask = xindex < (xoffset + valid_indices)
    x0 = xindex % hidden_dim

    b_dxr = tl.load(dxr + xindex, None).to(DTYPE)
    b_dxw = tl.load(dxw + xindex, None).to(DTYPE)
    b_dxk = tl.load(dxk + xindex, None).to(DTYPE)
    b_dxv = tl.load(dxv + xindex, None).to(DTYPE)
    b_dxa = tl.load(dxa + xindex, None).to(DTYPE)
    b_ixr = tl.load(ixr + x0, eviction_policy='evict_last').to(DTYPE)
    b_ixw = tl.load(ixw + x0, eviction_policy='evict_last').to(DTYPE)
    b_iwk = tl.load(ixk + x0, eviction_policy='evict_last').to(DTYPE)
    b_ixv = tl.load(ixv + x0, eviction_policy='evict_last').to(DTYPE)
    b_ixa = tl.load(ixa + x0, eviction_policy='evict_last').to(DTYPE)

    if use_xg:
        b_dxg = tl.load(dxg + xindex, None).to(DTYPE)
        b_ixg = tl.load(ixg + x0, eviction_policy='evict_last').to(DTYPE)
        g_hidden = b_dxr + b_dxw + b_dxk + b_dxv + b_dxa + b_dxg
        g_x = b_dxr * b_ixr + b_dxw * b_ixw + b_dxk * b_iwk + b_dxv * b_ixv + b_dxa * b_ixa + b_dxg * b_ixg
    else:
        g_hidden = b_dxr + b_dxw + b_dxk + b_dxv + b_dxa
        g_x = b_dxr * b_ixr + b_dxw * b_ixw + b_dxk * b_iwk + b_dxv * b_ixv + b_dxa * b_ixa

    tl.store(ghidden + xindex, g_hidden.to(ghidden.dtype.element_ty), xmask)
    tl.store(gx + xindex, g_x.to(gx.dtype.element_ty), xmask)


def addcmul_bwd1(d_xr, d_xw, d_xk, d_xv, d_xa, d_xg,
                 x_r, x_w, x_k, x_v, x_a, x_g, hidden_states, delta, use_xg, inplace=True):
    g_hiddn = hidden_states if inplace else torch.empty_like(hidden_states)
    g_delta = torch.empty_like(delta)
    numel = hidden_states.numel()
    def grid(meta): return (triton.cdiv(meta['xnumel'], meta['BLOCK_SIZE']),)
    addcmul_bwd_kernel1[grid](
        ixr=x_r,
        ixw=x_w,
        ixk=x_k,
        ixv=x_v,
        ixa=x_a,
        ixg=x_g,
        dxr=d_xr,
        dxw=d_xw,
        dxk=d_xk,
        dxv=d_xv,
        dxa=d_xa,
        dxg=d_xg,
        ghidden=g_hiddn,
        gx=g_delta,
        use_xg=use_xg,
        xnumel=numel,
        hidden_dim=hidden_states.size(-1),
        DTYPE=tl.float16 if hidden_states.dtype == torch.float16 else tl.float32,
    )
    return g_hiddn, g_delta


@torch.compile(fullgraph=True)
def addcmul_bwd2(d_oxr, d_xw, d_xk, d_xv, d_xa, d_xg, delta, use_xg: bool):
    g_xr = (d_oxr * delta).sum(dim=(0, 1), keepdim=True)
    g_xw = (d_xw * delta).sum(dim=(0, 1), keepdim=True)
    g_xk = (d_xk * delta).sum(dim=(0, 1), keepdim=True)
    g_xv = (d_xv * delta).sum(dim=(0, 1), keepdim=True)
    g_xa = (d_xa * delta).sum(dim=(0, 1), keepdim=True)
    g_xg = (d_xg * delta).sum(dim=(0, 1), keepdim=True) if use_xg else None
    return g_xr, g_xw, g_xk, g_xv, g_xa, g_xg


class Rwkv7FusedAddcmul(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx, hidden_states, delta,
        x_r, x_w, x_k, x_v, x_a, x_g,
        num_elements
    ):
        oxr = torch.empty_like(hidden_states)
        oxw = torch.empty_like(hidden_states)
        oxk = torch.empty_like(hidden_states)
        oxv = torch.empty_like(hidden_states)
        oxa = torch.empty_like(hidden_states)
        if x_g is not None:
            use_xg = True
            oxg = torch.empty_like(hidden_states)
        else:
            use_xg = False
            oxg = None
        ctx.save_for_backward(hidden_states, delta,
                              x_r, x_w, x_k, x_v, x_a, x_g)
        ctx.use_xg = use_xg

        def grid(meta): return (triton.cdiv(meta['xnumel'], meta['BLOCK_SIZE']),)
        fused_addcmul_fwd_kernel[grid](
            hidden_states,
            delta,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            oxr,
            oxw,
            oxk,
            oxv,
            oxa,
            oxg,
            use_xg,
            num_elements,
            hidden_states.size(-1),
        )
        return oxr, oxw, oxk, oxv, oxa, oxg

    @staticmethod
    @input_guard
    def backward(ctx, dxr,
                 dxw, dxk, dxv, dxa, dxg):
        hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g = ctx.saved_tensors

        d_hiddn, d_xx = addcmul_bwd1(dxr, dxw, dxk, dxv, dxa, dxg, x_r, x_w, x_k, x_v, x_a, x_g,
                                     hidden_states, delta, ctx.use_xg)

        d_ixr, d_ixw, d_ixk, d_ixv, d_ixa, d_ixg = addcmul_bwd2(dxr, dxw, dxk, dxv, dxa, dxg, delta, ctx.use_xg)

        return d_hiddn, d_xx, d_ixr, d_ixw, d_ixk, d_ixv, d_ixa, d_ixg, None


def fused_addcmul_rwkv7(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    xr: torch.Tensor,
    xw: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    xa: torch.Tensor,
    xg: Optional[torch.Tensor] = None
):
    num_elements = hidden_states.numel()
    if num_elements < 16777216 and device == "cuda":
        return torch_addcmul_rwkv7(hidden_states, delta, xr, xw, xk, xv, xa, xg)
    else:
        return Rwkv7FusedAddcmul.apply(hidden_states, delta, xr, xw, xk, xv, xa, xg, num_elements)


def torch_addcmul_rwkv7(hidden_states, delta, xr, xw, xk, xv, xa, xg=None):
    oxr = torch.addcmul(hidden_states, delta, xr)
    oxw = torch.addcmul(hidden_states, delta, xw)
    oxk = torch.addcmul(hidden_states, delta, xk)
    oxv = torch.addcmul(hidden_states, delta, xv)
    oxa = torch.addcmul(hidden_states, delta, xa)
    if xg is not None:
        oxg = torch.addcmul(hidden_states, delta, xg)
        return oxr, oxw, oxk, oxv, oxa, oxg
    else:
        return oxr, oxw, oxk, oxv, oxa, None
