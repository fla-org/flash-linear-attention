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
    hidden_ptr,
    x_ptr,
    ixr_ptr,
    ixw_ptr,
    ixk_ptr,
    ixv_ptr,
    ixa_ptr,
    ixg_ptr,
    oxr_ptr,
    oxw_ptr,
    oxk_ptr,
    oxv_ptr,
    oxa_ptr,
    oxg_ptr,
    use_xg: tl.constexpr,
    xnumel,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]

    valid_indices = xnumel - xoffset
    xmask = xindex < (xoffset + valid_indices)
    x0 = xindex % hidden_dim
    b_hiddn = tl.load(hidden_ptr + (xindex), xmask, other=0.).to(DTYPE)
    b_x = tl.load(x_ptr + (xindex), xmask, other=0.).to(DTYPE)
    b_ixr = tl.load(ixr_ptr + (x0), eviction_policy='evict_last').to(DTYPE)
    b_ixw = tl.load(ixw_ptr + (x0), eviction_policy='evict_last').to(DTYPE)
    b_ixk = tl.load(ixk_ptr + (x0), eviction_policy='evict_last').to(DTYPE)
    b_ixv = tl.load(ixv_ptr + (x0), eviction_policy='evict_last').to(DTYPE)
    b_ixa = tl.load(ixa_ptr + (x0), eviction_policy='evict_last').to(DTYPE)
    b_oxr = b_hiddn + b_x * b_ixr
    b_oxw = b_hiddn + b_x * b_ixw
    b_oxk = b_hiddn + b_x * b_ixk
    b_oxv = b_hiddn + b_x * b_ixv
    b_oxa = b_hiddn + b_x * b_ixa

    tl.store(oxr_ptr + (xindex), b_oxr.to(oxr_ptr.dtype.element_ty), xmask)
    tl.store(oxw_ptr + (xindex), b_oxw.to(oxw_ptr.dtype.element_ty), xmask)
    tl.store(oxk_ptr + (xindex), b_oxk.to(oxk_ptr.dtype.element_ty), xmask)
    tl.store(oxv_ptr + (xindex), b_oxv.to(oxv_ptr.dtype.element_ty), xmask)
    tl.store(oxa_ptr + (xindex), b_oxa.to(oxa_ptr.dtype.element_ty), xmask)

    if use_xg:
        b_ixg = tl.load(ixg_ptr + (x0), eviction_policy='evict_last').to(DTYPE)
        b_oxg = b_hiddn + b_x * b_ixg
        tl.store(oxg_ptr + (xindex), b_oxg.to(oxg_ptr.dtype.element_ty), xmask)


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
    ixr_ptr,
    ixw_ptr,
    ixk_ptr,
    ixv_ptr,
    ixa_ptr,
    ixg_ptr,
    dxr_ptr,
    dxw_ptr,
    dxk_ptr,
    dxv_ptr,
    dxa_ptr,
    dxg_ptr,
    ghidden_ptr,
    gx_ptr,
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

    b_dxr = tl.load(dxr_ptr + (xindex), None).to(DTYPE)
    b_dxw = tl.load(dxw_ptr + (xindex), None).to(DTYPE)
    b_dxk = tl.load(dxk_ptr + (xindex), None).to(DTYPE)
    b_dxv = tl.load(dxv_ptr + (xindex), None).to(DTYPE)
    b_dxa = tl.load(dxa_ptr + (xindex), None).to(DTYPE)
    b_ixr = tl.load(ixr_ptr + (x0), eviction_policy='evict_last').to(DTYPE)
    b_ixw = tl.load(ixw_ptr + (x0), eviction_policy='evict_last').to(DTYPE)
    b_iwk = tl.load(ixk_ptr + (x0), eviction_policy='evict_last').to(DTYPE)
    b_ixv = tl.load(ixv_ptr + (x0), eviction_policy='evict_last').to(DTYPE)
    b_ixa = tl.load(ixa_ptr + (x0), eviction_policy='evict_last').to(DTYPE)

    if use_xg:
        b_dxg = tl.load(dxg_ptr + (xindex), None).to(DTYPE)
        b_ixg = tl.load(ixg_ptr + (x0), eviction_policy='evict_last').to(DTYPE)
        g_hidden = b_dxr + b_dxw + b_dxk + b_dxv + b_dxa + b_dxg
        g_x = b_dxr * b_ixr + b_dxw * b_ixw + b_dxk * b_iwk + b_dxv * b_ixv + b_dxa * b_ixa + b_dxg * b_ixg
    else:
        g_hidden = b_dxr + b_dxw + b_dxk + b_dxv + b_dxa
        g_x = b_dxr * b_ixr + b_dxw * b_ixw + b_dxk * b_iwk + b_dxv * b_ixv + b_dxa * b_ixa

    tl.store(ghidden_ptr + (xindex), g_hidden.to(ghidden_ptr.dtype.element_ty), xmask)
    tl.store(gx_ptr + (xindex), g_x.to(gx_ptr.dtype.element_ty), xmask)


def addcmul_bwd1(d_oxr, d_oxw, d_oxk, d_oxv, d_oxa, d_oxg,
                 x_r, x_w, x_k, x_v, x_a, x_g, hidden_states, xx, use_xg, inplace=True):
    d_hiddn = hidden_states if inplace else torch.empty_like(hidden_states)
    d_xx = torch.empty_like(xx)
    numel = hidden_states.numel()
    def grid(meta): return (triton.cdiv(meta['xnumel'], meta['BLOCK_SIZE']),)
    addcmul_bwd_kernel1[grid](
        ixr_ptr=x_r,
        ixw_ptr=x_w,
        ixk_ptr=x_k,
        ixv_ptr=x_v,
        ixa_ptr=x_a,
        ixg_ptr=x_g,
        dxr_ptr=d_oxr,
        dxw_ptr=d_oxw,
        dxk_ptr=d_oxk,
        dxv_ptr=d_oxv,
        dxa_ptr=d_oxa,
        dxg_ptr=d_oxg,
        ghidden_ptr=d_hiddn,
        gx_ptr=d_xx,
        use_xg=use_xg,
        xnumel=numel,
        hidden_dim=hidden_states.size(-1),
        DTYPE=tl.float16 if hidden_states.dtype == torch.float16 else tl.float32,
    )
    return d_hiddn, d_xx


@torch.compile(fullgraph=True)
def addcmul_bwd2(d_oxr, d_oxw, d_oxk, d_oxv, d_oxa, d_oxg, xx, use_xg: bool):
    g_xr = (d_oxr * xx).sum(dim=(0, 1), keepdim=True)
    g_xw = (d_oxw * xx).sum(dim=(0, 1), keepdim=True)
    g_xk = (d_oxk * xx).sum(dim=(0, 1), keepdim=True)
    g_xv = (d_oxv * xx).sum(dim=(0, 1), keepdim=True)
    g_xa = (d_oxa * xx).sum(dim=(0, 1), keepdim=True)
    g_xg = (d_oxg * xx).sum(dim=(0, 1), keepdim=True) if use_xg else None
    return g_xr, g_xw, g_xk, g_xv, g_xa, g_xg


class Rwkv7FusedAddcmul(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx, hidden_states, xx,
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
        ctx.save_for_backward(hidden_states, xx,
                              x_r, x_w, x_k, x_v, x_a, x_g)
        ctx.use_xg = use_xg

        def grid(meta): return (triton.cdiv(meta['xnumel'], meta['BLOCK_SIZE']),)
        fused_addcmul_fwd_kernel[grid](
            hidden_states,
            xx,
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
            DTYPE=tl.float16 if hidden_states.dtype == torch.float16 else tl.float32,
        )
        return oxr, oxw, oxk, oxv, oxa, oxg

    @staticmethod
    @input_guard
    def backward(ctx, dxr,
                 dxw, dxk, dxv, dxa, dxg):
        hidden_states, xx, x_r, x_w, x_k, x_v, x_a, x_g = ctx.saved_tensors

        d_hiddn, d_xx = addcmul_bwd1(dxr, dxw, dxk, dxv, dxa, dxg, x_r, x_w, x_k, x_v, x_a, x_g, hidden_states, xx, ctx.use_xg)

        d_ixr, d_ixw, d_ixk, d_ixv, d_ixa, d_ixg = addcmul_bwd2(dxr, dxw, dxk, dxv, dxa, dxg, xx, ctx.use_xg)

        return d_hiddn, d_xx, d_ixr, d_ixw, d_ixk, d_ixv, d_ixa, d_ixg, None


def fused_addcmul_rwkv7(
    hidden_states: torch.Tensor,
    xx: torch.Tensor,
    xr: torch.Tensor,
    xw: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    xa: torch.Tensor,
    xg: Optional[torch.Tensor] = None
):
    num_elements = hidden_states.numel()
    if num_elements < 16777216 and device == "cuda":
        return torch_addcmul_rwkv7(hidden_states, xx, xr, xw, xk, xv, xa, xg)
    else:
        return Rwkv7FusedAddcmul.apply(hidden_states, xx, xr, xw, xk, xv, xa, xg, num_elements)


def torch_addcmul_rwkv7(hidden_states, xx, xr, xw, xk, xv, xa, xg=None):
    oxr = torch.addcmul(hidden_states, xx, xr)
    oxw = torch.addcmul(hidden_states, xx, xw)
    oxk = torch.addcmul(hidden_states, xx, xk)
    oxv = torch.addcmul(hidden_states, xx, xv)
    oxa = torch.addcmul(hidden_states, xx, xa)
    if xg is not None:
        oxg = torch.addcmul(hidden_states, xx, xg)
        return oxr, oxw, oxk, oxv, oxa, oxg
    else:
        return oxr, oxw, oxk, oxv, oxa, None
