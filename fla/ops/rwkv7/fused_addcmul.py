# -*- coding: utf-8 -*-

import logging
import sys
from typing import Optional

import torch
import triton
import triton.language as tl
from packaging.version import Version

from fla.utils import check_pytorch_version, input_guard, is_amd, use_cuda_graph

logger = logging.getLogger(__name__)

if not check_pytorch_version('2.4'):
    logger.warning('PyTorch < 2.4 detected - computations may be slower due to lack of optimizations')


def identity_decorator(fn):
    return fn


current_python_version = Version(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
min_torch_compile_version = Version("3.11")

if current_python_version >= min_torch_compile_version:
    torch_compile = torch.compile(fullgraph=True)
else:
    logger.warning('torch.compile is not available in Python 3.10, using identity decorator instead')
    torch_compile = identity_decorator

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [2, 4, 8, 16, 32]


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
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
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)
    xoffset = i_b * T * D + i_t * D
    indices = tl.arange(0, BD)
    xindex = xoffset + indices
    xmask = indices < D
    b_hiddn = tl.load(hidden + xindex, xmask)
    b_x = tl.load(delta + xindex, xmask)
    b_ixr = tl.load(ixr + indices, xmask)
    b_ixw = tl.load(ixw + indices, xmask)
    b_ixk = tl.load(ixk + indices, xmask)
    b_ixv = tl.load(ixv + indices, xmask)
    b_ixa = tl.load(ixa + indices, xmask)
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
        b_ixg = tl.load(ixg + indices)
        b_oxg = tl.fma(b_x, b_ixg, b_hiddn)
        tl.store(oxg + xindex, b_oxg.to(oxg.dtype.element_ty), xmask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
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
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    DTYPE: tl.constexpr
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)
    xoffset = i_b * T * D + i_t * D
    indices = tl.arange(0, BD)
    xindex = xoffset + indices
    xmask = indices < D
    b_dxr = tl.load(dxr + xindex, xmask).to(DTYPE)
    b_dxw = tl.load(dxw + xindex, xmask).to(DTYPE)
    b_dxk = tl.load(dxk + xindex, xmask).to(DTYPE)
    b_dxv = tl.load(dxv + xindex, xmask).to(DTYPE)
    b_dxa = tl.load(dxa + xindex, xmask).to(DTYPE)
    b_ixr = tl.load(ixr + indices, xmask).to(DTYPE)
    b_ixw = tl.load(ixw + indices, xmask).to(DTYPE)
    b_iwk = tl.load(ixk + indices, xmask).to(DTYPE)
    b_ixv = tl.load(ixv + indices, xmask).to(DTYPE)
    b_ixa = tl.load(ixa + indices, xmask).to(DTYPE)

    if use_xg:
        b_dxg = tl.load(dxg + xindex, xmask).to(DTYPE)
        b_ixg = tl.load(ixg + indices, xmask).to(DTYPE)
        g_hidden = b_dxr + b_dxw + b_dxk + b_dxv + b_dxa + b_dxg
        g_x = b_dxr * b_ixr + b_dxw * b_ixw + b_dxk * b_iwk + b_dxv * b_ixv + b_dxa * b_ixa + b_dxg * b_ixg
    else:
        g_hidden = b_dxr + b_dxw + b_dxk + b_dxv + b_dxa
        g_x = b_dxr * b_ixr + b_dxw * b_ixw + b_dxk * b_iwk + b_dxv * b_ixv + b_dxa * b_ixa

    tl.store(ghidden + xindex, g_hidden.to(ghidden.dtype.element_ty), xmask)
    tl.store(gx + xindex, g_x.to(gx.dtype.element_ty), xmask)


def addcmul_bwd1(d_xr, d_xw, d_xk, d_xv, d_xa, d_xg,
                 x_r, x_w, x_k, x_v, x_a, x_g, hidden_states, delta, use_xg, inplace=True):
    B, T, D = hidden_states.size()
    g_hiddn = hidden_states if inplace else torch.empty_like(hidden_states)
    g_delta = torch.empty_like(delta)
    addcmul_bwd_kernel1[(B, T)](
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
        T=T,
        D=D,
        BD=triton.next_power_of_2(D),
        DTYPE=tl.float16 if hidden_states.dtype == torch.float16 else tl.float32,
    )
    return g_hiddn, g_delta


@torch_compile
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
    ):
        B, T, D = hidden_states.size()
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

        fused_addcmul_fwd_kernel[(B, T)](
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
            T=T,
            D=D,
            BD=triton.next_power_of_2(D),
        )
        ctx.save_for_backward(hidden_states, delta,
                              x_r, x_w, x_k, x_v, x_a, x_g)
        ctx.use_xg = use_xg
        return oxr, oxw, oxk, oxv, oxa, oxg

    @staticmethod
    @input_guard
    def backward(ctx, dxr,
                 dxw, dxk, dxv, dxa, dxg):
        hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g = ctx.saved_tensors

        d_hiddn, d_xx = addcmul_bwd1(dxr, dxw, dxk, dxv, dxa, dxg, x_r, x_w, x_k, x_v, x_a, x_g,
                                     hidden_states, delta, ctx.use_xg)

        d_ixr, d_ixw, d_ixk, d_ixv, d_ixa, d_ixg = addcmul_bwd2(dxr, dxw, dxk, dxv, dxa, dxg, delta, ctx.use_xg)

        return d_hiddn, d_xx, d_ixr, d_ixw, d_ixk, d_ixv, d_ixa, d_ixg


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
    if num_elements < 16777216:
        return torch_addcmul_rwkv7(hidden_states, delta, xr, xw, xk, xv, xa, xg)
    else:
        return Rwkv7FusedAddcmul.apply(hidden_states, delta, xr, xw, xk, xv, xa, xg)


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
