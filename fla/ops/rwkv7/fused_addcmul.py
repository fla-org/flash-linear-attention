
import logging
import os
import sys

import torch
import triton
import triton.language as tl
from packaging.version import Version

from fla.utils import autotune_cache_kwargs, check_pytorch_version, input_guard, is_amd, use_cuda_graph

logger = logging.getLogger(__name__)

if not check_pytorch_version('2.4'):
    logger.warning('PyTorch < 2.4 detected - computations may be slower due to lack of optimizations')


def identity_decorator(fn):
    return fn


current_python_version = Version(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
min_torch_compile_version = Version("3.11")
fla_use_compile = os.getenv('FLA_USE_COMPILE', '1').lower() in ('1', 'true', 'yes')

if current_python_version >= min_torch_compile_version and fla_use_compile:
    torch_compile = torch.compile(fullgraph=True)
else:
    logger.warning('torch.compile is not available in Python 3.10, using identity decorator instead')
    torch_compile = identity_decorator

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [2, 4, 8, 16, 32]


@triton.autotune(
    configs=[
        triton.Config({'BT': BT},  num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
        for BT in [2, 4, 8]
    ],
    key=['BD'],
    use_cuda_graph=use_cuda_graph,
    **autotune_cache_kwargs,
)
@triton.jit
def fused_addcmul_fwd_kernel(
    hidden,
    delta,
    ixr, ixw, ixk, ixv, ixa, ixg,
    oxr, oxw, oxk, oxv, oxa, oxg,
    use_xg: tl.constexpr,
    T,
    T_OFFSET,
    BT: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1) * BT

    bos = i_b * (T + T_OFFSET)
    t_vec = i_t + T_OFFSET + tl.arange(0, BT)
    mask_t = t_vec < (T + T_OFFSET)
    o_d = tl.arange(0, BD)[None, :]
    off_vec = (bos + t_vec)[:, None] * D + o_d
    m_d = o_d < D
    mask = mask_t[:, None] & m_d

    b_h = tl.load(hidden + off_vec, mask=mask, other=0.)
    b_x = tl.load(delta + off_vec, mask=mask, other=0.)
    b_r = tl.load(ixr + o_d, mask=m_d)
    b_w = tl.load(ixw + o_d, mask=m_d)
    b_k = tl.load(ixk + o_d, mask=m_d)
    b_v = tl.load(ixv + o_d, mask=m_d)
    b_a = tl.load(ixa + o_d, mask=m_d)

    o_r = tl.fma(b_x, b_r, b_h)
    o_w = tl.fma(b_x, b_w, b_h)
    o_k = tl.fma(b_x, b_k, b_h)
    o_v = tl.fma(b_x, b_v, b_h)
    o_a = tl.fma(b_x, b_a, b_h)

    tl.store(oxr + off_vec, o_r.to(oxr.dtype.element_ty), mask=mask)
    tl.store(oxw + off_vec, o_w.to(oxw.dtype.element_ty), mask=mask)
    tl.store(oxk + off_vec, o_k.to(oxk.dtype.element_ty), mask=mask)
    tl.store(oxv + off_vec, o_v.to(oxv.dtype.element_ty), mask=mask)
    tl.store(oxa + off_vec, o_a.to(oxa.dtype.element_ty), mask=mask)

    if use_xg:
        b_g = tl.load(ixg + o_d, mask=m_d)
        o_g = tl.fma(b_x, b_g, b_h)
        tl.store(oxg + off_vec, o_g.to(oxg.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
        for BT in [2, 4, 8]
    ],
    key=['BD'],
    use_cuda_graph=use_cuda_graph,
    **autotune_cache_kwargs,
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
    T_OFFSET,
    BT: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    DTYPE: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)

    t_idx = T_OFFSET + i_t * BT + tl.arange(0, BT)[:, None]
    mask_t = t_idx < (T + T_OFFSET)

    d_idx = tl.arange(0, BD)[None, :]
    mask_d = d_idx < D
    mask = mask_t & mask_d

    offset_base = i_b * (T + T_OFFSET) * D
    x_idx = (offset_base + t_idx * D + d_idx).to(tl.uint32)

    b_dxr = tl.load(dxr + x_idx, mask=mask).to(DTYPE)
    b_dxw = tl.load(dxw + x_idx, mask=mask).to(DTYPE)
    b_dxk = tl.load(dxk + x_idx, mask=mask).to(DTYPE)
    b_dxv = tl.load(dxv + x_idx, mask=mask).to(DTYPE)
    b_dxa = tl.load(dxa + x_idx, mask=mask).to(DTYPE)

    b_ixr = tl.load(ixr + d_idx, mask=mask_d).to(DTYPE)
    b_ixw = tl.load(ixw + d_idx, mask=mask_d).to(DTYPE)
    b_ixk = tl.load(ixk + d_idx, mask=mask_d).to(DTYPE)
    b_ixv = tl.load(ixv + d_idx, mask=mask_d).to(DTYPE)
    b_ixa = tl.load(ixa + d_idx, mask=mask_d).to(DTYPE)

    g_hidden = b_dxr + b_dxw + b_dxk + b_dxv + b_dxa
    g_x = b_dxr * b_ixr + b_dxw * b_ixw + b_dxk * b_ixk + b_dxv * b_ixv + b_dxa * b_ixa

    if use_xg:
        b_dxg = tl.load(dxg + x_idx, mask=mask).to(DTYPE)
        b_ixg = tl.load(ixg + d_idx, mask=mask_d).to(DTYPE)
        g_hidden += b_dxg
        g_x += b_dxg * b_ixg

    tl.store(ghidden + x_idx, g_hidden.to(ghidden.dtype.element_ty), mask=mask)
    tl.store(gx + x_idx, g_x.to(gx.dtype.element_ty), mask=mask)


def addcmul_bwd1(d_xr, d_xw, d_xk, d_xv, d_xa, d_xg,
                 x_r, x_w, x_k, x_v, x_a, x_g, hidden_states, delta, use_xg, inplace=True):
    B, T, D = hidden_states.size()
    g_hiddn = hidden_states if inplace else torch.empty_like(hidden_states)
    g_delta = torch.empty_like(delta)
    for t in range(0, T, 65536):
        T_OFFSET = t
        T_SIZE = min(65536, T - t)
        def grid(meta): return (B, triton.cdiv(T_SIZE, meta['BT']))
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
            T=T_SIZE,
            T_OFFSET=T_OFFSET,
            D=D,
            BD=triton.next_power_of_2(D),
            DTYPE=tl.float16 if hidden_states.dtype == torch.float16 else tl.float32,
        )
    return g_hiddn, g_delta


@torch_compile
def addcmul_bwd2(d_oxr, d_xw, d_xk, d_xv, d_xa, d_xg, delta, use_xg: bool):
    g_xr = (d_oxr * delta).sum(dim=(0, 1), keepdim=True, dtype=torch.float32)
    g_xw = (d_xw * delta).sum(dim=(0, 1), keepdim=True, dtype=torch.float32)
    g_xk = (d_xk * delta).sum(dim=(0, 1), keepdim=True, dtype=torch.float32)
    g_xv = (d_xv * delta).sum(dim=(0, 1), keepdim=True, dtype=torch.float32)
    g_xa = (d_xa * delta).sum(dim=(0, 1), keepdim=True, dtype=torch.float32)
    g_xg = (d_xg * delta).sum(dim=(0, 1), keepdim=True, dtype=torch.float32) if use_xg else None
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

        for t in range(0, T, 65536):
            T_OFFSET = t
            T_SIZE = min(65536, T - t)
            def grid(meta): return (B, triton.cdiv(T_SIZE, meta['BT']))
            fused_addcmul_fwd_kernel[grid](
                hidden_states, delta,
                x_r, x_w, x_k, x_v, x_a, x_g,
                oxr, oxw, oxk, oxv, oxa, oxg,
                use_xg,
                T=T_SIZE,
                T_OFFSET=T_OFFSET,
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
    xg: torch.Tensor | None = None,
):
    if hidden_states.shape[1] == 1:
        # Special case for decode
        return torch_addcmul_rwkv7(hidden_states, delta, xr, xw, xk, xv, xa, xg)
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
