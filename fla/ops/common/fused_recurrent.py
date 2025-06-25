# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [4, 8]
    ],
    key=['BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_GK', 'USE_GV'],
)
@triton.jit(do_not_specialize=['B', 'T'])
def fused_recurrent_fwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    gk,
    gv,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_o = o + ((i_k * all + bos) + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    if USE_G:
        p_g = g + (bos + ((T-1) if REVERSE else 0)) * H + i_h
    if USE_GK:
        p_gk = gk + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    if USE_GV:
        p_gv = gv + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    if USE_G_GAMMA:
        b_g_gamma = tl.load(g_gamma + i_h)

    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * exp(b_g)
        if USE_G_GAMMA:
            b_h = b_h * exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
            b_h = b_h * exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            b_h = b_h * exp(b_gv[None, :])
        b_h += b_k[:, None] * b_v[None, :]
        b_o = b_h * b_q[:, None]
        b_o = tl.sum(b_o, axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)
        p_q += (-1 if REVERSE else 1) * H*K
        p_k += (-1 if REVERSE else 1) * H*K
        p_v += (-1 if REVERSE else 1) * H*V
        p_o += (-1 if REVERSE else 1) * H*V
        if USE_G:
            p_g += (-1 if REVERSE else 1) * H
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * H*K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * H*V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=m_h)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [4]
    ],
    key=['BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_GK', 'USE_GV'],
)
@triton.jit(do_not_specialize=['B', 'T'])
def fused_recurrent_bwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    gk,
    gv,
    o,
    h0,
    do,
    dq,
    dk,
    dv,
    dg,
    dgk,
    dgv,
    dht,
    dh0,
    cu_seqlens,
    scale,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    NV = tl.cdiv(V, BV)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]

    p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_do = do + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_dq = dq + ((i_v * all + bos) + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    if USE_G:
        p_g = g + (bos + ((T-1) if REVERSE else 0)) * H + i_h
    if USE_GK:
        p_gk = gk + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    if USE_GV:
        p_gv = gv + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    if USE_G_GAMMA:
        b_g_gamma = tl.load(g_gamma + i_h)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=m_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * exp(b_g)
        if USE_G_GAMMA:
            b_h = b_h * exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
            b_h = b_h * exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            b_h = b_h * exp(b_gv[None, :])
        b_h += b_k[:, None] * b_v[None, :]
        b_dq = b_h * b_do[None, :]
        b_dq = tl.sum(b_dq, axis=1) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_k)

        p_k += (-1 if REVERSE else 1) * H*K
        p_v += (-1 if REVERSE else 1) * H*V
        p_do += (-1 if REVERSE else 1) * H*V
        p_dq += (-1 if REVERSE else 1) * H*K
        if USE_G:
            p_g += (-1 if REVERSE else 1) * H
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * H*K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * H*V

    # sync threads
    tl.debug_barrier()

    p_q = q + (bos + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_k = k + (bos + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_v = v + (bos + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + o_v

    p_do = do + (bos + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + o_v
    p_dq = dq + ((i_v * all + bos) + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_dk = dk + ((i_v * all + bos) + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_dv = dv + ((i_k * all + bos) + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + o_v
    if USE_G:
        p_g = g + (bos + ((T - 1) if not REVERSE else 0)) * H + i_h
        p_dg = dg + ((i_k * NV + i_v) * all + bos + ((T - 1) if not REVERSE else 0)) * H + i_h
    if USE_GK:
        p_gk = gk + (bos + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + o_k
        p_dgk = dgk + ((i_v * all + bos) + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    if USE_GV:
        p_o = o + (bos + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + o_v
        p_gv = gv + (bos + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + o_v
        p_dgv = dgv + ((i_k * all + bos) + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + o_v

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_dh += tl.load(p_dht, mask=m_h, other=0).to(tl.float32)

    if USE_G:
        b_dg = tl.sum(b_h * b_dh)
    if USE_GK:
        b_dgk = tl.sum(b_h * b_dh, 1)
    if USE_GV:
        b_dgv = tl.sum(b_h * b_dh, 0)

    for _ in range(T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=m_v, other=0).to(tl.float32)
        b_dh += (b_q * scale)[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_dq = tl.load(p_dq, mask=m_k, other=0).to(tl.float32)
            b_dg += tl.sum(b_q * b_dq - b_k * b_dk)
            b_dh *= exp(b_g)
            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty))
        if USE_G_GAMMA:
            b_dh *= exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
            b_dq = tl.load(p_dq, mask=m_k, other=0).to(tl.float32)
            b_dgk += b_q * b_dq - b_k * b_dk
            b_dh *= exp(b_gk)[:, None]
            tl.store(p_dgk, b_dgk.to(p_dgk.dtype.element_ty), mask=m_k)
        if USE_GV:
            b_o = tl.load(p_o, mask=m_v, other=0).to(tl.float32)
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            if i_k == 0:
                b_dgv += b_o * b_do
            b_dgv -= b_v * b_dv
            b_dh *= exp(b_gv)[None, :]
            tl.store(p_dgv, b_dgv.to(p_dgv.dtype.element_ty), mask=m_v)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_v)

        p_q += (1 if REVERSE else -1) * H*K
        p_k += (1 if REVERSE else -1) * H*K
        p_v += (1 if REVERSE else -1) * H*V

        p_do += (1 if REVERSE else -1) * H*V
        p_dq += (1 if REVERSE else -1) * H*K
        p_dk += (1 if REVERSE else -1) * H*K
        p_dv += (1 if REVERSE else -1) * H*V
        if USE_G:
            p_g += (1 if REVERSE else -1) * H
            p_dg += (1 if REVERSE else -1) * H
        if USE_GK:
            p_gk += (1 if REVERSE else -1) * H*K
            p_dgk += (1 if REVERSE else -1) * H*K
        if USE_GV:
            p_o += (1 if REVERSE else -1) * H*V
            p_gv += (1 if REVERSE else -1) * H*V
            p_dgv += (1 if REVERSE else -1) * H*V

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = dh0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=m_h)


def fused_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    o = q.new_empty(NK, *v.shape, dtype=torch.float32)

    grid = (NV, NK, N * H)
    fused_recurrent_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        o=o,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        REVERSE=reverse,
    )
    o = o.sum(0)
    return o, ht


def fused_recurrent_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    o: Optional[torch.Tensor] = None,
    do: Optional[torch.Tensor] = None,
    dht: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    dq = q.new_empty(NV, *q.shape, dtype=torch.float32)
    dk = q.new_empty(NV, *k.shape, dtype=torch.float32)
    dv = q.new_empty(NK, *v.shape, dtype=torch.float32)
    dh0 = torch.empty_like(h0) if h0 is not None else None

    dg, dgk, dgv = None, None, None
    if g is not None:
        dg = g.new_empty(NK*NV, *g.shape, dtype=torch.float32)
    if gk is not None:
        dgk = gk.new_empty(NV, *gk.shape, dtype=torch.float32)
    if gv is not None:
        dgv = gv.new_empty(NK, *gv.shape, dtype=torch.float32)

    grid = (NV, NK, N * H)
    fused_recurrent_bwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        o=o,
        h0=h0,
        do=do,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        dgk=dgk,
        dgv=dgv,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        REVERSE=reverse,
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    if g is not None:
        dg = dg.sum(0).to(g)
    if gk is not None:
        dgk = dgk.sum(0).to(gk)
    if gv is not None:
        dgv = dgv.sum(0).to(gv)

    return dq, dk, dv, dg, dgk, dgv, dh0


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        g_gamma: Optional[torch.Tensor] = None,
        gk: Optional[torch.Tensor] = None,
        gv: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None
    ):
        o, ht = fused_recurrent_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            gk=gk,
            gv=gv,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v, g, g_gamma, gk, gv, initial_state, o)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype), ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, g, g_gamma, gk, gv, initial_state, o = ctx.saved_tensors
        dq, dk, dv, dg, dgk, dgv, dh0 = fused_recurrent_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            gk=gk,
            gv=gv,
            o=o,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            reverse=ctx.reverse,
            cu_seqlens=ctx.cu_seqlens,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg, None, dgk, dgv, None, dh0, None, None, None


def fused_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        g_gamma,
        gk,
        gv,
        scale,
        initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
    )
