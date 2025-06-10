# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_global_cumsum
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
        for num_warps in [1, 2, 4]
    ],
    key=["BK", "BV", "USE_GK", "USE_GV", "USE_G"],
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    p_q = q + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
    p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
    p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
    p_o = o + ((i_k * all + bos) + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
    if USE_G:
        p_g = g + (bos + ((T-1) if REVERSE else 0)) * H + i_h
    if USE_GK:
        p_gk = gk + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
    if USE_GV:
        p_gv = gv + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

    mask_k = (i_k * BK + tl.arange(0, BK)) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_h = b_h * exp(b_gk[None, :])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            b_h = b_h * exp(b_gv[:, None])
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * exp(b_g)
        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += (-1 if REVERSE else 1) * H*K
        p_k += (-1 if REVERSE else 1) * H*K
        p_v += (-1 if REVERSE else 1) * H*V
        p_o += (-1 if REVERSE else 1) * H*V
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * H*K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * H*V
        if USE_G:
            p_g += (-1 if REVERSE else 1) * H

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BK', 'BV', 'USE_GK', 'USE_GV', 'USE_G'],
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_bwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    h0,
    do,
    dq,
    dk,
    dv,
    dht,
    dh0,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
    p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
    p_do = do + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
    p_dq = dq + ((i_v * all + bos) + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
    if USE_G:
        p_g = g + (bos + ((T-1) if REVERSE else 0)) * H + i_h
    if USE_GK:
        p_gk = gk + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
    if USE_GV:
        p_gv = gv + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * exp(b_g)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_h = b_h * exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            b_h = b_h * exp(b_gv[None, :])
        b_h += b_k[:, None] * b_v[None, :]
        b_dq = b_h * b_do[None, :]
        b_dq = tl.sum(b_dq, axis=1) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)

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

    p_q = q + (bos + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
    p_k = k + (bos + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
    p_v = v + (bos + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
    p_do = do + (bos + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
    p_dk = dk + ((i_v * all + bos) + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
    p_dv = dv + ((i_k * all + bos) + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
    if USE_G:
        p_g = g + (bos + ((T - 1) if not REVERSE else 0)) * H + i_h
    if USE_GK:
        p_gk = gk + (bos + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
    if USE_GV:
        p_gv = gv + (bos + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_nh * K*V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_dh += tl.load(p_dht, mask=mask_h, other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_dh *= exp(b_g)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_dh *= exp(b_gk)[:, None]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            b_dh *= exp(b_gv)[None, :]
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)

        p_q += (1 if REVERSE else -1) * H*K
        p_k += (1 if REVERSE else -1) * H*K
        p_v += (1 if REVERSE else -1) * H*V
        p_do += (1 if REVERSE else -1) * H*V
        p_dk += (1 if REVERSE else -1) * H*K
        p_dv += (1 if REVERSE else -1) * H*V
        if USE_G:
            p_g += (1 if REVERSE else -1) * H
        if USE_GK:
            p_gk += (1 if REVERSE else -1) * H*K
        if USE_GV:
            p_gv += (1 if REVERSE else -1) * H*V

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = dh0 + i_nh * K*V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_h)


def fused_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
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
    BK, BV = min(K, 64), min(V, 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    o = q.new_empty(NK, *v.shape, dtype=torch.float32)

    grid = (NV, NK, N * H)
    fused_recurrent_fwd_kernel[grid](
        q,
        k,
        v,
        g,
        gk,
        gv,
        o,
        h0,
        ht,
        cu_seqlens,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
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

    BK, BV = min(K, 64), min(V, 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    dq = q.new_empty(NV, *q.shape, dtype=torch.float32)
    dk = q.new_empty(NV, *k.shape, dtype=torch.float32)
    dv = q.new_empty(NK, *v.shape, dtype=torch.float32)
    h0 = initial_state
    dh0 = torch.empty_like(initial_state) if initial_state is not None else None

    grid = (NV, NK, N * H)
    fused_recurrent_bwd_kernel[grid](
        q,
        k,
        v,
        g,
        gk,
        gv,
        h0,
        do,
        dq,
        dk,
        dv,
        dht,
        dh0,
        cu_seqlens,
        scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        REVERSE=reverse,
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    dg, dgk, dgv = None, None, None
    if g is not None:
        dg = chunk_global_cumsum((dq * q.float() - dk * k.float()).sum(-1), reverse=not reverse, cu_seqlens=cu_seqlens)
    if gk is not None:
        dgk = chunk_global_cumsum(dq * q.float() - dk * k.float(), reverse=not reverse, cu_seqlens=cu_seqlens)
    if gv is not None:
        dgv = chunk_global_cumsum(do.float() * o.float() - dv * v.float(), reverse=not reverse, cu_seqlens=cu_seqlens)

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
            gk=gk,
            gv=gv,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v, g, gk, gv, initial_state, o)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype), ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, g, gk, gv, initial_state, o = ctx.saved_tensors
        # not supported yet.
        if dht is not None:
            if not dht.eq(0).all():
                if g is not None:
                    assert g.requires_grad is False, "Cannot load final state gradient and use gates at the same time"
                if gk is not None:
                    assert gk.requires_grad is False, "Cannot load final state gradient and use gates at the same time"
                if gv is not None:
                    assert gv.requires_grad is False, "Cannot load final state gradient and use gates at the same time"
        dq, dk, dv, dg, dgk, dgv, dh0 = fused_recurrent_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
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
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg, dgk, dgv, None, dh0, None, None, None


def fused_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
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
        gk,
        gv,
        scale,
        initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
    )
