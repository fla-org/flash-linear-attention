# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from packaging import version

from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.exp import safe_exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


@triton.jit
def prepare_qg_kg(
    q,
    k,
    g,
    qg,
    kg,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * T*K + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * T*K + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * T*K + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * T*K + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * T*K + i_c * BT * K + i_k * BK + tl.arange(0, BK)

    mask = (i_k * BK + tl.arange(0, BK)) < K

    last_decay = tl.load(g + i_bh * T*K + (i_c * BT + BT - 1) * K + i_k * BK + tl.arange(0, BK))

    for _ in range(BT):
        b_q = tl.load(p_q, mask=mask, other=0)
        b_k = tl.load(p_k, mask=mask, other=0)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_q *= tl.exp(b_g) * scale
        b_k *= tl.exp(last_decay - b_g)
        tl.store(p_kg, b_k.to(p_kg.dtype.element_ty), mask=mask)
        tl.store(p_qg, b_q.to(p_qg.dtype.element_ty), mask=mask)
        p_q += K
        p_g += K
        p_k += K
        p_kg += K
        p_qg += K


@triton.jit
def bwd_decay_global_cumsum(
    dq_inner,
    dq_inter,
    dk_inner,
    dk_inter,
    q,
    k,
    g,
    dg,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_g = g + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dg = dg + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dq_inner = dq_inner + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dk_inner = dk_inner + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dq_inter = dq_inter + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dk_inter = dk_inter + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    cum_grad_dg = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < K
    last_g = tl.zeros([BK], dtype=tl.float32)
    for j in range(BT-1, -1, -1):
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        if j == (BT-1):
            last_g = b_g
        b_dq1 = tl.load(p_dq_inner, mask=mask, other=0)
        b_dq2 = tl.load(p_dq_inter, mask=mask, other=0)
        b_dq2 *= tl.exp(b_g)
        b_dq = b_dq1 + b_dq2
        tl.store(p_dq_inter, b_dq, mask=mask)
        b_dk1 = tl.load(p_dk_inner, mask=mask, other=0)
        b_dk2 = tl.load(p_dk_inter, mask=mask, other=0)
        b_dk2 *= safe_exp(last_g - b_g)
        b_dk = b_dk1 + b_dk2
        tl.store(p_dk_inter, b_dk, mask=mask)
        b_q = tl.load(p_q, mask=mask, other=0)
        b_k = tl.load(p_k, mask=mask, other=0)
        b_dg = b_dq * b_q - b_dk * b_k
        cum_grad_dg += b_dg
        tl.store(p_dg, cum_grad_dg.to(p_dg.dtype.element_ty), mask=mask)
        p_g -= K
        p_k -= K
        p_q -= K
        p_dq_inner -= K
        p_dk_inner -= K
        p_dq_inter -= K
        p_dk_inter -= K
        p_dg -= K


@triton.jit
def fused_chunk_gla_fwd_kernel(
    q,  # query [B, H, T, K]
    k,  # key [B, H, T, K]
    v,  # value [B, H, T, V]
    g,  # cumulative sum of log decay [B, H, T, K]
    o,  # output [B, H, T, V]

    h0,  # initial state of the chunk [B, H, K, V]
    ht,  # final state of the chunk [B, H, K, V]


    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # H
    T: tl.constexpr,  # T
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    # make block pointers
    p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (0, i_k * BK), (BT, BK), (1, 0))
    p_gn = g + i_bh * T*K + (BT - 1) * K + i_k * BK + tl.arange(0, BK)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * T*V, (T, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0))

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    mask = (i_k * BK + tl.arange(0, BK)) < K

    for i in range(0, tl.cdiv(T, BT)):
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gn = tl.load(p_gn, mask=mask, other=0).to(tl.float32)
        if CHECK and i == 0:
            b_o = tl.dot(b_q.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=False)
            b_h = b_h * tl.exp(b_gn)[:, None] + tl.dot(b_k.to(b_v.dtype), b_v, allow_tf32=False)
        else:
            b_o = tl.dot(b_q.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=False)
            b_h = b_h * tl.exp(b_gn)[:, None] + tl.dot(b_k.to(b_v.dtype), b_v, allow_tf32=False)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_gn += BT * K

    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h.to(p_final.dtype.element_ty), boundary_check=(0, 1))


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_chunk_gla_bwd_kernel(
    q, k, v, g,
    do,  # gradient of output [B, H, T, V]
    dq,  # gradient of query [NV, B, H, T, K]
    dk,  # gradient of key [NV, B, H, T, K]
    dv,  # gradient of value [NK, B, H, T, V]

    h0,  # initial state of the chunk [B, H, K, V]

    scale,  # K ** -0.5

    B: tl.constexpr,  # B
    H: tl.constexpr,  # H
    T: tl.constexpr,  # T
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    # clamp_min, # minimum log value of the gate for numerical stability. default: -5
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # [BV, BK]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    mask = (i_k * BK + tl.arange(0, BK)) < K
    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = g + i_bh * T*K + ((i+1) * BT - 1) * K + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh+i_v*B*H)*T*K, (T, K), (K, 1), (i * BT, i_k * BK), (BT, BK), (1, 0))
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        # [BT, K]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gn = tl.load(p_gn, mask=mask, other=0).to(tl.float32)

        # [V, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, V]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [V, K]
        if CHECK and i == 0:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h * tl.exp(b_gn)[None, :] + tl.dot(b_v, b_k.to(b_v.dtype), allow_tf32=False)
        else:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h * tl.exp(b_gn)[None, :] + tl.dot(b_v, b_k.to(b_v.dtype), allow_tf32=False)
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    # sync threads
    b_h = None
    tl.debug_barrier()
    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)

    # cum = tl.zeros([BK], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = g + i_bh * T*K + (T - (i-1) * BT - 1) * K + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * T*K, (T, K),
                                 (K, 1), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * T*V, (T, V),
                                 (V, 1), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        # [K, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, K]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, V]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_db = tl.load(p_gn, mask=mask, other=0).to(tl.float32)

        # inter-chunk
        # [K, V]
        if CHECK and i == 1:
            b_dk = tl.trans(tl.dot(b_dh.to(b_v.dtype), tl.trans(b_v), allow_tf32=False))
            b_dv = tl.dot((b_k).to(b_v.dtype), b_dh.to(b_v.dtype), allow_tf32=False)
            b_dh = b_dh * tl.exp(b_db)[:, None] + tl.dot(b_q.to(b_do.dtype), b_do, allow_tf32=False)
        else:
            b_dk = tl.trans(tl.dot(b_dh.to(b_v.dtype), tl.trans(b_v), allow_tf32=False))
            b_dv = tl.dot((b_k).to(b_v.dtype), b_dh.to(b_v.dtype), allow_tf32=False)
            b_dh = b_dh * tl.exp(b_db)[:, None] + tl.dot(b_q.to(b_do.dtype), b_do, allow_tf32=False)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def fwd_inner_chunk(
    q, k, g, A,
    scale,  # K ** -0.5
    B: tl.constexpr,  # B
    H: tl.constexpr,  # H
    T: tl.constexpr,  # T
    K: tl.constexpr,  # K
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr  # BLOCK SIZE along the K dimension
):

    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    mask = (i_k * BK + tl.arange(0, BK)) < K
    o_i = tl.arange(0, BT)

    p_q = q + i_bh * T*K + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_gq = g + i_bh * T*K + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_A = A + (i_bh + (i_k * B * H)) * (tl.cdiv(T, BT) * BT * BT) + i_t * BT * BT + tl.arange(0, BT)

    for i in range(BT):
        b_q = tl.load(p_q, mask=mask, other=0) * scale
        b_gq = tl.load(p_gq, mask=mask, other=0).to(tl.float32)
        s = b_q[None, :] * b_k * safe_exp(b_gq[None, :] - b_g)
        score = tl.sum(s, axis=1)
        score = tl.where(o_i <= i, score, 0)
        tl.store(p_A, score.to(p_A.dtype.element_ty))
        p_q += K
        p_gq += K
        p_A += BT


@triton.jit
def bwd_inner_chunk(
    q,
    k,
    g,
    dA,
    dq,
    dk,
    T: tl.constexpr,  # T
    K: tl.constexpr,  # K
    # clamp_min, # minimum log value of the gate for numerical stability. default: -5
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    p_g = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    mask = (i_k * BK + tl.arange(0, BK)) < K
    o_i = tl.arange(0, BT)

    p_q = q + i_bh * T*K + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_dq = dq + (i_bh) * T*K + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_gq = g + i_bh * T*K + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_dA = dA + i_bh * (tl.cdiv(T, BT) * BT * BT) + i_t * BT * BT + tl.arange(0, BT)

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)

    for i in range(BT):
        b_q = tl.load(p_q, mask=mask, other=0)
        b_gq = tl.load(p_gq, mask=mask, other=0).to(tl.float32)
        score = safe_exp(b_gq[None, :] - b_g)
        score = tl.where(o_i[:, None] <= i, score, 0)
        b_dA = tl.load(p_dA)
        b_dA = tl.where(o_i <= i, b_dA, 0)
        b_dk += (b_dA[:, None] * score * b_q[None, :])
        b_dq = tl.sum(b_dA[:, None] * score * b_k, axis=0)
        tl.store(p_dq, b_dq, mask=mask)
        p_q += K
        p_dq += K
        p_gq += K
        p_dA += BT

    p_dk = tl.make_block_ptr(dk + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dk, b_dk.to(dk.dtype.element_ty), boundary_check=(0, 1))


class FusedChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state):
        ctx.g_dtype = g.dtype
        ctx.scale = scale
        B, H, T, K, V = *k.shape, v.shape[-1]
        BT = 16  # chunk_size
        BK, BV = min(K, 64), min(V, 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 2

        g_org = g
        # cumulative decay should be in float32, otherwise the err will be accumulated and amplified.
        g = chunk_local_cumsum(g_org, chunk_size=BT)
        o = q.new_empty(NK, B, H, T, V)
        q_g = torch.empty_like(q)
        k_g = torch.empty_like(k)

        grid = (NK, triton.cdiv(T, BT), B * H)
        prepare_qg_kg[grid](
            q,
            k,
            g,
            q_g,
            k_g,
            scale,
            T=T,
            K=K,
            BT=BT,
            BK=BK,
            num_warps=1
        )

        if output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float, requires_grad=False)
        else:
            final_state = None
        # the bug still exists even for Triton 2.2 on H100 GPUs
        # so we always enable initial checks
        CHECK = True
        if version.parse(triton.__version__) < version.parse('2.2.0'):
            import warnings
            warnings.warn(
                "Triton<2.2.0 detected for running this kernel, "
                "which is known to have some weird compiler issues (refer to https://github.com/openai/triton/issues/2852) "
                "that lead to significant precision loss. "
                "We've add some initial condition checks to resolve this, sadly at the sacrifice of the speed. "
                "For optimal performance, it is recommended to install Triton>=2.2.0 (if possible)."
            )
            CHECK = True

        grid = (NV, NK, B * H)
        fused_chunk_gla_fwd_kernel[grid](
            q_g, k_g, v, g, o, initial_state, final_state,
            B=B,
            H=H,
            T=T,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            CHECK=CHECK,
            num_warps=num_warps,
            num_stages=num_stages
        )

        o = o.sum(0)

        # intra-chunk
        chunk_size = 16
        num_chunk = T // chunk_size
        v2 = rearrange(v, 'b h (n c) d -> b h n c d', n=num_chunk)
        BK = min(K, 64)
        NK = triton.cdiv(K, BK)
        A = q.new_empty(NK, B, H, triton.cdiv(T, BT), BT, BT)
        grid = (NK, triton.cdiv(T, BT), B * H)
        fwd_inner_chunk[grid](
            q, k, g, A,
            scale,
            B=B,
            H=H,
            T=T,
            K=K,
            BT=BT,
            BK=BK,
            num_stages=3,
            num_warps=4
        )
        A = A.sum(0)
        o2 = A @ v2
        o2 = rearrange(o2, 'b h n c d -> b h (n c) d')
        # combine inner and inter
        o.add_(o2)
        ctx.save_for_backward(q, k, v, g_org, A, initial_state)
        ctx.CHECK = CHECK
        return o.to(v), final_state

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht=None):
        q, k, v, g_org, A, initial_state = ctx.saved_tensors
        B, H, T, K, V = *k.shape, v.shape[-1]
        scale = ctx.scale

        # recomputation
        # inter-chunk
        BT = 16  # chunk_size
        g = chunk_local_cumsum(g_org, chunk_size=BT)
        BK, BV = min(K, 64), min(V, 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        q_g = torch.empty_like(q)
        k_g = torch.empty_like(k)
        grid = (NK, triton.cdiv(T, BT), B * H)
        prepare_qg_kg[grid](
            q,
            k,
            g,
            q_g,
            k_g,
            scale,
            T=T,
            K=K,
            BT=BT,
            BK=BK,
            num_warps=1
        )

        BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 2
        dq = q.new_empty(NV, B, H, T, K)
        dk = q.new_empty(NV, B, H, T, K)
        dv = q.new_empty(NK, B, H, T, V)

        grid = (NV, NK, B * H)

        fused_chunk_gla_bwd_kernel[grid](
            q_g, k_g, v, g, do, dq, dk, dv, initial_state,
            scale,
            B=B,
            H=H,
            T=T,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            CHECK=ctx.CHECK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)

        # intra chunk
        NT = T // BT
        v2 = rearrange(v, 'b h (n c) d -> b h n c d', n=NT)
        do2 = rearrange(do, 'b h (n c) d -> b h n c d', n=NT)
        dA2 = (do2 @ v2.transpose(-2, -1)) * scale
        dv2 = A.transpose(-1, -2) @ do2
        dv2 = rearrange(dv2, 'b h n c d -> b h (n c) d', n=NT)

        BK = min(triton.next_power_of_2(K), 16)
        NK = triton.cdiv(K, BK)
        dk2 = torch.empty_like(k)
        dq2 = torch.empty_like(q)

        grid = (NK, NT, B * H)
        bwd_inner_chunk[grid](
            q, k, g,
            dA2,
            dq2,
            dk2,
            T=T,
            K=K,
            BT=BT,
            BK=BK,
            num_warps=1,
            num_stages=3
        )

        BK = min(triton.next_power_of_2(K), 32)
        NK = triton.cdiv(K, BK)
        dg = torch.empty_like(g, dtype=torch.float32)
        grid = (NK, triton.cdiv(T, BT), B * H)
        bwd_decay_global_cumsum[grid](
            dq2,
            dq,
            dk2,
            dk,
            q,
            k,
            g,
            dg,
            T=T,
            K=K,
            BT=BT,
            BK=BK,
            num_warps=1,
            num_stages=1
        )
        dg = rearrange(dg, 'b h (n c) d -> b h n c d', c=BT)

        def rev_cumsum_exclusive(x):
            cumsum_x = x.cumsum(-2)
            rev_cumsum_x = cumsum_x[..., -1, None, :] - cumsum_x
            return rev_cumsum_x

        rev_cumsum_dg = rev_cumsum_exclusive(dg[..., 0, :])
        dg.add_(rev_cumsum_dg.unsqueeze(-2))
        dv.add_(dv2)
        dg = rearrange(dg, 'b h n c d -> b h (n c) d')

        return dq.to(q), dk.to(k), dv.to(v), dg.to(ctx.g_dtype), None, None, None


def ceildiv(a, b):
    return -(a // -b)


def pad(x, chunk_size=16):
    T = x.shape[-2]
    padded_seq_len = ceildiv(T, chunk_size) * chunk_size
    if x.shape[-2] % chunk_size != 0:
        x = F.pad(x, (0, 0, 0, padded_seq_len - T))
    return x


def fused_chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: int = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale == -1:
        scale = q.shape[-1] ** -0.5
    if not head_first:
        q, k, v, g = map(lambda x: x.transpose(1, 2), (q, k, v, g))
    seq_len = q.shape[-2]
    q, k, v, g = map(lambda x: pad(x), [q, k, v, g])
    o, final_state = FusedChunkGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state)
    o = o[..., :seq_len, :].contiguous()
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state
