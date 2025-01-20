# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.utils import contiguous


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.jit
def fused_recurrent_delta_rule_fwd_kernel(
    q,
    k,
    v,
    u,
    beta,
    o,
    h0,
    ht,
    offsets,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if HEAD_FIRST:
        p_q = q + i_nh * T*K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_nh * T*K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_nh * T*V + i_v * BV + tl.arange(0, BV)
        p_u = u + i_nh * T*V + i_v * BV + tl.arange(0, BV)
        if IS_BETA_HEADWISE:
            p_beta = beta + i_nh * T*V + i_v * BV + tl.arange(0, BV)
        else:
            p_beta = beta + i_nh * T
        p_o = o + (i_k * B*H + i_nh) * T*V + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
        p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
        p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
        p_u = u + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
        if IS_BETA_HEADWISE:
            p_beta = beta + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
        else:
            p_beta = beta + bos * H + i_h
        p_o = o + ((i_k * all + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV)

    mask_k = (i_k * BK + tl.arange(0, BK)) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_v_minus = tl.sum(b_h * b_k[None, :], axis=1)
        b_v -= b_v_minus
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        tl.store(p_u, b_v.to(p_v.dtype.element_ty), mask=mask_v)
        b_v *= b_beta
        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += K if HEAD_FIRST else H*K
        p_k += K if HEAD_FIRST else H*K
        p_o += V if HEAD_FIRST else H*V
        p_v += V if HEAD_FIRST else H*V
        p_u += V if HEAD_FIRST else H*V
        p_beta += (1 if HEAD_FIRST else H) * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.jit
def fused_recurrent_delta_rule_bwd_kernel(
    q,
    k,
    v,
    beta,
    h0,
    dh0,
    dht,
    do,
    dq,
    dk,
    dv,
    db,
    offsets,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar
    USE_INITIAL_STATE: tl.constexpr,  # whether to use dh0
    USE_FINAL_STATE_GRADIENT: tl.constexpr,  # whether to use dht
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V

    if HEAD_FIRST:
        p_q = q + i_nh * T*K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
        p_k = k + i_nh * T*K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
        p_v = v + i_nh * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
        p_do = do + i_nh * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
        p_dk = dk + (i_v * B*H + i_nh) * T*K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
        p_dv = dv + (i_k * B*H + i_nh) * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
        if IS_BETA_HEADWISE:
            p_beta = beta + i_nh * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
            p_dbeta = db + (i_v * NK*B*H + i_k * B*H + i_nh) * T*V + tl.arange(0, BV) + (T - 1) * V
        else:
            p_beta = beta + i_nh * T + T - 1
            p_dbeta = db + (i_v * B*H + i_nh) * T + T - 1
    else:
        p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
        p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
        p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
        p_do = do + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
        p_dk = dk + ((i_v * all + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
        p_dv = dv + ((i_k * all + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
        if IS_BETA_HEADWISE:
            p_beta = beta + (bos + T - 1) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
            p_dbeta = db + ((i_v * NK + i_k) * all + bos + T - 1) * H*V + i_h * V + tl.arange(0, BV)
        else:
            p_beta = beta + (bos + T - 1) * H + i_h
            p_dbeta = db + (i_v * all + bos + T - 1) * H + i_h

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_ht = dht + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_dh += tl.load(p_ht, mask=mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * (b_v * b_beta)[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)

        b_db = b_dv * b_v if IS_BETA_HEADWISE else tl.sum(b_dv * b_v)
        b_dv = b_dv * b_beta

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)
        if IS_BETA_HEADWISE:
            tl.store(p_dbeta, b_db.to(p_dbeta.dtype.element_ty), mask=mask_v)
        else:
            tl.store(p_dbeta, b_db.to(p_dbeta.dtype.element_ty))

        b_dh -= b_k[:, None] * b_dv[None, :]

        p_q -= K if HEAD_FIRST else H*K
        p_k -= K if HEAD_FIRST else H*K
        p_v -= V if HEAD_FIRST else H*V
        p_do -= V if HEAD_FIRST else H*V
        p_dk -= K if HEAD_FIRST else H*K
        p_dv -= V if HEAD_FIRST else H*V
        p_dbeta -= (1 if HEAD_FIRST else H) * (V if IS_BETA_HEADWISE else 1)
        p_beta -= (1 if HEAD_FIRST else H) * (V if IS_BETA_HEADWISE else 1)

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_k[:, None] & mask_v[None, :])

    tl.debug_barrier()

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if HEAD_FIRST:
        p_q = q + i_nh * T*K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_nh * T*K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_nh * T*V + i_v * BV + tl.arange(0, BV)
        if IS_BETA_HEADWISE:
            p_beta = beta + i_nh * T*V + i_v * BV + tl.arange(0, BV)
        else:
            p_beta = beta + i_nh * T
        p_do = do + i_nh * T*V + i_v * BV + tl.arange(0, BV)
        p_dq = dq + (i_v * B*H + i_nh) * T*K + i_k * BK + tl.arange(0, BK)
        p_dk = dk + (i_v * B*H + i_nh) * T*K + i_k * BK + tl.arange(0, BK)
        p_dv = dv + (i_k * B*H + i_nh) * T*V + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
        p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
        p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
        if IS_BETA_HEADWISE:
            p_beta = beta + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
        else:
            p_beta = beta + bos * H + i_h
        p_do = do + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
        p_dq = dq + ((i_v * all + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK)
        p_dk = dk + ((i_v * all + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK)
        p_dv = dv + ((i_k * all + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV)

    if USE_INITIAL_STATE:
        mask_h = mask_k[:, None] & mask_v[None, :]
        p_h0 = h0 + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_dk = tl.load(p_dk, mask=mask_k, other=0).to(tl.float32)
        b_dv = tl.load(p_dv, mask=mask_v, other=0).to(tl.float32)
        b_dk -= tl.sum(b_dv[None, :] * b_h, axis=1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)

        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta

        b_h += b_k[:, None] * b_v[None, :]
        b_dq = b_h * b_do[None, :]
        d_q = tl.sum(b_dq, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_k)

        p_k += K if HEAD_FIRST else H*K
        p_v += V if HEAD_FIRST else H*V
        p_do += V if HEAD_FIRST else H*V
        p_dq += K if HEAD_FIRST else H*K
        p_dk += K if HEAD_FIRST else H*K
        p_dv += V if HEAD_FIRST else H*V
        p_beta += (1 if HEAD_FIRST else H) * (V if IS_BETA_HEADWISE else 1)


def fused_recurrent_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if offsets is None else len(offsets) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if output_final_state:
        final_state = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        final_state = None

    grid = (NV, NK, N * H)
    u = torch.empty_like(v)
    fused_recurrent_delta_rule_fwd_kernel[grid](
        q,
        k,
        v,
        u,
        beta,
        o,
        initial_state,
        final_state,
        offsets,
        scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        HEAD_FIRST=head_first,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, u, final_state


def fused_recurrent_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    dht: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if offsets is None else len(offsets) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 2

    beta_vector = beta.ndim == v.ndim

    dq = q.new_empty(NV, *q.shape)
    dk = q.new_empty(NV, *k.shape)
    dv = q.new_empty(NK, *v.shape)
    if beta_vector:
        db = q.new_empty(NV, NK, B, H, T, V) if head_first else q.new_empty(NV, NK, B, T, H, V)
    else:
        db = q.new_empty(NV, B, H, T) if head_first else q.new_empty(NV, B, T, H)
    grid = (NV, NK, N * H)

    if initial_state is not None and initial_state.requires_grad:
        dh0 = torch.empty_like(initial_state, dtype=torch.float32)
    else:
        dh0 = None

    fused_recurrent_delta_rule_bwd_kernel[grid](
        q,
        k,
        v,
        beta,
        initial_state,
        dh0,
        dht,
        do,
        dq,
        dk,
        dv,
        db,
        offsets,
        scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        NK=NK,
        IS_BETA_HEADWISE=beta_vector,
        HEAD_FIRST=head_first,
        num_warps=num_warps,
        num_stages=num_stages
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    db = db.sum((0, 1)) if beta_vector else db.sum(0)

    return dq, dk, dv, db, dh0


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True,
        use_qk_l2norm_in_kernel: bool = False
    ):
        q_orig = q
        k_orig = k

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        o, u, final_state = fused_recurrent_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            head_first=head_first
        )

        ctx.save_for_backward(q_orig, k_orig, u, beta, initial_state)
        ctx.scale = scale
        ctx.offsets = offsets
        ctx.head_first = head_first
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        q, k, v, beta, initial_state = ctx.saved_tensors
        if ctx.use_qk_l2norm_in_kernel:
            q, q_orig = l2norm_fwd(q), q
            k, k_orig = l2norm_fwd(k), k
        dq, dk, dv, db, dh0 = fused_recurrent_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            dht=dht,
            do=do,
            scale=ctx.scale,
            initial_state=initial_state,
            offsets=ctx.offsets,
            head_first=ctx.head_first
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq, dk = l2norm_bwd(q_orig, dq), l2norm_bwd(k_orig, dk)
        return dq.to(q), dk.to(k), dv.to(v), db.to(beta), None, dh0, None, None, None, None


def fused_recurrent_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    use_qk_l2norm_in_kernel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        beta (torch.Tensor):
             betas of shape `[B, H, T]` if `head_first=True` else `(B, T, H)`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.delta_rule import fused_recurrent_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> beta = torch.rand(B, T, H, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = fused_recurrent_delta_rule(q, k, v, beta,
                                               initial_state=h0,
                                               output_final_state=True,
                                               head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_recurrent_delta_rule(q, k, v, beta,
                                                       initial_state=h0,
                                                       output_final_state=True,
                                                       cu_seqlens=cu_seqlens,
                                                       head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first,
        use_qk_l2norm_in_kernel
    )
    return o, final_state
