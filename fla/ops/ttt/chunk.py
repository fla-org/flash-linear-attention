# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.utils import (autocast_custom_bwd, autocast_custom_fwd, contiguous,
                       tensor_cache)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit
def chunk_ttt_linear_fwd_kernel_h(
    k,
    v,
    v_new,
    eta,
    w,
    b,
    eps,
    h,
    h0,
    ht,
    offsets,
    chunk_offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    offs = tl.arange(0, BV)
    b_w = tl.load(w + i_h * V + offs, mask=offs < V, other=0.).to(tl.float32)
    b_b = tl.load(b + i_h * V + offs, mask=offs < V, other=0.).to(tl.float32)

    for i_t in range(NT):
        if HEAD_FIRST:
            p_h = tl.make_block_ptr(h + (i_nh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_h = tl.make_block_ptr(h + ((boh + i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_hc = tl.zeros([BK, BV], dtype=tl.float32)
        # since we need to make all DK in the SRAM. we face serve SRAM memory burden. By subchunking we allievate such burden
        for i_c in range(tl.cdiv(min(BT, T - i_t * BT), BC)):
            is_last_c = (i_t == NT-1 and i_c == tl.cdiv(min(BT, T - i_t * BT), BC)-1)
            if HEAD_FIRST:
                p_k = tl.make_block_ptr(k+i_nh*T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_v = tl.make_block_ptr(v+i_nh*T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_v_new = tl.make_block_ptr(v_new+i_nh*T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_eta_last = eta+i_nh*T + T - 1 if is_last_c else eta+i_nh*T + i_t*BT + i_c*BC + BC - 1
            else:
                p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_v_new = tl.make_block_ptr(v_new+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT+i_c*BC, i_v * BV), (BC, BV), (1, 0))
                p_eta_last = eta+bos*H+i_h + (T-1)*H if is_last_c else eta+bos*H+i_h+(i_t*BT+i_c*BC+BC-1)*H
            b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
            b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
            # b_eta = tl.load(p_eta, boundary_check=(0, ))

            b_kh = tl.dot(tl.trans(b_k), b_h.to(b_k.dtype), allow_tf32=False).to(tl.float32)
            b_kh = tl.where((offs < V)[None, :], b_kh, 0.)
            mean = tl.sum(b_kh, axis=1, keep_dims=True) / V
            xbar = tl.where((offs < V)[None, :], b_kh - mean, 0.)
            var = tl.sum(xbar * xbar, axis=1, keep_dims=True) / V
            rstd = 1 / tl.sqrt(var.to(tl.float32) + eps)
            b_kh_hat = (b_kh - mean) * rstd

            b_v = b_kh_hat * b_w[None, :].to(b_k.dtype) + b_b[None, :].to(b_k.dtype) - b_v + tl.trans(b_k)
            b_v = tl.where((offs < V)[None, :], b_v * b_w[None, :].to(b_k.dtype), 0.)
            b_v2 = rstd * (V * b_v - tl.sum(b_v, axis=1, keep_dims=True) - b_kh_hat *
                           tl.sum(b_v * b_kh_hat, axis=1, keep_dims=True)) / V
            tl.store(p_v_new, b_v2.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
            b_eta_last = tl.load(p_eta_last)
            b_hc = b_hc - 2 * tl.dot(b_eta_last * b_k, b_v2.to(b_k.dtype), allow_tf32=False)
        b_h += b_hc

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=["BT"],
)
@triton.jit
def chunk_ttt_linear_fwd_kernel_o(
    q,
    k,
    v,
    eta,
    w,
    b,
    eps,
    h,
    o,
    offsets,
    indices,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    offs = tl.arange(0, BV)
    b_w = tl.load(w + i_h * V + offs, mask=offs < V, other=0.).to(tl.float32)
    b_b = tl.load(b + i_h * V + offs, mask=offs < V, other=0.).to(tl.float32)

    # offset calculation
    q += (i_bh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    k += (i_bh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    v += (i_bh * T * V) if HEAD_FIRST else ((bos * H + i_h) * V)
    eta += (i_bh * T) if HEAD_FIRST else (bos * H + i_h)
    o += (i_bh * T * V) if HEAD_FIRST else ((bos * H + i_h) * V)
    h += ((i_bh * NT + i_t) * K * V) if HEAD_FIRST else ((i_tg * H + i_h) * K * V)
    stride_qk = K if HEAD_FIRST else H*K
    stride_vo = V if HEAD_FIRST else H*V
    stride_eta = 1 if HEAD_FIRST else H

    p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (K, T), (1, stride_qk), (0, i_t * BT), (BK, BT), (0, 1))
    p_eta = tl.make_block_ptr(eta, (T,), (stride_eta,), (i_t * BT,), (BT,), (0,))
    p_h = tl.make_block_ptr(h, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    # [BK, BT]
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    # [BT, 1]
    b_eta = tl.load(p_eta, boundary_check=(0,), padding_option="zero")
    # [BK, BV]
    b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
    # [BT, BK] @ [BK, BV] -> [BT, BV]
    b_o = tl.dot(b_q, b_h, allow_tf32=False)
    # [BT, BK] @ [BK, BT] -> [BT, BT]
    b_A = tl.dot(b_q, b_k, allow_tf32=False)

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_o = (b_o - 2 * tl.dot(b_eta[:, None] * b_A.to(b_v.dtype), b_v, allow_tf32=False)) * scale

    b_o = tl.where((offs < V)[None, :], b_o, 0.)
    mean = tl.sum(b_o, axis=1, keep_dims=True) / V
    xbar = tl.where((offs < V)[None, :], b_o - mean, 0.)
    var = (tl.sum(xbar * xbar, axis=1, keep_dims=True) / V).to(tl.float32)
    rstd = (1 / tl.sqrt(var + eps)).to(tl.float32)
    b_o_hat = (b_o - mean) * rstd
    b_o += b_o_hat * b_w[None, :].to(b_o.dtype) + b_b[None, :].to(b_o.dtype)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@tensor_cache
def prepare_varlen_inputs_fwd_h(
    offsets: torch.Tensor,
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.cat([offsets.new_tensor([0]), triton.cdiv(offsets[1:] - offsets[:-1], chunk_size)]).cumsum(-1)


def chunk_ttt_linear_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    eps: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        chunk_offsets = prepare_varlen_inputs_fwd_h(offsets, BT)
        NT = chunk_offsets[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 128, "current kernel does not support head dimension larger than 128."
    # H100 can have larger block size
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = triton.next_power_of_2(V)
        BC = 64
    # A100
    elif torch.cuda.get_device_capability() == (8, 0):
        BV = triton.next_power_of_2(V)
        BC = 64
    else:
        BV = triton.next_power_of_2(V)
        BC = 64 if K <= 128 else 32
    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    assert NV == 1, 'NV > 1 is not supported by TTT update rule.'

    if head_first:
        h = k.new_empty(B, H, NT, K, V)
    else:
        h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(v)
    grid = (NK, NV, N * H)

    chunk_ttt_linear_fwd_kernel_h[grid](
        k=k,
        v=v,
        v_new=v_new,
        eta=eta,
        w=w,
        b=b,
        eps=eps,
        h=h,
        h0=initial_state,
        ht=final_state,
        offsets=offsets,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        NT=NT,
        HEAD_FIRST=head_first
    )
    return h, v_new, final_state


def chunk_ttt_linear_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eta: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    h: torch.Tensor,
    scale: Optional[float] = None,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> torch.Tensor:
    if head_first:
        B, H, T, K, V = *q.shape, v.shape[-1]
    else:
        B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = k.shape[-1] ** -0.5
    BT = chunk_size
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
        NT = len(indices)
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    assert NV == 1, 'NV > 1 is not supported by TTT update rule.'

    o = torch.empty_like(v)

    grid = (NV, NT, B * H)
    chunk_ttt_linear_fwd_kernel_o[grid](
        q,
        k,
        v,
        eta,
        w,
        b,
        eps,
        h,
        o,
        offsets,
        indices,
        scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first
    )
    return o


def chunk_ttt_linear_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    eps: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    BT: int = 16
):
    h, v_new, final_state = chunk_ttt_linear_fwd_h(
        k=k,
        v=v,
        w=w,
        b=b,
        eta=eta,
        eps=eps,
        initial_state=initial_state,
        output_final_state=output_final_state,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT
    )
    o = chunk_ttt_linear_fwd_o(
        q=q,
        k=k,
        v=v_new,
        eta=eta,
        w=w,
        b=b,
        eps=eps,
        h=h,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    return o, final_state


@tensor_cache
def prepare_varlen_inputs(
    offsets: torch.LongTensor,
    chunk_size: int
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
    indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
    return indices


class ChunkTTTLinearFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, ln_w, ln_b, BT, eta, scale, eps, initial_state, output_final_state, offsets, head_first):
        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = prepare_varlen_inputs(offsets, BT) if offsets is not None else None
        o, final_state = chunk_ttt_linear_fwd(
            q=q,
            k=k,
            v=v,
            w=ln_w,
            b=ln_b,
            eta=eta,
            scale=scale,
            eps=eps,
            BT=BT,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
        )
        ctx.save_for_backward(q, k, v, eta, ln_w, ln_b, initial_state)
        ctx.BT = BT
        ctx.scale = scale
        ctx.eps = eps
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        raise NotImplementedError("TTT-Linear backward is not yet implemented.")


def chunk_ttt_linear(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    eta: torch.Tensor,
    scale: float = None,
    eps: float = 1e-6,
    BT: int = 16,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `(B, H, T, K)`
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        ln_w (torch.Tensor):
            values of shape `(H, V)`
        ln_b (torch.Tensor):
            values of shape `(H, V)`
        eta (float):
            Learning rate for hidden state. Default: `1 / 2`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        BT (int):
            chunk size. Default: `16`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]`
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`
    """
    assert BT == 16, "Currently BT is chosen to be only 16."
    assert q.dtype == k.dtype == v.dtype
    assert k.shape[-1] == v.shape[-1], "DK must equal to DV."
    if isinstance(eta, float):
        eta = torch.full_like(q[:, :, :, :1], eta)
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
        assert scale > 0, "Scale must be positive."
    o, final_state = ChunkTTTLinearFunction.apply(
        q,
        k,
        v,
        ln_w,
        ln_b,
        BT,
        eta,
        scale,
        eps,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first,
    )
    return o, final_state
