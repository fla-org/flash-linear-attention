# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang, Yuqi Pan

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=['BT', 'BK'],
)
@triton.jit(do_not_specialize=['T'])
def fused_chunk_ttt_linear_fwd_kernel(
    q,
    k,
    v,
    w,
    b,
    o,
    initial_state,
    final_state,
    eta,
    scale,
    eps,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    # indices
    i_bh = tl.program_id(0)

    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # make block pointers
    p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (0, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (0, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (0, 0), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * T*V, (T, V), (V, 1), (0, 0), (BT, BV), (1, 0))
    p_eta = tl.make_block_ptr(eta + i_bh * T, (T, 1), (1, 1), (0, 0), (BT, 1), (1, 0))

    v_i = tl.arange(0, BV)
    i_h = i_bh % H
    b_w = tl.load(w + i_h * V + v_i, mask=v_i < V, other=0.).to(tl.float32)
    b_b = tl.load(b + i_h * V + v_i, mask=v_i < V, other=0.).to(tl.float32)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * K * V, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        # [BT, 1]
        b_eta = tl.load(p_eta, boundary_check=(0, 1), padding_option="zero")
        b_q = (b_q * scale).to(b_k.dtype)

        # [BT, BV]
        b_kh = tl.dot(tl.trans(b_k), b_h.to(b_q.dtype), allow_tf32=False).to(tl.float32)
        b_kh = tl.where((v_i < V)[None, :], b_kh, 0.)

        mean = tl.sum(b_kh, axis=1, keep_dims=True) / V
        xbar = tl.where((v_i < V)[None, :], b_kh - mean, 0.)
        var = tl.sum(xbar * xbar, axis=1, keep_dims=True) / V
        rstd = 1 / tl.sqrt(var.to(tl.float32) + eps)
        b_ln = (b_kh - mean) * rstd

        b_v = b_ln * b_w[None, :].to(b_q.dtype) + b_b[None, :].to(b_q.dtype) - b_v + tl.trans(b_k)
        b_v = tl.where((v_i < V)[None, :], b_v * b_w[None, :].to(b_q.dtype), 0.)
        # v_new here
        b_v = rstd * (V * b_v - tl.sum(b_v, axis=1, keep_dims=True) - b_ln * tl.sum(b_v * b_ln, axis=1, keep_dims=True)) / V

        # [BT, BT]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)

        b_o = - 2 * tl.dot(b_eta * b_s.to(b_q.dtype), b_v.to(b_q.dtype), allow_tf32=False)
        b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
        if i == tl.cdiv(T, BT) - 1:
            b_eta_last = tl.load(eta + i_bh * T + T - 1)
        else:
            b_eta_last = tl.load(eta + i_bh * T + i * BT + BT - 1)
        b_h = b_h - 2 * tl.dot(b_eta_last * b_k, b_v.to(b_k.dtype), allow_tf32=False)
        b_h = tl.where((v_i < V)[None, :], b_h, 0.)

        b_o = tl.where((v_i < V)[None, :], b_o, 0.)
        mean = tl.sum(b_o, axis=1, keep_dims=True) / V
        xbar = tl.where((v_i < V)[None, :], b_o - mean, 0.)
        var = (tl.sum(xbar * xbar, axis=1, keep_dims=True) / V).to(tl.float32)
        rstd = (1 / tl.sqrt(var + eps)).to(tl.float32)
        b_o_hat = (b_o - mean) * rstd
        b_o += b_o_hat * b_w[None, :].to(b_o.dtype) + b_b[None, :].to(b_o.dtype)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_eta = tl.advance(p_eta, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(final_state + i_bh * K * V, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


class FusedChunkTTTLinearFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, ln_w, ln_b, chunk_size, eta, scale, eps, initial_state, output_final_state):
        B, H, T, K, V = *q.shape, v.shape[-1]
        BK, BV = triton.next_power_of_2(K), triton.next_power_of_2(V)
        o = q.new_empty(B, H, T, V)
        if output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float32, requires_grad=False)
        else:
            final_state = None
        grid = (B * H,)
        fused_chunk_ttt_linear_fwd_kernel[grid](
            q, k, v, ln_w, ln_b, o, initial_state, final_state,
            eta, scale, eps,
            H=H,
            T=T,
            K=K,
            V=V,
            BT=chunk_size,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state
        )
        ctx.save_for_backward(q, k, v, eta, ln_w, ln_b, initial_state)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.eps = eps
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        raise NotImplementedError("TTT-Linear backward is not yet implemented.")


def fused_chunk_ttt_linear(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    eta: torch.Tensor,
    scale: float = None,
    eps: float = 1e-6,
    chunk_size: int = 16,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
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
            layer norm weight of shape `(V)`
        ln_b (torch.Tensor):
            layer norm bias of shape `(V)`
        eta (float):
            Learning rate for hidden state. Default: `1 / 2`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        chunk_size (int):
            chunk size. Default: `16`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]`
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`
    """
    assert chunk_size == 16, "Currently chunk_size is chosen to be only 16."
    assert q.dtype == k.dtype == v.dtype
    assert k.shape[-1] == v.shape[-1], "K must equal to V."
    if isinstance(eta, float):
        eta = torch.full_like(q[:, :, :, :1], eta)

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "Scale must be positive."
    if not head_first:
        q, k, v, eta = map(lambda x: x.transpose(1, 2), (q, k, v, eta))
    o, final_state = FusedChunkTTTLinearFunction.apply(
        q,
        k,
        v,
        ln_w,
        ln_b,
        chunk_size,
        eta,
        scale,
        eps,
        initial_state,
        output_final_state,
    )
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state
