# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.gka.chunk import check_gka_inputs
from fla.ops.simple_gla import fused_recurrent_simple_gla
from fla.ops.utils.op import exp
from fla.utils import input_guard


@triton.heuristics({
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gka_solve_fwd_kernel(
    q,
    k,
    gk,
    x,
    h0,
    ht,
    cu_seqlens,
    ridge_ratio,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    NUM_ITER: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_kk = m_k[:, None] & m_k[None, :]
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_x = x + (bos * H + i_h) * K + o_k
    if USE_GK:
        p_gk = gk + bos * H + i_h

    # the running state, which is H_0 before the loop and H_t after token t
    # [BK, BK]
    b_h = tl.zeros([BK, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(h0 + i_nh * K*K + o_k[:, None] * K + o_k[None, :], mask=m_kk, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        if USE_GK:
            b_h = b_h * exp(tl.load(p_gk).to(tl.float32))
        b_h += b_k[:, None] * b_k[None, :]

        # Chebyshev iteration on (H_t + lamb_t I) x = q_t; the matrix's spectrum lies in [lamb_t, lamb_t + fro_t]
        b_fro = tl.sqrt(tl.sum(b_h * b_h))
        b_lamb = ridge_ratio * b_fro
        b_step = 2 / (2 * b_lamb + b_fro)
        b_rho = b_fro / (2 * b_lamb + b_fro) / 2.
        b_rho = b_rho * b_rho
        b_x_prev = tl.zeros([BK], dtype=tl.float32)
        b_x = b_step * b_q
        b_w = 2.
        for _ in range(NUM_ITER):
            b_w = 1. / (1. - b_rho * b_w)
            b_r = tl.sum(b_h * b_x[:, None], axis=0) + b_lamb * b_x - b_q
            b_d = b_step * b_w * b_r + (b_w - 1) * b_x_prev
            b_x_prev = b_x
            b_x = b_w * b_x - b_d
        tl.store(p_x, b_x.to(p_x.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_k)

        p_q += H*K
        p_k += H*K
        p_x += H*K
        if USE_GK:
            p_gk += H

    if STORE_FINAL_STATE:
        tl.store(ht + i_nh * K*K + o_k[:, None] * K + o_k[None, :], b_h.to(ht.dtype.element_ty), mask=m_kk)


class FusedRecurrentGKAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None,
        gk: torch.Tensor | None,
        alpha: torch.Tensor | None,
        scale: float,
        ridge_ratio: float,
        num_iter: int,
        h_kk0: torch.Tensor | None,
        h_kv0: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None,
    ):
        B, T, H, K = q.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        x = torch.empty_like(q)
        h_kk = q.new_empty(N, H, K, K, dtype=torch.float) if output_final_state else None
        BK = max(triton.next_power_of_2(K), 16)
        # one warp is 2-4x faster for BK <= 64; at BK = 128 fewer than 4 warps spill the [BK, BK] state
        fused_recurrent_gka_solve_fwd_kernel[(N * H,)](
            q=q,
            k=k,
            gk=gk,
            x=x,
            h0=h_kk0,
            ht=h_kk,
            cu_seqlens=cu_seqlens,
            ridge_ratio=ridge_ratio,
            T=T,
            H=H,
            K=K,
            BK=BK,
            NUM_ITER=num_iter,
            num_warps=1 if BK <= 64 else 4,
        )
        q_mix = x if alpha is None else (q + alpha[..., None] * (x - q)).to(q.dtype)
        o, h_kv = fused_recurrent_simple_gla(
            q=q_mix,
            k=k,
            v=v,
            g=g,
            scale=scale,
            initial_state=h_kv0,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        return o, h_kk, h_kv

    @staticmethod
    def backward(ctx, do, dht_kk, dht_kv):
        raise NotImplementedError("`fused_recurrent_gka` is inference-only; use `chunk_gka` for training.")


@torch.compiler.disable
def fused_recurrent_gka(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    scale: float | None = None,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
):
    r"""
    Token-by-token Gated KalmaNet for inference and decoding; it matches `chunk_gka` up to floating-point rounding,
    with the same states `H_t` and `U_t`.
    It has no backward pass.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor, Optional):
            Log-space decay of `U_t`, of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor, Optional):
            Log-space decay of `H_t`, of shape `[B, T, H]`.
            Pass the same tensor as `g` to decay both states together. Default: `None`.
        alpha (torch.Tensor, Optional):
            Mixing weights of shape `[B, T, H]`; the readout query is `q + alpha * (x - q)`.
            If `None`, the readout query is the ridge solution `x`, as with `alpha = 1`. Default: `None`.
        scale (float, Optional):
            Scale applied to the readout. Default: `1 / sqrt(K)`.
        ridge_ratio (float, Optional):
            Sets the ridge strength `lamb_t = ridge_ratio * ||H_t||_F`. Default: `0.02`.
        num_iter (int, Optional):
            Number of Chebyshev iterations, at least 1. Default: `30`.
        initial_state (tuple[torch.Tensor, torch.Tensor], Optional):
            Initial states `(h_kk, h_kv)` of shapes `[N, H, K, K]` and `[N, H, K, V]` in `float32`, for `N` input
            sequences. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final states. Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length inputs,
            consistent with the FlashAttention API. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (tuple[torch.Tensor, torch.Tensor] | None):
            Final states `(h_kk, h_kv)` of shapes `[N, H, K, K]` and `[N, H, K, V]` in `float32` if
            `output_final_state=True`, else `None`.
    """
    h_kk0, h_kv0, scale = check_gka_inputs(q, k, v, g, gk, alpha, scale, num_iter, initial_state, cu_seqlens)
    o, h_kk, h_kv = FusedRecurrentGKAFunction.apply(
        q,
        k,
        v,
        g,
        gk,
        alpha,
        scale,
        ridge_ratio,
        num_iter,
        h_kk0,
        h_kv0,
        output_final_state,
        cu_seqlens,
    )
    final_state = (h_kk, h_kv) if output_final_state else None
    return o, final_state
