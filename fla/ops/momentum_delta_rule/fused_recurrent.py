# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch
import triton

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.delta_rule.fused_recurrent import (
    fused_recurrent_delta_rule_bwd_kernel,
    fused_recurrent_delta_rule_fwd_kernel,
)
from fla.ops.delta_rule.wy_fast import prepare_wy_repr_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

# Simplified fused_recurrent kernel: see note in chunk.py. Reuses
# fused_recurrent_delta_rule_{fwd,bwd}_kernel with u = v - k·h.
# Full momentum (log_alpha, log_mu, eta, p, dual state [S,M]) lives in
# MomentumDeltaNet/fla/ops/momentum_delta_rule/fused_recurrent.py
# (fused_recurrent_mode_rule_fwd_kernel with b_S/b_M). Current path is
# the mu-> -inf / alpha=1 reduction; see chunk.py header.


def fused_recurrent_mode_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    """Forward pass for MomentumDeltaNet fused_recurrent mode."""
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
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

    # Prepare WY representation (kept for parity with chunk path; kernel
    # recomputes u = v - k·h internally and overwrites this buffer)
    w, u, A = prepare_wy_repr_fwd(
        k=k,
        v=v,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=64,
    )
    u = u.to(q.dtype)

    grid = (NV * NK * N * H,)
    fused_recurrent_delta_rule_fwd_kernel[grid](
        q,
        k,
        v,
        u,
        beta,
        o,
        initial_state,
        final_state,
        cu_seqlens,
        scale=1.0 / math.sqrt(K),
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, u, final_state


def fused_recurrent_mode_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    dht: torch.Tensor,
    do: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
):
    """Backward pass for MomentumDeltaNet fused_recurrent mode."""
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
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
        db = q.new_empty(NV, NK, B, T, H, V)
    else:
        db = q.new_empty(NV, B, T, H)
    grid = (NV * NK * N * H,)

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
        cu_seqlens,
        scale=1.0 / math.sqrt(K),
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        NK=NK,
        IS_BETA_HEADWISE=beta_vector,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    db = db.sum((0, 1)) if beta_vector else db.sum(0)

    return dq, dk, dv, db, dh0


class FusedRecurrentModeRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        o, u, final_state = fused_recurrent_mode_rule_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, initial_state, u)
        ctx.scale = 1.0 / math.sqrt(q.shape[-1])
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        q, q_rstd, k, k_rstd, v, beta, initial_state, u = ctx.saved_tensors

        # Fused recurrent backward expects u = v - k·h (saved from forward),
        # not the raw v. See fused_recurrent_delta_rule_bwd_kernel which
        # operates on u. Passing raw v yields wrong dk/db/dq (P0 #2).
        dq, dk, dv, db, dh0 = fused_recurrent_mode_rule_bwd(
            q=q,
            k=k,
            v=u,
            beta=beta,
            dht=dht,
            do=do,
            initial_state=initial_state,
            cu_seqlens=ctx.cu_seqlens,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        # Forward inputs: q,k,v,beta,initial_state,output_final_state,
        # use_qk_l2norm_in_kernel,cu_seqlens (8). dh0 is 5th.
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), dh0, None, None, None


@torch.compiler.disable
def fused_recurrent_mode_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
):
    r"""MomentumDeltaNet fused_recurrent mode rule.

    Args:
        q (torch.Tensor): queries of shape `[B, T, H, K]`.
        k (torch.Tensor): keys of shape `[B, T, H, K]`.
        v (torch.Tensor): values of shape `[B, T, H, V]`.
        beta (torch.Tensor): betas of shape `[B, T, H]`.
        initial_state (Optional[torch.Tensor]): Initial state of shape `[N, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]): Whether to output the final state. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]): Whether to use qk l2norm in kernel. Default: `False`.
        cu_seqlens (torch.LongTensor): Cumulative sequence lengths for variable-length training.
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "FusedRecurrentModeRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape (batch size, num of head, seq len)."

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    o, final_state = FusedRecurrentModeRuleFunction.apply(
        q,
        k,
        v,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
    )
    return o, final_state
