# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.momentum_delta_rule.naive import recurrent_momentum_delta_rule_ref
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def fused_recurrent_mode_rule_fwd(q, k, v, beta, initial_state, output_final_state, cu_seqlens=None):
    import triton

    from fla.ops.delta_rule.fused_recurrent import fused_recurrent_delta_rule_fwd_kernel
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 1
    o = q.new_empty(NK, *v.shape)
    final_state = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    u = torch.empty_like(v)
    grid = (NV * NK * N * H,)
    fused_recurrent_delta_rule_fwd_kernel[grid](q, k, v, u, beta, o, initial_state, final_state, cu_seqlens, scale=1.0 / math.sqrt(
        K), T=T, B=B, H=H, K=K, V=V, BK=BK, BV=BV, IS_BETA_HEADWISE=beta.ndim == v.ndim, num_warps=num_warps, num_stages=num_stages)
    return o.squeeze(0), u, final_state


def fused_recurrent_mode_rule_bwd(q, k, v, beta, dht, do, initial_state, cu_seqlens=None):
    import triton

    from fla.ops.delta_rule.fused_recurrent import fused_recurrent_delta_rule_bwd_kernel
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
    db = q.new_empty(NV, NK, B, T, H, V) if beta_vector else q.new_empty(NV, B, T, H)
    grid = (NV * NK * N * H,)
    dh0 = torch.empty_like(
        initial_state, dtype=torch.float32) if initial_state is not None and initial_state.requires_grad else None
    fused_recurrent_delta_rule_bwd_kernel[grid](q, k, v, beta, initial_state, dh0, dht, do, dq, dk, dv, db, cu_seqlens, scale=1.0 / math.sqrt(
        K), T=T, B=B, H=H, K=K, V=V, BK=BK, BV=BV, NK=NK, IS_BETA_HEADWISE=beta_vector, num_warps=num_warps, num_stages=num_stages)
    return dq.sum(0), dk.sum(0), dv.sum(0), db.sum((0, 1)) if beta_vector else db.sum(0), dh0


class FusedRecurrentModeRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False, cu_seqlens=None):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None
        o, u, final_state = fused_recurrent_mode_rule_fwd(
            q=q, k=k, v=v, beta=beta, initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens)
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, initial_state, u)
        ctx.scale = 1.0 / math.sqrt(q.shape[-1])
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, q_rstd, k, k_rstd, v, beta, initial_state, u = ctx.saved_tensors
        dq, dk, dv, db, dh0 = fused_recurrent_mode_rule_bwd(
            q=q, k=k, v=u, beta=beta, dht=dht, do=do, initial_state=initial_state, cu_seqlens=ctx.cu_seqlens)
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), dh0, None, None, None


@torch.compiler.disable
def fused_recurrent_mode_rule(q, k, v, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs):
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "FusedRecurrentModeRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape (batch size, seq len, num heads)."
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
    o, final_state = FusedRecurrentModeRuleFunction.apply(
        q, k, v, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens)
    return o, final_state


class FusedRecurrentMomentumDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, p, log_alpha, log_mu, beta, eta, scale, initial_S, initial_M, output_final_state, cu_seqlens, use_qk_l2norm_in_kernel, use_p_times_alpha):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
            p, p_rstd = l2norm_fwd(p)
        else:
            q_rstd, k_rstd, p_rstd = None, None, None
        if cu_seqlens is not None:
            raise NotImplementedError("Variable-length `cu_seqlens` not yet supported for full momentum PyTorch path.")
        k_eta = k if eta is None else (k * eta.unsqueeze(-1)).to(q.dtype)
        p_eff = p if not use_p_times_alpha else (p * log_alpha.exp().unsqueeze(-1)).to(q.dtype)
        o, final_state = recurrent_momentum_delta_rule_ref(q=q, k=k_eta, v=v, p=p_eff, log_alpha=log_alpha, log_mu=log_mu, beta=beta, eta=torch.ones_like(
            beta), scale=scale, initial_S=initial_S, initial_M=initial_M, output_final_state=output_final_state)
        ctx.save_for_backward(q, k, v, p, eta, beta, log_alpha, log_mu, initial_S,
                              initial_M, cu_seqlens, q_rstd, k_rstd, p_rstd)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_p_times_alpha = use_p_times_alpha
        final_S, final_M = (final_state[0], final_state[1]) if final_state is not None else (None, None)
        return o.to(q.dtype), final_S, final_M

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dst, dmt):
        q, k, v, p, eta, beta, log_alpha, log_mu, initial_S, initial_M, cu_seqlens, q_rstd, k_rstd, p_rstd = ctx.saved_tensors
        with torch.enable_grad():
            q_r = q.detach().requires_grad_(True)
            k_r = k.detach().requires_grad_(True)
            v_r = v.detach().requires_grad_(True)
            p_r = p.detach().requires_grad_(True)
            log_alpha_r = log_alpha.detach().requires_grad_(True)
            log_mu_r = log_mu.detach().requires_grad_(True)
            beta_r = beta.detach().requires_grad_(True)
            eta_r = eta.detach().requires_grad_(True) if eta is not None else None
            if ctx.use_qk_l2norm_in_kernel:
                q_n, _ = l2norm_fwd(q_r)
                k_n, _ = l2norm_fwd(k_r)
                p_n, _ = l2norm_fwd(p_r)
            else:
                q_n, k_n, p_n = q_r, k_r, p_r
            k_eta = k_n if eta_r is None else (k_n * eta_r.unsqueeze(-1))
            p_eff = p_n if not ctx.use_p_times_alpha else (p_n * log_alpha_r.exp().unsqueeze(-1))
            o, _ = recurrent_momentum_delta_rule_ref(q=q_n, k=k_eta, v=v_r, p=p_eff, log_alpha=log_alpha_r, log_mu=log_mu_r, beta=beta_r, eta=torch.ones_like(
                beta_r), scale=ctx.scale, initial_S=initial_S, initial_M=initial_M, output_final_state=False)
            grads = torch.autograd.grad(o, (q_r, k_r, v_r, p_r, log_alpha_r, log_mu_r, beta_r) + ((eta_r,)
                                        if eta_r is not None else ()), grad_outputs=do, retain_graph=False, allow_unused=True)
        dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta = grads[:7]
        deta = grads[7] if eta is not None else None
        if ctx.use_qk_l2norm_in_kernel:
            if dq is not None:
                dq = l2norm_bwd(q, q_rstd, dq)
            if dk is not None:
                dk = l2norm_bwd(k, k_rstd, dk)
            if dp is not None:
                dp = l2norm_bwd(p, p_rstd, dp)
        ds0 = torch.zeros_like(initial_S, dtype=torch.float32) if initial_S is not None else None
        dm0 = torch.zeros_like(initial_M, dtype=torch.float32) if initial_M is not None else None
        return dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta, deta, None, ds0, dm0, None, None, None, None, None


@torch.compiler.disable
def fused_recurrent_momentum_delta_rule(q, k, v, log_alpha, log_mu, p=None, beta=None, eta=None, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None, use_qk_l2norm_in_kernel=True, use_p_times_alpha=True):
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    if eta is None:
        eta = torch.ones_like(q[..., 0])
    if p is None:
        p = k
    if initial_state is not None:
        initial_S, initial_M = initial_state[0], initial_state[1]
    else:
        initial_S, initial_M = None, None
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.")
        if initial_S is not None and initial_S.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_S.shape[0]}.")
    o, final_S, final_M = FusedRecurrentMomentumDeltaRuleFunction.apply(
        q, k, v, p, log_alpha, log_mu, beta, eta, scale, initial_S, initial_M, output_final_state, cu_seqlens, use_qk_l2norm_in_kernel, use_p_times_alpha)
    final_state = torch.stack([final_S, final_M], dim=0) if output_final_state else None
    return o, final_state
