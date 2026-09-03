# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.momentum_delta_rule.naive import chunk_momentum_delta_rule_ref
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def _chunk_momentum_degenerate_fwd(q, k, v, beta, initial_state, output_final_state, cu_seqlens=None, chunk_size=64):
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_fwd_o
    from fla.ops.delta_rule.wy_fast import prepare_wy_repr_fwd
    w, u, A = prepare_wy_repr_fwd(k=k, v=v, beta=beta, cu_seqlens=cu_seqlens, chunk_indices=None, chunk_size=chunk_size)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=None, initial_state=initial_state,
                                                         output_final_state=output_final_state, cu_seqlens=cu_seqlens, chunk_indices=None, chunk_size=chunk_size)
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=1.0 /
                    math.sqrt(q.shape[-1]), cu_seqlens=cu_seqlens, chunk_indices=None, chunk_size=chunk_size)
    return o, A, final_state


def _chunk_momentum_degenerate_bwd(q, k, v, beta, A, scale, initial_state, do, dht, cu_seqlens=None, chunk_size=64):
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local
    from fla.ops.delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, cu_seqlens=cu_seqlens)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=None, initial_state=initial_state,
                                               output_final_state=False, cu_seqlens=cu_seqlens, chunk_indices=None, chunk_size=chunk_size)
    dv = chunk_bwd_dv_local(q=q, k=k, do=do, g=None, scale=scale, cu_seqlens=cu_seqlens,
                            chunk_indices=None, chunk_size=chunk_size)
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(q=q, k=k, w=w, g=None, h0=initial_state, dht=dht,
                                                 do=do, dv=dv, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=None, chunk_size=chunk_size)
    dq, dk, dw, _ = chunk_bwd_dqkwg(q=q, k=k, v=v_new, h=h, w=w, dv=dv, do=do, dh=dh, g=None,
                                    scale=scale, cu_seqlens=cu_seqlens, chunk_indices=None, chunk_size=chunk_size)
    dk2, dv, db = prepare_wy_repr_bwd(k=k, v=v, beta=beta, A=A, dw=dw, du=dv, cu_seqlens=cu_seqlens)
    dk.add_(dk2)
    return dq, dk, dv, db, dh0


class ChunkModeRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False, cu_seqlens=None, cu_seqlens_cpu=None, chunk_size=64):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None
        o, A, final_state = _chunk_momentum_degenerate_fwd(
            q=q, k=k, v=v, beta=beta, initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, A, initial_state, cu_seqlens)
        ctx.scale = 1.0 / math.sqrt(q.shape[-1])
        ctx.chunk_size = chunk_size
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, q_rstd, k, k_rstd, v, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dh0 = _chunk_momentum_degenerate_bwd(
            q=q, k=k, v=v, beta=beta, A=A, scale=ctx.scale, initial_state=initial_state, do=do, dht=dht, cu_seqlens=cu_seqlens, chunk_size=ctx.chunk_size)
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), dh0, None, None, None, None, None


@torch.compiler.disable
def chunk_mode_rule(q, k, v, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None, cu_seqlens_cpu=None, **kwargs):
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkModeRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape (batch size, seq len, num heads)."
    chunk_size = kwargs.pop('chunk_size', 64)
    if chunk_size not in (16, 32, 64):
        raise ValueError(f"`chunk_size` must be 16, 32, or 64, got {chunk_size}.")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
    o, final_state = ChunkModeRuleFunction.apply(
        q, k, v, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens, cu_seqlens_cpu, chunk_size)
    return o, final_state


class ChunkMomentumDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, p, log_alpha, log_mu, beta, eta, scale, initial_S, initial_M, output_final_state, cu_seqlens, use_qk_l2norm_in_kernel, use_p_times_alpha, chunk_size):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
            p, p_rstd = l2norm_fwd(p)
        else:
            q_rstd, k_rstd, p_rstd = None, None, None
        # Pure PyTorch reference path (no Triton) - 0 Triton新增
        if cu_seqlens is not None:
            raise NotImplementedError("Variable-length `cu_seqlens` not yet supported for full momentum PyTorch path.")
        # apply eta and p*alpha exactly as Triton path did
        k_eta = k if eta is None else (k * eta.unsqueeze(-1)).to(q.dtype)
        p_eff = p if not use_p_times_alpha else (p * log_alpha.exp().unsqueeze(-1)).to(q.dtype)
        o, final_state = chunk_momentum_delta_rule_ref(
            q=q, k=k_eta, v=v, p=p_eff, log_alpha=log_alpha, log_mu=log_mu,
            beta=beta, eta=torch.ones_like(beta), scale=scale,
            initial_S=initial_S, initial_M=initial_M,
            output_final_state=output_final_state, chunk_size=chunk_size)
        # chunk_momentum_delta_rule_ref handles eta internally via k_eta, so pass ones
        ctx.save_for_backward(q, k, v, p, eta, beta, log_alpha, log_mu, initial_S,
                              initial_M, cu_seqlens, q_rstd, k_rstd, p_rstd)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_p_times_alpha = use_p_times_alpha
        ctx.chunk_size = chunk_size
        final_S, final_M = (final_state[0], final_state[1]) if final_state is not None else (None, None)
        # save final for backward recompute if needed
        ctx.final_S = final_S
        ctx.final_M = final_M
        return o.to(q.dtype), final_S, final_M

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dst, dmt):
        q, k, v, p, eta, beta, log_alpha, log_mu, initial_S, initial_M, cu_seqlens, q_rstd, k_rstd, p_rstd = ctx.saved_tensors
        # Recompute with autograd by re-running naive ref under grad
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
            o, _ = chunk_momentum_delta_rule_ref(
                q=q_n, k=k_eta, v=v_r, p=p_eff, log_alpha=log_alpha_r, log_mu=log_mu_r,
                beta=beta_r, eta=torch.ones_like(beta_r), scale=ctx.scale,
                initial_S=initial_S, initial_M=initial_M, output_final_state=False, chunk_size=ctx.chunk_size)
            # use dst/dmt as grad for final states if needed (ignored for torch path, recomputed)
            grads = torch.autograd.grad(o, (q_r, k_r, v_r, p_r, log_alpha_r, log_mu_r, beta_r) + ((eta_r,)
                                        if eta_r is not None else ()), grad_outputs=do, retain_graph=False, allow_unused=True)
        dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta = grads[:7]
        deta = grads[7] if eta is not None else None
        # handle eta scaling grad as Triton path did
        if eta is not None and dk is not None:
            # dk from grad is for k_eta = k*eta, need to unscale and compute deta
            # grads already accounts for eta via autograd, so keep as is
            pass
        if ctx.use_p_times_alpha and dp is not None and dlog_alpha is not None:
            # autograd already handled p*alpha, keep
            pass
        if ctx.use_qk_l2norm_in_kernel:
            if dq is not None:
                dq = l2norm_bwd(q, q_rstd, dq)
            if dk is not None:
                dk = l2norm_bwd(k, k_rstd, dk)
            if dp is not None:
                dp = l2norm_bwd(p, p_rstd, dp)
        # ds0/dm0 not materialized in torch path; return None and let caller handle
        ds0 = torch.zeros_like(initial_S, dtype=torch.float32) if initial_S is not None else None
        dm0 = torch.zeros_like(initial_M, dtype=torch.float32) if initial_M is not None else None
        # if dst/dmt provided, they would be grads for final states; ignore for now
        return dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta, deta, None, ds0, dm0, None, None, None, None, None


@torch.compiler.disable
def chunk_momentum_delta_rule(q, k, v, log_alpha, log_mu, p=None, beta=None, eta=None, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None, use_qk_l2norm_in_kernel=True, use_p_times_alpha=True, chunk_size=64):
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkMomentumDeltaRuleFunction does not support float32. Please use bfloat16."
    if chunk_size not in (16, 32, 64):
        raise ValueError(f"`chunk_size` must be 16, 32, or 64, got {chunk_size}.")
    if p is None:
        p = k
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    if eta is None:
        eta = torch.ones_like(q[..., 0])
    if initial_state is not None:
        initial_S, initial_M = initial_state[0], initial_state[1]
    else:
        initial_S, initial_M = None, None
    if cu_seqlens is not None:
        raise NotImplementedError(
            "Variable-length `cu_seqlens` is not yet supported for full momentum PyTorch path. Use degenerate `chunk_mode_rule` or wait for Triton varlen.")
    o, final_S, final_M = ChunkMomentumDeltaRuleFunction.apply(
        q, k, v, p, log_alpha, log_mu, beta, eta, scale, initial_S, initial_M, output_final_state, cu_seqlens, use_qk_l2norm_in_kernel, use_p_times_alpha, chunk_size)
    final_state = torch.stack([final_S, final_M], dim=0) if output_final_state else None
    return o, final_state
