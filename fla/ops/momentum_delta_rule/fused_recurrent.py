# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.momentum_delta_rule.naive import recurrent_momentum_delta_rule_ref
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


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
            raise NotImplementedError("Variable-length `cu_seqlens` is not yet supported for full momentum PyTorch path.")
        k_eta = k if eta is None else (k * eta.unsqueeze(-1)).to(q.dtype)
        p_eff = p if not use_p_times_alpha else (p * log_alpha.exp().unsqueeze(-1)).to(q.dtype)
        o, final_state = recurrent_momentum_delta_rule_ref(
            q=q, k=k_eta, v=v, p=p_eff, log_alpha=log_alpha, log_mu=log_mu, beta=beta,
            eta=torch.ones_like(beta), scale=scale, initial_S=initial_S, initial_M=initial_M,
            output_final_state=output_final_state,
        )
        ctx.save_for_backward(
            q, k, v, p, eta, beta, log_alpha, log_mu, initial_S, initial_M, cu_seqlens, q_rstd, k_rstd, p_rstd,
        )
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
            q_r, k_r, v_r, p_r = [x.detach().requires_grad_(True) for x in (q, k, v, p)]
            log_alpha_r = log_alpha.detach().requires_grad_(True)
            log_mu_r = log_mu.detach().requires_grad_(True)
            beta_r = beta.detach().requires_grad_(True)
            eta_r = eta.detach().requires_grad_(True) if eta is not None else None
            k_eta = k_r if eta_r is None else k_r * eta_r.unsqueeze(-1)
            p_eff = p_r if not ctx.use_p_times_alpha else p_r * log_alpha_r.exp().unsqueeze(-1)
            initial_S_r = initial_S.detach().requires_grad_(True) if initial_S is not None else None
            initial_M_r = initial_M.detach().requires_grad_(True) if initial_M is not None else None
            o, final_state = recurrent_momentum_delta_rule_ref(
                q=q_r, k=k_eta, v=v_r, p=p_eff, log_alpha=log_alpha_r, log_mu=log_mu_r,
                beta=beta_r, eta=torch.ones_like(beta_r), scale=ctx.scale, initial_S=initial_S_r,
                initial_M=initial_M_r, output_final_state=True,
            )
            outputs = (o, final_state[0], final_state[1])
            grads_out = (do, dst, dmt)
            loss = sum(
                (x * dx for x, dx in zip(outputs, grads_out) if dx is not None),
                torch.zeros((), device=o.device),
            )
            inputs = (q_r, k_r, v_r, p_r, log_alpha_r, log_mu_r, beta_r)
            if eta_r is not None:
                inputs += (eta_r,)
            if initial_S_r is not None:
                inputs += (initial_S_r, initial_M_r)
            grads = torch.autograd.grad(loss, inputs, allow_unused=True)
        dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta = grads[:7]
        deta = grads[7] if eta is not None else None
        state_offset = 8 if eta is not None else 7
        ds0, dm0 = grads[state_offset:state_offset + 2] if initial_S is not None else (None, None)
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
            dp = l2norm_bwd(p, p_rstd, dp)
        return dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta, deta, None, ds0, dm0, None, None, None, None


@torch.compiler.disable
def fused_recurrent_momentum_delta_rule(q, k, v, log_alpha, log_mu, p=None, beta=None, eta=None, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None, use_qk_l2norm_in_kernel=True, use_p_times_alpha=True):
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    beta = torch.ones_like(q[..., 0]) if beta is None else beta
    eta = torch.ones_like(q[..., 0]) if eta is None else eta
    p = k if p is None else p
    initial_S, initial_M = (initial_state[0], initial_state[1]) if initial_state is not None else (None, None)
    if cu_seqlens is not None:
        raise NotImplementedError("Variable-length `cu_seqlens` is not yet supported for full momentum PyTorch path.")
    o, final_S, final_M = FusedRecurrentMomentumDeltaRuleFunction.apply(
        q, k, v, p, log_alpha, log_mu, beta, eta, scale, initial_S, initial_M,
        output_final_state, cu_seqlens, use_qk_l2norm_in_kernel, use_p_times_alpha,
    )
    final_state = torch.stack([final_S, final_M], dim=0) if output_final_state else None
    return o, final_state
