# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend NPU backend implementation of ``chunk_momentum_delta_rule``.

The community entry is kept pristine; on NPU the ``momentum_delta_rule`` dispatch
routes here (``backends/triton_ascend/__init__.py``). The forward composes the
dispatched ``common`` momentum primitives (triton-ascend kernels on NPU, registered
under ``fla/ops/common/backends/triton_ascend``) with the k*eta / p*alpha products
kept in fp32: on NPU bf16 * bf16 rounds the product to bf16, and the rounding
amplifies through the state recurrence past the parity tolerance. The backward runs
the same fp32 torch reference autograd as the community path (fp32-promoted products).
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
import triton

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_momentum_delta import (
    chunk_momentum_delta_fwd_h,
    chunk_momentum_delta_fwd_o,
    chunk_momentum_delta_kkt_fwd,
    chunk_momentum_delta_wy_fwd,
)
from fla.ops.momentum_delta_rule.naive import chunk_momentum_delta_rule_ref
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def _build_k_eta(k: torch.Tensor, eta: torch.Tensor | None) -> torch.Tensor:
    if eta is None:
        return k
    # see module docstring: keep the product in fp32 on npu
    return k * eta.unsqueeze(-1).to(torch.float32)


def _build_p_eff(p: torch.Tensor, log_alpha: torch.Tensor, use_p_times_alpha: bool) -> torch.Tensor:
    if not use_p_times_alpha:
        return p
    return (p * log_alpha.exp().unsqueeze(-1)).to(torch.float32)


@dataclasses.dataclass
class MomentumChunkPrecompute:
    """All `[B, H, NT, BT]` fp32 vectors and `[B, H, NT]` scalars the chunk kernels need."""

    log_a_cum: torch.Tensor
    log_m_cum: torch.Tensor
    a_cum: torch.Tensor       # exp(log_a_cum)
    log_ct: torch.Tensor      # per-chunk logcumsumexp(log_beta + log_m_cum - log_a_cum)
    lct1: torch.Tensor        # log_ct shifted by one slot, -inf at slot 0
    b_t: torch.Tensor         # exp(log_a_cum + log_ct)
    b_tm1: torch.Tensor       # b_t shifted, 0 at slot 0
    bar_a_tm1: torch.Tensor   # a_cum shifted, 1 at slot 0 (exp of the log-space shift)
    cfac: torch.Tensor        # exp(-log_m_cum)
    # per-chunk scalars gathered at the last *real* row of each chunk (pad-safe)
    a_last: torch.Tensor
    b_last: torch.Tensor
    ct_last: torch.Tensor
    lm_last: torch.Tensor


def momentum_chunk_precompute(
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> MomentumChunkPrecompute:
    """Host-side fp32 log-space precompute, in `[B, H, NT, BT]` contiguous layout.

    Pads `T` to a multiple of `chunk_size` with zeros so cumsums freeze past the last
    real row; per-chunk scalars are gathered at the clamped last real row, which makes
    the pad path exact (no recurrent-ref shortcut needed downstream).
    """
    B, T, H = log_alpha.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    pT = NT * BT
    pad = pT - T
    pad3 = (0, 0, 0, pad)

    def pad_t(x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, pad3) if pad else x

    log_alpha_p = pad_t(log_alpha)
    log_mu_p = pad_t(log_mu)
    beta_p = pad_t(beta)

    # naive.py cumsums are *chunk-local*: the ref rearranges to [B, H, NT, BT] first and
    # cumsums along the chunk dim, so each chunk's decay factors restart at its first row.
    log_a_cum = log_alpha_p.reshape(B, NT, BT, H).cumsum(dim=2).permute(0, 3, 1, 2)
    log_m_cum = log_mu_p.reshape(B, NT, BT, H).cumsum(dim=2).permute(0, 3, 1, 2)
    log_beta = (beta_p + 1e-6).log().reshape(B, NT, BT, H).permute(0, 3, 1, 2)
    log_c_before = log_beta + log_m_cum - log_a_cum
    log_ct = torch.logcumsumexp(log_c_before, dim=-1)
    lct1 = torch.cat([torch.full_like(log_ct[..., :1], float('-inf')), log_ct[..., :-1]], dim=-1)

    a_cum = log_a_cum.exp()
    b_t = (log_a_cum + log_ct).exp()
    # naive.py shifts bar_a in *log* space before exp, so slot 0 is exp(0) == 1
    bar_a_tm1 = torch.cat([torch.ones_like(a_cum[..., :1]), a_cum[..., :-1]], dim=-1)
    b_tm1 = torch.cat([torch.zeros_like(b_t[..., :1]), b_t[..., :-1]], dim=-1)
    cfac = (-log_m_cum).exp()
    lm_cum = log_m_cum

    # per-chunk scalars gathered at the last *real* row of each chunk (pad-safe)
    last_idx = torch.full((NT,), BT - 1, dtype=torch.long, device=log_alpha.device)
    last_idx[NT - 1] = min(BT - 1, T - 1 - (NT - 1) * BT)

    def gather_last(x: torch.Tensor) -> torch.Tensor:
        return x.gather(-1, last_idx.view(1, 1, NT, 1).expand(B, H, NT, 1))[..., 0]   # [B, H, NT]

    a_last = gather_last(a_cum)
    b_last = gather_last(b_t)
    ct_last = gather_last(log_ct)
    lm_last = gather_last(lm_cum)

    return MomentumChunkPrecompute(
        log_a_cum=log_a_cum,
        log_m_cum=lm_cum,
        a_cum=a_cum,
        log_ct=log_ct,
        lct1=lct1,
        b_t=b_t,
        b_tm1=b_tm1,
        bar_a_tm1=bar_a_tm1,
        cfac=cfac,
        a_last=a_last,
        b_last=b_last,
        ct_last=ct_last,
        lm_last=lm_last,
    )


def prepare_wy_repr_fwd(
    p_eff: torch.Tensor,
    k_eta: torch.Tensor,
    v: torch.Tensor,
    pre: MomentumChunkPrecompute,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """attn = (p_eff @ k_eta^T) * gamma_mask, then the inverse and its three RHS.

    Returns ``(u_c, y_c, z_c)`` with ``u_c = attn_inv @ v``,
    ``y_c = attn_inv @ (bar_a_tm1 * p_eff)`` and ``z_c = attn_inv @ (b_tm1 * p_eff)``.
    """
    A = chunk_momentum_delta_kkt_fwd(
        p_eff=p_eff,
        k_eta=k_eta,
        cfac=pre.cfac,
        lct1=pre.lct1,
        b_tm1=pre.b_tm1,
        chunk_size=chunk_size,
    )
    u_c, y_c, z_c = chunk_momentum_delta_wy_fwd(
        A=A,
        v=v,
        p_eff=p_eff,
        bar_a_tm1=pre.bar_a_tm1,
        b_tm1=pre.b_tm1,
        chunk_size=chunk_size,
    )
    return u_c, y_c, z_c


class ChunkMomentumDeltaRuleFunctionNPU(torch.autograd.Function):
    """NPU override of the community ``ChunkMomentumDeltaRuleFunction``.

    Forward composes the dispatched ``common`` momentum primitives (triton-ascend
    kernels on NPU); backward keeps the community fp32 torch reference autograd (same
    saved-tensors / ctx contract), with the k*eta and p*alpha products promoted to
    fp32 so the grads match the fp32 forward recurrence.
    """

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
        if cu_seqlens is not None:
            raise NotImplementedError("Variable-length `cu_seqlens` not yet supported for the momentum NPU chunk path.")
        k_eta = _build_k_eta(k, eta)
        p_eff = _build_p_eff(p, log_alpha, use_p_times_alpha)
        pre = momentum_chunk_precompute(log_alpha, log_mu, beta, chunk_size)
        u_c, y_c, z_c = prepare_wy_repr_fwd(p_eff, k_eta, v, pre, chunk_size)
        h_s, h_m, v_i, final_S, final_M = chunk_momentum_delta_fwd_h(
            k_eta=k_eta, u_c=u_c, y_c=y_c, z_c=z_c,
            cfac=pre.cfac, lct1=pre.lct1, lm_cum=pre.log_m_cum,
            a_last=pre.a_last, b_last=pre.b_last, ct_last=pre.ct_last, lm_last=pre.lm_last,
            initial_S=initial_S, initial_M=initial_M,
            output_final_state=output_final_state, chunk_size=chunk_size,
        )
        o = chunk_momentum_delta_fwd_o(
            q=q, k_eta=k_eta, v_i=v_i, h_s=h_s, h_m=h_m,
            a_cum=pre.a_cum, b_t=pre.b_t, cfac=pre.cfac, lct=pre.log_ct, lct1=pre.lct1,
            scale=scale, chunk_size=chunk_size,
        )
        # same ctx contract as the community backward expects
        ctx.save_for_backward(q, k, v, p, eta, beta, log_alpha, log_mu, initial_S,
                              initial_M, cu_seqlens, q_rstd, k_rstd, p_rstd)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_p_times_alpha = use_p_times_alpha
        ctx.chunk_size = chunk_size
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
            k_eta = k_r if eta_r is None else _build_k_eta(k_r, eta_r)
            p_eff = p_r if not ctx.use_p_times_alpha else _build_p_eff(p_r, log_alpha_r, ctx.use_p_times_alpha)
            initial_S_r = initial_S.detach().requires_grad_(True) if initial_S is not None else None
            initial_M_r = initial_M.detach().requires_grad_(True) if initial_M is not None else None
            o, final_state = chunk_momentum_delta_rule_ref(
                q=q_r, k=k_eta, v=v_r, p=p_eff, log_alpha=log_alpha_r, log_mu=log_mu_r,
                beta=beta_r, eta=torch.ones_like(beta_r), scale=ctx.scale,
                initial_S=initial_S_r, initial_M=initial_M_r, output_final_state=True, chunk_size=ctx.chunk_size)
            differentiable_inputs = (q_r, k_r, v_r, p_r, log_alpha_r, log_mu_r, beta_r)
            if eta_r is not None:
                differentiable_inputs += (eta_r,)
            if initial_S_r is not None:
                differentiable_inputs += (initial_S_r, initial_M_r)
            grads = torch.autograd.grad(
                (o, final_state[0], final_state[1]), differentiable_inputs,
                grad_outputs=(
                    do,
                    torch.zeros_like(final_state[0]) if dst is None else dst,
                    torch.zeros_like(final_state[1]) if dmt is None else dmt,
                ),
                retain_graph=False, allow_unused=True,
            )
        dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta = grads[:7]
        deta = grads[7] if eta is not None else None
        state_offset = 8 if eta is not None else 7
        ds0, dm0 = (grads[state_offset:state_offset + 2] if initial_S is not None else (None, None))
        if ctx.use_qk_l2norm_in_kernel:
            if dq is not None:
                dq = l2norm_bwd(q, q_rstd, dq.contiguous())
            if dk is not None:
                dk = l2norm_bwd(k, k_rstd, dk.contiguous())
            if dp is not None:
                dp = l2norm_bwd(p, p_rstd, dp.contiguous())
        return dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta, deta, None, ds0, dm0, None, None, None, None, None


def chunk_momentum_delta_rule_npu(q, k, v, log_alpha, log_mu, p=None, beta=None, eta=None, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None, use_qk_l2norm_in_kernel=True, use_p_times_alpha=True, chunk_size=64):
    """NPU entry for the community ``chunk_momentum_delta_rule`` (same contract)."""
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
    o, final_S, final_M = ChunkMomentumDeltaRuleFunctionNPU.apply(
        q, k, v, p, log_alpha, log_mu, beta, eta, scale, initial_S, initial_M, output_final_state, cu_seqlens, use_qk_l2norm_in_kernel, use_p_times_alpha, chunk_size)
    final_state = torch.stack([final_S, final_M], dim=0) if output_final_state else None
    return o, final_state
