# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.delta_rule.wy_fast import prepare_wy_repr_bwd, prepare_wy_repr_fwd, recompute_w_u_fwd
from fla.ops.momentum_delta_rule.chunk_delta_h import (
    chunk_mode_rule_bwd_dhu,
    chunk_mode_rule_fwd_h_recompute_by_vnew,
    chunk_mode_rule_fwd_inter_qS_qM,
)
from fla.ops.momentum_delta_rule.chunk_o import chunk_mode_bwd_dv_local, chunk_mode_rule_bwd_dqkyz, chunk_mode_rule_fwd_o
from fla.ops.momentum_delta_rule.utils import chunk_mode_rule_cumsum_scalar_fwd
from fla.ops.momentum_delta_rule.wy_fast import chunk_scaled_dot_mode_rule_pkt_fwd, prepare_uyz_repr_bwd, recompute_u_y_z_fwd
from fla.ops.utils import solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

# Full momentum uses Triton kernels (utils, wy_fast, chunk_delta_h/o + solve_tril).
# Degenerate (mu->0) reuses DeltaRule WY kernels.


def _chunk_momentum_degenerate_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    w, u, A = prepare_wy_repr_fwd(
        k=k,
        v=v,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=None,
        scale=1.0 / math.sqrt(q.shape[-1]),
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    return o, A, final_state


def _chunk_momentum_degenerate_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        do=do,
        g=None,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=None,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    dq, dk, dw, _ = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        h=h,
        w=w,
        dv=dv,
        do=do,
        dh=dh,
        g=None,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    dk2, dv, db = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
    )
    dk.add_(dk2)
    return dq, dk, dv, db, dh0


class ChunkModeRuleFunction(torch.autograd.Function):
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
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_size: int = 64,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        o, A, final_state = _chunk_momentum_degenerate_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, A, initial_state, cu_seqlens)
        ctx.scale = 1.0 / math.sqrt(q.shape[-1])
        ctx.chunk_size = chunk_size
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        q, q_rstd, k, k_rstd, v, beta, A, initial_state, cu_seqlens = ctx.saved_tensors

        dq, dk, dv, db, dh0 = _chunk_momentum_degenerate_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), dh0, None, None, None, None, None


@torch.compiler.disable
def chunk_mode_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    **kwargs,
):
    r"""Degenerate MomentumDeltaNet chunk rule (mu->0, alpha=1).

    Kept for backward compatibility. For full stepwise momentum, use
    :func:`chunk_momentum_delta_rule`.
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkModeRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape (batch size, seq len, num heads)."

    chunk_size = kwargs.pop('chunk_size', 64)
    if chunk_size not in (16, 32, 64):
        raise ValueError(f"`chunk_size` must be 16, 32, or 64, got {chunk_size}.")
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
    o, final_state = ChunkModeRuleFunction.apply(
        q, k, v, beta, initial_state, output_final_state,
        use_qk_l2norm_in_kernel, cu_seqlens, cu_seqlens_cpu, chunk_size,
    )
    return o, final_state


# Full momentum Triton path.

def chunk_momentum_delta_fwd_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_S: torch.Tensor,
    initial_M: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    assert chunk_size in [16, 32, 64]
    log_a_cum, log_mu_cum, log_ct = chunk_mode_rule_cumsum_scalar_fwd(
        log_alpha=log_alpha,
        log_mu=log_mu,
        beta=beta,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )

    A, bt, gamma_mask_q = chunk_scaled_dot_mode_rule_pkt_fwd(
        k=k,
        p=p,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        log_ct=log_ct,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
        chunk_size=chunk_size,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype,
    )

    u, y, z = recompute_u_y_z_fwd(
        p=p,
        v=v,
        A=A,
        log_a_cum=log_a_cum,
        bt=bt,
        cu_seqlens=cu_seqlens,
    )

    o_inter, v_new, final_S, final_M = chunk_mode_rule_fwd_inter_qS_qM(
        q=q,
        k=k,
        v=v,
        u=u,
        y=y,
        z=z,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        bt=bt,
        gamma_mask_q=gamma_mask_q,
        initial_S=initial_S,
        initial_M=initial_M,
        output_final_state=output_final_state,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    o = chunk_mode_rule_fwd_o(
        q=q,
        k=k,
        v=v_new,
        o_inter=o_inter,
        gamma_mask_q=gamma_mask_q,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    o.add_(o_inter)
    return o, A, final_S, final_M, bt, log_a_cum, log_mu_cum, log_ct, gamma_mask_q, v_new


# Alias for compatibility with official naming
chunk_mode_rule_fwd = chunk_momentum_delta_fwd_triton


def chunk_momentum_delta_bwd_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    beta: torch.Tensor,
    log_ct: torch.Tensor,
    log_a_cum: torch.Tensor,
    log_mu_cum: torch.Tensor,
    gamma_mask_q: torch.Tensor,
    bt: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_S: torch.Tensor,
    initial_M: torch.Tensor,
    do: torch.Tensor,
    dst: torch.Tensor,
    dmt: torch.Tensor,
    hS: torch.Tensor = None,
    hM: torch.Tensor = None,
    v_new: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    assert chunk_size in [16, 32, 64]

    u, y, z = recompute_u_y_z_fwd(
        p=p,
        v=v,
        A=A,
        log_a_cum=log_a_cum,
        bt=bt,
        cu_seqlens=cu_seqlens,
    )

    hS, hM = chunk_mode_rule_fwd_h_recompute_by_vnew(
        k=k,
        v_new=v_new,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        bt=bt,
        gamma_mask_q=gamma_mask_q,
        initial_S=initial_S,
        initial_M=initial_M,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    dv = chunk_mode_bwd_dv_local(
        q=q,
        k=k,
        do=do,
        gamma_mask_q=gamma_mask_q,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    ds, dm, ds0, dm0, dv = chunk_mode_rule_bwd_dhu(
        q=q,
        k=k,
        u=u,
        y=y,
        z=z,
        log_mu_cum=log_mu_cum,
        log_a_cum=log_a_cum,
        bt=bt,
        gamma_mask_q=gamma_mask_q,
        s0=initial_S,
        m0=initial_M,
        dst=dst,
        dmt=dmt,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    dq, dk, dy, dz, d_log_mu_cum, d_log_a_cum, d_bt, d_Attn_do_v, d_decay_s = chunk_mode_rule_bwd_dqkyz(
        q=q,
        k=k,
        v=v_new,
        do=do,
        s=hS,
        m=hM,
        ds=ds,
        dm=dm,
        log_mu_cum=log_mu_cum,
        log_a_cum=log_a_cum,
        bt=bt,
        gamma_mask_q=gamma_mask_q,
        dv=dv,
        y=y,
        z=z,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    del hS, hM, ds, dm

    dk2, dv, dp, dlog_alpha, dlog_mu, dbeta = prepare_uyz_repr_bwd(
        q=q,
        k=k,
        v=v,
        p=p,
        beta=beta,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        log_ct=log_ct,
        gamma_mask_q=gamma_mask_q,
        d_Attn_do_v=d_Attn_do_v,
        d_decay_s=d_decay_s,
        A=A,
        bt=bt,
        dbt=d_bt,
        d_log_mu_cum=d_log_mu_cum,
        d_log_a_cum=d_log_a_cum,
        du=dv,
        dy=dy,
        dz=dz,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )

    dk.add_(dk2)
    return dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta, ds0, dm0


# Alias for compatibility
chunk_mode_rule_bwd = chunk_momentum_delta_bwd_triton


class ChunkMomentumDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        log_alpha: torch.Tensor,
        log_mu: torch.Tensor,
        beta: torch.Tensor,
        eta: torch.Tensor,
        scale: float,
        initial_S: torch.Tensor,
        initial_M: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        use_p_times_alpha: bool = True,
        chunk_size: int = 64,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
            p, p_rstd = l2norm_fwd(p)
        else:
            q_rstd, k_rstd, p_rstd = None, None, None

        o, A, final_S, final_M, bt, log_a_cum, log_mu_cum, log_ct, gamma_mask_q, v_new = chunk_momentum_delta_fwd_triton(
            q=q,
            k=k if eta is None else (k * eta.unsqueeze(-1)).to(q.dtype),
            v=v,
            p=p if not use_p_times_alpha else (p * log_alpha.exp().unsqueeze(-1)).to(q.dtype),
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta,
            scale=scale,
            initial_S=initial_S,
            initial_M=initial_M,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )

        ctx.save_for_backward(q, k, v, p, eta, beta, A, log_a_cum, log_mu_cum, log_ct, bt, gamma_mask_q,
                              initial_S, initial_M, cu_seqlens, q_rstd, k_rstd, p_rstd, v_new, log_alpha)

        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_p_times_alpha = use_p_times_alpha
        ctx.chunk_size = chunk_size

        return o.to(q.dtype), final_S, final_M

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dst: torch.Tensor,
        dmt: torch.Tensor,
    ):
        q, k, v, p, eta, beta, A, log_a_cum, log_mu_cum, log_ct, bt, gamma_mask_q, initial_S, initial_M, cu_seqlens, q_rstd, k_rstd, p_rstd, v_new, log_alpha = ctx.saved_tensors

        dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta, ds0, dm0 = chunk_momentum_delta_bwd_triton(
            q=q,
            k=k if eta is None else (k * eta.unsqueeze(-1)).to(q.dtype),
            v=v,
            p=p if not ctx.use_p_times_alpha else (p * log_alpha.exp().unsqueeze(-1)).to(q.dtype),
            beta=beta,
            log_ct=log_ct,
            log_a_cum=log_a_cum,
            log_mu_cum=log_mu_cum,
            gamma_mask_q=gamma_mask_q,
            bt=bt,
            A=A,
            scale=ctx.scale,
            initial_S=initial_S,
            initial_M=initial_M,
            do=do,
            dst=dst,
            dmt=dmt,
            v_new=v_new,
            cu_seqlens=cu_seqlens,
            chunk_size=ctx.chunk_size,
        )

        if eta is not None:
            deta = (dk * k).sum(-1).to(beta.dtype if hasattr(beta, 'dtype') else dk.dtype)
            dk = dk * eta.unsqueeze(-1)
        else:
            deta = None

        if ctx.use_p_times_alpha:
            alpha = log_alpha.exp()
            dlog_alpha.add_((dp * p).sum(-1).to(alpha.dtype) * alpha)
            dp = dp * alpha.unsqueeze(-1)

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
            dp = l2norm_bwd(p, p_rstd, dp)

        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dp.to(p.dtype), dlog_alpha.to(beta.dtype), dlog_mu.to(beta.dtype), dbeta.to(beta.dtype), deta, None, ds0, dm0, None, None, None, None, None


@torch.compiler.disable
def chunk_momentum_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    p: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    eta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_p_times_alpha: bool = True,
    chunk_size: int = 64,
):
    r"""Full stepwise momentum delta rule (Stage 2 Triton).

    Args:
        q (torch.Tensor): queries `[B, T, H, K]`.
        k (torch.Tensor): keys `[B, T, H, K]`.
        v (torch.Tensor): values `[B, T, H, V]`.
        log_alpha (torch.Tensor): `[B, T, H]`.
        log_mu (torch.Tensor): `[B, T, H]`.
        p (torch.Tensor, Optional): auxiliary keys, defaults to `k`.
        beta (torch.Tensor, Optional): forget gate `[B, T, H]`.
        eta (torch.Tensor, Optional): per-token scale `[B, T, H]`.
        scale (float, Optional): attention scale, defaults to `1/sqrt(K)`.
        initial_state (torch.Tensor, Optional): `[2, N, H, K, V]` or `None`.
        output_final_state (bool): whether to return final state.
        cu_seqlens (torch.LongTensor, Optional): varlen lengths.
        use_qk_l2norm_in_kernel (bool): whether to l2norm q/k/p in kernel.
        use_p_times_alpha (bool): whether to scale `p *= exp(log_alpha)`.
        chunk_size (int): 16/32/64.

    Returns:
        o (torch.Tensor): `[B, T, H, V]`.
        final_state (torch.Tensor | None): `[2, N, H, K, V]` if requested.
    """
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
            "Variable-length `cu_seqlens` is not yet supported for full momentum Triton path. "
            "Use degenerate `chunk_mode_rule` or wait for Stage 2 varlen.")
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_S is not None and initial_S.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_S.shape[0]}.",
            )
    o, final_S, final_M = ChunkMomentumDeltaRuleFunction.apply(
        q,
        k,
        v,
        p,
        log_alpha,
        log_mu,
        beta,
        eta,
        scale,
        initial_S,
        initial_M,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        use_p_times_alpha,
        chunk_size,
    )
    if output_final_state:
        final_state = torch.stack([final_S, final_M], dim=0)
    else:
        final_state = None

    return o, final_state
