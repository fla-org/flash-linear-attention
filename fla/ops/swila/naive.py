# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from fla.ops.swila.temporal_math import (
    log_temporal_query_mix,
    mix_exact_initial_temporal_prior,
)


def apply_log_decay(
    state: torch.Tensor,
    log_decay_i: torch.Tensor,
) -> torch.Tensor:
    """Apply multiplicative ``exp(log_decay_i)`` without expanding over D_v."""
    expected = (state.shape[0], state.shape[1], state.shape[-1])
    if tuple(log_decay_i.shape) != expected:
        raise ValueError(f"log_decay_i must have shape {expected}, got {tuple(log_decay_i.shape)}")
    decay_i = log_decay_i.exp()
    return state * decay_i[:, :, None, None, :]


def swila_recurrence_non_temporal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]:
    output_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    beta = beta.to(torch.float32)
    pi_q = pi_q.to(torch.float32)
    pi_k = pi_k.to(torch.float32)
    initial_state = initial_state.to(torch.float32) if initial_state is not None else None

    B, T, H, D_k = q.shape
    D_v = v.shape[-1]
    K = pi_q.shape[-1]
    q = q.transpose(1, 2).contiguous() * scale
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous()
    pi_q = pi_q.transpose(1, 2).contiguous()
    pi_k = pi_k.transpose(1, 2).contiguous()

    if initial_state is not None:
        S = initial_state.clone()
    else:
        S = q.new_zeros(B, H, D_v, K, D_k)

    o = q.new_zeros(B, H, T, D_v)
    r_seq = pi_k.new_zeros(B, H, T, D_v, K)

    for i in range(T):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        beta_i = beta[:, :, i]
        pi_q_i = pi_q[:, :, i]
        pi_k_i = pi_k[:, :, i]

        pred = torch.einsum('bhd,bhvkd->bhvk', k_i, S)
        u = v_i.unsqueeze(-1) - pred
        r_i = F.softmax(pi_k_i - 0.5 * u.square(), dim=-1)

        S = S + beta_i.unsqueeze(-1) * torch.einsum(
            'bhd,bhvk->bhvkd', k_i, r_i * u
        )

        qS = torch.einsum('bhd,bhvkd->bhvk', q_i, S)
        pi_q_star = F.softmax(pi_q_i, dim=-1)
        o_i = (pi_q_star * qS).sum(dim=-1)

        o[:, :, i] = o_i
        r_seq[:, :, i] = r_i

    o = o.transpose(1, 2).contiguous()
    r_seq = r_seq.transpose(1, 2).contiguous()
    final_state = S if output_final_state else None
    return o.to(output_dtype), final_state, r_seq


def swila_recurrence_non_temporal_gated(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    log_decay: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]:
    output_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    beta = beta.to(torch.float32)
    pi_q = pi_q.to(torch.float32)
    pi_k = pi_k.to(torch.float32)
    log_decay = log_decay.to(torch.float32)
    initial_state = initial_state.to(torch.float32) if initial_state is not None else None

    B, T, H, D_k = q.shape
    D_v = v.shape[-1]
    K = pi_q.shape[-1]
    q = q.transpose(1, 2).contiguous() * scale
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous()
    pi_q = pi_q.transpose(1, 2).contiguous()
    pi_k = pi_k.transpose(1, 2).contiguous()
    log_decay = log_decay.transpose(1, 2).contiguous()

    if initial_state is not None:
        S = initial_state.clone()
    else:
        S = q.new_zeros(B, H, D_v, K, D_k)

    o = q.new_zeros(B, H, T, D_v)
    r_seq = pi_k.new_zeros(B, H, T, D_v, K)

    for i in range(T):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        beta_i = beta[:, :, i]
        pi_q_i = pi_q[:, :, i]
        pi_k_i = pi_k[:, :, i]
        log_decay_i = log_decay[:, :, i]

        # Prediction and routing use the decayed state (Equation 41).
        prediction_state = S
        prediction_state = apply_log_decay(
            S, log_decay_i
        )
        pred = torch.einsum('bhd,bhvkd->bhvk', k_i, prediction_state)
        u = v_i.unsqueeze(-1) - pred
        r_i = F.softmax(pi_k_i - 0.5 * u.square(), dim=-1)

        S_prior = (
            prediction_state
        )
        S = S_prior + beta_i.unsqueeze(-1) * torch.einsum(
            'bhd,bhvk->bhvkd', k_i, r_i * u
        )

        qS = torch.einsum('bhd,bhvkd->bhvk', q_i, S)
        pi_q_star = F.softmax(pi_q_i, dim=-1)
        o_i = (pi_q_star * qS).sum(dim=-1)

        o[:, :, i] = o_i
        r_seq[:, :, i] = r_i

    o = o.transpose(1, 2).contiguous()
    r_seq = r_seq.transpose(1, 2).contiguous()
    final_state = S if output_final_state else None
    return o.to(output_dtype), final_state, r_seq


def swila_recurrence_temporal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    initial_r: torch.Tensor = None,
    initial_rho: torch.Tensor = None,
    gate_kv: torch.Tensor = None,
    gate_q: torch.Tensor = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, dict[str, torch.Tensor]]:
    output_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    beta = beta.to(torch.float32)
    pi_q = pi_q.to(torch.float32)
    pi_k = pi_k.to(torch.float32)
    initial_state = initial_state.to(torch.float32) if initial_state is not None else None
    initial_r = initial_r.to(torch.float32) if initial_r is not None else None
    initial_rho = initial_rho.to(torch.float32) if initial_rho is not None else None
    gate_kv = gate_kv.to(torch.float32) if gate_kv is not None else None
    gate_q = gate_q.to(torch.float32) if gate_q is not None else None

    B, T, H, D_k = q.shape
    D_v = v.shape[-1]
    K = pi_q.shape[-1]
    q = q.transpose(1, 2).contiguous() * scale
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous()
    pi_q = pi_q.transpose(1, 2).contiguous()
    pi_k = pi_k.transpose(1, 2).contiguous()
    gate_kv = gate_kv.transpose(1, 2).contiguous()
    gate_q = gate_q.transpose(1, 2).contiguous()

    if initial_state is not None:
        S = initial_state.clone()
    else:
        S = q.new_zeros(B, H, D_v, K, D_k)

    exact_initial_r = initial_r.clone() if initial_r is not None else None
    if initial_r is not None:
        log_r_prev = None
    else:
        log_r_prev = torch.full(
            (B, H, D_v, K), -math.log(K), device=q.device, dtype=q.dtype
        )
    exact_initial_rho = initial_rho.clone() if initial_rho is not None else None
    # Carry log(K*rho), centered at the uniform prior. The fixed shift is
    # removed only for probability outputs/cache; it does not normalize mass.
    log_rho_offset = math.log(K)
    if initial_rho is not None:
        log_rho_prev = None
    else:
        log_rho_prev = torch.full(
            (B, H, D_v, K), 0.0, device=q.device, dtype=q.dtype
        )

    o = q.new_zeros(B, H, T, D_v)
    r_seq = pi_k.new_zeros(B, H, T, D_v, K)
    rho_seq = pi_q.new_zeros(B, H, T, D_v, K)
    last_r_probability = None
    last_rho_probability = None

    for i in range(T):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        beta_i = beta[:, :, i]
        pi_q_i = pi_q[:, :, i]
        pi_k_i = pi_k[:, :, i]

        pred = torch.einsum('bhd,bhvkd->bhvk', k_i, S)
        u = v_i.unsqueeze(-1) - pred

        log_g_kv = F.logsigmoid(gate_kv[:, :, i])
        log_1mg_kv = F.logsigmoid(-gate_kv[:, :, i])
        if exact_initial_r is not None:
            _, log_r_prior = mix_exact_initial_temporal_prior(
                exact_initial_r,
                gate_kv[:, :, i],
                pi_k_i,
            )
            exact_initial_r = None
        else:
            assert log_r_prev is not None
            log_r_prior = torch.logaddexp(
                log_1mg_kv + log_r_prev,
                log_g_kv + F.log_softmax(pi_k_i, dim=-1),
            )
        log_posterior = log_r_prior - 0.5 * u.square()
        r_i = F.softmax(log_posterior, dim=-1)
        log_r_prev = F.log_softmax(log_posterior, dim=-1)
        last_r_probability = r_i

        S = S + beta_i.unsqueeze(-1) * torch.einsum(
            'bhd,bhvk->bhvkd', k_i, r_i * u
        )

        qS = torch.einsum('bhd,bhvkd->bhvk', q_i, S)
        if exact_initial_rho is not None:
            rho_i, log_rho = mix_exact_initial_temporal_prior(
                exact_initial_rho,
                gate_q[:, :, i],
                pi_q_i,
            )
            log_rho = log_rho + log_rho_offset
            exact_initial_rho = None
        else:
            assert log_rho_prev is not None
            log_rho = log_temporal_query_mix(
                log_rho_prev,
                F.log_softmax(pi_q_i, dim=-1) + log_rho_offset,
                gate_q[:, :, i],
            )
            rho_i = torch.exp(log_rho - log_rho_offset)
        log_rho_prev = log_rho
        last_rho_probability = rho_i
        o_i = (rho_i * qS).sum(dim=-1)
        rho_seq[:, :, i] = rho_i

        o[:, :, i] = o_i
        r_seq[:, :, i] = r_i

    o = o.transpose(1, 2).contiguous()
    r_seq = r_seq.transpose(1, 2).contiguous()
    rho_seq = rho_seq.transpose(1, 2).contiguous()
    r_prev = (
        last_r_probability
        if last_r_probability is not None
        else exact_initial_r
        if exact_initial_r is not None
        else torch.exp(log_r_prev)
    )
    rho_prev = (
        last_rho_probability
        if last_rho_probability is not None
        else exact_initial_rho
        if exact_initial_rho is not None
        else torch.exp(log_rho_prev - log_rho_offset)
    )
    final_state = None
    if output_final_state:
        final_state = {
            'S': S,
            'r_prev': r_prev,
            'rho_prev': rho_prev,
        }
    return o.to(output_dtype), final_state, {'r': r_seq, 'rho': rho_seq}


def swila_recurrence_temporal_gated(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    log_decay: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    initial_r: torch.Tensor = None,
    initial_rho: torch.Tensor = None,
    gate_kv: torch.Tensor = None,
    gate_q: torch.Tensor = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, dict[str, torch.Tensor]]:
    output_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    beta = beta.to(torch.float32)
    pi_q = pi_q.to(torch.float32)
    pi_k = pi_k.to(torch.float32)
    log_decay = log_decay.to(torch.float32)
    initial_state = initial_state.to(torch.float32) if initial_state is not None else None
    initial_r = initial_r.to(torch.float32) if initial_r is not None else None
    initial_rho = initial_rho.to(torch.float32) if initial_rho is not None else None
    gate_kv = gate_kv.to(torch.float32) if gate_kv is not None else None
    gate_q = gate_q.to(torch.float32) if gate_q is not None else None

    B, T, H, D_k = q.shape
    D_v = v.shape[-1]
    K = pi_q.shape[-1]
    q = q.transpose(1, 2).contiguous() * scale
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous()
    pi_q = pi_q.transpose(1, 2).contiguous()
    pi_k = pi_k.transpose(1, 2).contiguous()
    log_decay = log_decay.transpose(1, 2).contiguous()
    gate_kv = gate_kv.transpose(1, 2).contiguous()
    gate_q = gate_q.transpose(1, 2).contiguous()

    if initial_state is not None:
        S = initial_state.clone()
    else:
        S = q.new_zeros(B, H, D_v, K, D_k)

    exact_initial_r = initial_r.clone() if initial_r is not None else None
    if initial_r is not None:
        log_r_prev = None
    else:
        log_r_prev = torch.full(
            (B, H, D_v, K), -math.log(K), device=q.device, dtype=q.dtype
        )
    exact_initial_rho = initial_rho.clone() if initial_rho is not None else None
    # As above, a fixed log-coordinate shift improves long query carries
    # without changing supplied prior mass, output dtype, or cache payloads.
    log_rho_offset = math.log(K)
    if initial_rho is not None:
        log_rho_prev = None
    else:
        log_rho_prev = torch.full(
            (B, H, D_v, K), 0.0, device=q.device, dtype=q.dtype
        )

    o = q.new_zeros(B, H, T, D_v)
    r_seq = pi_k.new_zeros(B, H, T, D_v, K)
    rho_seq = pi_q.new_zeros(B, H, T, D_v, K)
    last_r_probability = None
    last_rho_probability = None

    for i in range(T):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        beta_i = beta[:, :, i]
        pi_q_i = pi_q[:, :, i]
        pi_k_i = pi_k[:, :, i]
        log_decay_i = log_decay[:, :, i]

        # The temporal posterior uses the same post-decay residual convention
        # as the non-temporal recurrence. Q-side filtering/readout is unchanged.
        prediction_state = S
        prediction_state = apply_log_decay(
            S, log_decay_i
        )
        pred = torch.einsum('bhd,bhvkd->bhvk', k_i, prediction_state)
        u = v_i.unsqueeze(-1) - pred

        log_g_kv = F.logsigmoid(gate_kv[:, :, i])
        log_1mg_kv = F.logsigmoid(-gate_kv[:, :, i])
        if exact_initial_r is not None:
            _, log_r_prior = mix_exact_initial_temporal_prior(
                exact_initial_r,
                gate_kv[:, :, i],
                pi_k_i,
            )
            exact_initial_r = None
        else:
            assert log_r_prev is not None
            log_r_prior = torch.logaddexp(
                log_1mg_kv + log_r_prev,
                log_g_kv + F.log_softmax(pi_k_i, dim=-1),
            )
        log_posterior = log_r_prior - 0.5 * u.square()
        r_i = F.softmax(log_posterior, dim=-1)
        log_r_prev = F.log_softmax(log_posterior, dim=-1)
        last_r_probability = r_i

        S_prior = (
            prediction_state
        )
        S = S_prior + beta_i.unsqueeze(-1) * torch.einsum(
            'bhd,bhvk->bhvkd', k_i, r_i * u
        )

        qS = torch.einsum('bhd,bhvkd->bhvk', q_i, S)
        if exact_initial_rho is not None:
            rho_i, log_rho = mix_exact_initial_temporal_prior(
                exact_initial_rho,
                gate_q[:, :, i],
                pi_q_i,
            )
            log_rho = log_rho + log_rho_offset
            exact_initial_rho = None
        else:
            assert log_rho_prev is not None
            log_rho = log_temporal_query_mix(
                log_rho_prev,
                F.log_softmax(pi_q_i, dim=-1) + log_rho_offset,
                gate_q[:, :, i],
            )
            rho_i = torch.exp(log_rho - log_rho_offset)
        log_rho_prev = log_rho
        last_rho_probability = rho_i
        o_i = (rho_i * qS).sum(dim=-1)
        rho_seq[:, :, i] = rho_i

        o[:, :, i] = o_i
        r_seq[:, :, i] = r_i

    o = o.transpose(1, 2).contiguous()
    r_seq = r_seq.transpose(1, 2).contiguous()
    rho_seq = rho_seq.transpose(1, 2).contiguous()
    r_prev = (
        last_r_probability
        if last_r_probability is not None
        else exact_initial_r
        if exact_initial_r is not None
        else torch.exp(log_r_prev)
    )
    rho_prev = (
        last_rho_probability
        if last_rho_probability is not None
        else exact_initial_rho
        if exact_initial_rho is not None
        else torch.exp(log_rho_prev - log_rho_offset)
    )
    final_state = None
    if output_final_state:
        final_state = {
            'S': S,
            'r_prev': r_prev,
            'rho_prev': rho_prev,
        }
    return o.to(output_dtype), final_state, {'r': r_seq, 'rho': rho_seq}


def swila_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    temporal: bool = False,
    initial_r: torch.Tensor = None,
    initial_rho: torch.Tensor = None,
    gate_kv: torch.Tensor = None,
    gate_q: torch.Tensor = None,
    log_decay: torch.Tensor = None,
) -> tuple[torch.Tensor, dict | torch.Tensor | None, torch.Tensor]:
    """
    Sequential (recurrent) SwiLA forward with nonlinear gradient update.

    This implements the exact EM-derived gradient ascent update for the expected
    log-likelihood of a mixture of experts regression model:

        L(w_{t-1,d}) = -1/2 sum_k r_{tdk} (y_{td} - w_{t-1,dk}^T x_t)^2

    The gradient-based update is:
        w_{tdk} <- w_{t-1,dk} + beta_t * r_{tdk} * u_{tdk} * x_t

    where:
        u_{tdk} = y_{td} - w_{t-1,dk}^T x_t  (prediction error)
        r_{tdk} = softmax(pi_k - 0.5 * u^2)  (responsibilities)

    Args:
        q: [B, T, H, D_k] - Query tensor
        k: [B, T, H, D_k] - Key tensor
        v: [B, T, H, D_v] - Value tensor (used as target Y)
        beta: [B, T, H, D_v, K] - Learning rate per position/head/dim/expert
        pi_q: [B, T, H, D_v, K] - Prior responsibilities from queries (per dimension)
        pi_k: [B, T, H, D_v, K] - Prior responsibilities from keys (per dimension)
        scale: float - Scale factor for output (default: 1/sqrt(D_k))
        initial_state: [B, H, D_v, K, D_k] - Initial recurrent state
        output_final_state: bool - Whether to return final state
        temporal: bool - If True, use exact-identity temporal filtering with reset gates.
        initial_r: [B, H, D_v, K] - Initial responsibility state (temporal mode).
        initial_rho: [B, H, D_v, K] - Initial query gating state (temporal mode).
        gate_kv: Optional gate values for reset_gate temporal coupling (KV-side).
            [B, T, H, D_v, 1].
            Temporal SwiLA uses these reset gates to mix transition and reset logits.
        gate_q: Optional gate values for reset_gate temporal coupling (Q-side).
            Same shape as gate_kv.
        log_decay: Optional input-dependent log decay with shape [B, T, H, D_k],
            shared over values and experts and applied before prediction.

    Returns:
        o: [B, T, H, D_v] - Output tensor, returned in the input q dtype.
        final_state: dict or [B, H, D_v, K, D_k] - Final recurrent state (if requested).
            Returned in float32. When temporal=True, returns dict with 'S', 'r_prev',
            'rho_prev'.
        responsibilities: When temporal=True, returns dict with:
            - 'r': [B, T, H, D_v, K] - KV-side responsibilities
            - 'rho': [B, T, H, D_v, K] - Q-side responsibilities
            Otherwise, returns tensor [B, T, H, D_v, K] - responsibilities.
            Responsibilities are returned in float32.
    """
    D_k = q.shape[-1]

    if log_decay is not None and tuple(log_decay.shape) != tuple(q.shape):
        raise ValueError(f"log_decay must have shape {tuple(q.shape)}, got {tuple(log_decay.shape)}")
    if scale is None:
        scale = D_k ** -0.5

    if temporal:
        if log_decay is None:
            return swila_recurrence_temporal(
                q=q,
                k=k,
                v=v,
                beta=beta,
                pi_q=pi_q,
                pi_k=pi_k,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                initial_r=initial_r,
                initial_rho=initial_rho,
                gate_kv=gate_kv,
                gate_q=gate_q,
            )
        return swila_recurrence_temporal_gated(
            q=q,
            k=k,
            v=v,
            beta=beta,
            pi_q=pi_q,
            pi_k=pi_k,
            log_decay=log_decay,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            initial_r=initial_r,
            initial_rho=initial_rho,
            gate_kv=gate_kv,
            gate_q=gate_q,
        )

    if log_decay is None:
        return swila_recurrence_non_temporal(
            q=q,
            k=k,
            v=v,
            beta=beta,
            pi_q=pi_q,
            pi_k=pi_k,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
    return swila_recurrence_non_temporal_gated(
        q=q,
        k=k,
        v=v,
        beta=beta,
        pi_q=pi_q,
        pi_k=pi_k,
        log_decay=log_decay,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
