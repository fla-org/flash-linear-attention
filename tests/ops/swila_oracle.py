# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Equation-level factorial SwiLA recurrence oracle used only by tests.

This module intentionally imports no SwiLA implementation or kernel wrapper.
Keeping the recurrence in ordinary differentiable PyTorch gives CPU tests an
independent forward and backward oracle for the live eager/Triton paths.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _decayed_state(
    state: torch.Tensor,
    log_decay: torch.Tensor,
) -> torch.Tensor:
    decay = log_decay.exp()
    return state * decay[:, :, None, None, :]


def _initial_log_probabilities(
    *,
    batch: int,
    heads: int,
    values: int,
    experts: int,
    like: torch.Tensor,
) -> torch.Tensor:
    return like.new_full(
        (batch, heads, values, experts),
        -math.log(experts),
    )


def _mix_exact_initial_prior(
    initial: torch.Tensor,
    gate_logit: torch.Tensor,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact first-step gate mix without taking ``log(initial)``."""
    gate = torch.sigmoid(gate_logit)
    probability = (
        (1.0 - gate) * initial
        + gate * F.softmax(router_logits, dim=-1)
    )
    return probability, probability.log()


def swila_recurrence_oracle(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    temporal: bool = False,
    initial_r: torch.Tensor | None = None,
    initial_rho: torch.Tensor | None = None,
    gate_kv: torch.Tensor | None = None,
    gate_q: torch.Tensor | None = None,
    log_decay: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    """Evaluate factorial SwiLA directly from its sequential equations.

    Inputs use the public recurrence layout. The returned dictionary has a
    stable shape across modes: ``o``, ``state``, ``r``, and ``rho`` are always
    tensors. Temporal outputs also include final priors, residuals ``u``,
    and the pre-observation routing prior ``pi_bar``.
    """
    if temporal and (gate_kv is None or gate_q is None):
        raise ValueError("temporal recurrence requires gate_kv and gate_q")

    batch, length, heads, keys = q.shape
    values = v.shape[-1]
    experts = pi_q.shape[-1]
    if scale is None:
        scale = keys**-0.5

    state = (
        initial_state.clone()
        if initial_state is not None
        else q.new_zeros(batch, heads, values, experts, keys)
    )
    log_r_previous = log_rho_previous = None
    exact_initial_r = exact_initial_rho = None
    if temporal:
        exact_initial_r = initial_r
        exact_initial_rho = initial_rho
        if exact_initial_r is None:
            log_r_previous = _initial_log_probabilities(
                batch=batch,
                heads=heads,
                values=values,
                experts=experts,
                like=q,
            )
        if exact_initial_rho is None:
            log_rho_previous = _initial_log_probabilities(
                batch=batch,
                heads=heads,
                values=values,
                experts=experts,
                like=q,
            )

    outputs = []
    responsibilities = []
    query_responsibilities = []
    residuals = []
    temporal_priors = []
    for index in range(length):
        q_t = q[:, index]
        k_t = k[:, index]
        v_t = v[:, index]
        beta_t = beta[:, index]
        pi_q_t = pi_q[:, index]
        pi_k_t = pi_k[:, index]

        prediction_state = state
        if log_decay is not None:
            prediction_state = _decayed_state(
                state,
                log_decay[:, index],
            )
        prediction = torch.einsum(
            "bhd,bhved->bhve",
            k_t,
            prediction_state,
        )
        residual = v_t.unsqueeze(-1) - prediction

        if temporal:
            assert gate_kv is not None
            gate_kv_t = gate_kv[:, index]
            if exact_initial_r is not None:
                r_prior, log_r_prior = _mix_exact_initial_prior(
                    exact_initial_r,
                    gate_kv_t,
                    pi_k_t,
                )
                exact_initial_r = None
            else:
                assert log_r_previous is not None
                log_r_prior = torch.logaddexp(
                    F.logsigmoid(-gate_kv_t) + log_r_previous,
                    F.logsigmoid(gate_kv_t)
                    + F.log_softmax(pi_k_t, dim=-1),
                )
                r_prior = log_r_prior.exp()
            temporal_priors.append(r_prior)
            posterior = log_r_prior - 0.5 * residual.square()
            r_t = F.softmax(posterior, dim=-1)
            log_r_previous = F.log_softmax(posterior, dim=-1)
        else:
            r_t = F.softmax(
                pi_k_t - 0.5 * residual.square(),
                dim=-1,
            )

        state_prior = prediction_state
        write = torch.einsum(
            "bhd,bhve->bhved",
            k_t,
            r_t * residual,
        )
        state = state_prior + beta_t.unsqueeze(-1) * write

        expert_output = torch.einsum(
            "bhd,bhved->bhve",
            q_t * scale,
            state,
        )
        if temporal:
            assert gate_q is not None
            gate_q_t = gate_q[:, index]
            if exact_initial_rho is not None:
                rho_t, log_rho = _mix_exact_initial_prior(
                    exact_initial_rho,
                    gate_q_t,
                    pi_q_t,
                )
                exact_initial_rho = None
            else:
                assert log_rho_previous is not None
                log_rho = torch.logaddexp(
                    F.logsigmoid(-gate_q_t) + log_rho_previous,
                    F.logsigmoid(gate_q_t)
                    + F.log_softmax(pi_q_t, dim=-1),
                )
                rho_t = log_rho.exp()
            log_rho_previous = log_rho
        else:
            rho_t = F.softmax(pi_q_t, dim=-1)

        residuals.append(residual)
        outputs.append((rho_t * expert_output).sum(dim=-1))
        responsibilities.append(r_t)
        query_responsibilities.append(rho_t)

    result = {
        "o": torch.stack(outputs, dim=1),
        "state": state,
        "r": torch.stack(responsibilities, dim=1),
        "rho": torch.stack(query_responsibilities, dim=1),
        "r_final": None,
        "rho_final": None,
        "u": torch.stack(residuals, dim=1) if temporal else None,
        "pi_bar": torch.stack(temporal_priors, dim=1) if temporal else None,
    }
    if temporal:
        result["r_final"] = (
            exact_initial_r
            if exact_initial_r is not None
            else log_r_previous.exp()
        )
        result["rho_final"] = (
            exact_initial_rho
            if exact_initial_rho is not None
            else log_rho_previous.exp()
        )
    return result
