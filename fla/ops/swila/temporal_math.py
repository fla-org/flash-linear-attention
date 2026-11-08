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

import torch
import torch.nn.functional as F

from fla.ops.swila.autograd import autocast_custom_bwd, autocast_custom_fwd


def log_temporal_query_mix(
    log_prior: torch.Tensor,
    log_reset: torch.Tensor,
    gate_logit: torch.Tensor,
) -> torch.Tensor:
    """Log of the affine reset mixture, in any common log coordinate system.

    Combine the small softplus corrections before adding the carried log
    probability. Separately rounding ``log_prior + log(1-g)`` at every step
    introduces a systematic error for persistent gates. This expression does
    not normalize either branch or assume unit prior mass.
    """
    zero = torch.zeros_like(gate_logit)
    # maximum/minimum split the derivative at zero equally. clamp_min/max
    # would each use derivative one there, doubling the gate VJP at g=0.
    carry = log_prior - torch.maximum(gate_logit, zero)
    reset = log_reset + torch.minimum(gate_logit, zero)
    maximum = torch.maximum(carry, reset)
    difference = carry - reset
    same_infinity = torch.isnan(difference)
    safe_difference = torch.where(same_infinity, torch.zeros_like(difference), difference)
    correction = (
        torch.log1p(torch.exp(-safe_difference.abs()))
        - torch.log1p(torch.exp(-gate_logit.abs()))
    )
    return torch.where(same_infinity, maximum, maximum + correction)


class ExactInitialTemporalPriorMix(torch.autograd.Function):
    """Stable exact reset mix with an analytic zero-prior backward."""

    @staticmethod
    @autocast_custom_fwd
    def forward(
        ctx,
        initial: torch.Tensor,
        gate_logit: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        positive_initial = initial > 0
        safe_initial = torch.where(
            positive_initial,
            initial,
            torch.ones_like(initial),
        )
        log_initial = torch.where(
            positive_initial,
            torch.log(safe_initial),
            torch.full_like(initial, -float("inf")),
        )
        log_reset = F.log_softmax(router_logits, dim=-1)
        log_gate = F.logsigmoid(gate_logit)
        log_one_minus_gate = F.logsigmoid(-gate_logit)
        log_probability = torch.logaddexp(
            log_one_minus_gate + log_initial,
            log_gate + log_reset,
        )
        ctx.save_for_backward(
            initial,
            log_initial,
            log_reset,
            log_probability,
            log_gate,
            log_one_minus_gate,
        )
        return torch.exp(log_probability), log_probability

    @staticmethod
    @autocast_custom_bwd
    def backward(
        ctx,
        d_probability: torch.Tensor | None,
        d_log_probability: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            initial,
            log_initial,
            log_reset,
            log_probability,
            log_gate,
            log_one_minus_gate,
        ) = ctx.saved_tensors
        if d_probability is None:
            d_probability = torch.zeros_like(log_probability)
        if d_log_probability is None:
            d_log_probability = torch.zeros_like(log_probability)

        reset = torch.exp(log_reset)
        gate = torch.exp(log_gate)
        one_minus_gate = torch.exp(log_one_minus_gate)
        reset_branch_weight = torch.exp(
            log_gate + log_reset - log_probability
        )
        initial_branch_weight = torch.exp(
            log_one_minus_gate + log_initial - log_probability
        )

        # Probability-output contribution is ordinary affine mixing.
        d_initial = d_probability * one_minus_gate

        # For the log-output contribution, evaluate
        # d_log_probability * exp(log(1-g) - log_probability) as one signed
        # log-magnitude.  This retains cancellation when each factor alone is
        # outside FP32 range and maps an exactly-zero upstream to exactly zero
        # instead of forming ``0 * inf``.
        nonzero_log_gradient = d_log_probability != 0
        safe_abs_log_gradient = torch.where(
            nonzero_log_gradient,
            d_log_probability.abs(),
            torch.ones_like(d_log_probability),
        )
        log_abs_initial_gradient = (
            torch.log(safe_abs_log_gradient.double())
            + (log_one_minus_gate - log_probability).double()
        )
        abs_initial_gradient = torch.exp(log_abs_initial_gradient).to(
            initial.dtype
        )
        signed_initial_gradient = torch.where(
            d_log_probability < 0,
            -abs_initial_gradient,
            abs_initial_gradient,
        )
        d_initial = d_initial + torch.where(
            nonzero_log_gradient,
            signed_initial_gradient,
            torch.zeros_like(signed_initial_gradient),
        )

        # Differentiate through log-softmax without ever constructing a/p.
        d_log_reset = (
            d_log_probability * reset_branch_weight
            + d_probability * gate * reset
        )
        d_router_logits = d_log_reset - reset * d_log_reset.sum(
            dim=-1,
            keepdim=True,
        )

        # The log-mixture gate derivative is a difference of branch
        # responsibilities, which is bounded even for a subnormal mixture.
        d_gate_log = d_log_probability * (
            reset_branch_weight * one_minus_gate
            - initial_branch_weight * gate
        )
        d_gate_probability = d_probability * (
            gate * one_minus_gate * (reset - initial)
        )
        d_gate_logit = (d_gate_log + d_gate_probability).sum(
            dim=-1,
            keepdim=True,
        )
        return d_initial, d_gate_logit, d_router_logits


def mix_exact_initial_temporal_prior(
    initial: torch.Tensor,
    gate_logit: torch.Tensor,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix an exact initial prior without an ``exp(log(.))`` round trip.

    In particular, this keeps zero-probability coordinates differentiable.
    Applying ``log`` to a zero initial probability and then differentiating
    through ``logaddexp`` produces ``0 * inf`` in eager autograd, even though
    the equivalent probability-space gate mixture has a finite derivative.

    The forward is evaluated in log space, so any finite gate/router branch
    remains represented even when its probability-space value rounds to zero.
    The custom backward differentiates the equivalent probability mixture
    without dividing by the supplied prior, so a zero coordinate never forms
    ``0 * inf``.
    """
    return ExactInitialTemporalPriorMix.apply(
        initial,
        gate_logit,
        router_logits,
    )
