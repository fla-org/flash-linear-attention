# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""
Tier 3 Atom Operations: Activation Functions

Log-addexp and log-sigmoid helpers for gate computations.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def log_gate_mix(log_prior, log_reset, gate_logit):
    """Log affine gate mixture without rounding two large gate-shifted logs.

    Inputs may share an arbitrary additive log offset. In the query recurrence
    they represent log(E*rho), retaining more FP32 carry precision near the
    uniform prior. No probability normalization or floor is introduced.
    """
    carry = log_prior - tl.maximum(gate_logit, 0.0)
    reset = log_reset + tl.minimum(gate_logit, 0.0)
    maximum = tl.maximum(carry, reset)
    difference = carry - reset
    correction = (
        libdevice.log1p(tl.exp(-tl.abs(difference)))
        - libdevice.log1p(tl.exp(-tl.abs(gate_logit)))
    )
    return tl.where(difference != difference, maximum, maximum + correction)


@triton.jit
def logaddexp(a, b):
    """Numerically stable log(exp(a) + exp(b)).

    Uses: max(a,b) + log(1 + exp(-|a-b|)) to avoid overflow.
    Guards against both-infinity inputs (where a-b = NaN in IEEE 754).

    Args:
        a, b: Tensors of same shape (or broadcastable).
    Returns:
        log(exp(a) + exp(b)), same shape.
    """
    m = tl.maximum(a, b)
    diff = a - b
    safe_result = m + tl.log(1.0 + tl.exp(-tl.abs(diff)))
    # diff is NaN when a=b=-inf or a=b=+inf; m is correct in those cases
    return tl.where(diff != diff, m, safe_result)


@triton.jit
def logsigmoid_pair(x):
    """Returns (log(sigmoid(x)), log(1-sigmoid(x))).

    Stable computation using softplus identity:
        softplus(z) = max(z, 0) + log(1 + exp(-|z|))
        log(sigmoid(x))   = -softplus(-x)
        log(1-sigmoid(x)) = -softplus(x)

    Since exp(-|z|) <= 1, this never overflows even for large |x|.

    Args:
        x: Input tensor (any shape).
    Returns:
        Tuple of (log_sig, log_1m_sig), each same shape as x.
    """
    abs_x = tl.abs(x)
    log1p_term = tl.log(1.0 + tl.exp(-abs_x))
    log_sig = -(tl.maximum(-x, 0.0) + log1p_term)      # -softplus(-x)
    log_1m_sig = -(tl.maximum(x, 0.0) + log1p_term)     # -softplus(x)
    return log_sig, log_1m_sig
