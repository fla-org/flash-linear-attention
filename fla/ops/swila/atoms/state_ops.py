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
Tier 3 Atom Operations: State Manipulation

Operations on the 3D hidden state tensor [BV, BE, BK].
"""

import triton

from fla.ops.swila.atoms.reductions import dot_3d_axis2


@triton.jit
def prediction(h, k):
    """Compute prediction: pred = sum_k(h[v,e,k] * k[k]) -> [BV, BE].

    Matrix-vector product reading from state with key vector.

    Args:
        h: [BV, BE, BK] state tensor.
        k: [BK] key vector.
    Returns:
        [BV, BE] prediction per (value_dim, expert) pair.
    """
    return dot_3d_axis2(h, k)


@triton.jit
def readout(h, q):
    """Compute readout: qS = sum_k(h[v,e,k] * q[k]) -> [BV, BE].

    Matrix-vector product reading from state with query vector.

    Args:
        h: [BV, BE, BK] state tensor.
        q: [BK] query vector (should be pre-scaled).
    Returns:
        [BV, BE] readout per (value_dim, expert) pair.
    """
    return dot_3d_axis2(h, q)


@triton.jit
def state_update(h, coef, k):
    """Update state: h_new = h + coef[:,:,None] * k[None,None,:].

    Rank-1 update per (v, e) pair.

    Args:
        h: [BV, BE, BK] current state.
        coef: [BV, BE] update coefficient (beta * responsibility * residual).
        k: [BK] key vector.
    Returns:
        [BV, BE, BK] updated state.
    """
    return h + coef[:, :, None] * k[None, None, :]
