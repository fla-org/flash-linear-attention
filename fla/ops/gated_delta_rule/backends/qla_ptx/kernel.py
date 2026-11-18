# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
"""Host dispatch for the exported prefill specializations."""

from fla.ops.gated_delta_rule.backends.qla_ptx.chunk_async import run as short
from fla.ops.gated_delta_rule.backends.qla_ptx.context import run as context
from fla.ops.gated_delta_rule.backends.qla_ptx.context_packed import run as packed
from fla.ops.gated_delta_rule.backends.qla_ptx.context_sensitivity_zero import run as sensitivity


def run(q, k, v, g, beta, scale, initial_state=None, cu_seqlens=None):
    args = (q, k, v, g, beta, scale, initial_state, cu_seqlens)
    t, hv = q.shape[1], v.shape[2]
    if t <= 209:
        return short(*args)
    if cu_seqlens is not None:
        if t >= 16384:
            return packed(*args, length=384 if initial_state is not None else 512, workers=128)
        if hv >= 32:
            return packed(*args)
        return context(*args, length=512, groups=2, warps=2)
    if t >= 32768:
        return sensitivity(*args, workers=128)
    if hv == 4:
        return sensitivity(*args, length=512, groups=2, warps=4, workers=128)
    return sensitivity(
        *args,
        length=512,
        groups=2 if hv <= 8 or initial_state is not None else 4,
        warps=8 if hv <= 8 or initial_state is not None else 4,
    )
