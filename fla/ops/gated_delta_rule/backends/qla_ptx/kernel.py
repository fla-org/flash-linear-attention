# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

from fla.ops.gated_delta_rule.backends.qla_ptx.gdn import run as chunked
from fla.ops.gated_delta_rule.backends.qla_ptx.gdn_cp import run as chunked_cp
from fla.ops.gated_delta_rule.backends.qla_ptx.gdn_cp import run_plain as chunked_plain
from fla.ops.gated_delta_rule.backends.qla_ptx.gdn_cp import sm_count

CONTEXTS = {8: 14, 16: 8, 24: 6, 32: 4, 48: 3, 64: 2}
SHORT_CONTEXTS = {2048: {8: 8, 16: 4, 24: 3, 32: 3}, 4096: {8: 8, 16: 6, 24: 5}, 8192: {8: 10, 16: 6, 24: 5}, 16384: {8: 12}}


def context_budget(t, n, hv):
    for length, budgets in SHORT_CONTEXTS.items():
        if n == 1 and length // 2 < t <= length and (hv in budgets):
            return budgets[hv]
    budget = CONTEXTS.get(hv, 0)
    if hv == 8 and 2 <= n <= 4 and (t >= 32768):
        budget = 16
    if hv == 8 and n == 2 and (t <= 8192):
        budget = 12
    if budget <= n or (n > 2 and budget < 2 * n):
        return 0
    if t // n <= 2048 or (hv <= 8 and t <= 4096):
        return 0
    return budget


def run(q, k, v, g, beta, scale, initial_state=None, cu_seqlens=None):
    return dispatch(q, k, v, g, beta, scale, initial_state, cu_seqlens)


def dispatch(q, k, v, g, beta, scale, initial_state, cu_seqlens):
    if v.shape[2] % q.shape[2]:
        raise ValueError("The number of value heads must be divisible by q/k heads")
    args = (q, k, v, g, beta, scale, initial_state, cu_seqlens)
    t, hv = (q.shape[1], v.shape[2])
    if cu_seqlens is not None and q.shape[-1] == 128 and (v.shape[-1] == 128):
        n = cu_seqlens.numel() - 1
        hq = q.shape[2]
        if n == 4 and t >= 32768 and (hq == hv == 16 or (hv == 24 and hq == 8)):
            return chunked_plain(*args)
        if n == 1 and 2048 < t <= 4096 and (hv >= 64) and (hv >= 3 * hq):
            return chunked_plain(*args)
        budget = context_budget(t, n, hv)
        if t <= 2048 and hv == 32 and (q.shape[2] < hv):
            budget = 0
        if budget:
            overlap = 0
            spin = hv * (2 * budget - 2 * n - 1)
            sms = sm_count(q.device)
            if t >= 4096 and 3 * spin <= 2 * sms or (t >= 16384 and n >= 4 and (spin < sms)):
                overlap = 2
            elif t >= 16384 and (hv >= 16 or t >= 32768) or (t >= 2048 and q.shape[2] == hv) or (t >= 8192 and hv >= 24):
                overlap = 1
            return chunked_cp(*args, jmax=budget, overlap=overlap)
        wide = n * hv > 16 or (hv >= 16 and t // n <= 4096)
        if wide and (not (hv <= 16 and n <= 2 and (t // n >= 8192))):
            if n * hv <= 64 and hq == hv and ((t // 64 + n) * hq >= 1024):
                return chunked_plain(*args)
            if n * hv <= 64 and hv >= 3 * hq:
                return chunked_plain(*args)
            return chunked(*args)
        if t >= 4096 or (t >= 2048 and hv <= 32):
            raise ValueError("Input configuration was not exported")
    raise ValueError("Input configuration was not exported")
