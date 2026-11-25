# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

"""Gated DeltaNet forward: chunked CuTe DSL implementation (prep + recurrence).

Dispatch uses host-known shape values only (T, N, Hv), never device data, so
the call stays CUDA-graph capturable.  Long sequences with few heads use
context parallelism (pass 1 / combine / pass 2); the context budget keeps the
active pass-1 CTAs within one wave of the GPU.
"""

from fla.ops.gated_delta_rule.backends.qla_cute.gdn import run as _run

C = 64
LC_MIN = 16


def contexts_per_wave(hv, nchunks):
    """Contexts per sequence: pass 2 (2 Hv CTAs per context >= 1) fits one wave.
    Pass 1 is light since decaying heads only run a trailing window (gdn_recur)."""
    table = {
        8: 10 if nchunks >= 384 else 8,
        16: 5,
        24: 4 if nchunks >= 128 else 3,
        32: 3,
    }
    if hv in table:
        return table[hv]
    return 1 + SMS // (2 * max(hv, 1))


def context_length(t, n, hv):
    """Chunks per context, or 0 for the plain (whole-sequence) path."""
    nchunks = (t + C - 1) // C
    if n > 2:
        return 0
    if hv >= 32 and n > 1:
        return 0
    if n == 1 and nchunks <= 32:
        return 0
    p = contexts_per_wave(hv, nchunks)
    if p < 3:
        return 0
    lc_min = LC_MIN
    if n == 1 and (nchunks <= 32 and hv <= 16 or nchunks <= 64 and hv <= 8):
        lc_min = 8
    lc = max(lc_min, -(-nchunks // p))
    if -(-nchunks // lc) < 3:
        return 0
    return lc


SMS = 148


LC_TC_MIN = 27


def tc_context_length(t, n, hv, hq):
    """Chunks per context for context parallelism on the tcgen05 kernel, or 0.
    Pass 2 runs Hv (P - 1) CTAs (one per head and context), so P = 1 + SMS // Hv."""
    nchunks = (t + C - 1) // C
    if hv <= 8 and n <= 2 and 64 < nchunks <= 128 or 8 < hv <= 16 and n == 1 and 32 < nchunks <= 64:
        return 11
    if n > 2:
        per = nchunks // n
        if 2 * hv * n > SMS or per < 128 or hv > 8:
            return 0
        return max(LC_MIN, -(-per // (1 + SMS // (hv * n))))
    p = 1 + SMS // max(hv, 1)
    if p < 3 or (p == 3 and (n > 1 or nchunks < 128)):
        return 0

    q = p + n - 1 if hv < 48 and nchunks >= 384 and not 8 < hv <= 16 else p
    if hv <= 8 and (nchunks >= 384 or n == 1):
        q = min(q, 16 if nchunks >= 384 else 14)
    elif 8 < hv <= 16 and nchunks < 384:
        q = min(q, 8)
    elif 8 < hv <= 16 and hq == hv:
        q = 8
    elif 16 < hv <= 24 and nchunks < 256:
        q = min(q, 6)
    elif 16 < hv <= 24 and n == 2 and nchunks < 384:
        q = 8
    elif 24 < hv <= 32 and n == 1 and nchunks < 256:
        q = min(q, 4)
    lc = max(LC_MIN, -(-nchunks // q))
    if 24 < hv <= 32 and n == 2 and 256 <= nchunks < 384:
        lc = 3 * nchunks // 16
    if -(-nchunks // lc) < 3:
        return 0
    short_tc = n == 1 and 16 < hv <= 32 and nchunks >= 64
    if lc < LC_TC_MIN and nchunks < 256 and (nchunks < 128 or hv < 16) and not short_tc:
        return 0
    if n == 2 and hv >= 48 and lc < 128 and nchunks < 256:
        return 0
    return lc


def run(q, k, v, g, beta, scale, initial_state=None, cu_seqlens=None):
    B, T, HQ, _ = q.shape
    HV = v.shape[-2]
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    lct = tc_context_length(B * T, N, HV, HQ)
    if lct:
        pk = HV >= 48 or (HV >= 32 and B * T < 256 * C)
        cpick = 3 if N == 2 and cu_seqlens is not None and pk else 0
        return _run(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            cu_seqlens,
            lc=lct,
            tccp=True,
            cpick=cpick,
        )

    lc = context_length(B * T, N, HV)
    return _run(q, k, v, g, beta, scale, initial_state, cu_seqlens, lc=lc, tc=lc == 0)
