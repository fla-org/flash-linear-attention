# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton
import triton.language as tl

from fla.ops.utils.op import exp2


@triton.jit
def chunk_delta_rule_bwd_dh_state(
    q,
    k,
    w,
    g,
    gk,
    do,
    dv,
    dv2,
    b_dh,
    o_t,
    o_v,
    last_idx,
    T,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    """Propagate a state gradient through one chunk, optionally saving the value gradient."""
    m_t = o_t < T
    m_v = o_v < V
    stride_w: tl.constexpr = H * K if USE_BG else HV * K
    if USE_G:
        b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
        b_g = tl.load(g + o_t * HV, mask=m_t, other=0.).to(tl.float32)
        b_g_last_exp = exp2(b_g_last)
        b_g_exp = exp2(b_g)
    b_do = tl.load(
        do + o_t[:, None] * (HV * V) + o_v[None, :],
        mask=m_t[:, None] & m_v[None, :],
        other=0.0,
    )
    b_gk_last = (tl.full((), 0, tl.float32), tl.full((), 0, tl.float32),
                 tl.full((), 0, tl.float32), tl.full((), 0, tl.float32))
    for i in tl.static_range(4):
        if i * 64 < K:
            o_k = i * 64 + tl.arange(0, 64)
            b_k = tl.load(
                k + o_t[:, None] * (H * K) + o_k[None, :],
                mask=m_t[:, None] & (o_k < K)[None, :],
                other=0.0,
            )
            if USE_GK:
                b_gki = tl.load(gk + last_idx * HV * K + o_k, mask=o_k < K, other=0.).to(tl.float32)
                b_gk_last = b_gk_last[:i] + (b_gki,) + b_gk_last[i + 1:]
            b_dhi = tl.trans(b_dh[i]) if STATE_V_FIRST else b_dh[i]
            if i == 0:
                b_dv = tl.dot(b_k, b_dhi.to(b_k.dtype))
            else:
                b_dv = tl.dot(b_k, b_dhi.to(b_k.dtype), b_dv)
    if USE_G:
        b_dv *= tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
    p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
    b_dv += tl.load(p_dv, mask=m_t[:, None] & m_v[None, :], other=0.)
    if SAVE_NEW_VALUE:
        p_dv2 = dv2 + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

    for i in tl.static_range(4):
        if i * 64 < K:
            o_k = i * 64 + tl.arange(0, 64)
            p_w = w + o_k[:, None] + o_t[None, :] * stride_w
            p_q = q + o_k[:, None] + o_t[None, :] * (H * K)
            mask = (o_k < K)[:, None] & m_t[None, :]
            if i == 0:
                b_w = tl.load(p_w, mask=mask, other=0.)
                b_q = tl.load(p_q, mask=mask, other=0.)
            else:
                b_q = tl.load(p_q, mask=mask, other=0.)
                b_w = tl.load(p_w, mask=mask, other=0.)
            b_dhi = b_dh[i]
            if USE_G:
                b_dhi *= b_g_last_exp
                b_q *= b_g_exp[None, :]
            if USE_GK:
                b_decay = exp2(b_gk_last[i])[None, :] if STATE_V_FIRST else exp2(b_gk_last[i])[:, None]
                b_dhi *= b_decay
            if USE_BG:
                b_update = tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
            else:
                b_update = tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
            b_dhi += tl.trans(b_update) if STATE_V_FIRST else b_update
            b_dh = b_dh[:i] + (b_dhi,) + b_dh[i + 1:]
    return b_dh
