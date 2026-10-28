# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton
import triton.language as tl

from fla.ops.utils.op import exp2


@triton.jit(do_not_specialize=['T'])
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
    o_t, o_v, last_idx = o_t.to(tl.int64), o_v.to(tl.int64), last_idx.to(tl.int64)
    m_t, m_v = o_t < T, o_v < V
    stride_w: tl.constexpr = H * K if USE_BG else HV * K
    if USE_G:
        b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
        b_g = tl.load(g + o_t * HV, mask=m_t, other=0.).to(tl.float32)
        b_g_last_exp = exp2(b_g_last)
        b_g_exp = exp2(b_g)
    p_do = do + o_t[:, None] * (HV * V) + o_v[None, :]
    b_do = tl.load(p_do, mask=m_t[:, None] & m_v[None, :], other=0.)
    b_gk_last = ()
    for i_k in tl.static_range(tl.cdiv(K, 64)):
        o_k = i_k * 64 + tl.arange(0, 64).to(tl.int64)
        p_k = k + o_t[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & (o_k < K)[None, :], other=0.)
        if USE_GK:
            p_gk = gk + last_idx * HV * K + o_k
            b_gk_last += (tl.load(p_gk, mask=o_k < K, other=0.).to(tl.float32),)
        b_dh_i = tl.trans(b_dh[i_k]) if STATE_V_FIRST else b_dh[i_k]
        if i_k == 0:
            b_dv = tl.dot(b_k, b_dh_i.to(b_k.dtype))
        else:
            b_dv = tl.dot(b_k, b_dh_i.to(b_k.dtype), b_dv)
    if USE_G:
        b_dv *= tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
    p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
    b_dv += tl.load(p_dv, mask=m_t[:, None] & m_v[None, :], other=0.)
    if SAVE_NEW_VALUE:
        p_dv2 = dv2 + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_dv2, b_dv.to(p_dv2.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

    b_dh_new = ()
    for i_k in tl.static_range(tl.cdiv(K, 64)):
        o_k = i_k * 64 + tl.arange(0, 64).to(tl.int64)
        m_k = o_k < K
        p_q = q + o_k[:, None] + o_t[None, :] * (H * K)
        p_w = w + o_k[:, None] + o_t[None, :] * stride_w
        b_q = tl.load(p_q, mask=m_k[:, None] & m_t[None, :], other=0.)
        b_w = tl.load(p_w, mask=m_k[:, None] & m_t[None, :], other=0.)
        b_dh_i = b_dh[i_k]
        if USE_G:
            b_dh_i *= b_g_last_exp
            b_q *= b_g_exp[None, :]
        if USE_GK:
            b_decay = exp2(b_gk_last[i_k])
            b_dh_i *= b_decay[None, :] if STATE_V_FIRST else b_decay[:, None]
        if USE_BG:
            # the DPLR recurrence adds the state update and receives pre-scaled queries
            b_update = tl.dot(b_q, b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
        else:
            b_update = tl.dot(b_q, b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        b_dh_i += tl.trans(b_update) if STATE_V_FIRST else b_update
        b_dh_new += (b_dh_i,)
    return b_dh_new
