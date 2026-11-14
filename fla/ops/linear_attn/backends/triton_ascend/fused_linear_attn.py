# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# ruff: noqa: E501

"""Un-gated ``fused_chunk`` / ``fused_recurrent`` forward kernels for Ascend.

``fused_chunk`` reuses the upstream FLA kernel with gate inputs passed as
``None``; on Ascend its forward launch must disable auto-multi-buffer
(``multibuffer=False``), otherwise fp32 shapes can produce wrong results.
``fused_recurrent`` is a gate-free port.  Varlen is handled by the same kernels
through ``cu_seqlens`` as in FLA.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .linear_attn import normalize_with_z_state
from .utils import empty_attn_outputs, exp, exp2, normalize_chunk_size, zero_acc


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_chunk_fwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    o_buf,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    total = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)

    o_i = tl.arange(0, BT)

    if USE_G_GAMMA:
        # decay rate given the head index
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (o_i + 1)
        b_g_last = b_gamma * BT
        b_gq = exp2(b_g)
        b_gk = exp2(b_g_last - b_g)
        b_gn = exp2(b_g_last)

    # shape BT x BT
    m_s = o_i[:, None] >= o_i[None, :]

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    v = v + (bos * H + i_h) * V
    o_buf = o_buf + (i_k * total + bos).to(tl.int64) * H * V + i_h * V

    # shape BK x BV
    b_h = zero_acc(BK, BV)
    if USE_INITIAL_STATE:
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        p_h = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0.0).to(
            tl.float32
        )

    for i_t in range(0, NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t_int64 * BT + tl.arange(0, BT)
        m_t = o_t < T
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
        p_k = k + o_k[:, None] + o_t[None, :] * (H * K)
        p_v = v + o_t[:, None] * (H * V) + o_v[None, :]
        p_o = o_buf + o_t[:, None] * (H * V) + o_v[None, :]

        # shape BT x BK
        b_q = tl.load(p_q, mask=m_t[:, None] & (o_k < K)[None, :], other=0.0)
        b_q = (b_q * scale).to(b_q.dtype)
        # shape BK x BT
        b_k = tl.load(p_k, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0)
        # shape BT x BV
        b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
        last_idx = min(i_t_int64 * BT + BT, T) - 1

        # shape BT x BT
        b_s = tl.dot(b_q, b_k)

        # scalar decay
        if USE_G:
            p_g = g + (bos + o_t) * H + i_h
            b_g = tl.load(p_g, mask=(o_t < T), other=0.0)
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)

            b_gq = exp2(b_g)
            b_gk = exp2(b_g_last - b_g)
            b_gn = exp2(b_g_last)
        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t_int64 * BT)
            b_gk = exp2(b_g_last - b_g)
            b_gn = exp2(b_g_last)
        if USE_G or USE_G_GAMMA:
            b_gs = tl.where(m_s & m_t, exp2(b_g[:, None] - b_g[None, :]), 0)
            # shape BT x BT
            b_s *= b_gs
            # shape BT x BV
            b_o = (
                tl.dot(b_s.to(b_q.dtype), b_v)
                + tl.dot(b_q, b_h.to(b_q.dtype)) * b_gq[:, None]
            )
            b_v = (b_v * b_gk[:, None]).to(b_v.dtype)
            b_h *= b_gn
        else:
            # shape BT x BT
            b_s *= m_s & m_t
            # shape BT x BV
            b_o = tl.dot(b_s.to(b_q.dtype), b_v) + tl.dot(b_q, b_h.to(b_q.dtype))

        b_h = tl.dot(b_k, b_v, b_h)

        tl.store(
            p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & (o_v < V)[None, :]
        )

    if STORE_FINAL_STATE:
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(
            p_ht,
            b_h.to(p_ht.dtype.element_ty),
            mask=(o_k[:, None] < K) & (o_v[None, :] < V),
        )


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_INITIAL_STATE": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE": lambda args: args["dht"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_chunk_bwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    do,
    dq,
    dk,
    dv,
    dg,
    h0,
    dht,
    dh0,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    total = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)
    NV = tl.cdiv(V, BV)

    o_i = tl.arange(0, BT)
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (o_i + 1)
        b_g_last = b_gamma * BT
        b_gq = exp2(b_g)
        b_gk = exp2(b_g_last - b_g)
        b_gn = exp2(b_g_last)

    m_s = o_i[:, None] >= o_i[None, :]

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    v = v + (bos * H + i_h) * V
    do = do + (bos * H + i_h) * V
    dq = dq + (i_v * total + bos).to(tl.int64) * H * K + i_h * K
    dk = dk + (i_v * total + bos).to(tl.int64) * H * K + i_h * K
    dv = dv + (i_k * total + bos).to(tl.int64) * H * V + i_h * V

    # shape BV x BK
    b_h = zero_acc(BV, BK)
    if USE_INITIAL_STATE:
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        p_h = h0 + i_nh * K * V + o_v[:, None] + o_k[None, :] * V
        b_h = tl.load(p_h, mask=(o_v[:, None] < V) & (o_k[None, :] < K), other=0.0).to(
            tl.float32
        )

    for i_t in range(0, NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t_int64 * BT + tl.arange(0, BT)
        m_t = o_t < T
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
        p_k = k + o_t[:, None] * (H * K) + o_k[None, :]
        p_v = v + o_v[:, None] + o_t[None, :] * (H * V)
        p_do = do + o_t[:, None] * (H * V) + o_v[None, :]
        p_dq = dq + o_t[:, None] * (H * K) + o_k[None, :]

        # shape BT x BK
        b_k = tl.load(p_k, mask=m_t[:, None] & (o_k < K)[None, :], other=0.0)
        # shape BV x BT
        b_v = tl.load(p_v, mask=(o_v[:, None] < V) & m_t[None, :], other=0.0)
        # shape BT x BV
        b_do = tl.load(p_do, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
        last_idx = min(i_t_int64 * BT + BT, T) - 1

        # shape BT x BT
        b_ds = tl.dot(b_do, b_v) * scale

        # scalar decay
        if USE_G:
            p_g = g + (bos + o_t) * H + i_h
            b_g = tl.load(p_g, mask=(o_t < T), other=0.0)
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)

            b_gq = exp2(b_g)
            b_gk = exp2(b_g_last - b_g)
            b_gn = exp2(b_g_last)

            p_dg = dg + ((i_k * NV + i_v) * total + (bos + o_t)).to(tl.int64) * H + i_h
            # shape BT x BT
            b_gs = tl.where(m_s & m_t, exp2(b_g[:, None] - b_g[None, :]), 0)
            b_ds = b_ds * b_gs
            # shape BT x BK
            b_q = tl.load(p_q, mask=m_t[:, None] & (o_k < K)[None, :], other=0.0)
            b_dq = tl.dot(b_ds.to(b_k.dtype), b_k) + tl.dot(
                (b_do * b_gq[:, None] * scale).to(b_k.dtype), b_h.to(b_k.dtype)
            )
            # shape BT
            b_dg_t = tl.sum(b_q * b_dq, 1)
            tl.store(p_dg, b_dg_t.to(p_dg.dtype.element_ty), mask=m_t)
            # shape BV x BK
            b_h = b_h * b_gn + tl.dot(b_v, (b_k * b_gk[:, None]).to(b_k.dtype))

        elif USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t_int64 * BT)
            b_gk = exp2(b_g_last - b_g)
            b_gn = exp2(b_g_last)

            # shape BT x BT
            b_gs = tl.where(m_s & m_t, exp2(b_g[:, None] - b_g[None, :]), 0)
            b_ds = b_ds * b_gs
            # shape BT x BK
            b_q = tl.load(p_q, mask=m_t[:, None] & (o_k < K)[None, :], other=0.0)
            b_dq = tl.dot(b_ds.to(b_k.dtype), b_k) + tl.dot(
                (b_do * b_gq[:, None] * scale).to(b_k.dtype), b_h.to(b_k.dtype)
            )
            # shape BV x BK
            b_h = b_h * b_gn + tl.dot(b_v, (b_k * b_gk[:, None]).to(b_k.dtype))

        else:
            # shape BT x BT
            b_ds *= m_s & m_t
            # shape BT x BK
            b_dq = tl.dot(b_ds.to(b_k.dtype), b_k) + tl.dot(
                (b_do * scale).to(b_k.dtype), b_h.to(b_k.dtype)
            )
            # shape BV x BK
            b_h = tl.dot(b_v, b_k, b_h)

        tl.store(
            p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_t[:, None] & (o_k < K)[None, :]
        )

    # shape BK x BV
    b_dh = zero_acc(BK, BV)
    if USE_FINAL_STATE:
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        p_dh = dht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_dh += tl.load(
            p_dh, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0.0
        ).to(tl.float32)

    if USE_G:
        b_dg = tl.zeros([BT], dtype=tl.float32)
        b_dg_last = tl.sum(tl.trans(b_h) * b_dh)

    # sync threads
    b_h = None
    tl.debug_barrier()

    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t_int64 * BT + tl.arange(0, BT)
        m_t = o_t < T
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        p_q = q + o_k[:, None] + o_t[None, :] * (H * K)
        p_k = k + o_t[:, None] * (H * K) + o_k[None, :]
        p_v = v + o_t[:, None] * (H * V) + o_v[None, :]
        p_do = do + o_t[:, None] * (H * V) + o_v[None, :]
        p_dk = dk + o_t[:, None] * (H * K) + o_k[None, :]
        p_dv = dv + o_t[:, None] * (H * V) + o_v[None, :]
        # shape BK x BT
        b_q = tl.load(p_q, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0)
        # shape BT x BK
        b_k = tl.load(p_k, mask=m_t[:, None] & (o_k < K)[None, :], other=0.0)
        # shape BT x BV
        b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
        b_do = tl.load(p_do, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
        last_idx = min(i_t_int64 * BT + BT, T) - 1

        # shape BT x BT
        b_s = tl.dot(b_k, b_q)
        b_ds = tl.dot(b_v, tl.trans(b_do))

        if USE_G:
            p_g = g + (bos + o_t) * H + i_h
            p_dg = dg + ((i_k * NV + i_v) * total + (bos + o_t)).to(tl.int64) * H + i_h
            b_g = tl.load(p_g, mask=m_t, other=0.0)
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)

            b_gq = exp2(b_g)
            b_gk = exp2(b_g_last - b_g)
            b_gn = exp2(b_g_last)
            b_gs = (
                tl.trans(
                    tl.where(
                        m_s & (m_t[:, None] & m_t), exp2(b_g[:, None] - b_g[None, :]), 0
                    )
                )
                * scale
            )

            b_s = b_s * b_gs
            b_ds = b_ds * b_gs

            # shape BT x BK
            b_dk = (
                tl.dot(b_ds.to(b_k.dtype), tl.trans(b_q))
                + tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype)) * b_gk[:, None]
            )

            # shape BT
            b_dg_t = tl.where(
                m_t, tl.load(p_dg, mask=m_t, other=0.0) - tl.sum(b_k * b_dk, 1), 0
            )
            b_dg_last += tl.sum(b_dg_t, 0)
            b_dg = b_dg_last + b_dg_t - tl.cumsum(b_dg_t, 0)

            # shape BT x BV
            b_dv = (
                tl.dot(b_s.to(b_do.dtype), b_do)
                + tl.dot(b_k, b_dh.to(b_k.dtype)) * b_gk[:, None]
            )
            # shape BK x BV
            b_dh = b_dh * b_gn + tl.dot(
                b_q, (b_do * b_gq[:, None] * scale).to(b_do.dtype)
            )

            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_t)

        elif USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t_int64 * BT)
            b_gk = exp2(b_g_last - b_g)
            b_gn = exp2(b_g_last)
            b_gs = (
                tl.trans(
                    tl.where(
                        m_s & (m_t[:, None] & m_t), exp2(b_g[:, None] - b_g[None, :]), 0
                    )
                )
                * scale
            )

            b_s = b_s * b_gs
            b_ds = b_ds * b_gs

            b_dk = (
                tl.dot(b_ds.to(b_k.dtype), tl.trans(b_q))
                + tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype)) * b_gk[:, None]
            )
            # shape BT x BV
            b_dv = (
                tl.dot(b_s.to(b_do.dtype), b_do)
                + tl.dot(b_k, b_dh.to(b_k.dtype)) * b_gk[:, None]
            )
            # shape BK x BV
            b_dh = b_dh * b_gn + tl.dot(
                b_q, (b_do * b_gq[:, None] * scale).to(b_do.dtype)
            )

        else:
            mask = tl.trans(m_s & (m_t[:, None] & m_t))
            b_s = tl.where(mask, b_s * scale, 0).to(b_do.dtype)
            b_ds = tl.where(mask, b_ds * scale, 0).to(b_q.dtype)

            b_dk = tl.dot(b_ds, tl.trans(b_q)) + tl.dot(
                b_v, tl.trans(b_dh).to(b_v.dtype)
            )
            # shape BT x BV
            b_dv = tl.dot(b_s.to(b_do.dtype), b_do) + tl.dot(b_k, b_dh.to(b_k.dtype))
            # shape BK x BV
            b_dh = tl.dot(b_q, (b_do * scale).to(b_do.dtype), b_dh)

        tl.store(
            p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_t[:, None] & (o_k < K)[None, :]
        )
        tl.store(
            p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None] & (o_v < V)[None, :]
        )

    if USE_INITIAL_STATE:
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        p_dh0 = dh0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(
            p_dh0,
            b_dh.to(p_dh0.dtype.element_ty),
            mask=(o_k[:, None] < K) & (o_v[None, :] < V),
        )


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["B", "T"])
def fused_recurrent_fwd_kernel(
    q,
    k,
    v,
    o_buf,
    h0,
    ht,
    cu_seqlens,
    scale,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    NV, NK = tl.cdiv(V, BV), tl.cdiv(K, BK)
    i_v = pid % NV
    i_k = (pid // NV) % NK
    i_nh = pid // (NV * NK)
    i_n, i_h = i_nh // H, i_nh % H

    total = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]

    p_q = q + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_k = k + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_v = v + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    p_o = (
        o_buf
        + ((i_k * total + bos) + ((T - 1) if REVERSE else 0)) * H * V
        + i_h * V
        + o_v
    )

    b_h = zero_acc(BK, BV)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0.0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=m_k, other=0.0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=m_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0.0).to(tl.float32)
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)
        p_q += (-1 if REVERSE else 1) * H * K
        p_k += (-1 if REVERSE else 1) * H * K
        p_v += (-1 if REVERSE else 1) * H * V
        p_o += (-1 if REVERSE else 1) * H * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=m_h)


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_INITIAL_STATE_GRADIENT": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["B", "T"])
def fused_recurrent_bwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    gk,
    gv,
    o_buf,
    h0,
    do,
    dq,
    dk,
    dv,
    dg,
    dgk,
    dgv,
    dht,
    dh0,
    cu_seqlens,
    scale,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STATE_V_FIRST: tl.constexpr = False,
):
    pid = tl.program_id(0)
    NV, NK = tl.cdiv(V, BV), tl.cdiv(K, BK)
    i_v, i_k, i_nh = (
        (pid % NV).to(tl.int64),
        ((pid // NV) % NK).to(tl.int64),
        (pid // (NV * NK)).to(tl.int64),
    )
    i_n, i_h = i_nh // H, i_nh % H

    total = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    NV = tl.cdiv(V, BV)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    if STATE_V_FIRST:
        m_h = m_v[:, None] & m_k[None, :]
    else:
        m_h = m_k[:, None] & m_v[None, :]

    p_k = k + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_v = v + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    p_do = do + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    p_dq = (
        dq + ((i_v * total + bos) + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    )
    if USE_G:
        p_g = g + (bos + ((T - 1) if REVERSE else 0)) * H + i_h
    if USE_GK:
        p_gk = gk + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    if USE_GV:
        p_gv = gv + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    if USE_G_GAMMA:
        b_g_gamma = tl.load(g_gamma + i_h)

    if STATE_V_FIRST:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0 = h0 + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=m_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * exp(b_g)
        if USE_G_GAMMA:
            b_h = b_h * exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
            if STATE_V_FIRST:
                b_h = b_h * exp(b_gk[None, :])
            else:
                b_h = b_h * exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            if STATE_V_FIRST:
                b_h = b_h * exp(b_gv[:, None])
            else:
                b_h = b_h * exp(b_gv[None, :])
        if STATE_V_FIRST:
            b_h += b_v[:, None] * b_k[None, :]
            b_dq = tl.sum(b_h * b_do[:, None], axis=0) * scale
        else:
            b_h += b_k[:, None] * b_v[None, :]
            b_dq = tl.sum(b_h * b_do[None, :], axis=1) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_k)

        p_k += (-1 if REVERSE else 1) * H * K
        p_v += (-1 if REVERSE else 1) * H * V
        p_do += (-1 if REVERSE else 1) * H * V
        p_dq += (-1 if REVERSE else 1) * H * K
        if USE_G:
            p_g += (-1 if REVERSE else 1) * H
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * H * K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * H * V

    # sync threads
    tl.debug_barrier()

    p_q = q + (bos + ((T - 1) if not REVERSE else 0)) * H * K + i_h * K + o_k
    p_k = k + (bos + ((T - 1) if not REVERSE else 0)) * H * K + i_h * K + o_k
    p_v = v + (bos + ((T - 1) if not REVERSE else 0)) * H * V + i_h * V + o_v

    p_do = do + (bos + ((T - 1) if not REVERSE else 0)) * H * V + i_h * V + o_v
    p_dq = (
        dq
        + ((i_v * total + bos) + ((T - 1) if not REVERSE else 0)) * H * K
        + i_h * K
        + o_k
    )
    p_dk = (
        dk
        + ((i_v * total + bos) + ((T - 1) if not REVERSE else 0)) * H * K
        + i_h * K
        + o_k
    )
    p_dv = (
        dv
        + ((i_k * total + bos) + ((T - 1) if not REVERSE else 0)) * H * V
        + i_h * V
        + o_v
    )
    if USE_G:
        p_g = g + (bos + ((T - 1) if not REVERSE else 0)) * H + i_h
        p_dg = (
            dg
            + ((i_k * NV + i_v) * total + bos + ((T - 1) if not REVERSE else 0)) * H
            + i_h
        )
    if USE_GK:
        p_gk = gk + (bos + ((T - 1) if not REVERSE else 0)) * H * K + i_h * K + o_k
        p_dgk = (
            dgk
            + ((i_v * total + bos) + ((T - 1) if not REVERSE else 0)) * H * K
            + i_h * K
            + o_k
        )
    if USE_GV:
        p_o = o_buf + (bos + ((T - 1) if not REVERSE else 0)) * H * V + i_h * V + o_v
        p_gv = gv + (bos + ((T - 1) if not REVERSE else 0)) * H * V + i_h * V + o_v
        p_dgv = (
            dgv
            + ((i_k * total + bos) + ((T - 1) if not REVERSE else 0)) * H * V
            + i_h * V
            + o_v
        )

    if STATE_V_FIRST:
        b_dh = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dht = dht + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_dht = dht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_dh += tl.load(p_dht, mask=m_h, other=0).to(tl.float32)

    if USE_G:
        b_dg = tl.sum(b_h * b_dh)
    if USE_GK:
        if STATE_V_FIRST:
            b_dgk = tl.sum(b_h * b_dh, 0)
        else:
            b_dgk = tl.sum(b_h * b_dh, 1)
    if USE_GV:
        if STATE_V_FIRST:
            b_dgv = tl.sum(b_h * b_dh, 1)
        else:
            b_dgv = tl.sum(b_h * b_dh, 0)

    for _ in range(T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=m_v, other=0).to(tl.float32)
        if STATE_V_FIRST:
            b_dh += b_do[:, None] * (b_q * scale)[None, :]
            b_dk = tl.sum(b_dh * b_v[:, None], axis=0)
            b_dv = tl.sum(b_dh * b_k[None, :], axis=1)
        else:
            b_dh += (b_q * scale)[:, None] * b_do[None, :]
            b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
            b_dv = tl.sum(b_dh * b_k[:, None], axis=0)

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_dq = tl.load(p_dq, mask=m_k, other=0).to(tl.float32)
            b_dg += tl.sum(b_q * b_dq - b_k * b_dk)
            b_dh *= exp(b_g)
            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty))
        if USE_G_GAMMA:
            b_dh *= exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
            b_dq = tl.load(p_dq, mask=m_k, other=0).to(tl.float32)
            b_dgk += b_q * b_dq - b_k * b_dk
            if STATE_V_FIRST:
                b_dh *= exp(b_gk)[None, :]
            else:
                b_dh *= exp(b_gk)[:, None]
            tl.store(p_dgk, b_dgk.to(p_dgk.dtype.element_ty), mask=m_k)
        if USE_GV:
            b_o = tl.load(p_o, mask=m_v, other=0).to(tl.float32)
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            if i_k == 0:
                b_dgv += b_o * b_do
            b_dgv -= b_v * b_dv
            if STATE_V_FIRST:
                b_dh *= exp(b_gv)[:, None]
            else:
                b_dh *= exp(b_gv)[None, :]
            tl.store(p_dgv, b_dgv.to(p_dgv.dtype.element_ty), mask=m_v)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_v)

        p_q += (1 if REVERSE else -1) * H * K
        p_k += (1 if REVERSE else -1) * H * K
        p_v += (1 if REVERSE else -1) * H * V

        p_do += (1 if REVERSE else -1) * H * V
        p_dq += (1 if REVERSE else -1) * H * K
        p_dk += (1 if REVERSE else -1) * H * K
        p_dv += (1 if REVERSE else -1) * H * V
        if USE_G:
            p_g += (1 if REVERSE else -1) * H
            p_dg += (1 if REVERSE else -1) * H
        if USE_GK:
            p_gk += (1 if REVERSE else -1) * H * K
            p_dgk += (1 if REVERSE else -1) * H * K
        if USE_GV:
            p_o += (1 if REVERSE else -1) * H * V
            p_gv += (1 if REVERSE else -1) * H * V
            p_dgv += (1 if REVERSE else -1) * H * V

    if STORE_INITIAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dh0 = dh0 + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_dh0 = dh0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=m_h)


def _fused_chunk_fwd(
    q,
    k,
    v,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    chunk_size,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BT = chunk_size
    BK = min(max(triton.next_power_of_2(K), 16), 64)
    BV = min(max(triton.next_power_of_2(V), 16), 64)
    NK = triton.cdiv(K, BK)

    if NK > 1:
        o_buf = torch.empty(NK, *v.shape, device=v.device, dtype=torch.float32)
    else:
        o_buf = torch.empty_like(v, dtype=torch.float32)
    ht = (
        torch.empty(N, H, K, V, device=q.device, dtype=torch.float32)
        if output_final_state
        else None
    )
    grid = (triton.cdiv(V, BV), NK, N * H)
    fused_chunk_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=None,
        g_gamma=None,
        o_buf=o_buf,
        h0=initial_state,
        ht=ht,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=3,
        multibuffer=False,
    )
    if NK > 1:
        o_buf = o_buf.sum(0)
    return o_buf, ht


def _fused_recurrent_fwd(
    q,
    k,
    v,
    scale,
    initial_state,
    output_final_state,
    reverse,
    cu_seqlens,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    o_buf = torch.empty(NK, *v.shape, device=v.device, dtype=torch.float32)
    ht = (
        torch.empty(N, H, K, V, device=q.device, dtype=torch.float32)
        if output_final_state
        else None
    )
    grid = (NV * NK * N * H,)
    fused_recurrent_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o_buf=o_buf,
        h0=initial_state,
        ht=ht,
        cu_seqlens=cu_seqlens,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REVERSE=reverse,
        num_warps=4,
        num_stages=2,
    )
    o_buf = o_buf.sum(0)
    return o_buf, ht


def _fused_chunk_bwd(q, k, v, do, initial_state, dht, scale, cu_seqlens, chunk_size):
    do = do.contiguous()
    if dht is not None:
        dht = dht.contiguous()
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BT = chunk_size
    BK = min(max(triton.next_power_of_2(K), 16), 64)
    BV = min(max(triton.next_power_of_2(V), 16), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    dq = (
        torch.empty(NV, *q.shape, device=q.device, dtype=torch.float32)
        if NV > 1
        else torch.empty_like(q, dtype=torch.float32)
    )
    dk = (
        torch.empty(NV, *k.shape, device=k.device, dtype=torch.float32)
        if NV > 1
        else torch.empty_like(k, dtype=torch.float32)
    )
    dv = (
        torch.empty(NK, *v.shape, device=v.device, dtype=torch.float32)
        if NK > 1
        else torch.empty_like(v, dtype=torch.float32)
    )
    dh0 = torch.empty_like(initial_state) if initial_state is not None else None

    grid = (NV, NK, N * H)
    fused_chunk_bwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=None,
        g_gamma=None,
        do=do,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=None,
        h0=initial_state,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=3,
        multibuffer=False,
        enable_auto_bind_sub_block=False,
    )
    dq = dq.sum(0) if NV > 1 else dq
    dk = dk.sum(0) if NV > 1 else dk
    dv = dv.sum(0) if NK > 1 else dv
    return dq, dk, dv, dh0


class FusedChunkLinearAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_size,
    ):
        o_buf, final_state = _fused_chunk_fwd(
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.initial_state = initial_state
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_size = chunk_size
        return o_buf.to(q.dtype), final_state

    @staticmethod
    def backward(ctx, do, dht):
        do = do.contiguous()
        if dht is not None:
            dht = dht.contiguous()
        q, k, v = ctx.saved_tensors
        dq, dk, dv, dh0 = _fused_chunk_bwd(
            q=q,
            k=k,
            v=v,
            do=do,
            initial_state=ctx.initial_state,
            dht=dht,
            scale=ctx.scale,
            cu_seqlens=ctx.cu_seqlens,
            chunk_size=ctx.chunk_size,
        )
        grads = (dq, dk, dv, None, dh0, None, None, None)
        return grads


def _fused_recurrent_bwd(q, k, v, do, initial_state, dht, scale, reverse, cu_seqlens):
    do = do.contiguous()
    if dht is not None:
        dht = dht.contiguous()
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    dq = torch.empty(NV, *q.shape, device=q.device, dtype=torch.float32)
    dk = torch.empty(NV, *k.shape, device=k.device, dtype=torch.float32)
    dv = torch.empty(NK, *v.shape, device=v.device, dtype=torch.float32)
    dh0 = torch.empty_like(initial_state) if initial_state is not None else None

    grid = (NV * NK * N * H,)
    fused_recurrent_bwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=None,
        g_gamma=None,
        gk=None,
        gv=None,
        o_buf=None,
        h0=initial_state,
        do=do,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=None,
        dgk=None,
        dgv=None,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REVERSE=reverse,
        USE_G=False,
        USE_G_GAMMA=False,
        USE_GK=False,
        USE_GV=False,
        STATE_V_FIRST=False,
        multibuffer=False,
        num_warps=4,
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    return dq, dk, dv, dh0


class FusedRecurrentLinearAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        scale,
        initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
    ):
        o_buf, final_state = _fused_recurrent_fwd(
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.initial_state = initial_state
        ctx.reverse = reverse
        ctx.cu_seqlens = cu_seqlens
        return o_buf.to(q.dtype), final_state

    @staticmethod
    def backward(ctx, do, dht):
        do = do.contiguous()
        if dht is not None:
            dht = dht.contiguous()
        q, k, v = ctx.saved_tensors
        dq, dk, dv, dh0 = _fused_recurrent_bwd(
            q=q,
            k=k,
            v=v,
            do=do,
            initial_state=ctx.initial_state,
            dht=dht,
            scale=ctx.scale,
            reverse=ctx.reverse,
            cu_seqlens=ctx.cu_seqlens,
        )
        grads = (dq, dk, dv, None, dh0, None, None, None)
        return grads


def _validate_varlen(q, cu_seqlens, kv_init):
    if cu_seqlens is None:
        return
    if q.shape[0] != 1:
        raise ValueError(
            "The batch size is expected to be 1 rather than "
            f"{q.shape[0]} when using `cu_seqlens`."
        )
    if kv_init is not None and kv_init.shape[0] != len(cu_seqlens) - 1:
        raise ValueError(
            "The number of initial states is expected to be equal to the "
            f"number of input sequences, i.e., {len(cu_seqlens) - 1} rather "
            f"than {kv_init.shape[0]}."
        )


@torch.compiler.disable
def fused_chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | tuple | None = None,
    output_final_state: bool = False,
    normalize: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | tuple | None]:
    if normalize and isinstance(initial_state, tuple):
        kv_init, z_init = initial_state
    else:
        kv_init, z_init = initial_state, None

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if isinstance(kv_init, torch.Tensor):
        kv_init = kv_init.contiguous()
    if z_init is not None:
        z_init = z_init.contiguous()
    _validate_varlen(q, cu_seqlens, kv_init)

    if scale is None:
        scale = q.shape[-1] ** -0.5
    chunk_size = min(64, max(16, triton.next_power_of_2(q.shape[1])))
    # Keep the on-chip tile within the 950 UB budget. 64 is the largest
    # locally verified internal block.
    chunk_size = normalize_chunk_size(chunk_size)

    if q.shape[1] == 0:
        num_seqs = 1 if cu_seqlens is None else len(cu_seqlens) - 1
        o, final_state, z_state = empty_attn_outputs(
            q, k, v, kv_init, z_init, output_final_state, num_seqs
        )
        return (o, (final_state, z_state)) if normalize else (o, final_state)

    if not normalize:
        return FusedChunkLinearAttnFunction.apply(
            q,
            k,
            v,
            scale,
            kv_init,
            output_final_state,
            cu_seqlens,
            chunk_size,
        )

    o_buf, final_state = FusedChunkLinearAttnFunction.apply(
        q,
        k,
        v,
        scale,
        kv_init,
        output_final_state,
        cu_seqlens,
        chunk_size,
    )
    o_buf, z_state = normalize_with_z_state(
        o_buf, q, k, scale, z_init, reverse=False, cu_seqlens=cu_seqlens
    )
    return o_buf, (final_state, z_state)


@torch.compiler.disable
def fused_recurrent_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | tuple | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    normalize: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | tuple | None]:
    if normalize and isinstance(initial_state, tuple):
        kv_init, z_init = initial_state
    else:
        kv_init, z_init = initial_state, None

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if isinstance(kv_init, torch.Tensor):
        kv_init = kv_init.contiguous()
    if z_init is not None:
        z_init = z_init.contiguous()
    _validate_varlen(q, cu_seqlens, kv_init)

    if scale is None:
        scale = q.shape[-1] ** -0.5
    if q.shape[1] == 0:
        num_seqs = 1 if cu_seqlens is None else len(cu_seqlens) - 1
        o, final_state, z_state = empty_attn_outputs(
            q, k, v, kv_init, z_init, output_final_state, num_seqs
        )
        return (o, (final_state, z_state)) if normalize else (o, final_state)
    if not normalize:
        return FusedRecurrentLinearAttnFunction.apply(
            q,
            k,
            v,
            scale,
            kv_init,
            output_final_state,
            reverse,
            cu_seqlens,
        )

    o_buf, final_state = FusedRecurrentLinearAttnFunction.apply(
        q,
        k,
        v,
        scale,
        kv_init,
        output_final_state,
        reverse,
        cu_seqlens,
    )
    o_buf, z_state = normalize_with_z_state(
        o_buf, q, k, scale, z_init, reverse=reverse, cu_seqlens=cu_seqlens
    )
    return o_buf, (final_state, z_state)
