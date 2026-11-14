# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Copyright © 2025 Huawei Technologies Co., Ltd.
# Based on vLLM: https://github.com/vllm-project/vllm
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import normalize_chunk_size, prepare_chunk_offsets, safe_exp


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o_buf,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = T

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o_buf += (bos * H + i_h) * V

    for i_t in range(NT):
        i_tg = boh + i_t
        h_base = h + (i_tg * H + i_h).to(tl.int64) * K * V
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_A = tl.zeros([BT, BT], dtype=tl.float32)

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            p_q = q + o_t[:, None] * (Hg * K) + o_k[None, :]
            p_k = k + o_k[:, None] + o_t[None, :] * (Hg * K)
            p_h = h_base + o_k[:, None] * V + o_v[None, :]
            # shape BT x BK
            b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0)
            # shape BK x BT
            b_k = tl.load(p_k, mask=m_k[:, None] & m_t[None, :], other=0)
            # shape BK x BV
            b_h = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0)

            # [BT, BK] @ [BK, BV] -> [BT, BV]
            b_o += tl.dot(b_q, b_h)
            # [BT, BK] @ [BK, BT] -> [BT, BT]
            b_A += tl.dot(b_q, b_k)

        if USE_G:
            g_ptr = g + bos + i_h * T_max
            b_g = tl.load(g_ptr + o_t, mask=m_t, other=0.0)

            b_o = b_o * tl.exp(b_g)[:, None]
            b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

        o_i = tl.arange(0, BT).to(tl.float32)
        m_A = o_i[:, None] >= o_i[None, :]
        b_A = tl.where(m_A, b_A, 0)

        p_v = v + o_t[:, None] * (H * V) + o_v[None, :]
        p_o = o_buf + o_t[:, None] * (H * V) + o_v[None, :]

        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0)
        # to fix mma -> mma layout conversion
        # already solved by triton v3.2 or higher
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    h = h.contiguous()
    if g is not None:
        g = g.contiguous()
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = normalize_chunk_size(chunk_size)

    if scale is None:
        scale = k.shape[-1] ** -0.5

    o_buf = torch.empty_like(v)
    if cu_seqlens is None:
        N, chunk_offsets = B, None
    else:
        N, chunk_offsets = (
            len(cu_seqlens) - 1,
            prepare_chunk_offsets(cu_seqlens, BT),
        )

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    # The kernel has a USE_G branch; keep the host path symmetric for un-gated
    # linear attention (`g=None`), which is the trajectory-A use case.
    g = None if g is None else g.transpose(1, 2).contiguous()
    BK = BV = 128
    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        o_buf=o_buf,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=2,
    )
    return o_buf
