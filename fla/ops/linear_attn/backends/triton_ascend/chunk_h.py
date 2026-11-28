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

"""Un-gated linear-attention chunk state scan for Ascend NPU.

This is the trajectory-A counterpart of FLA's ``chunk_fwd_h`` with the
gating-related inputs removed.  The host wrapper handles packed varlen inputs
by splitting per sequence and reusing the equal-length kernel, so the kernel
itself only has to understand a dense ``[N, T, H, K]`` key and ``[N, T, H, V]``
value layout.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import normalize_chunk_size, prepare_chunk_offsets, zero_acc

# Keep the same conservative tile sizes as the validated FLA Ascend backend.
_BK = 64
_BV = 64


def _chunk_h_tile_size(K: int, V: int) -> tuple[int, int]:
    if K > 128 or V > 128:
        return 32, 32
    return _BK, _BV


@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_h_contrib(
    k,
    v,
    contribution,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    NT = tl.cdiv(T, BT)
    i_nt = tl.program_id(2).to(tl.int64)
    i_t = i_nt % NT
    i_nh = i_nt // NT
    i_n = i_nh // H
    i_h = i_nh % H
    bos = i_n * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    # Clamped index variants.  Everything that is masked off still has to form a
    # valid address: the Ascend backend issues the transfer for the whole tile
    # and the masked lanes are folded with `other=0.0` afterwards, so pointing
    # them past the end of the tensor leaks out-of-range data into the
    # accumulator (seen as NaN in the fp16 final state for T < BT).
    o_k_s = tl.minimum(o_k, K - 1)
    o_v_s = tl.minimum(o_v, V - 1)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_t_s = tl.minimum(o_t, T - 1)
    p_k = (
        k
        + (bos * H + i_h) * K
        + o_k_s[:, None].to(tl.int64)
        + o_t_s[None, :].to(tl.int64) * (H * K)
    )
    p_v = (
        v + (bos * H + i_h) * V + o_t_s[:, None].to(tl.int64) * (H * V) + o_v_s[None, :]
    )
    b_k = tl.load(
        p_k,
        mask=(o_k[:, None] < K) & m_t[None, :],
        other=0.0,
    ).to(tl.float32)
    b_v = tl.load(
        p_v,
        mask=m_t[:, None] & (o_v[None, :] < V),
        other=0.0,
    ).to(tl.float32)
    b_h = zero_acc(BK, BV)
    b_h = tl.dot(b_k, b_v, b_h)

    p_c = (
        contribution
        + ((i_n * NT + i_t) * H + i_h) * K * V
        + o_k_s[:, None].to(tl.int64) * V
        + o_v_s[None, :]
    )
    tl.store(p_c, b_h, mask=(o_k[:, None] < K) & (o_v[None, :] < V))


def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    states_in_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute per-chunk start states for un-gated linear attention.

    Args:
        k: keys of shape ``[B, T, H, K]``.
        v: values of shape ``[B, T, H, V]``.
        h0: optional initial state ``[N, H, K, V]``.
        output_final_state: whether to return the final state.
        cu_seqlens: cumulative sequence lengths for packed varlen input
            (``B`` must be 1 when provided).
        chunk_size: chunk length used for the scan and for the consumer
            ``chunk_fwd_o``.
        states_in_fp32: allocate ``h``/``ht`` in fp32 instead of ``k.dtype``.

    Returns:
        ``h`` of shape ``[B, NT, H, K, V]`` where ``NT = cdiv(T, chunk_size)``,
        and optional final state of shape ``[N, H, K, V]``.
    """
    k = k.contiguous()
    v = v.contiguous()
    if h0 is not None:
        h0 = h0.contiguous()
    B, T, H, K = k.shape
    BT = normalize_chunk_size(chunk_size)

    if cu_seqlens is not None:
        assert B == 1, "packed varlen chunk_fwd_h expects B=1"
        offsets = prepare_chunk_offsets(cu_seqlens, BT)
        N = len(cu_seqlens) - 1
        NS = int(offsets[-1].item())
        state_dtype = torch.float32 if states_in_fp32 else k.dtype
        h = k.new_zeros(B, NS, H, K, v.shape[-1], dtype=state_dtype)
        ht = (
            k.new_zeros(N, H, K, v.shape[-1], dtype=torch.float32)
            if output_final_state
            else None
        )
        for i_n in range(N):
            bos = int(cu_seqlens[i_n].item())
            eos = int(cu_seqlens[i_n + 1].item())
            boh = int(offsets[i_n].item())
            n_i = int(offsets[i_n + 1].item()) - boh
            if eos <= bos or n_i <= 0:
                if ht is not None and h0 is not None:
                    ht[i_n].copy_(h0[i_n])
                continue
            i_n_end = i_n + 1
            boh_end = boh + n_i
            h_i, ht_i = chunk_fwd_h(
                k=k[:, bos:eos],
                v=v[:, bos:eos],
                h0=None if h0 is None else h0[i_n:i_n_end],
                output_final_state=output_final_state,
                cu_seqlens=None,
                chunk_size=chunk_size,
                states_in_fp32=states_in_fp32,
            )
            h[:, boh:boh_end] = h_i
            if ht is not None:
                ht[i_n].copy_(ht_i[0])
        return h, ht

    N = B
    if T == 0:
        state_dtype = torch.float32 if states_in_fp32 else k.dtype
        h = k.new_zeros(B, 0, H, K, v.shape[-1], dtype=state_dtype)
        ht = None
        if output_final_state:
            ht = (
                h0.detach()
                .to(torch.float32)
                .clone(memory_format=torch.contiguous_format)
                if h0 is not None
                else k.new_zeros(B, H, K, v.shape[-1], dtype=torch.float32)
            )
        return h, ht

    NT = triton.cdiv(T, BT)
    state_dtype = torch.float32 if states_in_fp32 else k.dtype
    contribution = torch.empty(
        B, NT, H, K, v.shape[-1], device=k.device, dtype=torch.float32
    )

    BK, BV = _chunk_h_tile_size(K, v.shape[-1])
    grid = (triton.cdiv(K, BK), triton.cdiv(v.shape[-1], BV), N * H * NT)
    chunk_fwd_kernel_h_contrib[grid](
        k=k,
        v=v,
        contribution=contribution,
        T=T,
        H=H,
        K=K,
        V=v.shape[-1],
        BT=BT,
        BK=BK,
        BV=BV,
        multibuffer=False,
        enable_auto_bind_sub_block=False,
    )

    cumulative = torch.cumsum(contribution, dim=1)
    h = cumulative - contribution
    final_state = cumulative[:, -1]
    if h0 is not None:
        h0_fp32 = h0.detach().to(torch.float32)
        h = h + h0_fp32[:, None]
        final_state = final_state + h0_fp32
    h = h.to(state_dtype)
    return h, final_state if output_final_state else None
