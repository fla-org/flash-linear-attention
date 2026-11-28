# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2025 sgl-project
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

import functools

import torch
import triton
import triton.language as tl


def input_guard(fn):
    """Make tensor arguments contiguous before entering pointer-based kernels."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = tuple(
            arg.contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args
        )
        kwargs = {
            key: value.contiguous() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        return fn(*args, **kwargs)

    return wrapper


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def normalize_chunk_size(chunk_size: int) -> int:
    """Validate the public chunk contract and cap the on-chip block size."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(f"`chunk_size` must be an integer, got {chunk_size!r}.")
    if chunk_size < 16:
        raise ValueError(f"`chunk_size` must be at least 16, got {chunk_size}.")
    if chunk_size & (chunk_size - 1):
        raise ValueError(f"`chunk_size` must be a power of 2, got {chunk_size}.")
    return min(chunk_size, 64)


def empty_attn_outputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None,
    z_init: torch.Tensor | None,
    output_final_state: bool,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Outputs of a linear-attention call whose input carries no token.

    Launching the chunked kernels on a zero-length sequence would publish an
    empty grid (``coreDim == 0``) and fault on Ascend, so the empty case is
    answered directly.  The output still carries zero-valued gradients to
    ``q``/``k``/``v`` and the states are forwarded unchanged, which keeps the
    call transparent inside a training loop.
    """
    o = torch.zeros_like(v)
    zero = None
    for t in (q, k, v):
        if t.requires_grad:
            term = t.sum() * 0
            zero = term if zero is None else zero + term
    if zero is not None:
        o = o + zero

    final_state = None
    if output_final_state:
        if initial_state is not None:
            final_state = initial_state.to(torch.float32).contiguous().clone()
        else:
            final_state = v.new_zeros(
                num_seqs,
                v.shape[2],
                k.shape[-1],
                v.shape[-1],
                dtype=torch.float32,
            )

    z_state = None
    if z_init is not None:
        z_dtype = torch.promote_types(k.dtype, z_init.dtype)
        z_state = z_init.to(z_dtype).contiguous().clone()
    else:
        z_state = k.new_zeros(num_seqs, 1, v.shape[2], k.shape[-1])
    return o, final_state, z_state


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    """``[chunk, (sequence index, chunk index inside the sequence)]`` pairs.

    The sequence index is derived from the per-sequence chunk counts rather
    than from ``indices.eq(0)``: an empty sequence contributes no chunk, so the
    run-detection trick that upstream uses would see two consecutive zeros and
    shift every following sequence by one, silently attributing the first chunk
    of a real sequence to the empty one (which produced all-zero gradients).
    """
    counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    total = int(counts.sum().item())
    if total == 0:
        return torch.zeros((0, 2), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    seq = torch.arange(counts.numel(), device=cu_seqlens.device)
    i_n = torch.repeat_interleave(seq, counts)
    starts = torch.cumsum(counts, 0) - counts
    i_t = torch.arange(total, device=cu_seqlens.device) - torch.repeat_interleave(
        starts, counts
    )
    return torch.stack([i_n, i_t], 1).to(cu_seqlens)


def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


@triton.jit
def zero_acc(BK: tl.constexpr, BV: tl.constexpr):
    """Exact zero for a loop-carried fp32 accumulator on Ascend.

    A constant ``tl.zeros`` seed can be lowered into dynamically shaped UB
    temporaries and make BiShengIR PlanMemory abort.  Routing the zero through
    a never-true runtime predicate avoids that path while preserving 0.0.
    """
    return tl.zeros([BK, BV], dtype=tl.float32) + tl.where(
        tl.program_id(0) < 0, 1.0, 0.0
    ).to(tl.float32)


@triton.jit
def zero_acc_1d(BS: tl.constexpr):
    """1-D variant of :func:`zero_acc` for loop-carried cumsum carries."""
    return tl.zeros([BS], dtype=tl.float32) + tl.where(
        tl.program_id(0) < 0, 1.0, 0.0
    ).to(tl.float32)


@triton.jit
def exp(x):
    return tl.exp(x)


@triton.jit
def exp2(x):
    return tl.math.exp2(x)


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))
