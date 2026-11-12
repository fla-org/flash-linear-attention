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

"""Global cumulative sums used by the normalized linear-attention path.

The kernels are a compact trajectory-A port of FLA's Ascend global-cumsum
implementation.  Packed varlen inputs are handled on the host by slicing each
sequence and reusing the dense kernel, which keeps the device kernel simple and
avoids a second varlen-specific launch path in this first milestone.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .ascend_limits import MAX_VECTOR_TILE_ELEMS
from .utils import zero_acc_1d


@triton.heuristics({"HAS_SCALE": lambda args: args["scale"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_global_cumsum_scalar_kernel(
    s,
    o_buf,
    scale,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    REVERSE: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n = i_nh // H
    i_h = i_nh % H
    bos = i_n * T
    b_z = tl.where(tl.program_id(0) < 0, 1.0, 0.0).to(tl.float32)

    for i_c in tl.range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(
            o_buf + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        if REVERSE:
            b_o = b_z + tl.cumsum(b_s, axis=0, reverse=True)
        else:
            b_o = b_z + tl.cumsum(b_s, axis=0)
        if HAS_SCALE:
            b_o *= scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))
        b_z += tl.sum(b_s, 0)


@triton.heuristics({"HAS_SCALE": lambda args: args["scale"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_global_cumsum_vector_kernel(
    s,
    o_buf,
    scale,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    NT: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    REVERSE: tl.constexpr,
):
    i_s, i_nh = tl.program_id(0), tl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H
    bos = i_n * T
    b_z = zero_acc_1d(BS)

    for i_c in tl.range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        p_s = tl.make_block_ptr(
            s + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o_buf + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        if REVERSE:
            b_o = b_z[None, :] + tl.cumsum(b_s, axis=0, reverse=True)
        else:
            b_o = b_z[None, :] + tl.cumsum(b_s, axis=0)
        if HAS_SCALE:
            b_o *= scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        b_z += tl.sum(b_s, 0)


# Peak UB footprint of a 2-D scan tile is dominated by the ``tl.cumsum``
# temporaries rather than by the tile payload itself: on Ascend 910B (192 KiB UB)
# a [64, 64] tile asks for ~386 KiB and aborts BiShengIR PlanMemory, a [32, 64]
# or [64, 32] tile (~198 KiB) still overflows, and [32, 32] (~96 KiB) fits.  That
# is ~96 bytes per element and is independent of the input dtype (the scan runs
# in fp32 either way), so cap the tile product instead of trusting the UB size.
# The cap itself lives in kernel/ascend_limits.py with the other UB budgets.
_MAX_VECTOR_TILE_ELEMS = MAX_VECTOR_TILE_ELEMS


def _vector_tile_sizes(T: int, S: int) -> tuple[int, int]:
    """Pick ``(BT, BS)`` so the vector cumsum tile fits the UB budget."""
    # ``triton.next_power_of_2(0)`` returns 0, which would make ``BT`` zero and
    # turn the later ``triton.cdiv(T, BT)`` into a ZeroDivisionError.
    BT = max(1, min(64, triton.next_power_of_2(T)))
    BS = max(1, min(64, triton.next_power_of_2(S)))
    while BT * BS > _MAX_VECTOR_TILE_ELEMS and (BT > 1 or BS > 1):
        if BT >= BS:
            BT //= 2
        else:
            BS //= 2
    return BT, BS


def _dense_global_cumsum(
    s: torch.Tensor,
    reverse: bool,
    scale: float | None,
    output_dtype: torch.dtype | None,
) -> torch.Tensor:
    if s.ndim == 3:
        B, T, H = s.shape
        S = None
    elif s.ndim == 4:
        B, T, H, S = s.shape
    else:
        raise ValueError(f"chunk_global_cumsum expects 3-D or 4-D input, got {s.shape}")

    out = torch.empty_like(s, dtype=output_dtype or s.dtype)
    if T == 0:
        # Empty input: the cumsum is empty too, and launching a kernel would
        # produce an all-zero grid.
        return out
    if S is None:
        BT = max(1, min(64, triton.next_power_of_2(T)))
        NT = triton.cdiv(T, BT)
        grid = (B * H,)
        chunk_global_cumsum_scalar_kernel[grid](
            s=s,
            o_buf=out,
            scale=scale,
            T=T,
            H=H,
            BT=BT,
            NT=NT,
            REVERSE=reverse,
            num_warps=4,
            num_stages=2,
        )
    else:
        BT, BS = _vector_tile_sizes(T, S)
        NT = triton.cdiv(T, BT)
        grid = (triton.cdiv(S, BS), B * H)
        chunk_global_cumsum_vector_kernel[grid](
            s=s,
            o_buf=out,
            scale=scale,
            T=T,
            H=H,
            S=S,
            BT=BT,
            BS=BS,
            NT=NT,
            REVERSE=reverse,
            num_warps=4,
            num_stages=2,
        )
    return out


def chunk_global_cumsum(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    scale: float | None = None,
    output_dtype: torch.dtype | None = torch.float,
    **kwargs,
) -> torch.Tensor:
    """Global (whole-sequence) cumulative sum along the time dimension.

    ``s`` must be either ``[B, T, H]`` or ``[B, T, H, S]``.  With
    ``cu_seqlens`` the batch dimension must be 1 and the cumsum is reset per
    sequence.
    """
    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed; inputs must be [B, T, H, ...]"
        )
    if s.ndim not in (3, 4):
        raise ValueError(f"chunk_global_cumsum expects 3-D or 4-D input, got {s.shape}")

    if cu_seqlens is None:
        return _dense_global_cumsum(s, reverse, scale, output_dtype)

    if s.shape[0] != 1:
        raise ValueError(
            "Only batch size 1 is supported when cu_seqlens are provided, "
            f"current size is {s.shape[0]}"
        )
    out = torch.empty_like(s, dtype=output_dtype or s.dtype)
    for _, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        bos = int(bos.item())
        eos = int(eos.item())
        if eos <= bos:
            continue
        out[:, bos:eos] = _dense_global_cumsum(
            s[:, bos:eos], reverse, scale, output_dtype
        )
    return out
