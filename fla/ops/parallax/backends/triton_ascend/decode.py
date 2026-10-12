# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend launch wrapper for the Parallax single-token decode kernel.

The mainline ``BS = BK = 128`` block needs 2118912 bits of UB against the
192KiB budget (hivm-plan-memory ``ub overflow``), so the backend relaunches
the unchanged kernel with a UB-budgeted ``BS``.
"""

from __future__ import annotations

import torch
import triton

from fla.ops.parallax.decode import parallax_decode_one_step_kernel
from fla.utils.ascend_ub_manager import compute_row_tile_block_size

# Per-tile-element live set of one KV block: two dtype loads (b_k, b_v) plus
# four fp32 products, ~16 B/element calibrated against the 2118912-bit report.
_ONE_STEP_MEM_MULT = 4.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BS = 64
_MIN_BS = 16
_MAX_BS = 128


def _one_step_bs(BK: int) -> int:
    return compute_row_tile_block_size(
        _MAX_BS,
        BK,
        _ONE_STEP_MEM_MULT,
        tiling_row=True,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BS,
        min_block=_MIN_BS,
        max_block=_MAX_BS,
    )


def parallax_decode_one_step_npu(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cache_start: torch.LongTensor | None = None,
) -> torch.Tensor:
    # Mirror of the mainline host orchestration (fla/ops/parallax/decode.py).
    B, Sq, HQ, K = q.shape
    if Sq != 1:
        raise ValueError(f"parallax_decode_one_step expects a single query (Sq=1), got Sq={Sq}")
    Skv, H = k.shape[1], k.shape[2]
    if H == 0 or HQ % H != 0:
        raise ValueError(f"The number of query heads ({HQ}) must be divisible by the number of key/value heads ({H}).")
    G = HQ // H
    if scale is None:
        scale = K ** -0.5
    window_size_left = -1 if window_size is None else window_size

    q, r, k, v = (x.contiguous() for x in (q, r, k, v))
    if cache_start is not None:
        cache_start = cache_start.to(device=q.device, dtype=torch.int32).contiguous()
    BK = triton.next_power_of_2(K)
    o = torch.empty_like(q)
    grid = (B * HQ,)
    parallax_decode_one_step_kernel[grid](
        q, r, k, v, o, float(scale), cache_start, Skv,
        HQ=HQ, H=H, G=G, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left,
        USE_CACHE_START=cache_start is not None,
        BS=_one_step_bs(BK),
        num_warps=4, num_stages=2,
    )
    return o


__all__ = ['parallax_decode_one_step_npu']
