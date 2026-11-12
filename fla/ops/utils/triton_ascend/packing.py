# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from fla.utils import ascend_compile_kwargs


@triton.jit(do_not_specialize=['T', 'T_OUT'])
def _pack_head_first_kernel(
    x, y, indices, valid, T, T_OUT,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, D_OUT: tl.constexpr,
    BT: tl.constexpr, BD: tl.constexpr, GATHER: tl.constexpr, ZERO_PAD: tl.constexpr,
    FULL_ROWS: tl.constexpr,
):
    nt = tl.cdiv(T_OUT, BT)
    rows = tl.arange(0, BT)
    columns = tl.arange(0, BD)
    for task in tl.range(tl.program_id(0), B * H * nt, tl.num_programs(0)):
        batch = (task // (H * nt)).to(tl.int64)
        head = ((task // nt) % H).to(tl.int64)
        start = (task % nt).to(tl.int64) * BT
        positions = start + rows
        if FULL_ROWS:
            mask_rows = tl.full((BT,), True, tl.int1)
        else:
            mask_rows = positions < T_OUT
        if GATHER:
            source = tl.load(indices + positions, mask_rows, other=0).to(tl.int64)
            pointers = x + (batch * T * H + head) * D + source[:, None] * H * D + columns[None, :]
        else:
            base = x + (batch * T * H + start * H + head) * D
            pointers = base + rows[:, None] * H * D + columns[None, :]
        values = tl.load(pointers, mask_rows[:, None] & (columns[None, :] < D), other=0)
        if ZERO_PAD:
            keep = tl.load(valid + positions, mask_rows, other=0).to(tl.int1)
            values = tl.where(keep[:, None], values, 0)
        target = y + ((batch * H + head) * T_OUT + start) * D_OUT
        tl.store(target + rows[:, None] * D_OUT + columns[None, :], values,
                 mask_rows[:, None] & (columns[None, :] < D_OUT))


def pack_head_first(
    x: torch.Tensor,
    width: int | None = None,
    indices: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a logical BTH[D] view backed by contiguous BHT[D] storage."""
    if x.ndim not in (3, 4):
        raise ValueError('head-first packing requires a BTH or BTHD tensor')
    x = x.contiguous()
    B, T, H = x.shape[:3]
    D = x.shape[-1] if x.ndim == 4 else 1
    width = D if width is None else width
    if width < D:
        raise ValueError('packed width must cover all input features')
    T_out = len(indices) if indices is not None else T
    packed = x.new_empty(B, H, T_out, width)
    if B * H * T_out:
        block_rows = 64 if width <= 128 else 16
        properties = driver.active.utils.get_device_properties(torch.npu.current_device())
        _pack_head_first_kernel[(properties['num_vectorcore'],)](
            x, packed, indices, valid, T, T_out,
            B=B, H=H, D=D, D_OUT=width, BT=block_rows, BD=triton.next_power_of_2(width),
            GATHER=indices is not None, ZERO_PAD=valid is not None,
            FULL_ROWS=T_out % block_rows == 0,
            **ascend_compile_kwargs(),
        )
    return (packed.squeeze(-1) if x.ndim == 3 else packed).transpose(1, 2)
