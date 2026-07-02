# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""L2 normalization kernels adapted for triton-ascend on Ascend NPU."""

import torch
import triton
import triton.language as tl

# Ascend Triton rejects grids whose product exceeds 65535.
_NPU_MAX_TRITON_GRID = 65535
_STATIC_NUM_WARPS = 2


def _iter_row_chunks(num_rows: int):
    for row_off in range(0, num_rows, _NPU_MAX_TRITON_GRID):
        yield row_off, min(_NPU_MAX_TRITON_GRID, num_rows - row_off)


@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    rstd,
    eps,
    D,
    BD: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + ROW_OFFSET
    x += i_t * D
    y += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D

    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)
    tl.store(rstd + i_t, b_rstd)


@triton.jit
def l2norm_bwd_kernel1(
    y,
    rstd,
    dy,
    dx,
    eps,
    D,
    BD: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + ROW_OFFSET
    y += i_t * D
    dx += i_t * D
    dy += i_t * D

    cols = tl.arange(0, BD)
    mask = cols < D
    b_y = tl.load(y + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + i_t).to(tl.float32)
    b_dy = tl.load(dy + cols, mask=mask, other=0.0).to(tl.float32)
    b_dx = b_dy * b_rstd - tl.sum(b_dy * b_y) * b_y * b_rstd
    tl.store(dx + cols, b_dx, mask=mask)


def l2norm_fwd_npu(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    for row_off, chunk in _iter_row_chunks(T):
        l2norm_fwd_kernel1[(chunk,)](
            x=x,
            y=y,
            rstd=rstd,
            eps=eps,
            D=D,
            BD=BD,
            ROW_OFFSET=row_off,
            num_warps=_STATIC_NUM_WARPS,
        )
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def l2norm_bwd_npu(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
):
    y_shape_og = y.shape
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    assert dy.shape == y.shape
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]

    MAX_FUSED_SIZE = 65536 // y.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    for row_off, chunk in _iter_row_chunks(T):
        l2norm_bwd_kernel1[(chunk,)](
            y=y,
            rstd=rstd,
            dy=dy,
            dx=dx,
            eps=eps,
            D=D,
            BD=BD,
            ROW_OFFSET=row_off,
            num_warps=_STATIC_NUM_WARPS,
        )
    return dx.view(y_shape_og)
