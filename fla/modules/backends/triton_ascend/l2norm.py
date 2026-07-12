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

from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM, compute_ub_block_size, iter_axis_launch_chunks

# Peak live fp32 vectors in row-wise kernel1 (same order as layer_norm).
_FWD_MEM_MULT = 6.0
_BWD_MEM_MULT = 8.0
_UB_SAFETY_MARGIN = 0.85
# Legacy byte cap when UB capacity cannot be detected (65536 // fp32).
_FALLBACK_MAX_BD = 65536 // 4


def _get_l2norm_bd(D: int, is_forward: bool) -> int:
    """Return power-of-2 block size for feature dim D under UB constraints."""
    memory_multiplier = _FWD_MEM_MULT if is_forward else _BWD_MEM_MULT
    return compute_ub_block_size(
        D,
        memory_multiplier,
        safety_margin=_UB_SAFETY_MARGIN,
        fallback=_FALLBACK_MAX_BD,
        desired=triton.next_power_of_2(D),
    )


@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    rstd,
    eps,
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
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
):
    i_t = tl.program_id(0)
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


def _launch_l2norm_fwd_kernel1(
    x: torch.Tensor,
    y: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    D: int,
    BD: int,
):
    chunk_T = x.shape[0]
    l2norm_fwd_kernel1[(chunk_T,)](
        x=x,
        y=y,
        rstd=rstd,
        eps=eps,
        D=D,
        BD=BD,
    )


def _launch_l2norm_bwd_kernel1(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    dx: torch.Tensor,
    eps: float,
    D: int,
    BD: int,
):
    chunk_T = y.shape[0]
    l2norm_bwd_kernel1[(chunk_T,)](
        y=y,
        rstd=rstd,
        dy=dy,
        dx=dx,
        eps=eps,
        D=D,
        BD=BD,
    )


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

    BD = _get_l2norm_bd(D, is_forward=True)
    if D > BD:
        raise RuntimeError(
            f"L2Norm feature dim {D} exceeds UB-safe block size {BD}. "
            "Column-tiled kernels are not yet implemented for this size."
        )

    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    for row_start, row_len in iter_axis_launch_chunks(T, 1, max_grid=ASCEND_MAX_GRID_DIM):
        row_end = row_start + row_len
        _launch_l2norm_fwd_kernel1(
            x[row_start:row_end],
            y[row_start:row_end],
            rstd[row_start:row_end],
            eps,
            D,
            BD,
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
    rstd = rstd.reshape(-1)
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    assert rstd.numel() == T

    BD = _get_l2norm_bd(D, is_forward=False)
    if D > BD:
        raise RuntimeError(
            f"L2Norm feature dim {D} exceeds UB-safe block size {BD}. "
            "Column-tiled kernels are not yet implemented for this size."
        )

    for row_start, row_len in iter_axis_launch_chunks(T, 1, max_grid=ASCEND_MAX_GRID_DIM):
        row_end = row_start + row_len
        _launch_l2norm_bwd_kernel1(
            y[row_start:row_end],
            rstd[row_start:row_end],
            dy[row_start:row_end],
            dx[row_start:row_end],
            eps,
            D,
            BD,
        )
    return dx.view(y_shape_og)
