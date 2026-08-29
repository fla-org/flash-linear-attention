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

from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    compute_ub_block_size,
    iter_axis_launch_chunks,
)

# Peak live fp32 tiles relative to [BT, BD].
# Forward keeps ~b_x/b_y; backward keeps b_y/b_dy/b_dx plus reduction temps.
# Multipliers are calibrated so small-D shapes can use large BT without UB overflow
# under Ascend multi-buffering (bwd BT=128 @ BD=128 overflows; BT=64 is safe).
_FWD_MEM_MULT = 2.0
_BWD_MEM_MULT = 4.0
_UB_SAFETY_MARGIN = 0.85
# Legacy byte cap when UB capacity cannot be detected (65536 // fp32).
_FALLBACK_MAX_BD = 65536 // 4
# Cap row tile to keep compile variants small and match the CUDA BT list.
_MAX_BT = 128


def _get_l2norm_tiles(D: int, is_forward: bool) -> tuple[int, int]:
    """Return (BD, BT) under UB constraints. BT may be 1 for large D."""
    memory_multiplier = _FWD_MEM_MULT if is_forward else _BWD_MEM_MULT
    BD = compute_ub_block_size(
        D,
        memory_multiplier,
        safety_margin=_UB_SAFETY_MARGIN,
        fallback=_FALLBACK_MAX_BD,
        desired=triton.next_power_of_2(D),
    )
    if D > BD:
        raise RuntimeError(
            f"L2Norm feature dim {D} exceeds UB-safe block size {BD}. "
            "Column-tiled kernels are not yet implemented for this size."
        )
    # Large synthetic row dim so BT is limited by UB, not by a host-side T guess.
    BT = compute_row_tile_block_size(
        1 << 20,
        BD,
        memory_multiplier,
        tiling_row=True,
        safety_margin=_UB_SAFETY_MARGIN,
        fallback=16,
        min_block=1,
        max_block=_MAX_BT,
    )
    return BD, BT


@triton.jit(do_not_specialize=['T'])
def l2norm_fwd_kernel(
    x,
    y,
    rstd,
    eps,
    T,
    T_OFFSET,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0) + T_OFFSET
    rows = i_t * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    mask = (rows[:, None] < T) & (cols[None, :] < D)

    b_x = tl.load(x + rows[:, None] * D + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(y + rows[:, None] * D + cols[None, :], b_y.to(y.dtype.element_ty), mask=mask)
    tl.store(rstd + rows, b_rstd.to(rstd.dtype.element_ty), mask=rows < T)


@triton.jit(do_not_specialize=['T'])
def l2norm_bwd_kernel(
    y,
    rstd,
    dy,
    dx,
    T,
    T_OFFSET,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0) + T_OFFSET
    rows = i_t * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    mask = (rows[:, None] < T) & (cols[None, :] < D)

    b_y = tl.load(y + rows[:, None] * D + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + rows, mask=rows < T, other=0.0).to(tl.float32)
    b_dy = tl.load(dy + rows[:, None] * D + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    b_dx = b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
    tl.store(dx + rows[:, None] * D + cols[None, :], b_dx.to(dx.dtype.element_ty), mask=mask)


def _launch_l2norm_fwd_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    T: int,
    D: int,
    BD: int,
    BT: int,
):
    NT = triton.cdiv(T, BT)
    for nt_off, nt_len in iter_axis_launch_chunks(NT, 1, max_grid=ASCEND_MAX_GRID_DIM):
        l2norm_fwd_kernel[(nt_len,)](
            x=x,
            y=y,
            rstd=rstd,
            eps=eps,
            T=T,
            T_OFFSET=nt_off,
            D=D,
            BD=BD,
            BT=BT,
        )


def _launch_l2norm_bwd_kernel(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    dx: torch.Tensor,
    T: int,
    D: int,
    BD: int,
    BT: int,
):
    NT = triton.cdiv(T, BT)
    for nt_off, nt_len in iter_axis_launch_chunks(NT, 1, max_grid=ASCEND_MAX_GRID_DIM):
        l2norm_bwd_kernel[(nt_len,)](
            y=y,
            rstd=rstd,
            dy=dy,
            dx=dx,
            T=T,
            T_OFFSET=nt_off,
            D=D,
            BD=BD,
            BT=BT,
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

    BD, BT = _get_l2norm_tiles(D, is_forward=True)
    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    _launch_l2norm_fwd_kernel(x, y, rstd, eps, T, D, BD, BT)
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def l2norm_bwd_npu(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
):
    y_shape_og = y.shape
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    assert dy.shape == y.shape
    rstd = rstd.reshape(-1)
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    assert rstd.numel() == T

    BD, BT = _get_l2norm_tiles(D, is_forward=False)
    _launch_l2norm_bwd_kernel(y, rstd, dy, dx, T, D, BD, BT)
    return dx.view(y_shape_og)
