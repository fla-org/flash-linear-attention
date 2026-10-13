# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused cross-entropy kernels adapted for triton-ascend on Huawei NPU."""

import torch
import triton
import triton.language as tl
from triton.language.math import tanh

from fla.backends import TritonAscendBackend, register
from fla.modules.fused_cross_entropy.common import _cross_entropy_reduce
from fla.modules.fused_cross_entropy.ops import FusedCrossEntropyFunction, cross_entropy_loss
from fla.ops.utils.op import exp, log
from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM, compute_vocab_block_size, iter_axis_launch_chunks

# cross-entropy fwd/bwd peak fp32 buffers along vocab dimension.
_CE_FWD_MEM_MULT = 8.0
_CE_BWD_MEM_MULT = 12.0


@triton.heuristics({"HAS_SMOOTHING": lambda args: args["label_smoothing"] > 0.0})
@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr,
    lse_ptr,
    z_loss_ptr,
    logits_ptr,
    labels_ptr,
    label_smoothing,
    logit_scale,
    lse_square_scale,
    logit_softcapping: tl.constexpr,
    ignore_index,
    total_classes,
    class_start_idx,
    n_cols,
    n_rows,
    logits_row_stride,
    ROW_OFFSET,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    SPLIT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    abs_row_idx = row_idx + ROW_OFFSET
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + tl.cast(row_idx, tl.int64) * logits_row_stride
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf"))
    logits = logits.to(tl.float32) * logit_scale
    if HAS_SOFTCAPPING:
        logits = logit_softcapping * tanh(logits / logit_softcapping)
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = log(tl.sum(exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + abs_row_idx, lse)
    if label_idx == ignore_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(n_cols, (col_block_idx + 1) * BLOCK_SIZE):
            logits_label = tl.load(logits_ptr + label_idx).to(tl.float32) * logit_scale
            if HAS_SOFTCAPPING:
                logits_label = logit_softcapping * tanh(logits_label / logit_softcapping)
            if HAS_SMOOTHING:
                loss = (
                    (lse if not SPLIT else 0.0)
                    - label_smoothing * sum_logits / total_classes
                    - (1 - label_smoothing) * logits_label
                )
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        else:
            if HAS_SMOOTHING:
                loss = label_smoothing * ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
            else:
                loss = 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + col_block_idx * n_rows + abs_row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + col_block_idx * n_rows + abs_row_idx, z_loss)


@triton.heuristics({"HAS_SMOOTHING": lambda args: args["label_smoothing"] > 0.0})
@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    label_smoothing,
    logit_scale,
    lse_square_scale,
    logit_softcapping: tl.constexpr,
    ignore_index,
    total_classes,
    class_start_idx,
    n_cols,
    logits_row_stride,
    dlogits_row_stride,
    dloss_row_stride,
    ROW_OFFSET,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    row_idx = tl.program_id(0)
    abs_row_idx = row_idx + ROW_OFFSET
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + tl.cast(row_idx, tl.int64) * logits_row_stride
    dlogits_ptr = dlogits_ptr + tl.cast(row_idx, tl.int64) * dlogits_row_stride
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignore_index:
        dloss = tl.load(dloss_ptr + abs_row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32) * logit_scale
    if HAS_SOFTCAPPING:
        t = tanh(logits / logit_softcapping)
        logits = logit_softcapping * t
    lse = tl.load(lse_ptr + abs_row_idx)
    probs = exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx -= class_start_idx
    if HAS_SMOOTHING:
        smooth_negative = label_smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - label_smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    if HAS_SOFTCAPPING:
        probs = probs * (1.0 - t * t)
    tl.store(dlogits_ptr + col_offsets, (dloss * logit_scale) * probs, mask=col_offsets < n_cols)


def _npu_block_size(n_cols: int, n_rows: int, is_backward: bool = False) -> tuple[int, int]:
    memory_multiplier = _CE_BWD_MEM_MULT if is_backward else _CE_FWD_MEM_MULT
    block_size = compute_vocab_block_size(vocab_size=n_cols, num_rows=n_rows, memory_multiplier=memory_multiplier)
    num_warps = 2 if block_size <= 2048 else 4
    return block_size, num_warps


def _launch_cross_entropy_fwd(
    losses,
    lse,
    z_losses,
    logits,
    target,
    *,
    label_smoothing,
    logit_scale,
    lse_square_scale,
    softcap_val,
    ignore_index,
    total_classes,
    class_start_idx,
    n_cols,
    n_rows,
    logits_stride,
    BLOCK_SIZE,
    has_softcapping,
    num_warps,
    split,
):
    n_splits = triton.cdiv(n_cols, BLOCK_SIZE)
    for row_off, row_len in iter_axis_launch_chunks(
        axis_size=n_rows,
        other_grid_product=n_splits,
        max_grid=ASCEND_MAX_GRID_DIM,
    ):
        cross_entropy_fwd_kernel[(row_len, n_splits)](
            loss_ptr=losses,
            lse_ptr=lse,
            z_loss_ptr=z_losses,
            logits_ptr=logits[row_off:row_off + row_len],
            labels_ptr=target[row_off:row_off + row_len],
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            lse_square_scale=lse_square_scale,
            logit_softcapping=softcap_val,
            ignore_index=ignore_index,
            total_classes=total_classes,
            class_start_idx=class_start_idx,
            n_cols=n_cols,
            n_rows=n_rows,
            logits_row_stride=logits_stride,
            ROW_OFFSET=row_off,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SOFTCAPPING=has_softcapping,
            SPLIT=split,
            num_warps=num_warps,
        )


def _launch_cross_entropy_bwd(
    dlogits,
    grad_losses,
    logits,
    lse,
    target,
    *,
    label_smoothing,
    logit_scale,
    lse_square_scale,
    softcap_val,
    ignore_index,
    total_classes,
    class_start_idx,
    n_cols,
    n_rows,
    logits_stride,
    dlogits_stride,
    grad_losses_stride,
    BLOCK_SIZE,
    has_softcapping,
    num_warps,
):
    n_splits = triton.cdiv(n_cols, BLOCK_SIZE)
    for row_off, row_len in iter_axis_launch_chunks(
        axis_size=n_rows,
        other_grid_product=n_splits,
        max_grid=ASCEND_MAX_GRID_DIM,
    ):
        cross_entropy_bwd_kernel[(row_len, n_splits)](
            dlogits_ptr=dlogits[row_off:row_off + row_len],
            dloss_ptr=grad_losses,
            logits_ptr=logits[row_off:row_off + row_len],
            lse_ptr=lse,
            labels_ptr=target[row_off:row_off + row_len],
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            lse_square_scale=lse_square_scale,
            logit_softcapping=softcap_val,
            ignore_index=ignore_index,
            total_classes=total_classes,
            class_start_idx=class_start_idx,
            n_cols=n_cols,
            logits_row_stride=logits_stride,
            dlogits_row_stride=dlogits_stride,
            dloss_row_stride=grad_losses_stride,
            ROW_OFFSET=row_off,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SOFTCAPPING=has_softcapping,
            num_warps=num_warps,
        )


def _cross_entropy_fwd_npu(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    logit_softcapping: float | None = None,
    ignore_index: int = -100,
    process_group=None,
):
    n_rows, n_cols = logits.shape
    assert target.shape == (n_rows,)
    world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
    total_classes = world_size * n_cols
    rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
    class_start_idx = rank * n_cols

    if logits.stride(-1) != 1:
        logits = logits.contiguous()

    MAX_BLOCK_SIZE = 64 * 1024
    BLOCK_SIZE, num_warps = _npu_block_size(n_cols=n_cols, n_rows=n_rows)
    has_softcapping = logit_softcapping is not None
    softcap_val = float(logit_softcapping) if has_softcapping else 0.0
    n_splits = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    split = world_size > 1 or n_cols > MAX_BLOCK_SIZE or n_splits > 1
    loss_shape = (n_splits, n_rows) if n_splits > 1 else (n_rows,)
    losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
    lse = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
    z_losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)

    _launch_cross_entropy_fwd(
        losses=losses,
        lse=lse,
        z_losses=z_losses,
        logits=logits,
        target=target,
        label_smoothing=label_smoothing,
        logit_scale=logit_scale,
        lse_square_scale=lse_square_scale,
        softcap_val=softcap_val,
        ignore_index=ignore_index,
        total_classes=total_classes,
        class_start_idx=class_start_idx,
        n_cols=n_cols,
        n_rows=n_rows,
        logits_stride=logits.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        has_softcapping=has_softcapping,
        num_warps=num_warps,
        split=split,
    )

    if split:
        losses, z_losses, lse = _cross_entropy_reduce(
            loss=losses,
            lse=lse,
            target=target,
            lse_square_scale=lse_square_scale,
            ignore_index=ignore_index,
            world_size=world_size,
            process_group=process_group,
        )

    return losses, z_losses, lse, total_classes, class_start_idx


def _cross_entropy_bwd_npu(
    logits: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    dloss: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    logit_softcapping: float | None = None,
    ignore_index: int = -100,
    total_classes: int | None = None,
    class_start_idx: int = 0,
    inplace_backward: bool = False,
) -> torch.Tensor:
    dlogits = logits if inplace_backward else torch.empty_like(logits)
    n_rows, n_cols = logits.shape
    BLOCK_SIZE, num_warps = _npu_block_size(n_cols=n_cols, n_rows=n_rows, is_backward=True)
    has_softcapping = logit_softcapping is not None
    softcap_val = float(logit_softcapping) if has_softcapping else 0.0

    _launch_cross_entropy_bwd(
        dlogits=dlogits,
        grad_losses=dloss,
        logits=logits,
        lse=lse,
        target=target,
        label_smoothing=label_smoothing,
        logit_scale=logit_scale,
        lse_square_scale=lse_square_scale,
        softcap_val=softcap_val,
        ignore_index=ignore_index,
        total_classes=logits.shape[-1] if total_classes is None else total_classes,
        class_start_idx=class_start_idx,
        n_cols=n_cols,
        n_rows=n_rows,
        logits_stride=logits.stride(0),
        dlogits_stride=dlogits.stride(0),
        grad_losses_stride=dloss.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        has_softcapping=has_softcapping,
        num_warps=num_warps,
    )
    return dlogits


class CrossEntropyLossFunctionNPU(FusedCrossEntropyFunction):

    @staticmethod
    def forward(ctx, *args):
        return FusedCrossEntropyFunction.forward(ctx, *args, fwd_fn=_cross_entropy_fwd_npu, bwd_fn=_cross_entropy_bwd_npu)


@register(cross_entropy_loss, backend=TritonAscendBackend)
def cross_entropy_loss_npu(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    logit_softcapping: float | None = None,
    ignore_index: int = -100,
    inplace_backward: bool = False,
    process_group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return CrossEntropyLossFunctionNPU.apply(
        logits,
        target,
        label_smoothing,
        logit_scale,
        lse_square_scale,
        logit_softcapping,
        ignore_index,
        inplace_backward,
        process_group,
    )
