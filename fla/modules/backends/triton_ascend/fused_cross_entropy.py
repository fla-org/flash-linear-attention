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

from fla.ops.utils.op import exp, log
from fla.utils import input_guard


@triton.heuristics({
    "HAS_SMOOTHING": lambda args: args["label_smoothing"] > 0.0,
})
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
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    SPLIT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
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
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignore_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(
            n_cols, (col_block_idx + 1) * BLOCK_SIZE,
        ):
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
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)


@triton.heuristics({
    "HAS_SMOOTHING": lambda args: args["label_smoothing"] > 0.0,
})
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
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignore_index:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32,
    ) * logit_scale
    if HAS_SOFTCAPPING:
        t = tanh(logits / logit_softcapping)
        logits = logit_softcapping * t
    lse = tl.load(lse_ptr + row_idx)
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


def _npu_block_size(n_cols: int, n_rows: int) -> tuple[int, int]:
    block_size = 1024 if n_cols <= 32768 else (2048 if n_cols <= 131072 else 4096)
    max_splits = max(1, 65535 // max(n_rows, 1))
    block_size = min(
        max(block_size, triton.next_power_of_2((n_cols + max_splits - 1) // max_splits)),
        8192,
    )
    num_warps = 2 if block_size <= 2048 else 4
    return block_size, num_warps


def fused_cross_entropy_forward_npu(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    logit_softcapping: float = None,
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
    BLOCK_SIZE, num_warps = _npu_block_size(n_cols, n_rows)
    has_softcapping = logit_softcapping is not None
    softcap_val = float(logit_softcapping) if has_softcapping else 0.0
    n_splits = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    split = world_size > 1 or n_cols > MAX_BLOCK_SIZE or n_splits > 1
    loss_shape = (n_splits, n_rows) if n_splits > 1 else (n_rows,)
    losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
    lse = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
    z_losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)

    cross_entropy_fwd_kernel[(n_rows, n_splits)](
        losses,
        lse,
        z_losses,
        logits,
        target,
        label_smoothing,
        logit_scale,
        lse_square_scale,
        softcap_val,
        ignore_index,
        total_classes,
        class_start_idx,
        n_cols,
        n_rows,
        logits.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_SOFTCAPPING=has_softcapping,
        num_warps=num_warps,
        SPLIT=split,
    )

    if split:
        if n_splits > 1:
            lse = torch.logsumexp(lse, dim=0)
            losses = losses.sum(dim=0)
        if world_size > 1:
            lse_allgather = torch.empty(world_size, n_rows, dtype=lse.dtype, device=lse.device)
            torch.distributed.all_gather_into_tensor(lse_allgather, lse, group=process_group)
            handle_losses = torch.distributed.all_reduce(
                losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True,
            )
            lse = torch.logsumexp(lse_allgather, dim=0)
            handle_losses.wait()
        losses += lse
        if lse_square_scale != 0.0:
            z_losses = lse_square_scale * lse.square()
            z_losses.masked_fill_(target == ignore_index, 0.0)
            losses += z_losses
        else:
            z_losses = torch.zeros_like(losses)
        losses.masked_fill_(target == ignore_index, 0.0)

    return losses, z_losses, lse, total_classes, class_start_idx


def fused_cross_entropy_backward_npu(
    dlogits: torch.Tensor,
    grad_losses: torch.Tensor,
    logits: torch.Tensor,
    lse: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    logit_scale: float,
    lse_square_scale: float,
    logit_softcapping: float | None,
    ignore_index: int,
    total_classes: int,
    class_start_idx: int,
) -> torch.Tensor:
    n_rows, n_cols = logits.shape
    BLOCK_SIZE, num_warps = _npu_block_size(n_cols, n_rows)
    has_softcapping = logit_softcapping is not None
    softcap_val = float(logit_softcapping) if has_softcapping else 0.0

    def grid(META): return (n_rows, triton.cdiv(n_cols, META["BLOCK_SIZE"]))  # noqa
    cross_entropy_bwd_kernel[grid](
        dlogits,
        grad_losses,
        logits,
        lse,
        target,
        label_smoothing,
        logit_scale,
        lse_square_scale,
        softcap_val,
        ignore_index,
        total_classes,
        class_start_idx,
        n_cols,
        logits.stride(0),
        dlogits.stride(0),
        grad_losses.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_SOFTCAPPING=has_softcapping,
        num_warps=num_warps,
    )
    return dlogits


class CrossEntropyLossFunctionNPU(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        logits,
        target,
        label_smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        logit_softcapping=None,
        ignore_index=-100,
        inplace_backward=False,
        process_group=None,
    ):
        losses, z_losses, lse, total_classes, class_start_idx = fused_cross_entropy_forward_npu(
            logits,
            target,
            label_smoothing,
            logit_scale,
            lse_square_scale,
            logit_softcapping,
            ignore_index,
            process_group,
        )
        ctx.save_for_backward(logits, lse, target)
        ctx.mark_non_differentiable(z_losses)
        ctx.label_smoothing = label_smoothing
        ctx.logit_scale = logit_scale
        ctx.lse_square_scale = lse_square_scale
        ctx.logit_softcapping = logit_softcapping
        ctx.ignore_index = ignore_index
        ctx.total_classes = total_classes
        ctx.class_start_idx = class_start_idx
        ctx.inplace_backward = inplace_backward

        return losses, z_losses

    @staticmethod
    @input_guard
    def backward(ctx, grad_losses, grad_z_losses):
        del grad_z_losses

        logits, lse, target = ctx.saved_tensors
        dlogits = logits if ctx.inplace_backward else torch.empty_like(logits)
        fused_cross_entropy_backward_npu(
            dlogits,
            grad_losses,
            logits,
            lse,
            target,
            ctx.label_smoothing,
            ctx.logit_scale,
            ctx.lse_square_scale,
            ctx.logit_softcapping,
            ctx.ignore_index,
            ctx.total_classes,
            ctx.class_start_idx,
        )
        return dlogits, None, None, None, None, None, None, None, None, None


def cross_entropy_loss_npu(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    logit_softcapping: float = None,
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
