# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fla.modules.backends import dispatch
from fla.ops.utils.op import exp, log, tanh
from fla.utils import input_guard

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@triton.jit(do_not_specialize=['N'])
def cross_entropy_fwd_kernel(
    logits,  # [N, V]
    target,  # [N]
    loss,  # [NV, N] or [N] when NV == 1
    lse,  # [NV, N] or [N] when NV == 1
    z_loss,  # [N]
    s_logits,
    scale: tl.constexpr,
    softcap: tl.constexpr,
    smoothing: tl.constexpr,
    z_scale: tl.constexpr,
    ignore_index: tl.constexpr,
    N,
    V: tl.constexpr,
    V_TOTAL: tl.constexpr,
    V_START: tl.constexpr,
    BV: tl.constexpr,
    SPLIT: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    i_v = tl.program_id(1)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    p_logits = logits + i_n * s_logits

    # [BV]
    b_logits = tl.load(p_logits + o_v, mask=m_v, other=-float('inf')).to(tl.float32) * scale
    if softcap is not None:
        b_logits = softcap * tanh(b_logits / softcap)
    # these transforms do not preserve -inf in padded lanes
    if softcap is not None or scale <= 0:
        b_logits = tl.where(m_v, b_logits, -float('inf'))
    if smoothing > 0:
        b_sum = tl.sum(tl.where(m_v, b_logits, 0.0), 0)
    b_max = tl.max(b_logits, 0)
    b_lse = log(tl.sum(exp(b_logits - b_max), 0)) + b_max

    b_target = tl.load(target + i_n).to(tl.int64)
    b_loss = 0.0
    b_z_loss = 0.0
    if b_target != ignore_index:
        b_target -= V_START
        m_target = (b_target >= i_v * BV) & (b_target < tl.minimum(V, (i_v + 1) * BV))
        b_target_logit = tl.load(p_logits + b_target, mask=m_target, other=0).to(tl.float32) * scale
        if softcap is not None:
            b_target_logit = softcap * tanh(b_target_logit / softcap)
        b_loss = -b_target_logit
        if smoothing > 0:
            b_loss = (1 - smoothing) * b_loss - smoothing * b_sum / V_TOTAL
        if not SPLIT:
            b_z_loss = z_scale * b_lse * b_lse
            b_loss += b_lse + b_z_loss

    tl.store(loss + i_v * N + i_n, b_loss)
    tl.store(lse + i_v * N + i_n, b_lse)
    if not SPLIT:
        tl.store(z_loss + i_n, b_z_loss)


@triton.jit
def cross_entropy_bwd_kernel(
    logits,  # [N, V]
    target,  # [N]
    lse,  # [N]
    dloss,  # [N]
    dlogits,  # [N, V]
    s_logits,
    s_dloss,
    s_dlogits,
    scale: tl.constexpr,
    softcap: tl.constexpr,
    smoothing: tl.constexpr,
    z_scale: tl.constexpr,
    ignore_index: tl.constexpr,
    V: tl.constexpr,
    V_TOTAL: tl.constexpr,
    V_START: tl.constexpr,
    BV: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    i_v = tl.program_id(1).to(tl.int64)
    o_v = i_v * BV + tl.arange(0, BV).to(tl.int64)
    m_v = o_v < V
    p_logits = logits + i_n * s_logits + o_v
    p_dlogits = dlogits + i_n * s_dlogits + o_v

    b_target = tl.load(target + i_n).to(tl.int64)
    b_dloss = tl.load(dloss + i_n * s_dloss, mask=b_target != ignore_index, other=0).to(tl.float32)
    # [BV]
    b_logits = tl.load(p_logits, mask=m_v, other=0).to(tl.float32) * scale
    if softcap is not None:
        b_tanh = tanh(b_logits / softcap)
        b_logits = softcap * b_tanh
    b_lse = tl.load(lse + i_n)
    # [BV]
    b_probs = exp(b_logits - b_lse)
    b_dlogits = b_probs + 2.0 * z_scale * b_lse * b_probs
    b_target -= V_START
    if smoothing > 0:
        b_dlogits -= tl.where(o_v == b_target, 1 - smoothing, 0.0)
        b_dlogits -= smoothing / V_TOTAL
    else:
        b_dlogits -= tl.where(o_v == b_target, 1.0, 0.0)
    if softcap is not None:
        b_dlogits *= 1.0 - b_tanh * b_tanh
    b_dlogits *= b_dloss * scale
    tl.store(p_dlogits, b_dlogits, mask=m_v)


def cross_entropy_fwd(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    logit_softcapping: float | None = None,
    ignore_index: int = -100,
    process_group: ProcessGroup | None = None,
):
    N, V = logits.shape
    assert target.shape == (N,)
    world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
    total_classes = world_size * V
    rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
    class_start_idx = rank * V

    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    BV = min(triton.next_power_of_2(V), 64 * 1024)
    # reduce softcap register pressure without introducing an additional split reduction
    if logit_softcapping is not None and V > BV:
        BV = 8 * 1024
    num_warps = 4 if BV < 2048 else (8 if BV < 8192 else 16)
    # vocab partitions contribute partial loss until the global LSE is available
    NV = triton.cdiv(V, BV)
    split = world_size > 1 or NV > 1
    shape = (NV, N) if NV > 1 else (N,)
    loss = logits.new_empty(shape, dtype=torch.float)
    lse = torch.empty_like(loss)
    z_loss = logits.new_empty(N, dtype=torch.float)

    cross_entropy_fwd_kernel[(N, NV)](
        logits=logits,
        target=target,
        loss=loss,
        lse=lse,
        z_loss=z_loss,
        s_logits=logits.stride(0),
        scale=logit_scale,
        softcap=logit_softcapping,
        smoothing=label_smoothing,
        z_scale=lse_square_scale,
        ignore_index=ignore_index,
        N=N,
        V=V,
        V_TOTAL=total_classes,
        V_START=class_start_idx,
        BV=BV,
        SPLIT=split,
        num_warps=num_warps,
    )

    if split:
        if NV > 1:
            lse = torch.logsumexp(lse, dim=0)
            loss = loss.sum(dim=0)
        if world_size > 1:
            gathered_lse = torch.empty(world_size, N, dtype=lse.dtype, device=lse.device)
            torch.distributed.all_gather_into_tensor(gathered_lse, lse, group=process_group)
            work = torch.distributed.all_reduce(loss, group=process_group, async_op=True)
            lse = torch.logsumexp(gathered_lse, dim=0)
            work.wait()
        loss += lse
        if lse_square_scale != 0:
            z_loss = lse_square_scale * lse.square()
            z_loss.masked_fill_(target == ignore_index, 0.0)
            loss += z_loss
        else:
            z_loss = torch.zeros_like(loss)
        loss.masked_fill_(target == ignore_index, 0.0)

    return loss, z_loss, lse, total_classes, class_start_idx


def cross_entropy_bwd(
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
):
    N, V = logits.shape
    BV = min(triton.next_power_of_2(V), 4 * 1024)
    dlogits = logits if inplace_backward else torch.empty_like(logits)
    cross_entropy_bwd_kernel[(N, triton.cdiv(V, BV))](
        logits=logits,
        target=target,
        lse=lse,
        dloss=dloss,
        dlogits=dlogits,
        s_logits=logits.stride(0),
        s_dloss=dloss.stride(0),
        s_dlogits=dlogits.stride(0),
        scale=logit_scale,
        softcap=logit_softcapping,
        smoothing=label_smoothing,
        z_scale=lse_square_scale,
        ignore_index=ignore_index,
        V=V,
        V_TOTAL=V if total_classes is None else total_classes,
        V_START=class_start_idx,
        BV=BV,
        num_warps=4 if BV < 2048 else 8,
    )
    return dlogits


class FusedCrossEntropyFunction(torch.autograd.Function):

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
        process_group: ProcessGroup | None = None,
    ):
        loss, z_loss, lse, total_classes, class_start_idx = cross_entropy_fwd(
            logits=logits,
            target=target,
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            lse_square_scale=lse_square_scale,
            logit_softcapping=logit_softcapping,
            ignore_index=ignore_index,
            process_group=process_group,
        )
        ctx.save_for_backward(logits, target, lse)
        ctx.mark_non_differentiable(z_loss)
        ctx.label_smoothing = label_smoothing
        ctx.logit_scale = logit_scale
        ctx.lse_square_scale = lse_square_scale
        ctx.logit_softcapping = logit_softcapping
        ctx.ignore_index = ignore_index
        ctx.total_classes = total_classes
        ctx.class_start_idx = class_start_idx
        ctx.inplace_backward = inplace_backward

        return loss, z_loss

    @staticmethod
    @input_guard
    def backward(ctx, dloss, dz_loss):
        # z-loss is returned for logging and is already included in the differentiable loss
        del dz_loss

        logits, target, lse = ctx.saved_tensors
        dlogits = cross_entropy_bwd(
            logits=logits,
            target=target,
            lse=lse,
            dloss=dloss,
            label_smoothing=ctx.label_smoothing,
            logit_scale=ctx.logit_scale,
            lse_square_scale=ctx.lse_square_scale,
            logit_softcapping=ctx.logit_softcapping,
            ignore_index=ctx.ignore_index,
            total_classes=ctx.total_classes,
            class_start_idx=ctx.class_start_idx,
            inplace_backward=ctx.inplace_backward,
        )
        return dlogits, None, None, None, None, None, None, None, None


fused_cross_entropy_forward = cross_entropy_fwd
CrossEntropyLossFunction = FusedCrossEntropyFunction


@dispatch('modules')
def cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    logit_softcapping: float | None = None,
    ignore_index: int = -100,
    inplace_backward: bool = False,
    process_group: ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute unreduced cross entropy and its non-differentiable z-loss component.

    Args:
        logits (torch.Tensor):
            Input logits of shape `[N, V]`.
        target (torch.Tensor):
            Global target indices of shape `[N]`.
        label_smoothing (float, Optional):
            Uniform label smoothing coefficient. Default: 0.0.
        logit_scale (float, Optional):
            Scale applied before softcapping. Default: 1.0.
        lse_square_scale (float, Optional):
            Coefficient of the squared logsumexp regularizer. Default: 0.0.
        logit_softcapping (float, Optional):
            Softcap applied as `softcap * tanh(logits / softcap)`. Default: `None`.
        ignore_index (int, Optional):
            Target index excluded from loss and gradients. Default: -100.
        inplace_backward (bool, Optional):
            Whether to overwrite logits with their gradients. Default: `False`.
        process_group (ProcessGroup, Optional):
            Group sharding the vocabulary into equal contiguous partitions. Default: `None`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            FP32 loss and z-loss tensors of shape `[N]`. The loss includes z-loss.
    """
    return FusedCrossEntropyFunction.apply(
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


class FusedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        lse_square_scale: float = 0.0,
        logit_softcapping: float | None = None,
        inplace_backward: bool = False,
        process_group: ProcessGroup | None = None,
        return_z_loss: bool = False,
    ):
        """Cross entropy with optional logit transforms and vocabulary parallelism.

        Args:
            ignore_index (int, Optional):
                Target index excluded from loss and gradients. Default: -100.
            reduction (str, Optional):
                Reduction over non-ignored targets: `mean`, `sum`, or `none`. Default: `mean`.
            label_smoothing (float, Optional):
                Uniform label smoothing coefficient. Default: 0.0.
            logit_scale (float, Optional):
                Scale applied before softcapping. Default: 1.0.
            lse_square_scale (float, Optional):
                Coefficient of the squared logsumexp regularizer. Default: 0.0.
            logit_softcapping (float, Optional):
                Softcap applied as `softcap * tanh(logits / softcap)`. Default: `None`.
            inplace_backward (bool, Optional):
                Whether to overwrite logits with their gradients. Default: `False`.
            process_group (ProcessGroup, Optional):
                Group sharding the vocabulary into equal contiguous partitions. Default: `None`.
            return_z_loss (bool, Optional):
                Whether to also return z-loss for logging, without its own backward. Default: `False`.
        """
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise NotImplementedError(f"Unsupported reduction: {reduction}")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.logit_softcapping = logit_softcapping
        self.inplace_backward = inplace_backward
        self.process_group = process_group
        self.return_z_loss = return_z_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Compute loss from `[N, V]` logits and `[N]` targets.

        Return FP32 scalars for `mean` or `sum`, and `[N]` tensors for `none`.
        When `return_z_loss=True`, also return the reduced z-loss component.
        """
        assert input.device.type in ('cuda', 'npu', 'xpu') and target.device.type in ('cuda', 'npu', 'xpu'), (
            "Only support CUDA/NPU/XPU tensors"
        )
        loss, z_loss = cross_entropy_loss(
            logits=input,
            target=target,
            label_smoothing=self.label_smoothing,
            logit_scale=self.logit_scale,
            lse_square_scale=self.lse_square_scale,
            logit_softcapping=self.logit_softcapping,
            ignore_index=self.ignore_index,
            inplace_backward=self.inplace_backward,
            process_group=self.process_group,
        )
        if self.reduction != 'none':
            loss = loss.sum()
            if self.return_z_loss:
                z_loss = z_loss.sum()
            if self.reduction == 'mean':
                total = (target != self.ignore_index).sum()
                loss = loss / total
                if self.return_z_loss:
                    z_loss = z_loss / total
        return (loss, z_loss) if self.return_z_loss else loss
