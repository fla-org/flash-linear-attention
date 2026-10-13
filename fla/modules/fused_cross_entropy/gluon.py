# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import extra

from fla.backends import GluonBackend, register
from fla.modules.fused_cross_entropy.ops import cross_entropy_loss
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    exp = extra.libdevice.fast_expf
    log = extra.libdevice.fast_logf
    tanh = extra.libdevice.fast_tanhf
else:
    exp = gl.exp
    log = gl.log
    tanh = extra.libdevice.tanh


@gluon.jit(do_not_specialize=['N'])
def cross_entropy_fwd_kernel_gluon(
    logits,  # [N, V]
    target,  # [N]
    loss,  # [NV, N] or [N] when NV == 1
    lse,  # [NV, N] or [N] when NV == 1
    z_loss,  # [N]
    s_logits,
    scale: gl.constexpr,
    softcap: gl.constexpr,
    smoothing: gl.constexpr,
    z_scale: gl.constexpr,
    ignore_index: gl.constexpr,
    N,
    V: gl.constexpr,
    V_TOTAL: gl.constexpr,
    V_START: gl.constexpr,
    BV: gl.constexpr,
    SPLIT: gl.constexpr,
):
    NW: gl.constexpr = 4
    R_MAX: gl.constexpr = 128 // logits.dtype.element_ty.primitive_bitwidth
    R: gl.constexpr = (V & -V) if (V & -V) < R_MAX else R_MAX
    layout: gl.constexpr = gl.BlockedLayout([R], [32], [NW], [0])
    i_n = gl.program_id(0).to(gl.int64)
    i_v = gl.program_id(1).to(gl.int64)
    o_v = i_v * BV + gl.arange(0, BV, layout=layout).to(gl.int64)
    m_v = o_v < V
    p_logits = logits + i_n * s_logits

    # [BV]
    b_logits = gl.load(p_logits + o_v, mask=m_v, other=-float('inf')).to(gl.float32) * scale
    if softcap is not None:
        b_logits = softcap * tanh(b_logits / softcap)
    # these transforms do not preserve -inf in padded lanes
    if softcap is not None or scale <= 0:
        b_logits = gl.where(m_v, b_logits, -float('inf'))
    if smoothing > 0:
        b_sum = gl.sum(gl.where(m_v, b_logits, 0.0), 0)
    b_max = gl.max(b_logits, 0)
    b_lse = log(gl.sum(exp(b_logits - b_max), 0)) + b_max

    b_target = gl.load(target + i_n).to(gl.int64)
    b_loss = 0.0
    b_z_loss = 0.0
    if b_target != ignore_index:
        b_target -= V_START
        m_target = (b_target >= i_v * BV) & (b_target < gl.minimum(V, (i_v + 1) * BV))
        b_target_logit = gl.load(p_logits + b_target, mask=m_target, other=0).to(gl.float32) * scale
        if softcap is not None:
            b_target_logit = softcap * tanh(b_target_logit / softcap)
        b_loss = -b_target_logit
        if smoothing > 0:
            b_loss = (1 - smoothing) * b_loss - smoothing * b_sum / V_TOTAL
        if not SPLIT:
            b_z_loss = z_scale * b_lse * b_lse
            b_loss += b_lse + b_z_loss

    gl.store(loss + i_v * N + i_n, b_loss)
    gl.store(lse + i_v * N + i_n, b_lse)
    if not SPLIT:
        gl.store(z_loss + i_n, b_z_loss)


@gluon.jit
def cross_entropy_bwd_kernel_gluon(
    logits,  # [N, V]
    target,  # [N]
    lse,  # [N]
    dloss,  # [N]
    dlogits,  # [N, V]
    s_logits,
    s_dloss,
    s_dlogits,
    scale: gl.constexpr,
    softcap: gl.constexpr,
    smoothing: gl.constexpr,
    z_scale: gl.constexpr,
    ignore_index: gl.constexpr,
    V: gl.constexpr,
    V_TOTAL: gl.constexpr,
    V_START: gl.constexpr,
    BV: gl.constexpr,
):
    NW: gl.constexpr = 4
    R_MAX: gl.constexpr = 128 // logits.dtype.element_ty.primitive_bitwidth
    R: gl.constexpr = (V & -V) if (V & -V) < R_MAX else R_MAX
    layout: gl.constexpr = gl.BlockedLayout([R], [32], [NW], [0])
    i_n = gl.program_id(0).to(gl.int64)
    i_v = gl.program_id(1).to(gl.int64)
    o_local = gl.arange(0, BV, layout=layout)
    o_v = i_v * BV + o_local.to(gl.int64)
    if R == R_MAX and V > BV:
        m_v = o_local < gl.minimum(V - i_v * BV, BV).to(gl.int32)
    else:
        m_v = o_v < V
    p_logits = logits + i_n * s_logits + o_v
    p_dlogits = dlogits + i_n * s_dlogits + o_v

    b_target = gl.load(target + i_n).to(gl.int64)
    b_dloss = gl.load(dloss + i_n * s_dloss, mask=b_target != ignore_index, other=0).to(gl.float32)
    # [BV]
    b_logits = gl.load(p_logits, mask=m_v, other=0).to(gl.float32) * scale
    if softcap is not None:
        b_tanh = tanh(b_logits / softcap)
        b_logits = softcap * b_tanh
    b_lse = gl.load(lse + i_n)
    # [BV]
    b_probs = exp(b_logits - b_lse)
    b_dlogits = b_probs + 2.0 * z_scale * b_lse * b_probs
    TARGET_WEIGHT: gl.constexpr = 1 - smoothing if smoothing > 0 else 1.0
    b_target -= V_START
    if R == R_MAX and V > BV:
        b_target -= i_v * BV
        if (b_target >= 0) & (b_target < BV):
            b_dlogits -= gl.where(o_local == b_target.to(gl.int32), TARGET_WEIGHT, 0.0)
    else:
        b_dlogits -= gl.where(o_v == b_target, TARGET_WEIGHT, 0.0)
    if smoothing > 0:
        b_dlogits -= smoothing / V_TOTAL
    if softcap is not None:
        b_dlogits *= 1.0 - b_tanh * b_tanh
    b_dlogits *= b_dloss * scale
    gl.store(p_dlogits, b_dlogits, mask=m_v)


@triton.heuristics({
    'BN': lambda args: 128 // max(
        1,
        min(8, triton.next_power_of_2(args['NV']) // 4, 16384 // triton.next_power_of_2(args['N'])),
    ),
})
@gluon.jit(do_not_specialize=['N'])
def cross_entropy_fwd_kernel_reduce_gluon(
    partial_loss,
    partial_lse,
    target,
    loss,
    lse,
    z_loss,
    N,
    NV: gl.constexpr,
    BN: gl.constexpr,
    BS: gl.constexpr,
    z_scale: gl.constexpr,
    ignore_index: gl.constexpr,
    SPLIT: gl.constexpr,
    REDUCE_LOSS: gl.constexpr,
):
    # keep partition reductions within a warp to avoid shared memory
    P: gl.constexpr = 128 // BN
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [P, 32 // P], [1, 4], [1, 0])
    o_n = gl.program_id(0).to(gl.int64) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, layout))
    o_s = gl.arange(0, BS, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    offsets = o_s[:, None] * N + o_n[None, :]
    mask = (o_s[:, None] < NV) & (o_n[None, :] < N)
    b_lse = gl.load(partial_lse + offsets, mask=mask, other=-float('inf')).to(gl.float32)
    b_max = gl.max(b_lse, 0)
    b_lse = b_max + log(gl.sum(exp(b_lse - b_max[None, :]), 0))
    if REDUCE_LOSS:
        b_loss = gl.sum(gl.load(partial_loss + offsets, mask=mask, other=0).to(gl.float32), 0)
    else:
        b_loss = gl.load(partial_loss + o_n, mask=o_n < N, other=0).to(gl.float32)
    if not SPLIT:
        valid = gl.load(target + o_n, mask=o_n < N, other=ignore_index) != ignore_index
        b_z_loss = gl.where(valid, z_scale * b_lse * b_lse, 0.)
        b_loss = gl.where(valid, b_loss + b_lse + b_z_loss, 0.)
        gl.store(z_loss + o_n, b_z_loss, mask=o_n < N)
    gl.store(loss + o_n, b_loss, mask=o_n < N)
    gl.store(lse + o_n, b_lse, mask=o_n < N)


def cross_entropy_reduce_gluon(
    loss: torch.Tensor,
    lse: torch.Tensor,
    z_loss: torch.Tensor,
    target: torch.Tensor,
    lse_square_scale: float,
    ignore_index: int,
    world_size: int,
    process_group: ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = target.shape[0]
    if lse.ndim == 2:
        NV = lse.shape[0]
        reduced_loss = loss.new_empty(N)
        reduced_lse = torch.empty_like(reduced_loss)
        cross_entropy_fwd_kernel_reduce_gluon[lambda meta: (triton.cdiv(N, meta['BN']),)](
            partial_loss=loss,
            partial_lse=lse,
            target=target,
            loss=reduced_loss,
            lse=reduced_lse,
            z_loss=z_loss,
            N=N,
            NV=NV,
            BN=128,
            BS=triton.next_power_of_2(NV),
            z_scale=lse_square_scale,
            ignore_index=ignore_index,
            SPLIT=world_size > 1,
            REDUCE_LOSS=True,
            num_warps=4,
        )
        loss, lse = reduced_loss, reduced_lse
    if world_size > 1:
        gathered_lse = lse.new_empty(world_size, N)
        torch.distributed.all_gather_into_tensor(gathered_lse, lse, group=process_group)
        torch.distributed.all_reduce(loss, group=process_group)
        cross_entropy_fwd_kernel_reduce_gluon[lambda meta: (triton.cdiv(N, meta['BN']),)](
            partial_loss=loss,
            partial_lse=gathered_lse,
            target=target,
            loss=loss,
            lse=lse,
            z_loss=z_loss,
            N=N,
            NV=world_size,
            BN=128,
            BS=triton.next_power_of_2(world_size),
            z_scale=lse_square_scale,
            ignore_index=ignore_index,
            SPLIT=False,
            REDUCE_LOSS=False,
            num_warps=4,
        )
    return loss, z_loss, lse


def cross_entropy_fwd_gluon(
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
    BV = min(triton.next_power_of_2(V), 2048 if logit_softcapping is not None else 8192)
    # vocab partitions contribute partial loss until the global LSE is available
    NV = triton.cdiv(V, BV)
    split = world_size > 1 or NV > 1
    shape = (NV, N) if NV > 1 else (N,)
    loss = logits.new_empty(shape, dtype=torch.float)
    lse = torch.empty_like(loss)
    z_loss = logits.new_empty(N, dtype=torch.float)

    cross_entropy_fwd_kernel_gluon[(N, NV)](
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
        num_warps=4,
    )

    if split:
        loss, z_loss, lse = cross_entropy_reduce_gluon(
            loss=loss,
            lse=lse,
            z_loss=z_loss,
            target=target,
            lse_square_scale=lse_square_scale,
            ignore_index=ignore_index,
            world_size=world_size,
            process_group=process_group,
        )

    return loss, z_loss, lse, total_classes, class_start_idx


def cross_entropy_bwd_gluon(
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
    N, V = logits.shape
    BV = min(triton.next_power_of_2(V), 1024 if logit_softcapping is not None else 4096)
    dlogits = logits if inplace_backward else torch.empty_like(logits)
    cross_entropy_bwd_kernel_gluon[(N, triton.cdiv(V, BV))](
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
        num_warps=4,
    )
    return dlogits


class FusedCrossEntropyGluonFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
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
        loss, z_loss, lse, total_classes, class_start_idx = cross_entropy_fwd_gluon(
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
    @autocast_custom_bwd
    def backward(ctx, dloss, dz_loss):
        # z-loss is returned for logging and is already included in the differentiable loss
        del dz_loss

        logits, target, lse = ctx.saved_tensors
        dlogits = cross_entropy_bwd_gluon(
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


def cross_entropy_loss_verifier(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    logit_softcapping: float | None = None,
    ignore_index: int = -100,
    inplace_backward: bool = False,
    process_group: ProcessGroup | None = None,
) -> tuple[bool, str | None]:
    if logits.device.type != 'cuda' or target.device != logits.device:
        return False, "Logits and targets must be on the same CUDA device"
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False, "Gluon cross entropy supports fp16, bf16, and fp32 logits"
    return True, None


@register(cross_entropy_loss, backend=GluonBackend, verifier=cross_entropy_loss_verifier, env_var='FLA_CE_GLUON')
def cross_entropy_loss_gluon(
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
    return FusedCrossEntropyGluonFunction.apply(
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
