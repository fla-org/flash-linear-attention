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
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import extra

from fla.backends import GluonBackend, register
from fla.modules.fused_cross_entropy.ops import cross_entropy_bwd, cross_entropy_fwd, cross_entropy_loss
from fla.utils import input_guard

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
    NW: gl.constexpr = 4 if BV < 2048 else (8 if BV < 8192 else 16)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
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
    NW: gl.constexpr = 4 if BV < 2048 else (8 if BV < 8192 else 16)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    i_n = gl.program_id(0).to(gl.int64)
    i_v = gl.program_id(1).to(gl.int64)
    o_v = i_v * BV + gl.arange(0, BV, layout=layout).to(gl.int64)
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
    b_target -= V_START
    if smoothing > 0:
        b_dlogits -= gl.where(o_v == b_target, 1 - smoothing, 0.0)
        b_dlogits -= smoothing / V_TOTAL
    else:
        b_dlogits -= gl.where(o_v == b_target, 1.0, 0.0)
    if softcap is not None:
        b_dlogits *= 1.0 - b_tanh * b_tanh
    b_dlogits *= b_dloss * scale
    gl.store(p_dlogits, b_dlogits, mask=m_v)


class GluonCrossEntropyFunction(torch.autograd.Function):

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
            kernel=cross_entropy_fwd_kernel_gluon,
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
            kernel=cross_entropy_bwd_kernel_gluon,
        )
        return dlogits, None, None, None, None, None, None, None, None


def cross_entropy_loss_verifier(
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
    return GluonCrossEntropyFunction.apply(
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
