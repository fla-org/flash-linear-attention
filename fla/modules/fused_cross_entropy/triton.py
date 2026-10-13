# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import triton
import triton.language as tl

from fla.modules.fused_cross_entropy.common import _cross_entropy_bwd, _cross_entropy_fwd
from fla.ops.utils.op import exp, log, tanh

if TYPE_CHECKING:
    import torch
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
    i_v = tl.program_id(1).to(tl.int64)
    o_v = tl.arange(0, BV)
    # a bounded lane count avoids wide comparisons without narrowing global addresses
    m_v = o_v < tl.minimum(V - i_v * BV, BV).to(tl.int32)
    p_logits = logits + i_n * s_logits

    # [BV]
    b_logits = tl.load(p_logits + i_v * BV + o_v.to(tl.int64), mask=m_v, other=-float('inf')).to(tl.float32) * scale
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
    V = logits.shape[-1]
    BV = min(triton.next_power_of_2(V), 64 * 1024)
    # reduce softcap register pressure without introducing an additional split reduction
    if logit_softcapping is not None and V > BV:
        BV = 8 * 1024
    return _cross_entropy_fwd(
        logits=logits,
        target=target,
        label_smoothing=label_smoothing,
        logit_scale=logit_scale,
        lse_square_scale=lse_square_scale,
        logit_softcapping=logit_softcapping,
        ignore_index=ignore_index,
        process_group=process_group,
        kernel=cross_entropy_fwd_kernel,
        block_size=BV,
        num_warps=4 if BV < 2048 else (8 if BV < 8192 else 16),
    )


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
) -> torch.Tensor:
    BV = min(triton.next_power_of_2(logits.shape[-1]), 4 * 1024)
    return _cross_entropy_bwd(
        logits=logits,
        target=target,
        lse=lse,
        dloss=dloss,
        label_smoothing=label_smoothing,
        logit_scale=logit_scale,
        lse_square_scale=lse_square_scale,
        logit_softcapping=logit_softcapping,
        ignore_index=ignore_index,
        total_classes=total_classes,
        class_start_idx=class_start_idx,
        inplace_backward=inplace_backward,
        kernel=cross_entropy_bwd_kernel,
        block_size=BV,
        num_warps=4 if BV < 2048 else 8,
    )
