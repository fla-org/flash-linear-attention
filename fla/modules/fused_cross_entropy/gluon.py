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
from fla.modules.fused_cross_entropy.ops import cross_entropy_bwd, cross_entropy_fwd

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    exp = extra.libdevice.fast_expf
    log = extra.libdevice.fast_logf
else:
    exp = gl.exp
    log = gl.log
tanh = extra.libdevice.tanh


@gluon.jit(do_not_specialize=['N'])
def cross_entropy_fwd_kernel_gluon(
    logits,  # [N, V]
    target,  # [N]
    loss,  # [NV, N] or [N] when NV == 1
    z_loss,  # [N]
    lse,  # [NV, N] or [N] when NV == 1
    s_logits,
    label_smoothing: gl.constexpr,
    logit_scale: gl.constexpr,
    lse_square_scale: gl.constexpr,
    logit_softcapping: gl.constexpr,
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
    b_logits = gl.load(p_logits + o_v, mask=m_v, other=-float('inf')).to(gl.float32) * logit_scale
    if logit_softcapping is not None:
        b_logits = logit_softcapping * tanh(b_logits / logit_softcapping)
    # these transforms do not preserve -inf in padded lanes
    if logit_softcapping is not None or logit_scale <= 0:
        b_logits = gl.where(m_v, b_logits, -float('inf'))
    if label_smoothing > 0:
        b_sum = gl.sum(gl.where(m_v, b_logits, 0.0), 0)
    b_max = gl.max(b_logits, 0)
    b_lse = log(gl.sum(exp(b_logits - b_max), 0)) + b_max

    b_target = gl.load(target + i_n).to(gl.int64)
    b_loss = 0.0
    b_z_loss = 0.0
    if b_target != ignore_index:
        b_target -= V_START
        m_target = (b_target >= i_v * BV) & (b_target < gl.minimum(V, (i_v + 1) * BV))
        b_target_logit = gl.load(p_logits + b_target, mask=m_target, other=0).to(gl.float32) * logit_scale
        if logit_softcapping is not None:
            b_target_logit = logit_softcapping * tanh(b_target_logit / logit_softcapping)
        b_loss = -b_target_logit
        if label_smoothing > 0:
            b_loss = (1 - label_smoothing) * b_loss - label_smoothing * b_sum / V_TOTAL
        if not SPLIT:
            b_z_loss = lse_square_scale * b_lse * b_lse
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
    label_smoothing: gl.constexpr,
    logit_scale: gl.constexpr,
    lse_square_scale: gl.constexpr,
    logit_softcapping: gl.constexpr,
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
    b_logits = gl.load(p_logits, mask=m_v, other=0).to(gl.float32) * logit_scale
    if logit_softcapping is not None:
        b_tanh = tanh(b_logits / logit_softcapping)
        b_logits = logit_softcapping * b_tanh
    b_lse = gl.load(lse + i_n)
    # [BV]
    b_probs = exp(b_logits - b_lse)
    b_dlogits = b_probs + 2.0 * lse_square_scale * b_lse * b_probs
    TARGET_WEIGHT: gl.constexpr = 1 - label_smoothing if label_smoothing > 0 else 1.0
    b_target -= V_START
    if R == R_MAX and V > BV:
        b_target -= i_v * BV
        if (b_target >= 0) & (b_target < BV):
            b_dlogits -= gl.where(o_local == b_target.to(gl.int32), TARGET_WEIGHT, 0.0)
    else:
        b_dlogits -= gl.where(o_v == b_target, TARGET_WEIGHT, 0.0)
    if label_smoothing > 0:
        b_dlogits -= label_smoothing / V_TOTAL
    if logit_softcapping is not None:
        b_dlogits *= 1.0 - b_tanh * b_tanh
    b_dlogits *= b_dloss * logit_scale
    gl.store(p_dlogits, b_dlogits, mask=m_v)


@gluon.jit(do_not_specialize=['N'])
def cross_entropy_fwd_kernel_reduce_gluon(
    partial_loss,
    partial_lse,
    target,
    loss,
    z_loss,
    lse,
    lse_square_scale: gl.constexpr,
    ignore_index: gl.constexpr,
    N,
    NV: gl.constexpr,
    BN: gl.constexpr,
    BS: gl.constexpr,
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
        b_z_loss = gl.where(valid, lse_square_scale * b_lse * b_lse, 0.)
        b_loss = gl.where(valid, b_loss + b_lse + b_z_loss, 0.)
        gl.store(z_loss + o_n, b_z_loss, mask=o_n < N)
    gl.store(loss + o_n, b_loss, mask=o_n < N)
    gl.store(lse + o_n, b_lse, mask=o_n < N)


def cross_entropy_reduce_gluon(
    loss: torch.Tensor,
    z_loss: torch.Tensor,
    lse: torch.Tensor,
    target: torch.Tensor,
    lse_square_scale: float,
    ignore_index: int,
    process_group: ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = target.shape[0]
    world_size = 1 if process_group is None else torch.distributed.get_world_size(group=process_group)
    if lse.ndim == 2:
        NV = lse.shape[0]
        BS = triton.next_power_of_2(NV)
        BN = 128 // max(1, min(8, BS // 4, 16384 // triton.next_power_of_2(N)))
        reduced_loss = loss.new_empty(N)
        reduced_lse = torch.empty_like(reduced_loss)
        cross_entropy_fwd_kernel_reduce_gluon[(triton.cdiv(N, BN),)](
            partial_loss=loss,
            partial_lse=lse,
            target=target,
            loss=reduced_loss,
            z_loss=z_loss,
            lse=reduced_lse,
            lse_square_scale=lse_square_scale,
            ignore_index=ignore_index,
            N=N,
            NV=NV,
            BN=BN,
            BS=BS,
            SPLIT=world_size > 1,
            REDUCE_LOSS=True,
            num_warps=4,
        )
        loss, lse = reduced_loss, reduced_lse
    if world_size > 1:
        BS = triton.next_power_of_2(world_size)
        BN = 128 // max(1, min(8, BS // 4, 16384 // triton.next_power_of_2(N)))
        gathered_lse = lse.new_empty(world_size, N)
        torch.distributed.all_gather_into_tensor(output_tensor=gathered_lse, input_tensor=lse, group=process_group)
        torch.distributed.all_reduce(tensor=loss, group=process_group)
        cross_entropy_fwd_kernel_reduce_gluon[(triton.cdiv(N, BN),)](
            partial_loss=loss,
            partial_lse=gathered_lse,
            target=target,
            loss=loss,
            z_loss=z_loss,
            lse=lse,
            lse_square_scale=lse_square_scale,
            ignore_index=ignore_index,
            N=N,
            NV=world_size,
            BN=BN,
            BS=BS,
            SPLIT=False,
            REDUCE_LOSS=False,
            num_warps=4,
        )
    return loss, z_loss, lse


def cross_entropy_fwd_verifier(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    logit_softcapping: float | None = None,
    ignore_index: int = -100,
    process_group: ProcessGroup | None = None,
) -> tuple[bool, str | None]:
    if logits.device.type != 'cuda' or target.device != logits.device:
        return False, "Logits and targets must be on the same CUDA device"
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False, "Gluon cross entropy supports fp16, bf16, and fp32 logits"
    return True, None


@register(cross_entropy_fwd, backend=GluonBackend, verifier=cross_entropy_fwd_verifier, env_var='FLA_CE_GLUON')
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
    world_size = 1 if process_group is None else torch.distributed.get_world_size(group=process_group)
    total_classes = world_size * V
    rank = 0 if process_group is None else torch.distributed.get_rank(group=process_group)
    class_start_idx = rank * V

    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    BV = triton.next_power_of_2(V)
    if logit_softcapping is not None:
        BV = min(BV, 2048)
    elif V > 16384:
        BV = 8192
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
        z_loss=z_loss,
        lse=lse,
        s_logits=logits.stride(0),
        label_smoothing=label_smoothing,
        logit_scale=logit_scale,
        lse_square_scale=lse_square_scale,
        logit_softcapping=logit_softcapping,
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
            z_loss=z_loss,
            lse=lse,
            target=target,
            lse_square_scale=lse_square_scale,
            ignore_index=ignore_index,
            process_group=process_group,
        )

    return loss, z_loss, lse, total_classes, class_start_idx


def cross_entropy_bwd_verifier(
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
) -> tuple[bool, str | None]:
    return cross_entropy_fwd_verifier(logits=logits, target=target)


@register(cross_entropy_bwd, backend=GluonBackend, verifier=cross_entropy_bwd_verifier, env_var='FLA_CE_GLUON')
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
        label_smoothing=label_smoothing,
        logit_scale=logit_scale,
        lse_square_scale=lse_square_scale,
        logit_softcapping=logit_softcapping,
        ignore_index=ignore_index,
        V=V,
        V_TOTAL=V if total_classes is None else total_classes,
        V_START=class_start_idx,
        BV=BV,
        num_warps=4,
    )
    return dlogits
