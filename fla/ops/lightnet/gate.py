# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.logcumsumexp import logcumsumexp
from fla.ops.utils.op import exp, unflatten_program_id
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=['S'],
    **autotune_cache_kwargs,
)
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'STORE_Z': lambda args: args['z'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_lightnet_gate_fwd_kernel(
    x,
    z,
    k,
    g,
    initial_state,
    final_state,
    cu_seqlens,
    T,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_Z: tl.constexpr,
):
    i_s, i_n = unflatten_program_id(tl.cdiv(S, BS))
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_n * T, (i_n + 1) * T
    o_s = i_s * BS + tl.arange(0, BS)
    o_i = tl.arange(0, BT)
    if USE_INITIAL_STATE:
        b_previous = tl.load(initial_state + i_n * S + o_s, mask=o_s < S, other=float('-inf'))
    else:
        b_previous = tl.full((BS,), float('-inf'), tl.float32)
    for i_t in range(tl.cdiv(eos - bos, BT)):
        o_t = i_t * BT + o_i
        offsets = (bos + o_t[:, None]) * S + o_s[None, :]
        mask = (o_t[:, None] < eos - bos) & (o_s[None, :] < S)
        b_x = tl.load(x + offsets, mask=mask, other=float('-inf')).to(tl.float32)
        b_z = logcumsumexp(x=b_x, initial_state=b_previous)
        previous_indices = tl.broadcast_to(tl.maximum(o_i - 1, 0)[:, None], (BT, BS))
        b_prev_z = tl.gather(b_z, previous_indices, axis=0)
        b_prev_z = tl.where(o_i[:, None] == 0, b_previous[None, :], b_prev_z)
        b_k = exp(b_x - b_z)
        b_g = tl.where(b_prev_z == float('-inf'), 0., b_prev_z - b_z)
        tl.store(k + offsets, b_k, mask=mask)
        tl.store(g + offsets, b_g, mask=mask)
        if STORE_Z:
            tl.store(z + offsets, b_z, mask=mask)
        last = tl.minimum((i_t + 1) * BT, eos - bos) - 1
        b_previous = tl.sum(tl.where(o_t[:, None] == last, b_z, 0.), axis=0)
    tl.store(final_state + i_n * S + o_s, b_previous, mask=o_s < S)


@triton.jit
def _reverse_recurrence(a_left, b_left, a_right, b_right):
    return a_left * a_right, b_left * a_right + b_right


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'HAS_DK': lambda args: args['dk'] is not None,
    'HAS_DG': lambda args: args['dg'] is not None,
    'HAS_DFINAL_STATE': lambda args: args['dfinal_state'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_lightnet_gate_bwd_kernel(
    x,
    z,
    initial_state,
    cu_seqlens,
    dk,
    dg,
    dfinal_state,
    dx,
    dinitial_state,
    T,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_DK: tl.constexpr,
    HAS_DG: tl.constexpr,
    HAS_DFINAL_STATE: tl.constexpr,
):
    i_s, i_n = unflatten_program_id(tl.cdiv(S, BS))
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_n * T, (i_n + 1) * T
    o_s = i_s * BS + tl.arange(0, BS)
    if USE_INITIAL_STATE:
        b_initial = tl.load(initial_state + i_n * S + o_s, mask=o_s < S, other=float('-inf'))
    else:
        b_initial = tl.full((BS,), float('-inf'), tl.float32)
    b_carry = tl.full((BS,), 0., tl.float32)
    if HAS_DFINAL_STATE:
        b_dfinal = tl.load(dfinal_state + i_n * S + o_s, mask=o_s < S, other=0.).to(tl.float32)
    else:
        b_dfinal = tl.full((BS,), 0., tl.float32)
    for i_t in range(tl.cdiv(eos - bos, BT)):
        o_t = eos - bos - 1 - i_t * BT - tl.arange(0, BT)
        offsets = (bos + o_t[:, None]) * S + o_s[None, :]
        mask = (o_t[:, None] >= 0) & (o_s[None, :] < S)
        next_mask = mask & (o_t[:, None] + 1 < eos - bos)
        b_x = tl.load(x + offsets, mask=mask, other=0.).to(tl.float32)
        b_z = tl.load(z + offsets, mask=mask, other=0.)
        b_z_next = tl.load(z + offsets + S, mask=next_mask, other=0.)
        b_a = exp(b_x - b_z)
        if HAS_DK:
            b_dk = tl.load(dk + offsets, mask=mask, other=0.).to(tl.float32)
        else:
            b_dk = tl.full((BT, BS), 0., tl.float32)
        b_dz = -b_dk * b_a
        if HAS_DG:
            b_dg = tl.load(dg + offsets, mask=mask, other=0.).to(tl.float32)
            b_dg = tl.where((o_t[:, None] > 0) | (b_initial[None, :] != float('-inf')), b_dg, 0.)
            b_dg_next = tl.load(dg + offsets + S, mask=next_mask, other=0.).to(tl.float32)
            b_dz += b_dg_next - b_dg
        b_dz += tl.where(o_t[:, None] == eos - bos - 1, b_dfinal[None, :], 0.)
        # all exponent differences in the reverse recurrence are nonpositive.
        b_decay = exp(tl.where(next_mask, b_z - b_z_next, 0.))
        b_decay, b_dz = tl.associative_scan((b_decay, b_dz), axis=0, combine_fn=_reverse_recurrence)
        b_r = b_dz + b_decay * b_carry[None, :]
        tl.store(dx + offsets, b_a * (b_dk + b_r), mask=mask)
        last = tl.maximum(eos - bos - (i_t + 1) * BT, 0)
        b_carry = tl.sum(tl.where(o_t[:, None] == last, b_r, 0.), axis=0)
    if USE_INITIAL_STATE:
        if eos > bos:
            b_first = tl.load(z + bos * S + o_s, mask=o_s < S, other=0.)
            b_dinitial = exp(b_initial - b_first) * b_carry
            if HAS_DG:
                b_dg_first = tl.load(dg + bos * S + o_s, mask=o_s < S, other=0.).to(tl.float32)
                b_dinitial += tl.where(b_initial != float('-inf'), b_dg_first, 0.)
        else:
            b_dinitial = b_dfinal
        tl.store(dinitial_state + i_n * S + o_s, b_dinitial, mask=o_s < S)


def fused_lightnet_gate_fwd(
    x: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    save_z: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K = x.shape
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    S, BS = H * K, 32
    k, g = torch.empty_like(x), torch.empty_like(x)
    z = torch.empty_like(x, dtype=torch.float32) if save_z else None
    final_state = x.new_empty((N, 1, H, K), dtype=torch.float32)
    fused_lightnet_gate_fwd_kernel[(N * triton.cdiv(S, BS),)](
        x=x,
        z=z,
        k=k,
        g=g,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        S=S,
        BS=BS,
    )
    return k, g, final_state, z


def fused_lightnet_gate_bwd(
    x: torch.Tensor,
    z: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    dk: torch.Tensor | None,
    dg: torch.Tensor | None,
    dfinal_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K = x.shape
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    S, BS = H * K, 32
    dx = torch.empty_like(x)
    dinitial_state = torch.empty_like(initial_state) if initial_state is not None else None
    fused_lightnet_gate_bwd_kernel[(N * triton.cdiv(S, BS),)](
        x=x,
        z=z,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        dk=dk,
        dg=dg,
        dfinal_state=dfinal_state,
        dx=dx,
        dinitial_state=dinitial_state,
        T=T,
        S=S,
        BT=32,
        BS=BS,
    )
    return dx, dinitial_state


class FusedLightNetGateFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, initial_state, cu_seqlens):
        k, g, final_state, z = fused_lightnet_gate_fwd(
            x=x,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            save_z=ctx.needs_input_grad[0] or ctx.needs_input_grad[1],
        )
        ctx.save_for_backward(x, z, initial_state, cu_seqlens)
        ctx.set_materialize_grads(False)
        return k, g, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dk, dg, dfinal_state):
        x, z, initial_state, cu_seqlens = ctx.saved_tensors
        dx, dinitial_state = fused_lightnet_gate_bwd(
            x=x,
            z=z,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            dk=dk,
            dg=dg,
            dfinal_state=dfinal_state,
        )
        return dx, dinitial_state, None


def fused_lightnet_gate(
    x: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute LightNet's cumulative key normalization and log-decay gates.

    Args:
        x (torch.Tensor):
            Key logits of shape `[B, T, H, K]`. Packed inputs have `B=1`.
        initial_state (torch.Tensor, Optional):
            FP32 log-normalizers of shape `[N, 1, H, K]`, where `N` is the number of sequences. Default: `None`.
        cu_seqlens (torch.Tensor, Optional):
            Cumulative lengths of the packed sequences, including empty sequences. Default: `None`.

    Returns:
        k (torch.Tensor):
            Normalized keys with the shape and dtype of `x`.
        g (torch.Tensor):
            Log-decay gates with the shape and dtype of `x`. An empty prefix has a zero first gate.
        final_state (torch.Tensor):
            FP32 log-normalizers of shape `[N, 1, H, K]`. Empty sequences preserve `initial_state`,
            or return `-inf` without one. Gradients are supported for `x` and `initial_state`.
    """
    if cu_seqlens is not None and x.shape[0] != 1:
        raise ValueError("Packed inputs with cu_seqlens must have batch size 1.")
    if initial_state is not None:
        N = x.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
        expected_shape = (N, 1, *x.shape[2:])
        if initial_state.shape != expected_shape:
            raise ValueError(f"initial_state must have shape {expected_shape}, got {tuple(initial_state.shape)}.")
    return FusedLightNetGateFunction.apply(x, initial_state, cu_seqlens)
