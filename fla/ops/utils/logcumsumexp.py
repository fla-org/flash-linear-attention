# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from fla.ops.utils.op import exp, log, unflatten_program_id
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard


@triton.jit
def _logaddexp(a, b):
    maximum = tl.maximum(a, b)
    difference = tl.where(maximum == float('-inf'), 0., -tl.abs(a - b))
    return maximum + tldevice.log1p(exp(difference))


@triton.jit
def _logcumsumexp_normalize_fwd(
    x, z, k, g, initial_state, final_state, cu_seqlens,
    T, S: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr,
    IS_VARLEN: tl.constexpr, HAS_INITIAL_STATE: tl.constexpr, SAVE_Z: tl.constexpr,
):
    i_s, i_n = unflatten_program_id(tl.cdiv(S, BS))
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_n * T, (i_n + 1) * T
    o_s = i_s * BS + tl.arange(0, BS)
    o_i = tl.arange(0, BT)
    if HAS_INITIAL_STATE:
        b_previous = tl.load(initial_state + i_n * S + o_s, o_s < S, other=float('-inf'))
    else:
        b_previous = tl.full((BS,), float('-inf'), tl.float32)
    for i_t in range(tl.cdiv(eos - bos, BT)):
        o_t = i_t * BT + o_i
        offsets = (bos + o_t[:, None]) * S + o_s[None, :]
        mask = (o_t[:, None] < eos - bos) & (o_s[None, :] < S)
        b_x = tl.load(x + offsets, mask, other=float('-inf')).to(tl.float32)
        b_z = tl.associative_scan(b_x, axis=0, combine_fn=_logaddexp)
        b_z = _logaddexp(b_z, b_previous[None, :])
        previous_indices = tl.broadcast_to(tl.maximum(o_i - 1, 0)[:, None], (BT, BS))
        b_prev_z = tl.gather(b_z, previous_indices, axis=0)
        b_prev_z = tl.where(o_i[:, None] == 0, b_previous[None, :], b_prev_z)
        b_k = exp(b_x - b_z)
        b_g = tl.where(b_prev_z == float('-inf'), 0., b_prev_z - b_z)
        tl.store(k + offsets, b_k, mask)
        tl.store(g + offsets, b_g, mask)
        if SAVE_Z:
            tl.store(z + offsets, b_z, mask)
        last = tl.minimum((i_t + 1) * BT, eos - bos) - 1
        b_previous = tl.sum(tl.where(o_t[:, None] == last, b_z, 0.), axis=0)
    tl.store(final_state + i_n * S + o_s, b_previous, o_s < S)


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=['S'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def logcumsumexp_fwd_kernel(
    s,
    z,
    T,
    S: tl.constexpr,
    BT: tl.constexpr,
    k=None,
    g=None,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    NORMALIZE: tl.constexpr = False,
    BS: tl.constexpr = 32,
    IS_VARLEN: tl.constexpr = False,
    HAS_INITIAL_STATE: tl.constexpr = False,
    SAVE_Z: tl.constexpr = True,
):
    if NORMALIZE:
        _logcumsumexp_normalize_fwd(
            s, z, k, g, initial_state, final_state, cu_seqlens,
            T, S, BT, BS, IS_VARLEN, HAS_INITIAL_STATE, SAVE_Z,
        )
    else:
        i_bh = tl.program_id(0).to(tl.int64)
        o_i = tl.arange(0, BT)
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

        b_mp = tl.full([S], float('-inf'), dtype=tl.float32)
        b_zp = tl.zeros([S], dtype=tl.float32)
        for i_t in range(tl.cdiv(T, BT)):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            p_s = s + i_bh * T*S + o_t[:, None] * S + tl.arange(0, S)[None, :]
            p_z = z + i_bh * T*S + o_t[:, None] * S + tl.arange(0, S)[None, :]

            # [BT, S]
            b_s = tl.load(p_s, mask=m_t[:, None], other=0.0).to(tl.float32)
            # [S,]
            b_mc = tl.max(b_s, 0)
            b_mc = tl.maximum(b_mp, b_mc)
            b_zp = b_zp * exp(b_mp - b_mc)
            # [BT, S]
            b_s = exp(b_s - b_mc)
            b_z = tl.dot(m_s, b_s, allow_tf32=False) + b_zp
            # [S,]
            b_zc = tl.max(b_z, 0)
            b_mp = b_mc
            b_zp = b_zc
            # [BT, BS]
            # small eps to prevent underflows
            b_z = log(tl.where(b_z != 0, b_z, 1e-20)) + b_mc
            tl.store(p_z, b_z.to(p_z.dtype.element_ty), mask=m_t[:, None])


@triton.jit
def _reverse_recurrence(a_left, b_left, a_right, b_right):
    return a_left * a_right, b_left * a_right + b_right


@triton.jit(do_not_specialize=['T'])
def logcumsumexp_normalize_bwd_kernel(
    x, z, initial_state, cu_seqlens, dk, dg, dfinal_state, dx, dinitial_state,
    T, S: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr,
    IS_VARLEN: tl.constexpr, HAS_INITIAL_STATE: tl.constexpr,
    HAS_DK: tl.constexpr, HAS_DG: tl.constexpr, HAS_DFINAL: tl.constexpr,
):
    i_s, i_n = unflatten_program_id(tl.cdiv(S, BS))
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_n * T, (i_n + 1) * T
    o_s = i_s * BS + tl.arange(0, BS)
    if HAS_INITIAL_STATE:
        b_initial = tl.load(initial_state + i_n * S + o_s, o_s < S, other=float('-inf'))
    else:
        b_initial = tl.full((BS,), float('-inf'), tl.float32)
    b_carry = tl.full((BS,), 0., tl.float32)
    if HAS_DFINAL:
        b_dfinal = tl.load(dfinal_state + i_n * S + o_s, o_s < S, other=0.).to(tl.float32)
    else:
        b_dfinal = tl.full((BS,), 0., tl.float32)
    for i_t in range(tl.cdiv(eos - bos, BT)):
        o_t = eos - bos - 1 - i_t * BT - tl.arange(0, BT)
        offsets = (bos + o_t[:, None]) * S + o_s[None, :]
        mask = (o_t[:, None] >= 0) & (o_s[None, :] < S)
        next_mask = mask & (o_t[:, None] + 1 < eos - bos)
        b_x = tl.load(x + offsets, mask, other=0.).to(tl.float32)
        b_z = tl.load(z + offsets, mask, other=0.)
        b_z_next = tl.load(z + offsets + S, next_mask, other=0.)
        b_a = exp(b_x - b_z)
        if HAS_DK:
            b_dk = tl.load(dk + offsets, mask, other=0.).to(tl.float32)
        else:
            b_dk = tl.full((BT, BS), 0., tl.float32)
        b_dz = -b_dk * b_a
        if HAS_DG:
            b_dg = tl.load(dg + offsets, mask, other=0.).to(tl.float32)
            b_dg = tl.where((o_t[:, None] > 0) | (b_initial[None, :] != float('-inf')), b_dg, 0.)
            b_dg_next = tl.load(dg + offsets + S, next_mask, other=0.).to(tl.float32)
            b_dz += b_dg_next - b_dg
        b_dz += tl.where(o_t[:, None] == eos - bos - 1, b_dfinal[None, :], 0.)
        # all exponent differences in the reverse recurrence are nonpositive.
        b_decay = exp(tl.where(next_mask, b_z - b_z_next, 0.))
        b_decay, b_dz = tl.associative_scan((b_decay, b_dz), axis=0, combine_fn=_reverse_recurrence)
        b_r = b_dz + b_decay * b_carry[None, :]
        tl.store(dx + offsets, b_a * (b_dk + b_r), mask)
        last = tl.maximum(eos - bos - (i_t + 1) * BT, 0)
        b_carry = tl.sum(tl.where(o_t[:, None] == last, b_r, 0.), axis=0)
    if HAS_INITIAL_STATE:
        if eos > bos:
            b_first = tl.load(z + bos * S + o_s, o_s < S, other=0.)
            b_dinitial = exp(b_initial - b_first) * b_carry
            if HAS_DG:
                b_dg_first = tl.load(dg + bos * S + o_s, o_s < S, other=0.).to(tl.float32)
                b_dinitial += tl.where(b_initial != float('-inf'), b_dg_first, 0.)
        else:
            b_dinitial = b_dfinal
        tl.store(dinitial_state + i_n * S + o_s, b_dinitial, o_s < S)


class LogcumsumexpNormalizeFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, initial_state, cu_seqlens):
        B, T, H, K = x.shape
        N = B if cu_seqlens is None else cu_seqlens.numel() - 1
        S, BS = H * K, 32
        k, g = torch.empty_like(x), torch.empty_like(x)
        save_z = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        z = torch.empty_like(x, dtype=torch.float32) if save_z else None
        final_state = x.new_empty((N, 1, H, K), dtype=torch.float32)
        logcumsumexp_fwd_kernel[(N * triton.cdiv(S, BS),)](
            s=x,
            z=z,
            T=T,
            S=S,
            k=k,
            g=g,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            NORMALIZE=True,
            BS=BS,
            IS_VARLEN=cu_seqlens is not None,
            HAS_INITIAL_STATE=initial_state is not None,
            SAVE_Z=save_z,
        )
        ctx.save_for_backward(x, z, initial_state, cu_seqlens)
        ctx.set_materialize_grads(False)
        return k, g, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dk, dg, dfinal_state):
        x, z, initial_state, cu_seqlens = ctx.saved_tensors
        B, T, H, K = x.shape
        N = B if cu_seqlens is None else cu_seqlens.numel() - 1
        S, BS = H * K, 32
        dx = torch.empty_like(x)
        dinitial_state = torch.empty_like(initial_state) if initial_state is not None else None
        logcumsumexp_normalize_bwd_kernel[(N * triton.cdiv(S, BS),)](
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
            IS_VARLEN=cu_seqlens is not None,
            HAS_INITIAL_STATE=initial_state is not None,
            HAS_DK=dk is not None,
            HAS_DG=dg is not None,
            HAS_DFINAL=dfinal_state is not None,
        )
        return dx, dinitial_state, None


def logcumsumexp_normalize(
    x: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cumulative normalized keys, log-decay gates, and FP32 final normalizers per sequence."""
    if cu_seqlens is not None and x.shape[0] != 1:
        raise ValueError("Packed inputs with cu_seqlens must have batch size 1.")
    if initial_state is not None:
        N = x.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
        expected_shape = (N, 1, *x.shape[2:])
        if initial_state.shape != expected_shape:
            raise ValueError(f"initial_state must have shape {expected_shape}, got {tuple(initial_state.shape)}.")
    return LogcumsumexpNormalizeFunction.apply(x, initial_state, cu_seqlens)
