# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Dense gated split backward kernels and their scheduled host launcher."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.swila.atoms import dot_3d_axis2, readout, softmax_2d, softmax_backward_2d, state_update
from fla.ops.swila.dispatch import (
    _is_nvidia_sm90_tensor,
    _use_dense_d64e4_key_bwd_fastpath,
)
from fla.utils import autotune_cache_kwargs

_DENSE_D64E4_BWD_VALUE_TILES = (16, 32)
_DENSE_D64E4_BWD_MIN_VALUE_TILE = min(_DENSE_D64E4_BWD_VALUE_TILES)


def _zero_dense_d64e4_reverse_outputs(args) -> None:
    """Reset reduction buffers before each dense fast-path autotune trial."""
    args["dk_update"].zero_()
    args["dlog_decay"].zero_()


def _zero_dense_d64e4_replay_outputs(args) -> None:
    """Reset replay partials before each dense fast-path autotune trial."""
    args["dq"].zero_()
    args["dk_prediction"].zero_()
    args["dlog_decay"].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BV": BV},
            num_warps=1,
            num_stages=2,
            pre_hook=_zero_dense_d64e4_reverse_outputs,
        )
        for BV in _DENSE_D64E4_BWD_VALUE_TILES
    ],
    key=[
        "H",
        "USE_INITIAL_STATE",
        "USE_FINAL_STATE_GRADIENT",
        "USE_AUX_LOSS_GRAD",
    ],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _fused_recurrent_gated_swila_dense_d64e4_bwd_reverse_kernel(
    q,
    k,
    beta,
    pi_q,
    log_decay,
    r,
    u,
    h0,
    do,
    dht,
    ht,
    d_r_seq,
    dk_update,
    dv,
    dbeta,
    dpi_k,
    dlog_decay,
    du_save,
    dh0,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    D_k: tl.constexpr,
    D_v: tl.constexpr,
    E: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_AUX_LOSS_GRAD: tl.constexpr,
):
    """Reverse half of the dense BF16 D64/E4 key-decay backward."""
    T = T.to(tl.int64)
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * T
    all_bt = B * T

    k_range = tl.arange(0, BK).to(tl.int64)
    v_range = i_v * BV + tl.arange(0, BV).to(tl.int64)
    e_range = tl.arange(0, BE).to(tl.int64)
    mask_k = k_range < D_k
    mask_v = v_range < D_v
    mask_e = e_range < E
    mask_ve = mask_v[:, None] & mask_e[None, :]
    mask_vek = (
        mask_v[:, None, None]
        & mask_e[None, :, None]
        & mask_k[None, None, :]
    )

    end = bos + T - 1
    p_q = q + (end * H + i_h) * D_k + k_range
    p_k = k + (end * H + i_h) * D_k + k_range
    p_do = do + (end * H + i_h) * D_v + v_range
    base_end = (
        (end * H + i_h) * D_v * E
        + v_range[:, None] * E
        + e_range[None, :]
    )
    p_pi_q = pi_q + base_end
    p_beta = beta + base_end
    p_r = r + base_end
    p_u = u + base_end
    if USE_AUX_LOSS_GRAD:
        p_d_r_seq = d_r_seq + base_end
    decay_end = (end * H + i_h) * D_k + k_range
    p_log_decay = log_decay + decay_end
    p_dlog_decay = dlog_decay + decay_end

    p_dv = dv + (end * H + i_h) * D_v + v_range
    p_dbeta = dbeta + base_end
    p_dpi_k = dpi_k + base_end
    p_dk_update = (
        dk_update
        + ((i_v * all_bt + end) * H + i_h) * D_k
        + k_range
    )
    p_du_save = du_save + base_end

    b_dh = tl.zeros([BV, BE, BK], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        state_offset = (
            i_nh * D_v * E * D_k
            + v_range[:, None, None] * E * D_k
            + e_range[None, :, None] * D_k
            + k_range[None, None, :]
        )
        b_dht = tl.load(
            dht + state_offset,
            mask=mask_vek,
            other=0.0,
        ).to(tl.float32)
        b_dh += b_dht
        b_ht = tl.load(
            ht + state_offset,
            mask=mask_vek,
            other=0.0,
        ).to(tl.float32)
        b_tail_per_expert = tl.sum(b_dht * b_ht, axis=0)
        b_tail = tl.sum(b_tail_per_expert, axis=0)
        tl.atomic_add(p_dlog_decay, b_tail, mask=mask_k)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0.0).to(tl.float32)
        b_pi_q = tl.load(p_pi_q, mask=mask_ve, other=0.0).to(tl.float32)
        b_beta = tl.load(p_beta, mask=mask_ve, other=0.0).to(tl.float32)
        b_r = tl.load(p_r, mask=mask_ve, other=0.0).to(tl.float32)
        b_u = tl.load(p_u, mask=mask_ve, other=0.0).to(tl.float32)
        b_log_decay = tl.load(
            p_log_decay,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

        b_rho = softmax_2d(b_pi_q, mask_e)
        b_dh += (
            (b_q * scale)[None, None, :]
            * (b_do[:, None] * b_rho)[:, :, None]
        )

        b_d_coef = dot_3d_axis2(b_dh, b_k)
        b_dbeta = b_d_coef * b_r * b_u
        b_d_r = b_d_coef * b_beta * b_u
        b_d_u = b_d_coef * b_beta * b_r
        if USE_AUX_LOSS_GRAD:
            b_d_r += tl.load(
                p_d_r_seq,
                mask=mask_ve,
                other=0.0,
            ).to(tl.float32)

        b_d_logits = softmax_backward_2d(b_r, b_d_r)
        b_dpi_k = b_d_logits
        b_d_u += b_d_logits * (-b_u)
        b_dv = tl.sum(tl.trans(b_d_u), axis=0)

        b_coef = b_beta * b_r * b_u
        b_dh_coef = b_dh * b_coef[:, :, None]
        b_dk_per_expert = tl.sum(b_dh_coef, axis=0)
        b_dk = tl.sum(b_dk_per_expert, axis=0)

        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)
        tl.store(
            p_dbeta,
            b_dbeta.to(p_dbeta.dtype.element_ty),
            mask=mask_ve,
        )
        tl.store(
            p_dpi_k,
            b_dpi_k.to(p_dpi_k.dtype.element_ty),
            mask=mask_ve,
        )
        tl.store(
            p_dk_update,
            b_dk.to(p_dk_update.dtype.element_ty),
            mask=mask_k,
        )
        tl.store(p_du_save, b_d_u, mask=mask_ve)

        b_update_per_expert = tl.sum(
            -b_dh_coef * b_k[None, None, :],
            axis=0,
        )
        b_update_local = tl.sum(b_update_per_expert, axis=0)
        tl.atomic_add(p_dlog_decay, b_update_local, mask=mask_k)

        b_dh -= b_k[None, None, :] * b_d_u[:, :, None]
        b_dh *= tl.exp(b_log_decay[None, None, :])

        p_q -= H * D_k
        p_k -= H * D_k
        p_do -= H * D_v
        p_pi_q -= H * D_v * E
        p_beta -= H * D_v * E
        p_r -= H * D_v * E
        p_u -= H * D_v * E
        if USE_AUX_LOSS_GRAD:
            p_d_r_seq -= H * D_v * E
        p_log_decay -= H * D_k
        p_dlog_decay -= H * D_k
        p_dv -= H * D_v
        p_dbeta -= H * D_v * E
        p_dpi_k -= H * D_v * E
        p_dk_update -= H * D_k
        p_du_save -= H * D_v * E

    if USE_INITIAL_STATE:
        state_offset = (
            i_nh * D_v * E * D_k
            + v_range[:, None, None] * E * D_k
            + e_range[None, :, None] * D_k
            + k_range[None, None, :]
        )
        tl.store(
            dh0 + state_offset,
            b_dh.to(dh0.dtype.element_ty),
            mask=mask_vek,
        )


@triton.autotune(
    configs=[
        triton.Config(
            {"BV": BV},
            num_warps=1,
            num_stages=1,
            pre_hook=_zero_dense_d64e4_replay_outputs,
        )
        for BV in _DENSE_D64E4_BWD_VALUE_TILES
    ],
    key=["H", "USE_INITIAL_STATE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _fused_recurrent_gated_swila_dense_d64e4_bwd_replay_kernel(
    q,
    k,
    beta,
    pi_q,
    log_decay,
    r,
    u,
    h0,
    do,
    du_save,
    dq,
    dk_prediction,
    dpi_q,
    dlog_decay,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    D_k: tl.constexpr,
    D_v: tl.constexpr,
    E: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    """Forward replay half of the dense BF16 D64/E4 key-decay backward."""
    T = T.to(tl.int64)
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * T
    all_bt = B * T

    k_range = tl.arange(0, BK).to(tl.int64)
    v_range = i_v * BV + tl.arange(0, BV).to(tl.int64)
    e_range = tl.arange(0, BE).to(tl.int64)
    mask_k = k_range < D_k
    mask_v = v_range < D_v
    mask_e = e_range < E
    mask_ve = mask_v[:, None] & mask_e[None, :]
    mask_vek = (
        mask_v[:, None, None]
        & mask_e[None, :, None]
        & mask_k[None, None, :]
    )

    b_h = tl.zeros([BV, BE, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        state_offset = (
            i_nh * D_v * E * D_k
            + v_range[:, None, None] * E * D_k
            + e_range[None, :, None] * D_k
            + k_range[None, None, :]
        )
        b_h += tl.load(
            h0 + state_offset,
            mask=mask_vek,
            other=0.0,
        ).to(tl.float32)

    p_q = q + (bos * H + i_h) * D_k + k_range
    p_k = k + (bos * H + i_h) * D_k + k_range
    p_do = do + (bos * H + i_h) * D_v + v_range
    base = (
        (bos * H + i_h) * D_v * E
        + v_range[:, None] * E
        + e_range[None, :]
    )
    p_beta = beta + base
    p_pi_q = pi_q + base
    p_r = r + base
    p_u = u + base
    p_du_save = du_save + base
    p_dpi_q = dpi_q + base
    p_log_decay = log_decay + (bos * H + i_h) * D_k + k_range
    p_dlog_decay = dlog_decay + (bos * H + i_h) * D_k + k_range
    p_dq = dq + ((i_v * all_bt + bos) * H + i_h) * D_k + k_range
    p_dk_prediction = (
        dk_prediction
        + ((i_v * all_bt + bos) * H + i_h) * D_k
        + k_range
    )

    for i_t in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0.0).to(tl.float32)
        b_beta = tl.load(p_beta, mask=mask_ve, other=0.0).to(tl.float32)
        b_pi_q = tl.load(p_pi_q, mask=mask_ve, other=0.0).to(tl.float32)
        b_r = tl.load(p_r, mask=mask_ve, other=0.0).to(tl.float32)
        b_u = tl.load(p_u, mask=mask_ve, other=0.0).to(tl.float32)
        b_d_u = tl.load(
            p_du_save,
            mask=mask_ve,
            other=0.0,
        ).to(tl.float32)
        b_log_decay = tl.load(
            p_log_decay,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

        b_h *= tl.exp(b_log_decay[None, None, :])

        # Prediction dk always uses the same state as the saved residual.
        b_dk_weighted = b_d_u[:, :, None] * b_h
        b_dk_per_expert = tl.sum(b_dk_weighted, axis=0)
        b_dk_correction = tl.sum(b_dk_per_expert, axis=0)
        tl.store(
            p_dk_prediction,
            (-b_dk_correction).to(p_dk_prediction.dtype.element_ty),
            mask=mask_k,
        )

        b_h = state_update(b_h, b_beta * b_r * b_u, b_k)
        b_rho = softmax_2d(b_pi_q, mask_e)
        b_d_qS = b_do[:, None] * b_rho
        b_output_per_expert = tl.sum(
            b_d_qS[:, :, None] * b_h,
            axis=0,
        )
        b_dq = tl.sum(b_output_per_expert, axis=0) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)

        b_qS = readout(b_h, b_q * scale)
        b_dpi_q = softmax_backward_2d(b_rho, b_do[:, None] * b_qS)
        tl.store(
            p_dpi_q,
            b_dpi_q.to(p_dpi_q.dtype.element_ty),
            mask=mask_ve,
        )

        b_replay_per_expert = (
            (b_q * scale)[None, :] * b_output_per_expert
            - b_k[None, :] * b_dk_per_expert
        )
        tl.atomic_add(
            p_dlog_decay,
            tl.sum(b_replay_per_expert, axis=0),
            mask=mask_k,
        )

        p_q += H * D_k
        p_k += H * D_k
        p_do += H * D_v
        p_beta += H * D_v * E
        p_pi_q += H * D_v * E
        p_r += H * D_v * E
        p_u += H * D_v * E
        p_du_save += H * D_v * E
        p_dpi_q += H * D_v * E
        p_log_decay += H * D_k
        p_dlog_decay += H * D_k
        p_dq += H * D_k
        p_dk_prediction += H * D_k


@triton.jit(do_not_specialize=["T"])
def _reverse_cumsum_log_decay_kernel(
    dlog_decay,
    cu_seqlens,
    T,
    H: tl.constexpr,
    D_k: tl.constexpr,
    E: tl.constexpr,
    BK: tl.constexpr,
    BE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Convert compact generalized-sigma local terms to dlog-decay."""
    T = T.to(tl.int64)
    i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
    else:
        bos = i_n * T
        seq_len = T

    k_range = i_k * BK + tl.arange(0, BK).to(tl.int64)
    mask_k = k_range < D_k

    p_dlog = dlog_decay + (
        ((bos + seq_len - 1) * H + i_h) * D_k + k_range
    )
    b_acc = tl.zeros([BK], dtype=tl.float32)
    for _ in range(seq_len):
        b_acc += tl.load(p_dlog, mask=mask_k, other=0.0).to(tl.float32)
        tl.store(p_dlog, b_acc, mask=mask_k)
        p_dlog -= H * D_k


def _validate_dense_split_schedule(
    schedule, q, k, v, beta, pi_q, pi_k, log_decay, cu_seqlens,
):
    """Bound the C2 experiment without changing either production table.

    These are predeclared trials, not an autotune space: control, reverse BV8,
    replay BV8, reverse stage1, and combined reverse/replay BV8. Dense kernels
    have no early-config pruner;
    this exact whitelist and the existing dense predicate bound the opt-in.
    The generic fallback and its safety pruner are untouched.
    """
    allowed = (
        ((16, 1, 2), (16, 1, 1)),
        ((8, 1, 2), (16, 1, 1)),
        ((16, 1, 2), (8, 1, 1)),
        ((16, 1, 1), (16, 1, 1)),
        ((8, 1, 2), (8, 1, 1)),
    )
    if (
        type(schedule) is not tuple or len(schedule) != 2
        or any(type(item) is not tuple or len(item) != 3
               or any(type(value) is not int for value in item) for item in schedule)
        or schedule not in allowed
    ):
        raise ValueError("dense_split_schedule requires one of the five predeclared C2 pairs")
    if (
        log_decay is None
        or not _use_dense_d64e4_key_bwd_fastpath(q, k, v, beta, pi_q, pi_k, log_decay, cu_seqlens)
        or not _is_nvidia_sm90_tensor(q)
        or any(size <= 0 for size in q.shape[:3])
        or any(tensor.device != q.device for tensor in (k, v, beta, pi_q, pi_k, log_decay))
    ):
        raise ValueError("C2 requires positive dense SM90 BF16 D64/E4 inputs and same-device FP32 decay")
    return schedule


def _fused_recurrent_gated_swila_bwd_scheduled(
    q, k, v, beta, pi_q, pi_k, r, u, initial_state, do, dht,
    log_decay, final_state, d_r_seq, scale, cu_seqlens, schedule,
):
    """Fixed-schedule C2 host path; original JIT equations/hooks.

    The independent partial leading dimensions are essential: the production
    table's four slabs cannot hold the eight programs of a BV8 launch.
    Automatic selection is restricted to the measured H100 workload; other
    callers may explicitly select a qualified internal schedule.
    """
    reverse_schedule, replay_schedule = _validate_dense_split_schedule(
        schedule, q, k, v, beta, pi_q, pi_k, log_decay, cu_seqlens,
    )
    B, T, H, D_k_dim = q.shape
    D_v_dim, E_dim = v.shape[-1], beta.shape[-1]
    routing_shape, state_shape = (B, T, H, D_v_dim, E_dim), (B, H, D_v_dim, E_dim, D_k_dim)

    def check_tensor(name, tensor, shape, dtypes):
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                or tensor.dtype not in dtypes or tensor.device != q.device or not tensor.is_contiguous()):
            raise ValueError("C2 saved tensor metadata differs: " + name)

    for name, tensor in (("r", r), ("u", u)):
        check_tensor(name, tensor, routing_shape, (torch.float32,))
    check_tensor("do", do, tuple(v.shape), (torch.bfloat16,))
    if initial_state is not None:
        check_tensor("initial_state", initial_state, state_shape, (torch.float16, torch.bfloat16, torch.float32))
    if dht is not None and final_state is None:
        raise RuntimeError("final_state is required when dht is provided")
    for name, tensor in (("final_state", final_state), ("dht", dht)):
        if tensor is not None:
            check_tensor(name, tensor, state_shape, (torch.float32,))
    if d_r_seq is not None:
        check_tensor("d_r_seq", d_r_seq, routing_shape, (torch.float32,))

    reverse_nv = triton.cdiv(D_v_dim, reverse_schedule[0])
    replay_nv = triton.cdiv(D_v_dim, replay_schedule[0])
    dq = q.new_zeros(replay_nv, B, T, H, D_k_dim, dtype=torch.float32)
    dk_update = q.new_zeros(reverse_nv, B, T, H, D_k_dim, dtype=torch.float32)
    dk_prediction = q.new_zeros(replay_nv, B, T, H, D_k_dim, dtype=torch.float32)
    dv = v.new_empty(B, T, H, D_v_dim)
    dbeta = beta.new_empty(*routing_shape)
    dpi_q = pi_q.new_empty(*routing_shape)
    dpi_k = pi_k.new_empty(*routing_shape)
    du_save = q.new_empty(*routing_shape, dtype=torch.float32)
    dlog_decay = torch.zeros_like(log_decay, dtype=torch.float32)
    dlog_decay_replay = torch.zeros_like(log_decay, dtype=torch.float32)
    dh0 = q.new_empty(*state_shape, dtype=torch.float32) if initial_state is not None else None
    common = dict(B=B, H=H, D_k=D_k_dim, D_v=D_v_dim, E=E_dim, BK=64, BE=4,
                  USE_INITIAL_STATE=initial_state is not None)

    # Direct JIT launches bypass Autotuner.run, so invoke the same original
    # Config hooks explicitly, once per real launch, with phase-owned buffers.
    _zero_dense_d64e4_reverse_outputs({"dk_update": dk_update, "dlog_decay": dlog_decay})
    _fused_recurrent_gated_swila_dense_d64e4_bwd_reverse_kernel.fn[(reverse_nv, B * H)](
        q, k, beta, pi_q, log_decay, r, u, initial_state, do, dht, final_state, d_r_seq,
        dk_update, dv, dbeta, dpi_k, dlog_decay, du_save, dh0, scale, T,
        USE_FINAL_STATE_GRADIENT=dht is not None, USE_AUX_LOSS_GRAD=d_r_seq is not None,
        BV=reverse_schedule[0], num_warps=reverse_schedule[1], num_stages=reverse_schedule[2], **common,
    )
    _zero_dense_d64e4_replay_outputs({"dq": dq, "dk_prediction": dk_prediction, "dlog_decay": dlog_decay_replay})
    _fused_recurrent_gated_swila_dense_d64e4_bwd_replay_kernel.fn[(replay_nv, B * H)](
        q, k, beta, pi_q, log_decay, r, u, initial_state, do, du_save,
        dq, dk_prediction, dpi_q, dlog_decay_replay, scale, T,
        BV=replay_schedule[0], num_warps=replay_schedule[1], num_stages=replay_schedule[2], **common,
    )
    dlog_decay.add_(dlog_decay_replay)
    _reverse_cumsum_log_decay_kernel[(1, B * H)](
        dlog_decay, None, T, H=H, D_k=D_k_dim, E=E_dim, BK=64, BE=4, IS_VARLEN=False, num_warps=4,
    )
    dq = dq.sum(0).to(q.dtype)
    dk = (dk_update.sum(0) + dk_prediction.sum(0)).to(k.dtype)
    return dq, dk, dv, dbeta, dpi_q, dpi_k, dh0, dlog_decay.to(log_decay)
