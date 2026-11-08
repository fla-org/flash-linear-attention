# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Temporal backward orchestration and decay-gradient accumulation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.swila.autotune import prune_temporal_value_configs
from fla.ops.swila.dispatch import (
    _use_temporal_dense_d64e4_key_bwd_fastpath as _use_dense_d64e4_key_bwd_fastpath,
)
from fla.ops.swila.temporal_bwd_state import (
    fused_recurrent_swila_temporal_bwd_kernel,
)
from fla.ops.swila.temporal_split_bwd import (
    fused_recurrent_swila_temporal_bwd_q_temporal_kernel,
    fused_recurrent_swila_temporal_bwd_replay_kernel,
)


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
    """Turn generalized-sigma local terms into segmented dlog-decay."""
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

    p_dlog = dlog_decay + ((bos + seq_len - 1) * H + i_h) * D_k + k_range
    b_acc = tl.zeros([BK], dtype=tl.float32)
    for _ in range(seq_len):
        b_acc += tl.load(p_dlog, mask=mask_k, other=0.0).to(tl.float32)
        tl.store(p_dlog, b_acc, mask=mask_k)
        p_dlog -= H * D_k


def fused_recurrent_swila_temporal_bwd(
    q: torch.Tensor,            # [B, T, H, D_k]
    k: torch.Tensor,            # [B, T, H, D_k]
    v: torch.Tensor,            # [B, T, H, D_v]
    beta: torch.Tensor,         # [B, T, H, D_v, E]
    pi_q: torch.Tensor,         # [B, T, H, D_v, E]
    pi_k: torch.Tensor,         # [B, T, H, D_v, E]
    # Saved intermediates from forward
    r: torch.Tensor,            # [B, T, H, D_v, E]
    u: torch.Tensor,            # [B, T, H, D_v, E]
    pi_bar: torch.Tensor,       # [B, T, H, D_v, E]
    rho: torch.Tensor,          # [B, T, H, D_v, E]
    qS: torch.Tensor,           # [B, T, H, D_v, E]
    log_decay: torch.Tensor | None,
    initial_state: torch.Tensor | None,  # [B, H, D_v, E, D_k]
    final_state: torch.Tensor | None,    # [B, H, D_v, E, D_k]
    r_prev_init: torch.Tensor | None,    # [B, H, D_v, E]
    rho_prev_init: torch.Tensor | None,  # [B, H, D_v, E]
    do: torch.Tensor,           # [B, T, H, D_v]
    dht: torch.Tensor | None,  # [B, H, D_v, E, D_k]
    scale: float,
    d_r_prev_final: torch.Tensor | None = None,
    d_rho_prev_final: torch.Tensor | None = None,
    # Auxiliary loss gradients (load balancing)
    d_r_seq: torch.Tensor | None = None,        # [B, T, H, D_v, E]
    d_rho_seq: torch.Tensor | None = None,      # [B, T, H, D_v, E]
    d_u_seq: torch.Tensor | None = None,        # [B, T, H, D_v, E]
    d_pi_bar_seq: torch.Tensor | None = None,   # [B, T, H, D_v, E]
    # Gate inputs for reset_gate temporal coupling
    gate_kv: torch.Tensor | None = None,        # [B, T, H, D_v]
    gate_q: torch.Tensor | None = None,         # [B, T, H, D_v]
    cu_seqlens: torch.Tensor | None = None,     # [N+1] cumulative sequence lengths
    debug_intermediates: bool = False,
    *,
    split_schedule: tuple[tuple[int, int, int], ...] | str | None = "auto",
):
    """
    Backward pass for fused recurrent SwiLA with temporal transitions.

    ``split_schedule`` defaults to automatic selection within the measured
    H100 workload. Explicit None retains native autotuning; three (BV, warps,
    stages) triples override query reverse, memory reverse and replay.
    Selection changes scheduling, not equations.
    """
    B, T, H, D_k_dim = q.shape
    D_v_dim = v.shape[-1]
    E_dim = beta.shape[-1]
    assert 1 <= E_dim <= 8, f"fused_recurrent SwiLA requires 1 <= num_experts <= 8, got E={E_dim}"
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # Block sizes
    BK = max(triton.next_power_of_2(D_k_dim), 16)
    BE = triton.next_power_of_2(E_dim)
    NK = triton.cdiv(D_k_dim, BK)

    # BE=16 uses a split, identity-specialized resource-safe backward. The
    # exact dense FineWeb E4 geometry reuses that decomposition with a small
    # BV16/BV32 internal table and right-sized partial buffers.
    resource_safe = BE == 16
    production_fast = (
        _use_dense_d64e4_key_bwd_fastpath(
            q,
            k,
            v,
            beta,
            pi_q,
            pi_k,
            log_decay,
            gate_kv,
            gate_q,
            cu_seqlens,
        )
        and r_prev_init is None
        and rho_prev_init is None
        and d_r_prev_final is None
        and d_rho_prev_final is None
        and dht is None
    )
    # Depend on the existing production predicate: forced-generic execution and
    # carried/final-prior gradients must not silently enter the automatic path.
    if isinstance(split_schedule, str):
        if split_schedule != "auto":
            raise ValueError("split_schedule must be 'auto', None, or three triples")
        split_schedule = None
        if (
            production_fast and tuple(q.shape) == (1, 4096, 16, 64)
            and torch.version.hip is None
            and torch.cuda.get_device_capability(q.device) == (9, 0)
            and "H100" in torch.cuda.get_device_name(q.device)
            and not debug_intermediates
            and all(tensor is None or tensor.device == q.device
                    for tensor in (k, v, beta, pi_q, pi_k, log_decay, gate_kv, gate_q))
        ):
            split_schedule = ((32, 1, 2), (8, 1, 2), (16, 1, 1))
    if split_schedule is not None:
        if (
            not isinstance(split_schedule, tuple)
            or len(split_schedule) != 3
            or any(
                not isinstance(item, tuple) or len(item) != 3
                or any(type(value) is not int for value in item)
                or item[0] not in (8, 16, 32)
                or item[1] not in (1, 2, 4)
                or item[2] not in (1, 2, 3, 4)
                for item in split_schedule
            )
        ):
            raise ValueError("split_schedule requires three (BV, warps, stages) triples")
        if BE > 8 or BK > 64:
            raise ValueError("experimental temporal split requires BE <= 8 and BK <= 64")
    split_identity = resource_safe or production_fast or split_schedule is not None
    split_bv = 8 if resource_safe else 16
    NV_max = triton.cdiv(D_v_dim, split_bv if split_identity else 8)
    query_meta = {"BV": split_bv, "num_warps": 1, "num_stages": 2}
    replay_meta = {"BV": split_bv, "num_warps": 1, "num_stages": 1}
    if split_schedule is not None:
        query_meta, candidate_reverse_meta, replay_meta = (
            dict(zip(("BV", "num_warps", "num_stages"), item))
            for item in split_schedule
        )
        # A fixed launch avoids autotuning, not its resource-safety contract.
        for meta in (candidate_reverse_meta, replay_meta):
            prune_temporal_value_configs(
                [triton.Config({"BV": meta["BV"]}, num_warps=meta["num_warps"],
                               num_stages=meta["num_stages"])],
                {"D_v": D_v_dim, "BE": BE, "BK": BK},
            )
    # Reverse and replay own separate FP32 partials; unequal value tiles must
    # never leave uninitialized slabs in a later reduction.
    reverse_nv = (triton.cdiv(D_v_dim, candidate_reverse_meta["BV"])
                  if split_schedule is not None else NV_max)
    replay_nv = (triton.cdiv(D_v_dim, replay_meta["BV"])
                 if split_schedule is not None else NV_max)

    assert NK == 1, f"NK > 1 not supported yet (D_k={D_k_dim}, BK={BK}, NK={NK})"
    if gate_kv is None or gate_q is None:
        raise ValueError("temporal SwiLA backward requires reset gate inputs")

    # Allocate gradient tensors
    # dq/dk: NV_max allocation; pre_hook zeros before each autotune benchmark
    dq = q.new_zeros(replay_nv, B, T, H, D_k_dim, dtype=torch.float32)
    dk = q.new_zeros(
        reverse_nv,
        B,
        T,
        H,
        D_k_dim,
        dtype=torch.float32,
    )
    dk_prediction = (
        q.new_empty(replay_nv, B, T, H, D_k_dim, dtype=torch.float32)
        if split_identity
        else None
    )
    # NK is asserted to be one. Every valid element is direct-written by the
    # sole key block and disjoint value blocks, so leading-singleton scratch,
    # zero fills, and post-kernel reductions are all unnecessary.
    dv = v.new_empty(B, T, H, D_v_dim)
    dbeta = beta.new_empty(B, T, H, D_v_dim, E_dim)
    # Exact-transition reverse temporarily stores its FP32 log-rho carry in
    # this buffer before overwriting it with d(pi_q). Avoid an internal
    # BF16/FP16 round trip and cast only the completed public input gradient.
    dpi_q = pi_q.new_empty(
        B,
        T,
        H,
        D_v_dim,
        E_dim,
        dtype=torch.float32,
    )
    dpi_k = pi_k.new_empty(B, T, H, D_v_dim, E_dim)

    # Temporary storage for grad_u
    du_save = q.new_empty(B, T, H, D_v_dim, E_dim, dtype=torch.float32)

    if log_decay is not None:
        dlog_decay = torch.zeros_like(log_decay, dtype=torch.float32)
        dlog_decay_replay = (
            torch.zeros_like(log_decay, dtype=torch.float32)
            if split_identity
            else None
        )
    else:
        dlog_decay = None
        dlog_decay_replay = None

    # Initial state gradient (N sequences, not B)
    if initial_state is not None:
        dh0 = q.new_empty(N, H, D_v_dim, E_dim, D_k_dim, dtype=torch.float32)
    else:
        dh0 = None

    # Initial temporal state gradients (N sequences, not B)
    if r_prev_init is not None:
        dr_prev_init = q.new_zeros(N, H, D_v_dim, E_dim, dtype=torch.float32)
        drho_prev_init = q.new_zeros(N, H, D_v_dim, E_dim, dtype=torch.float32)
    else:
        dr_prev_init = None
        drho_prev_init = None

    # Allocate gate gradients
    if gate_kv is not None:
        d_gate_kv = gate_kv.new_zeros(gate_kv.shape)
        d_gate_q = gate_q.new_zeros(gate_q.shape)
    else:
        d_gate_kv = None
        d_gate_q = None

    use_aux_loss_grad = d_r_seq is not None and d_rho_seq is not None
    is_varlen = cu_seqlens is not None
    use_log_decay = log_decay is not None

    # Allocate debug buffers if requested
    if debug_intermediates:
        debug_dh = q.new_zeros(B, T, H, D_v_dim, E_dim, D_k_dim, dtype=torch.float32)
    else:
        debug_dh = None

    def grid(meta):
        return (triton.cdiv(D_v_dim, meta['BV']), NK, N * H)
    if split_identity:
        # Q-side temporal reverse is independent of the recurrent state. Split
        # it out so the main reverse never holds both Q and KV transition state.
        q_grid = (triton.cdiv(D_v_dim, query_meta["BV"]), 1, N * H)
        fused_recurrent_swila_temporal_bwd_q_temporal_kernel[q_grid](
            pi_q,
            rho,
            qS,
            rho_prev_init,
            do,
            d_rho_prev_final,
            dpi_q,
            drho_prev_init,
            d_rho_seq,
            gate_q,
            d_gate_q,
            cu_seqlens,
            T,
            B=B,
            H=H,
            D_v=D_v_dim,
            E=E_dim,
            BE=BE,
            USE_INITIAL_TEMPORAL_STATE=rho_prev_init is not None,
            STORE_TEMPORAL_STATE_GRADIENT=drho_prev_init is not None,
            USE_FINAL_RHO_GRADIENT=d_rho_prev_final is not None,
            USE_AUX_LOSS_GRAD=d_rho_seq is not None,
            IS_VARLEN=is_varlen,
            **query_meta,
        )

        # BE=16 must not enter Triton's autotuner at all: even benchmarking an
        # unsafe candidate can poison the CUDA context. The decorator's ``fn``
        # is the underlying JITFunction, so the resource-safe path launches it
        # directly with fixed conservative metadata. The E4 production path
        # still uses its deliberately small internal candidate table.
        reverse_kernel = (
            fused_recurrent_swila_temporal_bwd_kernel.fn
            if resource_safe or split_schedule is not None
            else fused_recurrent_swila_temporal_bwd_kernel
        )
        reverse_grid = (
            (triton.cdiv(D_v_dim, 8), NK, N * H)
            if resource_safe
            else grid
        )
        fixed_reverse_meta = (
            {"BV": 8, "num_warps": 1, "num_stages": 2}
            if resource_safe
            else {}
        )
        if split_schedule is not None:
            reverse_grid = (reverse_nv, NK, N * H)
            fixed_reverse_meta = candidate_reverse_meta
        reverse_kernel[reverse_grid](
            q, k, v, beta, pi_q, pi_k,
            log_decay,
            r, u, pi_bar, rho, qS,
            initial_state,
            r_prev_init, rho_prev_init,
            do, dht, final_state,
            d_r_prev_final, d_rho_prev_final,
            dq, dk, dv, dbeta, dpi_q, dpi_k,
            du_save, dlog_decay, dh0,
            dr_prev_init, drho_prev_init,
            d_r_seq, d_rho_seq, d_u_seq, d_pi_bar_seq,
            gate_kv, gate_q, d_gate_kv, d_gate_q,
            debug_dh,
            cu_seqlens,
            scale,
            T,
            B=B,
            H=H,
            D_k=D_k_dim,
            D_v=D_v_dim,
            E=E_dim,
            BK=BK,
            BE=BE,
            NK=NK,
            USE_INITIAL_STATE=initial_state is not None,
            USE_FINAL_STATE_GRADIENT=dht is not None,
            USE_INITIAL_TEMPORAL_STATE=r_prev_init is not None,
            STORE_TEMPORAL_STATE_GRADIENT=dr_prev_init is not None,
            USE_FINAL_R_GRADIENT=d_r_prev_final is not None,
            USE_FINAL_RHO_GRADIENT=False,
            USE_AUX_LOSS_GRAD=d_r_seq is not None,
            USE_U_AUX_GRAD=d_u_seq is not None,
            USE_PI_BAR_AUX_GRAD=d_pi_bar_seq is not None,
            DEBUG_INTERMEDIATES=debug_intermediates,
            SKIP_REPLAY=True,
            SPLIT_Q_TEMPORAL=True,
            REUSE_DK_COEF=False,
            RESOURCE_SAFE=resource_safe,
            PRODUCTION_FAST=production_fast,
            USE_LOG_DECAY=use_log_decay,
            IS_VARLEN=is_varlen,
            **fixed_reverse_meta,
        )

        replay_grid = (replay_nv, NK, N * H)
        fused_recurrent_swila_temporal_bwd_replay_kernel[replay_grid](
            q,
            k,
            beta,
            r,
            u,
            rho,
            log_decay,
            initial_state,
            do,
            dq,
            dk_prediction,
            du_save,
            dlog_decay_replay,
            cu_seqlens,
            scale,
            T,
            B=B,
            H=H,
            D_k=D_k_dim,
            D_v=D_v_dim,
            E=E_dim,
            BK=BK,
            BE=BE,
            USE_INITIAL_STATE=initial_state is not None,
            USE_LOG_DECAY=use_log_decay,
            IS_VARLEN=is_varlen,
            **replay_meta,
        )
        if dlog_decay is not None:
            dlog_decay.add_(dlog_decay_replay)
    else:
        fused_recurrent_swila_temporal_bwd_kernel[grid](
            q, k, v, beta, pi_q, pi_k,
            log_decay,
            r, u, pi_bar, rho, qS,
            initial_state,
            r_prev_init, rho_prev_init,
            do, dht, final_state,
            d_r_prev_final, d_rho_prev_final,
            dq, dk, dv, dbeta, dpi_q, dpi_k,
            du_save, dlog_decay, dh0,
            dr_prev_init, drho_prev_init,
            d_r_seq, d_rho_seq, d_u_seq, d_pi_bar_seq,
            gate_kv, gate_q, d_gate_kv, d_gate_q,
            debug_dh,
            cu_seqlens,
            scale,
            T,
            B=B,
            H=H,
            D_k=D_k_dim,
            D_v=D_v_dim,
            E=E_dim,
            BK=BK,
            BE=BE,
            NK=NK,
            USE_INITIAL_STATE=initial_state is not None,
            USE_FINAL_STATE_GRADIENT=dht is not None,
            USE_INITIAL_TEMPORAL_STATE=r_prev_init is not None,
            STORE_TEMPORAL_STATE_GRADIENT=dr_prev_init is not None,
            USE_FINAL_R_GRADIENT=d_r_prev_final is not None,
            USE_FINAL_RHO_GRADIENT=d_rho_prev_final is not None,
            USE_AUX_LOSS_GRAD=use_aux_loss_grad,
            USE_U_AUX_GRAD=d_u_seq is not None,
            USE_PI_BAR_AUX_GRAD=d_pi_bar_seq is not None,
            DEBUG_INTERMEDIATES=debug_intermediates,
            SKIP_REPLAY=False,
            SPLIT_Q_TEMPORAL=False,
            REUSE_DK_COEF=False,
            RESOURCE_SAFE=False,
            PRODUCTION_FAST=False,
            USE_LOG_DECAY=use_log_decay,
            IS_VARLEN=is_varlen,
        )

    if dlog_decay is not None:
        _reverse_cumsum_log_decay_kernel[(NK, N * H)](
            dlog_decay,
            cu_seqlens,
            T,
            H=H,
            D_k=D_k_dim,
            E=E_dim,
            BK=BK,
            BE=BE,
            IS_VARLEN=is_varlen,
            num_warps=4,
        )

    # Reduce across blocks
    dq = dq.sum(0).to(q.dtype)
    if split_identity:
        dk = (dk.sum(0).float() + dk_prediction.sum(0)).to(k.dtype)
    else:
        dk = dk.sum(0)
    dk = dk.to(k.dtype)
    dpi_q = dpi_q.to(pi_q.dtype)

    # FIX: Sum temporal state gradients over k-blocks (they're stored per (i_v, i_k) block)
    # The kernel stores partial gradients, we need to sum over any extra dimensions
    # Since we use new_zeros, the kernel atomically adds to these, so no extra reduction needed
    # (each (i_v, i_k, i_nh) block writes to non-overlapping regions)

    if debug_intermediates:
        return dq, dk, dv, dbeta, dpi_q, dpi_k, dh0, dr_prev_init, drho_prev_init, dlog_decay, d_gate_kv, d_gate_q, debug_dh, du_save
    return dq, dk, dv, dbeta, dpi_q, dpi_k, dh0, dr_prev_init, drho_prev_init, dlog_decay, d_gate_kv, d_gate_q
