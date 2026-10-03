# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


# Portions adapted from CyclicFlowAttention, Copyright (c) 2026 Yixiao Chen.
# https://github.com/Chyxx/CyclicFlowAttention

from __future__ import annotations

import weakref
from functools import cache

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_o import NUM_WARPS
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp
from fla.utils import IS_NVIDIA, autotune_cache_kwargs, check_shared_mem, get_device_capability, input_guard

from .qk_norm import qk_rmsnorm_fwd
from .utils import (
    CyFAState,
    build_readout_table,
    cyfa_cos,
    cyfa_sin,
    modal_phase,
    prepare_cu_seqlens,
    validate_cyfa_inputs,
    validate_initial_state,
)

_READOUT_TABLE_CACHE: dict[
    tuple[int, int, int, str, torch.dtype],
    tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor],
] = {}
_READOUT_NUM_WARPS = (
    NUM_WARPS if check_shared_mem("ampere") else [w for w in NUM_WARPS if w <= 4]
)


def _tensor_version(x: torch.Tensor) -> int:
    try:
        return int(x._version)
    except RuntimeError:
        return -1


@cache
def _device_supports_pdl(device_index: int | None) -> bool:
    return IS_NVIDIA and get_device_capability(device_index or 0)[0] >= 9


def _k_recurrent_feature_block_size(d_k: int) -> int:
    return min(64, triton.next_power_of_2(d_k))


@triton.jit
def _cardinal_orbit_value(lam, o_m, M: tl.constexpr):
    is_cos = (o_m % 2) == 0
    phase = modal_phase(lam, o_m, M)
    scale = tl.sqrt(tl.where((o_m // 2) == 0, 1.0, 2.0) / (M - 1))
    return scale * tl.where(is_cos, cyfa_cos(phase), -cyfa_sin(phase))


@triton.jit
def _rotate_pos_1d(lam, x, BM: tl.constexpr, M: tl.constexpr):
    x_even, x_odd = tl.split(tl.reshape(x, (BM // 2, 2)))
    o_even = 2 * tl.arange(0, BM // 2)
    phase = modal_phase(lam, o_even, M)
    c = cyfa_cos(phase)
    s = cyfa_sin(phase)
    return tl.reshape(tl.join(c * x_even - s * x_odd, s * x_even + c * x_odd), (BM,))


@triton.jit
def _rotate_neg_1d(lam, x, BM: tl.constexpr, M: tl.constexpr):
    x_even, x_odd = tl.split(tl.reshape(x, (BM // 2, 2)))
    o_even = 2 * tl.arange(0, BM // 2)
    phase = modal_phase(lam, o_even, M)
    c = cyfa_cos(phase)
    s = cyfa_sin(phase)
    return tl.reshape(tl.join(c * x_even + s * x_odd, -s * x_even + c * x_odd), (BM,))


@triton.jit
def _rotate_pos_2d(lam, x, BR: tl.constexpr, BM: tl.constexpr, M: tl.constexpr):
    x_even, x_odd = tl.split(tl.reshape(x, (BR, BM // 2, 2)))
    o_even = 2 * tl.arange(0, BM // 2)
    phase = modal_phase(lam[:, None], o_even[None, :], M)
    c = cyfa_cos(phase)
    s = cyfa_sin(phase)
    return tl.reshape(
        tl.join(c * x_even - s * x_odd, s * x_even + c * x_odd),
        (BR, BM),
    )


@triton.jit
def _rotate_neg_2d(lam, x, BR: tl.constexpr, BM: tl.constexpr, M: tl.constexpr):
    x_even, x_odd = tl.split(tl.reshape(x, (BR, BM // 2, 2)))
    o_even = 2 * tl.arange(0, BM // 2)
    phase = modal_phase(lam[:, None], o_even[None, :], M)
    c = cyfa_cos(phase)
    s = cyfa_sin(phase)
    return tl.reshape(
        tl.join(c * x_even + s * x_odd, -s * x_even + c * x_odd),
        (BR, BM),
    )


@triton.jit
def _as_recurrent_key_dtype(x, ref):
    return x.to(ref.dtype.element_ty).to(tl.float32)


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial_k"] is not None,
        "USE_INITIAL_LAMBDA": lambda args: args["initial_lambda"] is not None,
        "STORE_FINAL_STATE": lambda args: args["state_k"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    },
)
@fla_cache_autotune(
    configs=[
        triton.Config({"BM": bm}, num_warps=num_warps, num_stages=2)
        for bm in [16, 32, 64]
        for num_warps in NUM_WARPS
    ],
    key=["K", "BK", "M", "USE_INITIAL_STATE", "STORE_FINAL_STATE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "TOTAL"])
def _fused_recurrent_cyfa_k_kernel(
    q,
    q_norm_weight,
    q_rstd,
    k,
    delta,
    beta,
    g,
    initial_k,
    initial_lambda,
    state_k,
    state_lambda,
    lambda_trace,
    decay_trace,
    raw_partial,
    slot_write_out,
    cu_seqlens,
    scale,
    T,
    TOTAL,
    H: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_LAMBDA: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NM: tl.constexpr = triton.cdiv(M, BM)
    NK: tl.constexpr = triton.cdiv(K, BK)
    i_m = (pid % NM).to(tl.int64)
    i_k = ((pid // NM) % NK).to(tl.int64)
    i_nh = (pid // (NM * NK)).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_m = i_m * BM + tl.arange(0, BM)
    mask_k = o_k < K
    mask_m = o_m < M
    mask_h = mask_k[:, None] & mask_m[None, :]

    if USE_INITIAL_LAMBDA:
        lam = tl.load(initial_lambda + i_n * H + i_h).to(tl.float32)
    else:
        lam = 0.0
    if USE_INITIAL_STATE:
        p_initial = initial_k + ((i_n * H + i_h) * K + o_k[:, None]) * M + o_m[None, :]
        state = tl.load(p_initial, mask=mask_h, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BK, BM), dtype=tl.float32)

    p_q = q + (bos * H + i_h) * K + o_k
    p_q_rstd = q_rstd + bos * H + i_h
    p_q_weight = q_norm_weight + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_delta = delta + bos * H + i_h
    p_beta = beta + bos * H + i_h
    p_g = g + bos * H + i_h
    p_raw = raw_partial + ((i_k * TOTAL + bos) * H + i_h) * M + o_m
    p_write = slot_write_out + (bos * H + i_h) * M + o_m
    p_lambda = lambda_trace + bos * H + i_h
    p_decay = decay_trace + bos * H + i_h

    for _ in tl.range(0, T, num_stages=2):
        d = tl.load(p_delta, eviction_policy="evict_last").to(tl.float32)
        b = tl.load(p_beta, eviction_policy="evict_last").to(tl.float32)
        lam += d
        write = _as_recurrent_key_dtype(
            b * _cardinal_orbit_value(lam, o_m, M),
            q,
        )
        if i_k == 0:
            tl.store(p_write, write, mask=mask_m)
        decay = exp(tl.load(p_g, eviction_policy="evict_last").to(tl.float32))
        state *= decay
        value = tl.load(p_k, mask=mask_k, other=0.0, eviction_policy="evict_last").to(tl.float32)
        state += value[:, None] * write[None, :]
        query = tl.load(p_q, mask=mask_k, other=0.0, eviction_policy="evict_last").to(tl.float32)
        query *= tl.load(p_q_rstd, eviction_policy="evict_last").to(tl.float32)
        query *= tl.load(
            p_q_weight,
            mask=mask_k,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        query = _as_recurrent_key_dtype(query, q) * scale
        raw = tl.sum(state * query[:, None], axis=0)
        tl.store(p_raw, raw, mask=mask_m)
        if i_k == 0 and i_m == 0:
            tl.store(p_lambda, lam)
            tl.store(p_decay, decay)

        p_q += H * K
        p_q_rstd += H
        p_k += H * K
        p_delta += H
        p_beta += H
        p_g += H
        p_raw += H * M
        p_write += H * M
        p_lambda += H
        p_decay += H

    if STORE_FINAL_STATE:
        p_state = state_k + ((i_n * H + i_h) * K + o_k[:, None]) * M + o_m[None, :]
        tl.store(p_state, state, mask=mask_h)
        if i_k == 0 and i_m == 0:
            tl.store(state_lambda + i_n * H + i_h, lam)


@fla_cache_autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=num_warps, num_stages=num_stages)
        for bt in [16, 32]
        for num_warps in _READOUT_NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=["M", "C", "NK"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["TOTAL"])
def _fused_recurrent_cyfa_readout_kernel(
    raw_partial,
    lambda_trace,
    readout_table,
    slot_read_out,
    TOTAL,
    H: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    BM: tl.constexpr,
    NK: tl.constexpr,
    BT: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NT = tl.cdiv(TOTAL, BT)
    i_t = pid % NT
    i_h = pid // NT
    o_t = i_t * BT + tl.arange(0, BT)
    o_m = tl.arange(0, BM)
    o_c = tl.arange(0, BM)
    mask_t = o_t < TOTAL
    mask_m = o_m < M
    mask_c = o_c < C
    mask = mask_t[:, None] & mask_m[None, :]

    raw = tl.zeros((BT, BM), dtype=tl.float32)
    for i_k in tl.static_range(0, NK):
        p_raw = raw_partial + ((i_k * TOTAL + o_t[:, None]) * H + i_h) * M + o_m[None, :]
        raw += tl.load(p_raw, mask=mask, other=0.0).to(tl.float32)
    lam = tl.load(lambda_trace + o_t * H + i_h, mask=mask_t, other=0.0).to(tl.float32)
    rotated = _rotate_pos_2d(lam, raw, BT, BM, M)
    table = tl.load(
        readout_table + i_h * C * M + o_c[:, None] * M + o_m[None, :],
        mask=mask_c[:, None] & mask_m[None, :],
        other=0.0,
    ).to(slot_read_out.dtype.element_ty)
    logits = tl.dot(rotated.to(slot_read_out.dtype.element_ty), tl.trans(table))
    logits = tl.where(
        mask_t[:, None] & mask_c[None, :],
        logits,
        0.0,
    )
    max_logits = tl.max(tl.where(mask_c[None, :], logits, -float("inf")), axis=1)
    probs = tl.where(mask_c[None, :], tl.exp(logits - max_logits[:, None]), 0.0)
    denom = tl.sum(probs, axis=1)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    probs *= inv_denom[:, None]
    probs = probs.to(slot_read_out.dtype.element_ty)
    coeff = tl.dot(probs, table)
    slot_read = _rotate_neg_2d(lam, coeff, BT, BM, M)
    tl.store(
        slot_read_out + (o_t[:, None] * H + i_h) * M + o_m[None, :],
        slot_read,
        mask=mask,
    )


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial_v"] is not None,
        "STORE_FINAL_STATE": lambda args: args["state_v"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    },
)
@fla_cache_autotune(
    configs=[
        triton.Config({"BV": bv}, num_warps=num_warps, num_stages=2)
        for bv in [8, 16, 32, 64]
        for num_warps in NUM_WARPS
    ],
    key=["BM", "V", "USE_INITIAL_STATE", "STORE_FINAL_STATE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _fused_recurrent_cyfa_v_kernel(
    slot_read,
    slot_write,
    v,
    decay_trace,
    initial_v,
    state_v,
    out,
    cu_seqlens,
    T,
    H: tl.constexpr,
    M: tl.constexpr,
    V: tl.constexpr,
    BM: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = triton.cdiv(V, BV)
    i_v = (pid % NV).to(tl.int64)
    i_nh = (pid // NV).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T

    o_m = tl.arange(0, BM)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_m = o_m < M
    mask_v = o_v < V
    mask_h = mask_m[:, None] & mask_v[None, :]
    if USE_INITIAL_STATE:
        p_initial = initial_v + ((i_n * H + i_h) * M + o_m[:, None]) * V + o_v[None, :]
        state = tl.load(p_initial, mask=mask_h, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BM, BV), dtype=tl.float32)

    p_read = slot_read + (bos * H + i_h) * M + o_m
    p_write = slot_write + (bos * H + i_h) * M + o_m
    p_v = v + (bos * H + i_h) * V + o_v
    p_decay = decay_trace + bos * H + i_h
    p_out = out + (bos * H + i_h) * V + o_v

    for _ in tl.range(0, T, num_stages=2):
        state *= tl.load(p_decay, eviction_policy="evict_last").to(tl.float32)
        write = tl.load(p_write, mask=mask_m, other=0.0, eviction_policy="evict_last").to(tl.float32)
        value = tl.load(p_v, mask=mask_v, other=0.0, eviction_policy="evict_first").to(tl.float32)
        state += write[:, None] * value[None, :]
        read = tl.load(p_read, mask=mask_m, other=0.0, eviction_policy="evict_last").to(tl.float32)
        result = tl.sum(state * read[:, None], axis=0)
        tl.store(p_out, result, mask=mask_v)

        p_read += H * M
        p_write += H * M
        p_v += H * V
        p_decay += H
        p_out += H * V

    if STORE_FINAL_STATE:
        p_state = state_v + ((i_n * H + i_h) * M + o_m[:, None]) * V + o_v[None, :]
        tl.store(p_state, state, mask=mask_h)


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial_k"] is not None,
        "USE_INITIAL_LAMBDA": lambda args: args["initial_lambda"] is not None,
        "STORE_FINAL_STATE": lambda args: args["state_k"] is not None,
    },
)
@triton.jit
def _fused_recurrent_cyfa_decode_k_kernel(
    q,
    q_norm_weight,
    k,
    k_norm_weight,
    delta,
    beta,
    g,
    initial_k,
    initial_lambda,
    state_k,
    workspace,
    lambda_value,
    scale,
    q_norm_eps,
    k_norm_eps,
    H: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    USE_PDL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_LAMBDA: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NM: tl.constexpr = triton.cdiv(M, BM)
    NK: tl.constexpr = triton.cdiv(K, BK)
    i_m = pid % NM
    i_k = (pid // NM) % NK
    i_bh = (pid // (NM * NK)).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    if USE_PDL:
        tl.inline_asm_elementwise(
            'griddepcontrol.launch_dependents; // dummy $0',
            '=r',
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    o_m = i_m * BM + tl.arange(0, BM)
    o_k = i_k * BK + tl.arange(0, BK)
    mask_m = o_m < M
    mask_k = o_k < K
    mask_h = mask_m[:, None] & mask_k[None, :]

    q_square = 0.0
    k_square = 0.0
    for k_start in tl.range(0, K, BK, num_stages=2):
        o_norm = k_start + tl.arange(0, BK)
        mask_norm = o_norm < K
        q_norm = tl.load(q + i_bh * K + o_norm, mask=mask_norm, other=0.0).to(tl.float32)
        k_norm = tl.load(k + i_bh * K + o_norm, mask=mask_norm, other=0.0).to(tl.float32)
        q_square += tl.sum(q_norm * q_norm)
        k_square += tl.sum(k_norm * k_norm)
    q_rstd = tl.rsqrt(q_square / K + q_norm_eps)
    k_rstd = tl.rsqrt(k_square / K + k_norm_eps)

    if USE_INITIAL_LAMBDA:
        lam = tl.load(initial_lambda + i_b * H + i_h).to(tl.float32)
    else:
        lam = 0.0
    lam += tl.load(delta + i_bh).to(tl.float32)
    b = tl.load(beta + i_bh).to(tl.float32)
    slot_write = _as_recurrent_key_dtype(
        b * _cardinal_orbit_value(lam, o_m, M),
        q,
    )
    decay = exp(tl.load(g + i_bh).to(tl.float32))

    if USE_INITIAL_STATE:
        p_initial = initial_k + ((i_bh * K + o_k[None, :]) * M + o_m[:, None])
        state = tl.load(p_initial, mask=mask_h, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BM, BK), dtype=tl.float32)

    key = tl.load(k + i_bh * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
    key *= k_rstd
    key *= tl.load(k_norm_weight + o_k, mask=mask_k, other=0.0).to(tl.float32)
    key = _as_recurrent_key_dtype(key, k)
    state = state * decay + slot_write[:, None] * key[None, :]

    query = tl.load(q + i_bh * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
    query *= q_rstd
    query *= tl.load(q_norm_weight + o_k, mask=mask_k, other=0.0).to(tl.float32)
    query = _as_recurrent_key_dtype(query, q) * scale
    raw = tl.sum(state * query[None, :], axis=1)
    if USE_PDL:
        tl.store(workspace + ((i_bh * NK + i_k) * M + o_m), raw, mask=mask_m)
    else:
        tl.store(workspace + ((i_bh * (NK + 1) + i_k) * M + o_m), raw, mask=mask_m)
        if i_k == 0:
            tl.store(workspace + ((i_bh * (NK + 1) + NK) * M + o_m), slot_write, mask=mask_m)
        if i_k == 0 and i_m == 0:
            tl.store(lambda_value + i_bh, lam)
    if STORE_FINAL_STATE:
        p_state = state_k + ((i_bh * K + o_k[None, :]) * M + o_m[:, None])
        tl.store(p_state, state, mask=mask_h)


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial_v"] is not None,
        "USE_INITIAL_LAMBDA": lambda args: args["initial_lambda"] is not None,
        "STORE_FINAL_STATE": lambda args: args["state_v"] is not None,
    },
)
@triton.jit
def _fused_recurrent_cyfa_decode_v_kernel(
    v,
    g,
    delta,
    beta,
    readout_table,
    workspace,
    lambda_value,
    initial_lambda,
    initial_v,
    state_v,
    state_lambda,
    out,
    H: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    NK: tl.constexpr,
    BV: tl.constexpr,
    BM: tl.constexpr,
    USE_PDL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_LAMBDA: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = triton.cdiv(V, BV)
    i_v = pid % NV
    i_bh = (pid // NV).to(tl.int64)
    i_h = i_bh % H
    o_m = tl.arange(0, BM)
    o_c = tl.arange(0, BM)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_m = o_m < M
    mask_c = o_c < C
    mask_v = o_v < V

    if USE_PDL:
        if USE_INITIAL_LAMBDA:
            lam = tl.load(initial_lambda + i_bh).to(tl.float32)
        else:
            lam = 0.0
        lam += tl.load(delta + i_bh).to(tl.float32)
        slot_write = _as_recurrent_key_dtype(
            tl.load(beta + i_bh).to(tl.float32) * _cardinal_orbit_value(lam, o_m, M),
            v,
        )
    else:
        lam = tl.load(lambda_value + i_bh).to(tl.float32)
        slot_write = tl.load(
            workspace + ((i_bh * (NK + 1) + NK) * M + o_m),
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
    table = tl.load(
        readout_table + i_h * C * M + o_c[:, None] * M + o_m[None, :],
        mask=mask_c[:, None] & mask_m[None, :],
        other=0.0,
    ).to(tl.float32)
    decay = exp(tl.load(g + i_bh).to(tl.float32))
    value = tl.load(v + i_bh * V + o_v, mask=mask_v, other=0.0).to(tl.float32)
    if USE_PDL:
        tl.inline_asm_elementwise(
            'griddepcontrol.wait; // dummy $0',
            '=r',
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    raw = tl.zeros((BM,), dtype=tl.float32)
    for i_k in tl.static_range(0, NK):
        if USE_PDL:
            raw += tl.load(
                workspace + ((i_bh * NK + i_k) * M + o_m),
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
        else:
            raw += tl.load(
                workspace + ((i_bh * (NK + 1) + i_k) * M + o_m),
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
    z = _as_recurrent_key_dtype(_rotate_pos_1d(lam, raw, BM, M), v)
    logits = tl.sum(table * z[None, :], axis=1)
    max_logits = tl.max(tl.where(mask_c, logits, -float("inf")), axis=0)
    probs = tl.where(mask_c, tl.exp(logits - max_logits), 0.0)
    denom = tl.sum(probs, axis=0)
    probs *= tl.where(denom > 0.0, 1.0 / denom, 0.0)
    coeff = tl.sum(
        table * probs.to(v.dtype.element_ty).to(tl.float32)[:, None],
        axis=0,
    )
    slot_read = _as_recurrent_key_dtype(_rotate_neg_1d(lam, coeff, BM, M), v)

    mask_state = mask_m[:, None] & mask_v[None, :]
    if USE_INITIAL_STATE:
        p_initial = initial_v + ((i_bh * M + o_m[:, None]) * V + o_v[None, :])
        state = tl.load(p_initial, mask=mask_state, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BM, BV), dtype=tl.float32)
    state = state * decay + slot_write[:, None] * value[None, :]
    result = tl.sum(state * slot_read[:, None], axis=0)
    tl.store(out + i_bh * V + o_v, result, mask=mask_v)

    if STORE_FINAL_STATE:
        p_state = state_v + ((i_bh * M + o_m[:, None]) * V + o_v[None, :])
        tl.store(p_state, state, mask=mask_state)
        if i_v == 0:
            tl.store(state_lambda + i_bh, lam)


def _fused_recurrent_cyfa_decode_fwd(
    q: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k: torch.Tensor,
    k_norm_weight: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    readout_table: torch.Tensor,
    scale: float,
    q_norm_eps: float,
    k_norm_eps: float,
    initial_state: CyFAState | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, CyFAState | None]:
    batch, _, n_heads, d_k = q.shape
    d_v = v.shape[-1]
    num_slots = readout_table.shape[-1]
    use_pdl = _device_supports_pdl(q.device.index)
    state = validate_initial_state(
        initial_state,
        batch=batch,
        n_heads=n_heads,
        num_slots=num_slots,
        d_k=d_k,
        d_v=d_v,
        device=q.device,
    )
    if state is None:
        initial_k, initial_v, initial_lambda = None, None, None
    else:
        initial_k, initial_v, initial_lambda = state
    state_k = state_v = state_lambda = None
    if output_final_state:
        if initial_k is None:
            state_k = torch.empty(batch, n_heads, d_k, num_slots, device=q.device, dtype=torch.float32)
            state_v = torch.empty(batch, n_heads, num_slots, d_v, device=q.device, dtype=torch.float32)
            state_lambda = torch.empty(batch, n_heads, device=q.device, dtype=torch.float32)
        else:
            state_k, state_v = initial_k, initial_v
            state_lambda = torch.empty_like(initial_lambda) if use_pdl else initial_lambda

    bk = min(128, triton.next_power_of_2(d_k))
    bm = min(64, triton.next_power_of_2(num_slots))
    bv = min(64, triton.next_power_of_2(d_v))
    nk = triton.cdiv(d_k, bk)
    nm = triton.cdiv(num_slots, bm)
    nv = triton.cdiv(d_v, bv)
    workspace = torch.empty(
        batch,
        n_heads,
        nk if use_pdl else nk + 1,
        num_slots,
        device=q.device,
        dtype=q.dtype,
    )
    lambda_value = None
    if not use_pdl:
        lambda_value = torch.empty(batch, n_heads, device=q.device, dtype=torch.float32)
    out = torch.empty_like(v)

    _fused_recurrent_cyfa_decode_k_kernel[(batch * n_heads * nk * nm,)](
        q,
        q_norm_weight,
        k,
        k_norm_weight,
        delta,
        beta,
        g,
        initial_k,
        initial_lambda,
        state_k,
        workspace,
        lambda_value,
        float(scale),
        float(q_norm_eps),
        float(k_norm_eps),
        H=n_heads,
        K=d_k,
        M=num_slots,
        BK=bk,
        BM=bm,
        USE_PDL=use_pdl,
        num_warps=4,
        num_stages=2,
    )
    _fused_recurrent_cyfa_decode_v_kernel[(batch * n_heads * nv,)](
        v,
        g,
        delta,
        beta,
        readout_table,
        workspace,
        lambda_value,
        initial_lambda,
        initial_v,
        state_v,
        state_lambda,
        out,
        H=n_heads,
        V=d_v,
        M=num_slots,
        C=readout_table.shape[-2],
        NK=nk,
        BV=bv,
        BM=triton.next_power_of_2(num_slots),
        USE_PDL=use_pdl,
        launch_pdl=use_pdl,
        num_warps=4,
        num_stages=2,
    )
    final_state = (state_k, state_v, state_lambda) if output_final_state else None
    return out, final_state


def _fused_recurrent_cyfa_fwd(
    q: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_rstd: torch.Tensor | None,
    k: torch.Tensor,
    k_norm_weight: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    readout_table: torch.Tensor,
    scale: float,
    q_norm_eps: float,
    k_norm_eps: float,
    initial_state: CyFAState | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, CyFAState | None]:
    batch, seq_len, n_heads, d_k = q.shape
    d_v = v.shape[-1]
    num_slots = readout_table.shape[-1]
    if seq_len == 1 and cu_seqlens is None:
        return _fused_recurrent_cyfa_decode_fwd(
            q=q,
            q_norm_weight=q_norm_weight,
            k=k,
            k_norm_weight=k_norm_weight,
            v=v,
            delta=delta,
            beta=beta,
            g=g,
            readout_table=readout_table,
            scale=scale,
            q_norm_eps=q_norm_eps,
            k_norm_eps=k_norm_eps,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
    if cu_seqlens is None:
        n_seq = batch
    else:
        cu_seqlens = prepare_cu_seqlens(
            cu_seqlens,
            batch_size=batch,
            seq_len=seq_len,
            device=q.device,
        )
        n_seq = int(cu_seqlens.numel() - 1)

    state = validate_initial_state(
        initial_state,
        batch=n_seq,
        n_heads=n_heads,
        num_slots=num_slots,
        d_k=d_k,
        d_v=d_v,
        device=q.device,
    )
    if state is None:
        initial_k, initial_v, initial_lambda = None, None, None
    else:
        initial_k, initial_v, initial_lambda = state

    state_k = None
    state_v = None
    state_lambda = None
    if output_final_state:
        state_k = torch.empty(n_seq, n_heads, d_k, num_slots, device=q.device, dtype=torch.float32)
        state_v = torch.empty(n_seq, n_heads, num_slots, d_v, device=q.device, dtype=torch.float32)
        state_lambda = torch.empty(n_seq, n_heads, device=q.device, dtype=torch.float32)
    out = torch.empty_like(v)
    total_tokens = batch * seq_len
    lambda_trace = torch.empty(batch, seq_len, n_heads, device=q.device, dtype=torch.float32)
    decay_trace = torch.empty_like(lambda_trace)
    slot_write = torch.empty(batch, seq_len, n_heads, num_slots, device=q.device, dtype=q.dtype)
    slot_read = torch.empty_like(slot_write)
    bk = _k_recurrent_feature_block_size(d_k)
    readout_slots = readout_table.shape[-2]
    bm = triton.next_power_of_2(num_slots)
    bm_state = triton.next_power_of_2(num_slots)
    nk = triton.cdiv(d_k, bk)
    raw_partial = torch.empty(
        nk,
        batch,
        seq_len,
        n_heads,
        num_slots,
        device=q.device,
        dtype=q.dtype,
    )

    _fused_recurrent_cyfa_k_kernel[
        lambda meta: (triton.cdiv(num_slots, meta["BM"]) * nk * n_seq * n_heads,)
    ](
        q,
        q_norm_weight,
        q_rstd,
        k,
        delta,
        beta,
        g,
        initial_k,
        initial_lambda,
        state_k,
        state_lambda,
        lambda_trace,
        decay_trace,
        raw_partial,
        slot_write,
        cu_seqlens,
        float(scale),
        seq_len,
        total_tokens,
        H=n_heads,
        K=d_k,
        M=num_slots,
        BK=bk,
    )
    _fused_recurrent_cyfa_readout_kernel[
        lambda meta: (triton.cdiv(total_tokens, meta["BT"]) * n_heads,)
    ](
        raw_partial,
        lambda_trace,
        readout_table,
        slot_read,
        total_tokens,
        H=n_heads,
        M=num_slots,
        C=readout_slots,
        BM=bm,
        NK=nk,
    )
    _fused_recurrent_cyfa_v_kernel[
        lambda meta: (triton.cdiv(d_v, meta["BV"]) * n_seq * n_heads,)
    ](
        slot_read,
        slot_write,
        v,
        decay_trace,
        initial_v,
        state_v,
        out,
        cu_seqlens,
        seq_len,
        H=n_heads,
        M=num_slots,
        V=d_v,
        BM=bm_state,
    )
    final_state = (state_k, state_v, state_lambda) if output_final_state else None
    return out, final_state


class FusedRecurrentCyFAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        q_norm_weight: torch.Tensor,
        q_rstd: torch.Tensor | None,
        k: torch.Tensor,
        k_norm_weight: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        delta: torch.Tensor,
        beta: torch.Tensor,
        readout_table: torch.Tensor,
        scale: float,
        q_norm_eps: float,
        k_norm_eps: float,
        initial_state: CyFAState | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
    ):
        return _fused_recurrent_cyfa_fwd(
            q=q,
            q_norm_weight=q_norm_weight,
            q_rstd=q_rstd,
            k=k,
            k_norm_weight=k_norm_weight,
            v=v,
            delta=delta,
            beta=beta,
            g=g,
            readout_table=readout_table,
            scale=float(scale),
            q_norm_eps=float(q_norm_eps),
            k_norm_eps=float(k_norm_eps),
            initial_state=initial_state,
            output_final_state=bool(output_final_state),
            cu_seqlens=cu_seqlens,
        )

    @staticmethod
    @input_guard
    def backward(ctx, *args):
        raise NotImplementedError("`fused_recurrent_cyfa` is inference-only; use chunk mode for training.")


def fused_recurrent_cyfa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    delta: torch.Tensor,
    beta: torch.Tensor,
    readout: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_norm_eps: float,
    k_norm_eps: float,
    scale: float | None = None,
    initial_state: CyFAState | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, CyFAState | None]:
    """Run the inference-only recurrent CyclicFlowAttention path with log-space decay ``g``."""
    num_slots = validate_cyfa_inputs(
        q,
        k,
        v,
        g,
        delta,
        beta,
        readout,
        require_cuda=True,
    )
    grad_inputs = (q, k, v, g, delta, beta, readout, q_norm_weight, k_norm_weight, *(initial_state or ()))
    if torch.is_grad_enabled() and any(x is not None and x.requires_grad for x in grad_inputs):
        raise NotImplementedError("`fused_recurrent_cyfa` is inference-only; use chunk mode for training.")
    cu_seqlens = prepare_cu_seqlens(
        cu_seqlens,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        device=q.device,
    )

    _, _, _, d_k = q.shape
    if scale is None:
        scale = d_k ** -0.5
    if q.shape[1] == 1 and cu_seqlens is None:
        q = q.contiguous()
        k = k.contiguous()
        q_rstd = None
    else:
        q, k, q_rstd, _ = qk_rmsnorm_fwd(
            q,
            k,
            q_norm_weight,
            k_norm_weight,
            float(q_norm_eps),
            float(k_norm_eps),
            store_q=False,
            store_rstd=True,
        )
    cache_key = (id(readout), _tensor_version(readout), int(num_slots), str(q.device), q.dtype)
    cache_entry = _READOUT_TABLE_CACHE.get(cache_key)
    readout_table = None
    if cache_entry is not None:
        readout_ref, cached_table = cache_entry
        if readout_ref() is readout:
            readout_table = cached_table
    if readout_table is None:
        readout_table = build_readout_table(readout).to(q.dtype)
        _READOUT_TABLE_CACHE[cache_key] = (weakref.ref(readout), readout_table)
    return FusedRecurrentCyFAFunction.apply(
        q,
        q_norm_weight,
        q_rstd,
        k,
        k_norm_weight,
        v,
        g,
        delta,
        beta,
        readout_table,
        float(scale),
        float(q_norm_eps),
        float(k_norm_eps),
        initial_state,
        bool(output_final_state),
        cu_seqlens,
    )
