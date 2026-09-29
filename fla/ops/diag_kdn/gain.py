# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard

NUM_WARPS = [1, 2, 4]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS
    ],
    key=['H', 'K', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def diag_kdn_gain_fwd_kernel_map(
    k,
    g,
    omega,
    r,
    m,
    s,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tg, i_h = tl.program_id(0).to(tl.int64) // H, tl.program_id(0).to(tl.int64) % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_tg * 2).to(tl.int64), tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_b, i_t = i_tg // NT, i_tg % NT
        bos, eos = i_b * T, i_b * T + T
    o_k = tl.arange(0, BK)
    m_k = o_k < K

    # compose the per-channel Möbius maps p_t = (r a p_{t-1} + r w) / (s k^2 a p_{t-1} + r + s k^2 w),
    # renormalized every step since only the ratio of the entries matters
    b_mA = tl.full([BK], 1., dtype=tl.float32)
    b_mB = tl.zeros([BK], dtype=tl.float32)
    b_mC = tl.zeros([BK], dtype=tl.float32)
    b_mD = tl.full([BK], 1., dtype=tl.float32)
    for t in range(i_t * BT, min(i_t * BT + BT, T)):
        o_t = (bos + t) * H + i_h
        b_k = tl.load(k + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_g = tl.load(g + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_w = tl.load(omega + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_r = tl.load(r + o_t).to(tl.float32)
        b_a = exp(2 * b_g)
        b_sk = s * b_k * b_k
        b_tA, b_tB, b_tC, b_tD = b_r * b_a, b_r * b_w, b_sk * b_a, b_r + b_sk * b_w
        b_nA = b_tA * b_mA + b_tB * b_mC
        b_nB = b_tA * b_mB + b_tB * b_mD
        b_nC = b_tC * b_mA + b_tD * b_mC
        b_nD = b_tC * b_mB + b_tD * b_mD
        b_scale = 1. / tl.maximum(tl.maximum(tl.maximum(tl.abs(b_nA), tl.abs(b_nB)),
                                  tl.maximum(tl.abs(b_nC), tl.abs(b_nD))), 1e-30)
        b_mA, b_mB, b_mC, b_mD = b_nA * b_scale, b_nB * b_scale, b_nC * b_scale, b_nD * b_scale

    o_m = (i_tg * H + i_h) * 4 * K
    tl.store(m + o_m + o_k, b_mA, mask=m_k)
    tl.store(m + o_m + K + o_k, b_mB, mask=m_k)
    tl.store(m + o_m + 2 * K + o_k, b_mC, mask=m_k)
    tl.store(m + o_m + 3 * K + o_k, b_mD, mask=m_k)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def diag_kdn_gain_fwd_kernel_carry(
    m,
    p0,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_n, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    o_k = tl.arange(0, BK)
    m_k = o_k < K

    # the scan runs on the covariance p = 1 / precision
    b_p = tl.full([BK], 1., dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_p = 1. / tl.load(h0 + (i_n * H + i_h) * K + o_k, mask=m_k, other=1.).to(tl.float32)
    for i_t in range(NT):
        o_t = (boh + i_t) * H + i_h
        tl.store(p0 + o_t * K + o_k, b_p, mask=m_k)
        b_mA = tl.load(m + o_t * 4 * K + o_k, mask=m_k, other=1.)
        b_mB = tl.load(m + o_t * 4 * K + K + o_k, mask=m_k, other=0.)
        b_mC = tl.load(m + o_t * 4 * K + 2 * K + o_k, mask=m_k, other=0.)
        b_mD = tl.load(m + o_t * 4 * K + 3 * K + o_k, mask=m_k, other=1.)
        b_p = (b_mA * b_p + b_mB) / (b_mC * b_p + b_mD)
    if STORE_FINAL_STATE:
        tl.store(ht + (i_n * H + i_h) * K + o_k, 1. / b_p, mask=m_k)


@triton.heuristics({
    'STORE_GAIN': lambda args: args['kappa'] is not None,
    'STORE_COVARIANCE': lambda args: args['p'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS
    ],
    key=['H', 'K', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def diag_kdn_gain_fwd_kernel_emit(
    k,
    g,
    omega,
    r,
    p0,
    kappa,
    p,
    s,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    STORE_GAIN: tl.constexpr,
    STORE_COVARIANCE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tg, i_h = tl.program_id(0).to(tl.int64) // H, tl.program_id(0).to(tl.int64) % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_tg * 2).to(tl.int64), tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_b, i_t = i_tg // NT, i_tg % NT
        bos, eos = i_b * T, i_b * T + T
    o_k = tl.arange(0, BK)
    m_k = o_k < K

    b_p = tl.load(p0 + (i_tg * H + i_h) * K + o_k, mask=m_k, other=0.)
    for t in range(i_t * BT, min(i_t * BT + BT, T)):
        o_t = (bos + t) * H + i_h
        b_k = tl.load(k + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_g = tl.load(g + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_w = tl.load(omega + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_r = tl.load(r + o_t).to(tl.float32)
        if STORE_COVARIANCE:
            tl.store(p + o_t * K + o_k, b_p, mask=m_k)
        b_z = exp(2 * b_g) * b_p + b_w
        if STORE_GAIN:
            b_d = b_r + tl.sum(b_z * b_k * b_k)
            tl.store(kappa + o_t * K + o_k, (b_z * b_k / b_d).to(kappa.dtype.element_ty), mask=m_k)
        b_p = b_r * b_z / (b_r + s * b_k * b_k * b_z)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS
    ],
    key=['H', 'K', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def diag_kdn_gain_bwd_kernel_map(
    k,
    g,
    omega,
    r,
    p,
    dkappa,
    m,
    s,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tg, i_h = tl.program_id(0).to(tl.int64) // H, tl.program_id(0).to(tl.int64) % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_tg * 2).to(tl.int64), tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_b, i_t = i_tg // NT, i_tg % NT
        bos, eos = i_b * T, i_b * T + T
    o_k = tl.arange(0, BK)
    m_k = o_k < K

    # the covariance adjoint obeys the per-channel affine recurrence lam_{t-1} = A_t lam_t + B_t
    b_mA = tl.full([BK], 1., dtype=tl.float32)
    b_mB = tl.zeros([BK], dtype=tl.float32)
    for t in range(min(i_t * BT + BT, T) - 1, i_t * BT - 1, -1):
        o_t = (bos + t) * H + i_h
        b_k = tl.load(k + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_g = tl.load(g + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_w = tl.load(omega + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_r = tl.load(r + o_t).to(tl.float32)
        b_p = tl.load(p + o_t * K + o_k, mask=m_k, other=0.)
        b_du = tl.load(dkappa + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_a = exp(2 * b_g)
        b_z = b_a * b_p + b_w
        b_id = 1. / (b_r + tl.sum(b_z * b_k * b_k))
        # sum_i dkappa_i * kappa_i
        b_su = tl.sum(b_du * b_z * b_k) * b_id
        b_q = b_r / (b_r + s * b_k * b_k * b_z)
        b_tA = b_a * b_q * b_q
        b_tB = b_a * b_k * b_id * (b_du - b_k * b_su)
        b_mA, b_mB = b_tA * b_mA, b_tA * b_mB + b_tB

    o_m = (i_tg * H + i_h) * 2 * K
    tl.store(m + o_m + o_k, b_mA, mask=m_k)
    tl.store(m + o_m + K + o_k, b_mB, mask=m_k)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def diag_kdn_gain_bwd_kernel_carry(
    m,
    lam,
    h0,
    ht,
    dht,
    dh0,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_n, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    o_k = tl.arange(0, BK)
    m_k = o_k < K

    b_lam = tl.zeros([BK], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        # ht = 1 / p_T, so dp_T = -dht * ht^2
        b_ht = tl.load(ht + (i_n * H + i_h) * K + o_k, mask=m_k, other=0.)
        b_lam = -tl.load(dht + (i_n * H + i_h) * K + o_k, mask=m_k, other=0.).to(tl.float32) * b_ht * b_ht
    for i_t in range(NT - 1, -1, -1):
        o_t = (boh + i_t) * H + i_h
        tl.store(lam + o_t * K + o_k, b_lam, mask=m_k)
        b_mA = tl.load(m + o_t * 2 * K + o_k, mask=m_k, other=0.)
        b_mB = tl.load(m + o_t * 2 * K + K + o_k, mask=m_k, other=0.)
        b_lam = b_mA * b_lam + b_mB
    if USE_INITIAL_STATE:
        b_h0 = tl.load(h0 + (i_n * H + i_h) * K + o_k, mask=m_k, other=1.).to(tl.float32)
        tl.store(dh0 + (i_n * H + i_h) * K + o_k, -b_lam / (b_h0 * b_h0), mask=m_k)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS
    ],
    key=['H', 'K', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def diag_kdn_gain_bwd_kernel_emit(
    k,
    g,
    omega,
    r,
    p,
    lam,
    dkappa,
    dk,
    dg,
    domega,
    dr,
    s,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tg, i_h = tl.program_id(0).to(tl.int64) // H, tl.program_id(0).to(tl.int64) % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_tg * 2).to(tl.int64), tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_b, i_t = i_tg // NT, i_tg % NT
        bos, eos = i_b * T, i_b * T + T
    o_k = tl.arange(0, BK)
    m_k = o_k < K

    b_lam = tl.load(lam + (i_tg * H + i_h) * K + o_k, mask=m_k, other=0.)
    for t in range(min(i_t * BT + BT, T) - 1, i_t * BT - 1, -1):
        o_t = (bos + t) * H + i_h
        b_k = tl.load(k + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_g = tl.load(g + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_w = tl.load(omega + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_r = tl.load(r + o_t).to(tl.float32)
        b_p = tl.load(p + o_t * K + o_k, mask=m_k, other=0.)
        b_du = tl.load(dkappa + o_t * K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_a = exp(2 * b_g)
        b_z = b_a * b_p + b_w
        b_k2 = b_k * b_k
        b_id = 1. / (b_r + tl.sum(b_z * b_k2))
        b_su = tl.sum(b_du * b_z * b_k) * b_id
        b_ip = 1. / (b_r + s * b_k2 * b_z)
        # lam / (r + s k^2 z)^2, so that dp/dz = r^2 / (r + s k^2 z)^2
        b_lp = b_lam * b_ip * b_ip
        b_dz = b_k * b_id * (b_du - b_k * b_su) + b_lp * b_r * b_r
        b_dk = b_z * b_id * (b_du - 2 * b_k * b_su) - 2 * s * b_k * b_z * b_z * b_lp * b_r
        b_dr = tl.sum(s * b_k2 * b_z * b_z * b_lp) - b_su * b_id
        tl.store(dk + o_t * K + o_k, b_dk.to(dk.dtype.element_ty), mask=m_k)
        tl.store(dg + o_t * K + o_k, (2 * b_a * b_p * b_dz).to(dg.dtype.element_ty), mask=m_k)
        tl.store(domega + o_t * K + o_k, b_dz.to(domega.dtype.element_ty), mask=m_k)
        tl.store(dr + o_t, b_dr.to(dr.dtype.element_ty))
        b_lam = b_a * b_dz


def diag_kdn_gain_fwd_emit(
    k: torch.Tensor,
    g: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    p0: torch.Tensor,
    s: float,
    store_gain: bool = True,
    store_covariance: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K = k.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    kappa = torch.empty(B, T, H, K, dtype=torch.float32, device=k.device) if store_gain else None
    p = torch.empty(B, T, H, K, dtype=torch.float32, device=k.device) if store_covariance else None
    diag_kdn_gain_fwd_kernel_emit[(B * NT * H,)](
        k=k,
        g=g,
        omega=omega,
        r=r,
        p0=p0,
        kappa=kappa,
        p=p,
        s=s,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=triton.next_power_of_2(K),
    )
    return kappa, p


def diag_kdn_gain_fwd(
    k: torch.Tensor,
    g: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    s: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    B, T, H, K = k.shape
    BT, BK = chunk_size, triton.next_power_of_2(K)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    final_state = k.new_empty(N, H, K, dtype=torch.float32) if output_final_state else None
    # per-chunk Möbius maps and the covariance entering each chunk
    m = k.new_empty(B * NT, H, 4, K, dtype=torch.float32)
    p0 = k.new_empty(B * NT, H, K, dtype=torch.float32)

    diag_kdn_gain_fwd_kernel_map[(B * NT * H,)](
        k=k,
        g=g,
        omega=omega,
        r=r,
        m=m,
        s=s,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
    )
    diag_kdn_gain_fwd_kernel_carry[(N, H)](
        m=m,
        p0=p0,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
    )
    kappa, _ = diag_kdn_gain_fwd_emit(
        k=k,
        g=g,
        omega=omega,
        r=r,
        p0=p0,
        s=s,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    return kappa, final_state, p0


def diag_kdn_gain_bwd(
    k: torch.Tensor,
    g: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    p0: torch.Tensor,
    s: float,
    dkappa: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K = k.shape
    BT, BK = chunk_size, triton.next_power_of_2(K)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    # recompute the covariance entering each token from the saved chunk-start covariances
    _, p = diag_kdn_gain_fwd_emit(
        k=k,
        g=g,
        omega=omega,
        r=r,
        p0=p0,
        s=s,
        store_gain=False,
        store_covariance=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    dk = torch.empty_like(k)
    dg = torch.empty_like(g)
    domega = torch.empty_like(omega)
    dr = torch.empty_like(r)
    dh0 = torch.empty_like(initial_state, dtype=torch.float32) if initial_state is not None else None
    # per-chunk affine adjoint maps and the covariance adjoint entering each chunk from the right
    m = k.new_empty(B * NT, H, 2, K, dtype=torch.float32)
    lam = k.new_empty(B * NT, H, K, dtype=torch.float32)

    grid = (B * NT * H,)
    diag_kdn_gain_bwd_kernel_map[grid](
        k=k,
        g=g,
        omega=omega,
        r=r,
        p=p,
        dkappa=dkappa,
        m=m,
        s=s,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
    )
    diag_kdn_gain_bwd_kernel_carry[(N, H)](
        m=m,
        lam=lam,
        h0=initial_state,
        ht=final_state,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
    )
    diag_kdn_gain_bwd_kernel_emit[grid](
        k=k,
        g=g,
        omega=omega,
        r=r,
        p=p,
        lam=lam,
        dkappa=dkappa,
        dk=dk,
        dg=dg,
        domega=domega,
        dr=dr,
        s=s,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
    )
    return dk, dg, domega, dr, dh0


class DiagKDNGainFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        k: torch.Tensor,
        g: torch.Tensor,
        omega: torch.Tensor,
        r: torch.Tensor,
        s: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None,
        chunk_indices: torch.LongTensor | None,
    ):
        kappa, final_state, p0 = diag_kdn_gain_fwd(
            k=k,
            g=g,
            omega=omega,
            r=r,
            s=s,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        ctx.save_for_backward(k, g, omega, r, p0, initial_state, final_state, cu_seqlens, chunk_indices)
        ctx.s = s
        return kappa, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dkappa: torch.Tensor, dht: torch.Tensor | None):
        k, g, omega, r, p0, initial_state, final_state, cu_seqlens, chunk_indices = ctx.saved_tensors
        dk, dg, domega, dr, dh0 = diag_kdn_gain_bwd(
            k=k,
            g=g,
            omega=omega,
            r=r,
            p0=p0,
            s=ctx.s,
            dkappa=dkappa,
            initial_state=initial_state,
            final_state=final_state,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        return dk, dg, domega, dr, None, dh0, None, None, None


@torch.compiler.disable
def diag_kdn_gain(
    k: torch.Tensor,
    g: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    info_scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Diagonal Kalman gain of DiagKDN, i.e., the per-channel write key of the delta-rule memory update.

    Each head tracks a diagonal posterior precision ``c`` over the key channels of its memory.
    With the covariance ``p = 1 / c``,

    .. math::
        z_t = \exp(2 g_t) \odot p_{t-1} + \omega_t, \quad
        \kappa_t = z_t \odot k_t / (r_t + \langle z_t, k_t^2 \rangle), \quad
        p_t = r_t z_t / (r_t + \text{info\_scale} \, k_t^2 \odot z_t).

    The covariance recurrence is solved as a chunked scan of per-channel Möbius maps.

    Args:
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        g (torch.Tensor):
            Forget gates (in log space) of shape ``[B, T, H, K]``.
        omega (torch.Tensor):
            Positive process noise of shape ``[B, T, H, K]``.
        r (torch.Tensor):
            Positive observation noise of shape ``[B, T, H]``.
        initial_state (torch.Tensor, Optional):
            Initial precision of shape ``[N, H, K]`` for ``N`` input sequences.
            A precision of 1 is used if `None`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final precision of shape ``[N, H, K]``. Default: `False`.
        info_scale (float, Optional):
            Information contributed by one unit-norm key. Default: `None`, i.e., ``K``.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape ``[N+1]`` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.
        cu_seqlens_cpu (torch.LongTensor, Optional):
            CPU copy of ``cu_seqlens`` that avoids a device synchronization. Default: `None`.

    Returns:
        kappa (torch.Tensor):
            Gains of shape ``[B, T, H, K]`` in ``float32``.
        final_state (torch.Tensor):
            Final precision of shape ``[N, H, K]`` if ``output_final_state=True`` else `None`.
    """
    B, T, H, K = k.shape
    assert K <= 256, "The key dimension must be at most 256."
    assert g.shape == k.shape, f"g must have shape [B, T, H, K]={list(k.shape)}, got {list(g.shape)}"
    assert omega.shape == k.shape, f"omega must have shape [B, T, H, K]={list(k.shape)}, got {list(omega.shape)}"
    assert r.shape == (B, T, H), f"r must have shape [B, T, H]={[B, T, H]}, got {list(r.shape)}"
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.shape[1:] == (H, K), \
            f"initial_state must have shape [N, H, K]=[N, {H}, {K}], got {list(initial_state.shape)}"
    if info_scale is None:
        info_scale = K
    if not info_scale > 0:
        raise ValueError(f"`info_scale` must be positive, got {info_scale}.")
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None
    return DiagKDNGainFunction.apply(
        k,
        g,
        omega,
        r,
        float(info_scale),
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
    )
