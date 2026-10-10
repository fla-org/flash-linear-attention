# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.jit(do_not_specialize=['NR'])
def iso_kdn_decay_fwd_kernel(
    g,
    a,
    NR,
    K: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
):
    i_r = tl.program_id(0).to(tl.int64)
    o_r = i_r * BR + tl.arange(0, BR).to(tl.int64)
    o_k = tl.arange(0, BK)
    m_r = o_r < NR
    m_g = m_r[:, None] & (o_k < K)[None, :]

    b_g = tl.load(g + o_r[:, None] * K + o_k[None, :], mask=m_g, other=0.)
    # [BR]
    b_a = tl.sum(tl.where(m_g, exp(2 * b_g), 0.), 1) / K
    tl.store(a + o_r, b_a, mask=m_r)


@triton.jit(do_not_specialize=['NR'])
def iso_kdn_decay_bwd_kernel(
    g,
    da,
    dg,
    NR,
    K: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
):
    i_r = tl.program_id(0).to(tl.int64)
    o_r = i_r * BR + tl.arange(0, BR).to(tl.int64)
    o_k = tl.arange(0, BK)
    m_r = o_r < NR
    m_g = m_r[:, None] & (o_k < K)[None, :]

    b_g = tl.load(g + o_r[:, None] * K + o_k[None, :], mask=m_g, other=0.).to(tl.float32)
    b_da = tl.load(da + o_r, mask=m_r, other=0.)
    # [BR, BK]
    b_dg = b_da[:, None] * (2. / K) * exp(2 * b_g)
    tl.store(dg + o_r[:, None] * K + o_k[None, :], b_dg.to(dg.dtype.element_ty), mask=m_g)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def iso_kdn_gain_fwd_kernel_map(
    a,
    omega,
    r,
    m,
    s,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # compose the per-token Möbius maps p_t = (r a p_{t-1} + r w) / (s a p_{t-1} + r + s w),
    # renormalized every step since only the ratio of the entries matters
    b_mA, b_mB, b_mC, b_mD = 1., 0., 0., 1.
    for t in range(i_t * BT, min(i_t * BT + BT, T)):
        o_t = (bos + t) * H + i_h
        b_a = tl.load(a + o_t).to(tl.float32)
        b_w = tl.load(omega + o_t).to(tl.float32)
        b_r = tl.load(r + o_t).to(tl.float32)
        b_tA, b_tB, b_tC, b_tD = b_r * b_a, b_r * b_w, s * b_a, b_r + s * b_w
        b_nA = b_tA * b_mA + b_tB * b_mC
        b_nB = b_tA * b_mB + b_tB * b_mD
        b_nC = b_tC * b_mA + b_tD * b_mC
        b_nD = b_tC * b_mB + b_tD * b_mD
        b_scale = 1. / tl.maximum(tl.maximum(tl.maximum(tl.abs(b_nA), tl.abs(b_nB)),
                                  tl.maximum(tl.abs(b_nC), tl.abs(b_nD))), 1e-30)
        b_mA, b_mB, b_mC, b_mD = b_nA * b_scale, b_nB * b_scale, b_nC * b_scale, b_nD * b_scale

    o_m = (i_tg * H + i_h) * 4
    tl.store(m + o_m, b_mA)
    tl.store(m + o_m + 1, b_mB)
    tl.store(m + o_m + 2, b_mC)
    tl.store(m + o_m + 3, b_mD)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def iso_kdn_gain_fwd_kernel_carry(
    m,
    p0,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
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

    # the scan runs on the covariance p = 1 / precision
    b_p = 1.
    if USE_INITIAL_STATE:
        b_p = 1. / tl.load(h0 + i_n * H + i_h).to(tl.float32)
    for i_t in range(NT):
        o_t = (boh + i_t) * H + i_h
        tl.store(p0 + o_t, b_p)
        b_mA = tl.load(m + o_t * 4)
        b_mB = tl.load(m + o_t * 4 + 1)
        b_mC = tl.load(m + o_t * 4 + 2)
        b_mD = tl.load(m + o_t * 4 + 3)
        b_p = (b_mA * b_p + b_mB) / (b_mC * b_p + b_mD)
    if STORE_FINAL_STATE:
        tl.store(ht + i_n * H + i_h, 1. / b_p)


@triton.heuristics({
    'STORE_COVARIANCE': lambda args: args['p'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def iso_kdn_gain_fwd_kernel_emit(
    a,
    omega,
    r,
    p0,
    beta,
    p,
    s,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    STORE_COVARIANCE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    b_p = tl.load(p0 + i_tg * H + i_h)
    for t in range(i_t * BT, min(i_t * BT + BT, T)):
        o_t = (bos + t) * H + i_h
        b_a = tl.load(a + o_t).to(tl.float32)
        b_w = tl.load(omega + o_t).to(tl.float32)
        b_r = tl.load(r + o_t).to(tl.float32)
        if STORE_COVARIANCE:
            tl.store(p + o_t, b_p)
        b_z = b_a * b_p + b_w
        tl.store(beta + o_t, (b_z / (b_r + b_z)).to(beta.dtype.element_ty))
        b_p = b_r * b_z / (b_r + s * b_z)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def iso_kdn_gain_bwd_kernel_map(
    a,
    omega,
    r,
    p,
    dbeta,
    m,
    s,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # the covariance adjoint obeys the affine recurrence lam_{t-1} = A_t lam_t + B_t
    b_mA, b_mB = 1., 0.
    for t in range(min(i_t * BT + BT, T) - 1, i_t * BT - 1, -1):
        o_t = (bos + t) * H + i_h
        b_a = tl.load(a + o_t).to(tl.float32)
        b_w = tl.load(omega + o_t).to(tl.float32)
        b_r = tl.load(r + o_t).to(tl.float32)
        b_p = tl.load(p + o_t)
        b_db = tl.load(dbeta + o_t).to(tl.float32)
        b_z = b_a * b_p + b_w
        b_q = b_r / (b_r + s * b_z)
        b_tA = b_a * b_q * b_q
        b_tB = b_a * b_db * b_r / ((b_r + b_z) * (b_r + b_z))
        b_mA, b_mB = b_tA * b_mA, b_tA * b_mB + b_tB

    o_m = (i_tg * H + i_h) * 2
    tl.store(m + o_m, b_mA)
    tl.store(m + o_m + 1, b_mB)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def iso_kdn_gain_bwd_kernel_carry(
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
    BT: tl.constexpr,
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

    b_lam = 0.
    if USE_FINAL_STATE_GRADIENT:
        # ht = 1 / p_T, so dp_T = -dht * ht^2
        b_ht = tl.load(ht + i_n * H + i_h)
        b_lam = -tl.load(dht + i_n * H + i_h).to(tl.float32) * b_ht * b_ht
    for i_t in range(NT - 1, -1, -1):
        o_t = (boh + i_t) * H + i_h
        tl.store(lam + o_t, b_lam)
        b_lam = tl.load(m + o_t * 2) * b_lam + tl.load(m + o_t * 2 + 1)
    if USE_INITIAL_STATE:
        b_h0 = tl.load(h0 + i_n * H + i_h).to(tl.float32)
        tl.store(dh0 + i_n * H + i_h, -b_lam / (b_h0 * b_h0))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def iso_kdn_gain_bwd_kernel_emit(
    a,
    omega,
    r,
    p,
    lam,
    dbeta,
    da,
    domega,
    dr,
    s,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    b_lam = tl.load(lam + i_tg * H + i_h)
    for t in range(min(i_t * BT + BT, T) - 1, i_t * BT - 1, -1):
        o_t = (bos + t) * H + i_h
        b_a = tl.load(a + o_t).to(tl.float32)
        b_w = tl.load(omega + o_t).to(tl.float32)
        b_r = tl.load(r + o_t).to(tl.float32)
        b_p = tl.load(p + o_t)
        b_db = tl.load(dbeta + o_t).to(tl.float32)
        b_z = b_a * b_p + b_w
        b_id = 1. / (b_r + b_z)
        b_ip = 1. / (b_r + s * b_z)
        b_id2, b_ip2 = b_id * b_id, b_ip * b_ip
        b_dz = b_r * (b_db * b_id2 + b_lam * b_r * b_ip2)
        b_dr = b_z * (b_lam * s * b_z * b_ip2 - b_db * b_id2)
        tl.store(da + o_t, (b_dz * b_p).to(da.dtype.element_ty))
        tl.store(domega + o_t, b_dz.to(domega.dtype.element_ty))
        tl.store(dr + o_t, b_dr.to(dr.dtype.element_ty))
        b_lam = b_a * b_dz


def iso_kdn_decay_fwd(g: torch.Tensor) -> torch.Tensor:
    B, T, H, K = g.shape
    NR = B * T * H
    BK = triton.next_power_of_2(K)
    BR = max(1, 4096 // BK)
    a = g.new_empty(B, T, H, dtype=torch.float32)
    iso_kdn_decay_fwd_kernel[(triton.cdiv(NR, BR),)](
        g=g,
        a=a,
        NR=NR,
        K=K,
        BR=BR,
        BK=BK,
    )
    return a


def iso_kdn_decay_bwd(g: torch.Tensor, da: torch.Tensor) -> torch.Tensor:
    B, T, H, K = g.shape
    NR = B * T * H
    BK = triton.next_power_of_2(K)
    BR = max(1, 4096 // BK)
    dg = torch.empty_like(g)
    iso_kdn_decay_bwd_kernel[(triton.cdiv(NR, BR),)](
        g=g,
        da=da,
        dg=dg,
        NR=NR,
        K=K,
        BR=BR,
        BK=BK,
    )
    return dg


def iso_kdn_gain_fwd(
    a: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    s: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    store_covariance: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, H = a.shape
    BT = chunk_size
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    beta = torch.empty(B, T, H, dtype=torch.float32, device=a.device)
    p = torch.empty(B, T, H, dtype=torch.float32, device=a.device) if store_covariance else None
    final_state = a.new_empty(N, H, dtype=torch.float32) if output_final_state else None
    # per-chunk Möbius maps and the covariance entering each chunk
    m = a.new_empty(B * NT, H, 4, dtype=torch.float32)
    p0 = a.new_empty(B * NT, H, dtype=torch.float32)

    grid = (NT, B * H)
    iso_kdn_gain_fwd_kernel_map[grid](
        a=a,
        omega=omega,
        r=r,
        m=m,
        s=s,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
    )
    iso_kdn_gain_fwd_kernel_carry[(N, H)](
        m=m,
        p0=p0,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        BT=BT,
    )
    iso_kdn_gain_fwd_kernel_emit[grid](
        a=a,
        omega=omega,
        r=r,
        p0=p0,
        beta=beta,
        p=p,
        s=s,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
    )
    return beta, final_state, p


def iso_kdn_gain_bwd(
    a: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    p: torch.Tensor,
    s: float,
    dbeta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H = a.shape
    BT = chunk_size
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    da = torch.empty_like(a)
    domega = torch.empty_like(omega)
    dr = torch.empty_like(r)
    dh0 = torch.empty_like(initial_state, dtype=torch.float32) if initial_state is not None else None
    # per-chunk affine adjoint maps and the covariance adjoint entering each chunk from the right
    m = a.new_empty(B * NT, H, 2, dtype=torch.float32)
    lam = a.new_empty(B * NT, H, dtype=torch.float32)

    grid = (NT, B * H)
    iso_kdn_gain_bwd_kernel_map[grid](
        a=a,
        omega=omega,
        r=r,
        p=p,
        dbeta=dbeta,
        m=m,
        s=s,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
    )
    iso_kdn_gain_bwd_kernel_carry[(N, H)](
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
        BT=BT,
    )
    iso_kdn_gain_bwd_kernel_emit[grid](
        a=a,
        omega=omega,
        r=r,
        p=p,
        lam=lam,
        dbeta=dbeta,
        da=da,
        domega=domega,
        dr=dr,
        s=s,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
    )
    return da, domega, dr, dh0


class IsoKDNGainFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        g: torch.Tensor,
        omega: torch.Tensor,
        r: torch.Tensor,
        s: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None,
        chunk_indices: torch.LongTensor | None,
    ):
        a = iso_kdn_decay_fwd(g)
        beta, final_state, p = iso_kdn_gain_fwd(
            a=a,
            omega=omega,
            r=r,
            s=s,
            initial_state=initial_state,
            output_final_state=output_final_state,
            store_covariance=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        ctx.save_for_backward(g, a, omega, r, p, initial_state, final_state, cu_seqlens, chunk_indices)
        ctx.s = s
        return beta, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dbeta: torch.Tensor, dht: torch.Tensor | None):
        g, a, omega, r, p, initial_state, final_state, cu_seqlens, chunk_indices = ctx.saved_tensors
        da, domega, dr, dh0 = iso_kdn_gain_bwd(
            a=a,
            omega=omega,
            r=r,
            p=p,
            s=ctx.s,
            dbeta=dbeta,
            initial_state=initial_state,
            final_state=final_state,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        dg = iso_kdn_decay_bwd(g, da)
        return dg, domega, dr, None, dh0, None, None, None


@torch.compiler.disable
def iso_kdn_gain(
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
    Isotropic Kalman gain of IsoKDN, i.e., the write strength of the delta-rule memory update.

    Each head tracks a scalar posterior precision ``c`` of its memory.
    With ``a_t = mean(exp(2 g_t))`` over the key dimension ``K``, and unit-norm keys,

    .. math::
        z_t = a_t / c_{t-1} + \omega_t, \quad
        \beta_t = z_t / (r_t + z_t), \quad
        c_t = 1 / z_t + \text{info\_scale} / (K r_t).

    The recurrence is solved as a chunked scan of Möbius maps.

    Args:
        g (torch.Tensor):
            Forget gates (in log space) of shape ``[B, T, H, K]``.
        omega (torch.Tensor):
            Positive process noise of shape ``[B, T, H]``.
        r (torch.Tensor):
            Positive observation noise of shape ``[B, T, H]``.
        initial_state (torch.Tensor, Optional):
            Initial precision of shape ``[N, H]`` for ``N`` input sequences.
            A precision of 1 is used if `None`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final precision of shape ``[N, H]``. Default: `False`.
        info_scale (float, Optional):
            Information contributed by one unit-norm key, relative to ``K``. Default: `None`, i.e., ``K``.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape ``[N+1]`` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.
        cu_seqlens_cpu (torch.LongTensor, Optional):
            CPU copy of ``cu_seqlens`` that avoids a device synchronization. Default: `None`.

    Returns:
        beta (torch.Tensor):
            Gains of shape ``[B, T, H]`` in ``float32``.
        final_state (torch.Tensor):
            Final precision of shape ``[N, H]`` if ``output_final_state=True`` else `None`.
    """
    B, T, H, K = g.shape
    assert omega.shape == (B, T, H), f"omega must have shape [B, T, H]={[B, T, H]}, got {list(omega.shape)}"
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
    if info_scale is None:
        info_scale = K
    if not info_scale > 0:
        raise ValueError(f"`info_scale` must be positive, got {info_scale}.")
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None
    return IsoKDNGainFunction.apply(
        g,
        omega,
        r,
        info_scale / K,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
    )
