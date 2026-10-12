# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp2, unflatten_program_id
from fla.utils import IS_TF32_SUPPORTED, autotune_cache_kwargs, input_guard

# rows of the per-chunk Householder system solved per Neumann-doubling step
BS_SOLVE = 32
MAX_COMPACT_K = 64
MAX_COMPACT_V = 128


def is_compact_fwd_supported(num_householder: int, K: int, V: int, *tensors: torch.Tensor | None) -> bool:
    """Whether a call may use the token-aligned forward, which has no backward."""
    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors)
    return IS_TF32_SUPPORTED and num_householder >= 2 and K <= MAX_COMPACT_K and V <= MAX_COMPACT_V and not needs_grad


@triton.jit
def _compact_unit_setup(
    k,
    beta,
    g,
    u0,
    m_end,
    H: tl.constexpr,
    K: tl.constexpr,
    NH: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
):
    o_u = tl.arange(0, BS)
    o_k = tl.arange(0, BK)
    o_m = u0 + o_u
    m_m = o_m < m_end
    # [BS, BK]
    b_k = tl.load(k + o_m[:, None] * (H*K) + o_k[None, :], mask=m_m[:, None] & (o_k[None, :] < K), other=0.)
    # [BS]
    b_beta = tl.load(beta + o_m * H, mask=m_m, other=0.).to(tl.float32)
    if USE_G:
        # the gate of a token is applied before its first sub-step; padded rows reuse the last valid row's gate
        b_g0 = tl.load(g + (tl.maximum(u0 - 1, 0) // NH) * H, mask=u0 > 0, other=0.)
        b_gu = tl.load(g + (tl.minimum(o_m, m_end - 1) // NH) * H) - b_g0
        b_gl = tl.load(g + ((tl.minimum(u0 + BS, m_end) - 1) // NH) * H) - b_g0
    else:
        b_gu = tl.zeros([BS], dtype=tl.float32)
        b_gl = tl.sum(b_gu, 0)
    b_kf = b_k.to(tl.float32)
    b_kg = b_kf * exp2(b_gu)[:, None]
    b_kw = b_kf * (b_beta * exp2(b_gl - b_gu))[:, None]
    return o_m, m_m, b_k, b_beta, b_gu, b_kg, b_kw, b_gl


@triton.jit
def _compact_unit_inverse(b_k, b_beta, b_gu, BS: tl.constexpr, NST: tl.constexpr):
    o_u = tl.arange(0, BS)
    m_A = o_u[:, None] > o_u[None, :]
    # [BS, BS] strictly lower P, so (I - P)^{-1} = prod_s (I + P^{2^s}); later stages only refine the tail
    b_P = tl.dot(b_k, tl.trans(b_k))
    b_P = tl.where(m_A, -b_P * b_beta[None, :] * exp2(tl.where(m_A, b_gu[:, None] - b_gu[None, :], 0.)), 0.)
    b_T = tl.where(o_u[:, None] == o_u[None, :], 1., 0.) + b_P
    for i in tl.static_range(NST):
        if i == 0:
            b_P = tl.dot(b_P, b_P)
            b_T += tl.dot(b_P, b_T)
        else:
            b_P = tl.dot(b_P.to(tl.bfloat16), b_P.to(tl.bfloat16))
            b_T += tl.dot(b_P.to(tl.bfloat16), b_T.to(tl.bfloat16))
    return b_T


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [1, 2]
    ],
    key=['H', 'K', 'NH', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_fwd_kernel_compact_m(
    k,
    beta,
    g,
    A,
    mp,
    phi,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    NH: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    NST: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = i_b * tl.cdiv(T, BT) + i_t
        bos = i_b * T
    M: tl.constexpr = BT * NH
    m_end = (tl.minimum(BT, T - i_t * BT) * NH).to(tl.int32)

    k += ((bos + i_t * BT) * NH * H + i_h) * K
    beta += (bos + i_t * BT) * NH * H + i_h
    if USE_G:
        g += (bos + i_t * BT) * H + i_h
    A += (i_tg * H + i_h) * M * BS
    mp += (i_tg * H + i_h) * M * K
    phi += (i_tg * H + i_h) * K * K

    o_s = tl.arange(0, BS)
    o_k = tl.arange(0, BK)
    m_kk = (o_k[:, None] < K) & (o_k[None, :] < K)
    # [BK, BK] state coefficient of the chunk-start state
    b_phi = tl.where(m_kk & (o_k[:, None] == o_k[None, :]), 1., 0.)
    for u0 in range(0, m_end, BS):
        o_m, m_m, b_k, b_beta, b_gu, b_kg, b_kw, b_gl = _compact_unit_setup(k, beta, g, u0, m_end, H, K, NH, BS, BK, USE_G)
        # [BS, BS]
        b_T = _compact_unit_inverse(b_k, b_beta, b_gu, BS, NST)
        tl.store(A + o_m[:, None] * BS + o_s[None, :], b_T, mask=m_m[:, None])
        b_x = tl.dot(b_T, -tl.dot(b_kg, b_phi))
        p_mp = mp + o_m[:, None] * K + o_k[None, :]
        tl.store(p_mp, (-b_x * b_beta[:, None]).to(p_mp.dtype.element_ty), mask=m_m[:, None] & (o_k[None, :] < K))
        b_phi = exp2(b_gl) * b_phi + tl.dot(tl.trans(b_kw), b_x)
    tl.store(phi + o_k[:, None] * K + o_k[None, :], b_phi, mask=m_kk)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [32, 64, 128]
        for num_warps in [2, 4]
        for num_stages in [1, 2]
    ],
    key=['H', 'K', 'V', 'NH', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_fwd_kernel_compact_u(
    k,
    v,
    beta,
    g,
    A,
    up,
    bc,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NH: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t = unflatten_program_id(tl.cdiv(V, BV))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = i_b * tl.cdiv(T, BT) + i_t
        bos = i_b * T
    M: tl.constexpr = BT * NH
    m_end = (tl.minimum(BT, T - i_t * BT) * NH).to(tl.int32)

    k += ((bos + i_t * BT) * NH * H + i_h) * K
    v += ((bos + i_t * BT) * NH * H + i_h) * V
    beta += (bos + i_t * BT) * NH * H + i_h
    if USE_G:
        g += (bos + i_t * BT) * H + i_h
    A += (i_tg * H + i_h) * M * BS
    up += (i_tg * H + i_h) * M * V
    bc += (i_tg * H + i_h) * K * V

    o_s = tl.arange(0, BS)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    # [BK, BV] additive part of the running state
    b_s = tl.zeros([BK, BV], dtype=tl.float32)
    for u0 in range(0, m_end, BS):
        o_m, m_m, b_k, b_beta, b_gu, b_kg, b_kw, b_gl = _compact_unit_setup(k, beta, g, u0, m_end, H, K, NH, BS, BK, USE_G)
        m_mv = m_m[:, None] & (o_v[None, :] < V)
        b_v = tl.load(v + o_m[:, None] * (H*V) + o_v[None, :], mask=m_mv, other=0.).to(tl.float32)
        b_T = tl.load(A + o_m[:, None] * BS + o_s[None, :], mask=m_m[:, None], other=0.)
        b_x = tl.dot(b_T, b_v - tl.dot(b_kg, b_s))
        tl.store(up + o_m[:, None] * V + o_v[None, :], (b_x * b_beta[:, None]).to(up.dtype.element_ty), mask=m_mv)
        b_s = exp2(b_gl) * b_s + tl.dot(tl.trans(b_kw), b_x)
    m_s = (o_k[:, None] < K) & (o_v[None, :] < V)
    tl.store(bc + o_k[:, None] * V + o_v[None, :], b_s, mask=m_s)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32]
        for num_warps in [2, 4]
        for num_stages in [2, 3]
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_fwd_kernel_compact_h(
    phi,
    bc,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = unflatten_program_id(tl.cdiv(V, BV))
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        boh = i_n * tl.cdiv(T, BT)
    NT = tl.cdiv(T, BT)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_kk = (o_k[:, None] < K) & (o_k[None, :] < K)
    m_h = (o_k[:, None] < K) & (o_v[None, :] < V)
    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :], mask=m_h, other=0.).to(tl.float32)
    p_h = h + (boh * H + i_h) * K*V + o_k[:, None] * V + o_v[None, :]
    p_bc = bc + (boh * H + i_h) * K*V + o_k[:, None] * V + o_v[None, :]
    p_phi = phi + (boh * H + i_h) * K*K + o_k[:, None] * K + o_k[None, :]
    for _ in range(NT):
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=m_h)
        b_phi = tl.load(p_phi, mask=m_kk, other=0.)
        b_bc = tl.load(p_bc, mask=m_h, other=0.)
        b_h = tl.dot(b_phi, b_h) + b_bc
        p_h += H*K*V
        p_bc += H*K*V
        p_phi += H*K*K
    if STORE_FINAL_STATE:
        tl.store(ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :], b_h.to(ht.dtype.element_ty), mask=m_h)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV, 'BM': BM}, num_warps=num_warps, num_stages=2)
        for BV in [32, 64, 128]
        for BM in [32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=['H', 'K', 'V', 'NH', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_fwd_kernel_compact_o(
    q,
    k,
    up,
    mp,
    g,
    h,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NH: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BM: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t = unflatten_program_id(tl.cdiv(V, BV))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = i_b * tl.cdiv(T, BT) + i_t
        bos = i_b * T
    M: tl.constexpr = BT * NH
    t_end = tl.minimum(BT, T - i_t * BT).to(tl.int32)
    m_end = t_end * NH

    q += ((bos + i_t * BT) * H + i_h) * K
    k += ((bos + i_t * BT) * NH * H + i_h) * K
    o += ((bos + i_t * BT) * H + i_h) * V
    if USE_G:
        g += (bos + i_t * BT) * H + i_h
    up += (i_tg * H + i_h) * M * V
    mp += (i_tg * H + i_h) * M * K
    h += (i_tg * H + i_h) * K * V

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < t_end
    # [BT, BK]
    b_q = tl.load(q + o_t[:, None] * (H*K) + o_k[None, :], mask=m_t[:, None] & (o_k[None, :] < K), other=0.)
    # [BK, BV] chunk-start state
    b_h = tl.load(h + o_k[:, None] * V + o_v[None, :], mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0.)
    # [BT, BV]
    b_o = tl.dot(b_q, b_h)
    if USE_G:
        b_g = tl.load(g + o_t * H, mask=m_t, other=0.).to(tl.float32)
        b_o = b_o * exp2(b_g)[:, None]
    for i_m in range(0, m_end, BM):
        o_m = i_m + tl.arange(0, BM)
        m_m = o_m < m_end
        m_mk = m_m[:, None] & (o_k[None, :] < K)
        b_mp = tl.load(mp + o_m[:, None] * K + o_k[None, :], mask=m_mk, other=0.)
        b_up = tl.load(up + o_m[:, None] * V + o_v[None, :], mask=m_m[:, None] & (o_v[None, :] < V), other=0.)
        # [BM, BV] pseudo-values corrected for the chunk-start state
        b_u = b_up - tl.dot(b_mp, b_h.to(tl.float32))
        b_k = tl.load(k + o_m[:, None] * (H*K) + o_k[None, :], mask=m_mk, other=0.)
        # [BT, BM]
        b_A = tl.dot(b_q, tl.trans(b_k))
        o_tm = o_m // NH
        m_A = (o_tm[None, :] <= o_t[:, None]) & m_m[None, :] & m_t[:, None]
        if USE_G:
            b_gm = tl.load(g + o_tm * H, mask=m_m, other=0.).to(tl.float32)
            b_A = b_A * exp2(tl.where(m_A, b_g[:, None] - b_gm[None, :], 0.))
        b_A = tl.where(m_A, b_A, 0.)
        b_o += tl.dot(b_A.to(b_k.dtype), b_u.to(b_k.dtype))
    p_o = o + o_t[:, None] * (H*V) + o_v[None, :]
    tl.store(p_o, (b_o * scale).to(p_o.dtype.element_ty), mask=m_t[:, None] & (o_v[None, :] < V))


@input_guard
def chunk_gated_delta_product_fwd_compact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    scale: float,
    num_householder: int,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward-only gated DeltaProduct whose sequential pass runs over `T / chunk_size` token chunks.

    Each chunk of `chunk_size` tokens (`chunk_size * num_householder` Householder sub-steps) is reduced in parallel
    to an affine map `S -> Phi_c S + B_c` of the chunk-start state, so the only sequential kernel applies one
    `[K, K] x [K, V]` product per chunk regardless of `num_householder`.
    """
    B, T, H, K, V = *q.shape, v.shape[-1]
    NH, BT = num_householder, chunk_size
    M = BT * NH
    assert M % BS_SOLVE == 0, f"chunk_size * num_householder must be divisible by {BS_SOLVE}"
    assert K <= MAX_COMPACT_K, f"the compact forward supports K <= {MAX_COMPACT_K}, got {K}"
    assert V <= MAX_COMPACT_V, f"the compact forward supports V <= {MAX_COMPACT_V}, got {V}"

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    NC = B * NT if cu_seqlens is None else NT
    if g is not None:
        g = chunk_local_cumsum(
            g,
            chunk_size=BT,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
            chunk_indices=chunk_indices,
        )

    BK = max(16, triton.next_power_of_2(K))
    NST = BS_SOLVE.bit_length() - 2
    A = k.new_empty(NC, H, M, BS_SOLVE, dtype=torch.float32)
    # fp32 rows: the output kernel forms `U' - M' S_0`, which cancels to the much smaller pseudo-values
    mp = k.new_empty(NC, H, M, K, dtype=torch.float32)
    up = v.new_empty(NC, H, M, V, dtype=torch.float32)
    phi = k.new_empty(NC, H, K, K, dtype=torch.float32)
    bc = v.new_empty(NC, H, K, V, dtype=torch.float32)
    h = k.new_empty(NC, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    o = v.new_empty(B, T, H, V)

    chunk_gated_delta_product_fwd_kernel_compact_m[(NT, B * H)](
        k=k,
        beta=beta,
        g=g,
        A=A,
        mp=mp,
        phi=phi,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        NH=NH,
        BT=BT,
        BS=BS_SOLVE,
        NST=NST,
        BK=BK,
    )
    def grid(meta): return (triton.cdiv(V, meta['BV']) * NT, B * H)
    chunk_gated_delta_product_fwd_kernel_compact_u[grid](
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        up=up,
        bc=bc,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        NH=NH,
        BT=BT,
        BS=BS_SOLVE,
        BK=BK,
    )
    def grid(meta): return (triton.cdiv(V, meta['BV']) * N * H,)
    chunk_gated_delta_product_fwd_kernel_compact_h[grid](
        phi=phi,
        bc=bc,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
    )
    def grid(meta): return (triton.cdiv(V, meta['BV']) * NT, B * H)
    chunk_gated_delta_product_fwd_kernel_compact_o[grid](
        q=q,
        k=k,
        up=up,
        mp=mp,
        g=g,
        h=h,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        NH=NH,
        BT=BT,
        BK=BK,
    )
    return o, final_state
