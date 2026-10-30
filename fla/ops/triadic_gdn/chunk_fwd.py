# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets, solve_tril
from fla.ops.utils.op import exp
from fla.utils import IS_TF32_SUPPORTED

# precision of the float32 products that build R, Rq and their gradients, as in the fused GDN kkt and solve kernel
DOT_PRECISION = tl.constexpr('tf32' if IS_TF32_SUPPORTED else 'ieee')


@triton.jit
def triadic_intra_r(
    k2,
    q2,
    g,
    bos,
    i_t,
    i_h,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BE: tl.constexpr,
):
    """
    `R[i, j] = sum_e k2[i, e] * k2[j, e] * D_e[i, j]` and `Rq[i, j] = sum_e q2[i, e] * k2[j, e] * D_e[i, j]` for `j <= i`,
    with `D_e[i, j] = exp(g[i, e] - g[j, e])`.

    Between sub-chunks of `BC` rows, `D_e` factors exactly through the first row `r_c` of the later sub-chunk `c`:
    `D_e[i, j] = exp(g[i, e] - g[r_c, e]) * exp(g[r_c, e] - g[j, e])`, where neither factor exceeds 1.
    These blocks come from one product over columns `(c, e)`; diagonal sub-chunks are formed from gate differences.
    """
    o_i = tl.arange(0, BT)
    o_c = tl.arange(0, BC)
    o_n = tl.arange(0, BT // BC * BE)
    o_nc, o_ne = o_n // BE, o_n % BE
    m_t = i_t * BT + o_i < T
    m_tn = m_t[:, None] & (o_ne < E)[None, :]
    m_r = (i_t * BT + o_nc * BC < T) & (o_ne < E)
    p_n = (bos + i_t * BT + o_i[:, None]) * H*E + i_h * E + o_ne[None, :]
    # [BT, BT // BC * BE]
    b_g = tl.load(g + p_n, mask=m_tn, other=0).to(tl.float32)
    b_k2 = tl.load(k2 + p_n, mask=m_tn, other=0).to(tl.float32)
    b_q2 = tl.load(q2 + p_n, mask=m_tn, other=0).to(tl.float32)
    b_gr = tl.load(g + (bos + i_t * BT + o_nc * BC) * H*E + i_h * E + o_ne, mask=m_r, other=0).to(tl.float32)
    # rows i of sub-chunk c, and rows j before it
    m_a = m_tn & (o_nc[None, :] == (o_i // BC)[:, None])
    m_b = m_tn & m_r[None, :] & (o_nc[None, :] > (o_i // BC)[:, None])
    b_a = tl.where(m_a, exp(tl.where(m_a, b_g - b_gr[None, :], 0)), 0)
    b_b = tl.where(m_b, b_k2 * exp(tl.where(m_b, b_gr[None, :] - b_g, 0)), 0)
    b_R = tl.dot(b_k2 * b_a, tl.trans(b_b), input_precision=DOT_PRECISION)
    b_Rq = tl.dot(b_q2 * b_a, tl.trans(b_b), input_precision=DOT_PRECISION)

    # diagonal sub-chunks as [BT, BC] tiles, whose column c holds j = BC * (i // BC) + c
    m_d = (o_c[None, :] <= (o_i % BC)[:, None]) & m_t[:, None]
    p_i = (bos + i_t * BT + o_i) * H*E + i_h * E
    p_j = (bos + i_t * BT + (o_i // BC * BC)[:, None] + o_c[None, :]) * H*E + i_h * E
    b_Rd = tl.zeros([BT, BC], dtype=tl.float32)
    b_Rqd = tl.zeros([BT, BC], dtype=tl.float32)
    for i_e in range(E):
        b_gi = tl.load(g + p_i + i_e, mask=m_t, other=0).to(tl.float32)
        b_k2i = tl.load(k2 + p_i + i_e, mask=m_t, other=0).to(tl.float32)
        b_q2i = tl.load(q2 + p_i + i_e, mask=m_t, other=0).to(tl.float32)
        b_gj = tl.load(g + p_j + i_e, mask=m_d, other=0).to(tl.float32)
        b_k2j = tl.load(k2 + p_j + i_e, mask=m_d, other=0).to(tl.float32)
        b_d = tl.where(m_d, exp(tl.where(m_d, b_gi[:, None] - b_gj, 0)), 0)
        b_Rd += b_k2i[:, None] * b_k2j * b_d
        b_Rqd += b_q2i[:, None] * b_k2j * b_d
    # scatter the diagonal tiles into place
    b_place = tl.where(o_i[None, :] % BC == o_c[:, None], 1., 0.)
    m_blk = (o_i // BC)[:, None] == (o_i // BC)[None, :]
    b_R += tl.where(m_blk, tl.dot(b_Rd, b_place, input_precision=DOT_PRECISION), 0)
    b_Rq += tl.where(m_blk, tl.dot(b_Rqd, b_place, input_precision=DOT_PRECISION), 0)
    return b_R, b_Rq


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_triadic_gdn_fwd_kernel_intra(
    q,
    k,
    k2,
    q2,
    g,
    beta,
    A,
    Aqk,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T
    p_A = (bos + o_t[:, None]) * H*BT + i_h * BT + o_i[None, :]

    b_R, b_Rq = triadic_intra_r(k2, q2, g, bos, i_t, i_h, T, H, E, BT, BC, BE)

    # A = beta * (k @ k^T) * R, strictly lower
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_k = k + (bos + o_t[:, None]) * H*K + i_h * K + o_k[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & (o_k < K)[None, :], other=0)
        b_A += tl.dot(b_k, tl.trans(b_k))
    b_beta = tl.load(beta + (bos + o_t) * H + i_h, mask=m_t, other=0).to(tl.float32)
    b_A = tl.where(o_i[:, None] > o_i[None, :], b_beta[:, None] * b_A * b_R, 0)
    tl.store(A + p_A, b_A.to(A.dtype.element_ty), mask=m_t[:, None])

    # Aqk = scale * (q @ k^T) * Rq, lower
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_q = q + (bos + o_t[:, None]) * H*K + i_h * K + o_k[None, :]
        p_k = k + (bos + o_t[:, None]) * H*K + i_h * K + o_k[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None] & (o_k < K)[None, :], other=0)
        b_k = tl.load(p_k, mask=m_t[:, None] & (o_k < K)[None, :], other=0)
        b_A += tl.dot(b_q, tl.trans(b_k))
    b_A = b_A * b_Rq * scale
    tl.store(Aqk + p_A, b_A.to(Aqk.dtype.element_ty), mask=m_t[:, None])


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_STATES': lambda args: args['h'] is not None,
    'STORE_OUTPUT': lambda args: args['o'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_triadic_gdn_fwd_kernel_h(
    q,
    k,
    v,
    k2,
    q2,
    g,
    beta,
    Ai,
    Aqk,
    o,
    v_new,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_STATES: tl.constexpr,
    STORE_OUTPUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Runs the chunk recurrence for one value block of one head, keeping the value block of all `E` slices in registers.

    Writes the output when `o` is given, and the chunk-start states `h` with the new values `v_new` when `h` is given.
    Every tile is transposed so that the narrow value block is the M dimension of each `tl.dot`.
    """
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_v[:, None] & m_k[None, :]
    # [BV, BK] tile of a [K, V] state
    p_h = o_k[None, :] * V + o_v[:, None]

    # one transposed [BV, BK] state tile per slice
    b_hs = ()
    for i_e in tl.static_range(E):
        if USE_INITIAL_STATE:
            b_s = tl.load(h0 + (i_nh * E + i_e) * K*V + p_h, mask=m_h, other=0).to(tl.float32)
        else:
            b_s = tl.zeros([BV, BK], dtype=tl.float32)
        b_hs = b_hs + (b_s,)

    for i_t in range(NT):
        o_t = i_t * BT + o_i
        m_t = o_t < T
        m_vt = m_v[:, None] & m_t[None, :]
        p_qk = (bos + o_t[:, None]) * H*K + i_h * K + o_k[None, :]
        p_kt = (bos + o_t[None, :]) * H*K + i_h * K + o_k[:, None]
        # [BV, BT] tile of a [T, V] tensor
        p_vt = (bos + o_t[None, :]) * H*V + i_h * V + o_v[:, None]
        # [BT, BT] tile of a transposed [T, BT] matrix
        p_At = (bos + o_t[None, :]) * H*BT + i_h * BT + o_i[:, None]
        p_e = (bos + o_t) * H*E + i_h * E
        p_last = (bos + tl.minimum(i_t * BT + BT, T) - 1) * H*E + i_h * E
        # [BT, BK] and [BK, BT]
        b_k = tl.load(k + p_qk, mask=m_t[:, None] & m_k[None, :], other=0)
        b_kt = tl.load(k + p_kt, mask=m_k[:, None] & m_t[None, :], other=0)
        b_beta = tl.load(beta + (bos + o_t) * H + i_h, mask=m_t, other=0).to(tl.float32)
        # [BV, BT]
        b_u = b_beta[None, :] * tl.load(v + p_vt, mask=m_vt, other=0).to(tl.float32)
        if STORE_OUTPUT:
            b_qt = tl.load(q + p_kt, mask=m_k[:, None] & m_t[None, :], other=0)
            b_o = tl.zeros([BV, BT], dtype=tl.float32)

        # u = (I + A)^-1 (beta * v - sum_e (beta * k2_e * exp(g_e)) * (k @ S_e))
        # o = scale * sum_e (q2_e * exp(g_e)) * (q @ S_e) + Aqk @ u
        for i_e in tl.static_range(E):
            b_h = b_hs[i_e].to(b_k.dtype)
            if SAVE_STATES:
                tl.store(h + (((boh + i_t) * H + i_h) * E + i_e) * K*V + p_h, b_h.to(h.dtype.element_ty), mask=m_h)
            b_k2 = tl.load(k2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
            b_g = tl.load(g + p_e + i_e, mask=m_t, other=0).to(tl.float32)
            b_u -= (b_beta * b_k2 * exp(b_g))[None, :] * tl.dot(b_h, b_kt)
            if STORE_OUTPUT:
                b_q2 = tl.load(q2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
                b_o += (b_q2 * exp(b_g))[None, :] * tl.dot(b_h, b_qt)
        b_u = tl.dot(b_u.to(b_k.dtype), tl.load(Ai + p_At, mask=m_t[None, :], other=0))
        if SAVE_STATES:
            tl.store(v_new + p_vt, b_u.to(v_new.dtype.element_ty), mask=m_vt)
        if STORE_OUTPUT:
            b_o = b_o * scale + tl.dot(b_u.to(b_k.dtype), tl.load(Aqk + p_At, mask=m_t[None, :], other=0).to(b_k.dtype))
            tl.store(o + p_vt, b_o.to(o.dtype.element_ty), mask=m_vt)

        # S_e <- exp(g_last_e) * S_e + k^T @ (k2_e * exp(g_last_e - g_e) * u)
        b_new = ()
        for i_e in tl.static_range(E):
            b_k2 = tl.load(k2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
            b_g = tl.load(g + p_e + i_e, mask=m_t, other=0).to(tl.float32)
            b_g_last = tl.load(g + p_last + i_e).to(tl.float32)
            b_w = ((b_k2 * exp(b_g_last - b_g))[None, :] * b_u).to(b_k.dtype)
            b_new = b_new + (b_hs[i_e] * exp(b_g_last) + tl.dot(b_w, b_k),)
        b_hs = b_new

    if STORE_FINAL_STATE:
        for i_e in tl.static_range(E):
            tl.store(ht + (i_nh * E + i_e) * K*V + p_h, b_hs[i_e].to(ht.dtype.element_ty), mask=m_h)


def chunk_triadic_gdn_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, E = *k.shape, k2.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = k.new_empty(B, T, H, BT, dtype=torch.float32)
    # consumed only as a bf16/fp16 tl.dot operand
    Aqk = k.new_empty(B, T, H, BT)
    chunk_triadic_gdn_fwd_kernel_intra[(NT, B * H)](
        q=q,
        k=k,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        A=A,
        Aqk=Aqk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        E=E,
        BT=BT,
        BC=16,
        BK=min(64, triton.next_power_of_2(K)),
        BE=max(16, triton.next_power_of_2(E)),
        num_warps=8,
        num_stages=1,
    )
    Ai = solve_tril(A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype)
    return Ai, Aqk


def get_state_block_size(K: int, E: int) -> tuple[int, int]:
    """Value block size and warp count of the state kernels, which keep `E` [BV, K] tiles in registers."""
    # tuned on H100 at K=V=128; a value block of 32 wins up to E=8 despite some register spilling
    BV = 32 if E * triton.next_power_of_2(K) <= 1024 else 16
    return BV, 4 if E <= 2 else 8


def chunk_triadic_gdn_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Ai: torch.Tensor,
    q: torch.Tensor | None = None,
    q2: torch.Tensor | None = None,
    Aqk: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    save_states: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    """
    Runs the chunk recurrence. Computes the output when `q` is given,
    and returns the chunk-start states `h` with the new values `v_new` when `save_states` is set.
    """
    B, T, H, K, V, E = *k.shape, v.shape[-1], k2.shape[-1]
    BT = chunk_size
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    BV, num_warps = get_state_block_size(K, E)

    o = torch.empty_like(v) if q is not None else None
    h = k.new_empty(B, NT, H, E, K, V) if save_states else None
    v_new = torch.empty_like(v) if save_states else None
    final_state = k.new_empty(N, H, E, K, V, dtype=torch.float32) if output_final_state else None
    chunk_triadic_gdn_fwd_kernel_h[(triton.cdiv(V, BV), N * H)](
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        Ai=Ai,
        Aqk=Aqk,
        o=o,
        v_new=v_new,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        E=E,
        BT=BT,
        BK=triton.next_power_of_2(K),
        BV=BV,
        num_warps=num_warps,
        num_stages=2,
    )
    return o, h, v_new, final_state


def chunk_triadic_gdn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    save_states: bool = False,
):
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    Ai, Aqk = chunk_triadic_gdn_fwd_intra(
        q=q,
        k=k,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    o, h, v_new, final_state = chunk_triadic_gdn_fwd_h(
        k=k,
        v=v,
        k2=k2,
        g=g,
        beta=beta,
        Ai=Ai,
        q=q,
        q2=q2,
        Aqk=Aqk,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        save_states=save_states,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    return g, Ai, Aqk, h, v_new, o, final_state
