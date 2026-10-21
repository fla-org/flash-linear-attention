# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.cp.chunk_merge import CP_DOT_PRECISION, _prune_cp_configs
from fla.ops.cp.context import _check_cp_device
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_BG': lambda args: args['bg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'USE_G', 'USE_GK', 'USE_BG', 'IS_VARLEN', 'MULTI_SEQS'],
    prune_configs_by={'early_config_prune': _prune_cp_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_cp_fwd_kernel_h(
    k,
    v,
    w,
    g,
    gk,
    bg,
    u,
    hm,
    cu_seqlens,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MULTI_SEQS: tl.constexpr,
):
    """Summarize a local sequence as h_out = M @ h_in + h_local.

    Each value head writes [h_local | M] with shape [K, V+K].
    The two column groups compute the zero-initial-state response and the incoming-state transform independently.
    """
    i_col, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    if MULTI_SEQS:
        i_n = tl.program_id(2).to(tl.int64)
        hm += i_n * HV * K * (K + V) + i_h * K * (K + V)
    else:
        i_n = 0
        hm += i_h * K * (K + V)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    # h_local and M have separately padded column grids, including when V is not a block multiple
    is_h_part = i_col * BLOCK_SIZE < V
    # for DPLR, w/bg use H heads; GDN/KDA w uses HV heads
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
        bg += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        g += (bos * HV + i_h).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    stride_k = H * K
    stride_w = H * K if USE_BG else HV * K

    if is_h_part:
        v += ((bos * HV + i_h) * V).to(tl.int64)
        if USE_BG:
            # for DPLR, u and v are distinct; GDN/KDA aliases u to v in the wrapper
            u += ((bos * HV + i_h) * V).to(tl.int64)
        stride_v = HV * V
        i_v = i_col

        # starting from zero isolates the local contribution from the unknown incoming rank state
        b_h1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        o_vb = i_v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m_vb = o_vb < V
        o_k1 = tl.arange(0, 64)
        m_k1 = o_k1 < K
        o_k2 = 64 + o_k1
        m_k2 = o_k2 < K
        o_k3 = 128 + o_k1
        m_k3 = o_k3 < K
        o_k4 = 192 + o_k1
        m_k4 = o_k4 < K

        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            p_w = w + o_t[:, None] * stride_w + o_k1[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
            b_v_decay = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = w + o_t[:, None] * stride_w + o_k2[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
                b_v_decay = tl.dot(b_w, b_h2.to(b_w.dtype), b_v_decay)
            if K > 128:
                p_w = w + o_t[:, None] * stride_w + o_k3[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
                b_v_decay = tl.dot(b_w, b_h3.to(b_w.dtype), b_v_decay)
            if K > 192:
                p_w = w + o_t[:, None] * stride_w + o_k4[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
                b_v_decay = tl.dot(b_w, b_h4.to(b_w.dtype), b_v_decay)

            p_v = v + o_t[:, None] * stride_v + o_vb[None, :]
            if USE_BG:
                # for DPLR, v2 = w @ h + u; the update also needs the original v
                b_v_orig = tl.load(p_v, mask=m_t[:, None] & m_vb[None, :], other=0.0)
                p_u = u + o_t[:, None] * stride_v + o_vb[None, :]
                b_v = b_v_decay + tl.load(p_u, mask=m_t[:, None] & m_vb[None, :], other=0.0)
            else:
                # the delta-rule residual subtracts the prediction from the current state
                b_v = tl.load(p_v, mask=m_t[:, None] & m_vb[None, :], other=0.0) - b_v_decay

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = g + o_t * HV
                b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
                b_h1 *= b_g_last
                if K > 64:
                    b_h2 *= b_g_last
                if K > 128:
                    b_h3 *= b_g_last
                if K > 192:
                    b_h4 *= b_g_last

            # vector gates decay each key row independently; scalar gates share one decay across rows
            if USE_GK:
                o_k1 = tl.arange(0, 64)
                p_gk_last = gk + last_idx * HV * K
                b_gk_last1 = tl.load(p_gk_last + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
                b_h1 *= exp2(b_gk_last1)[:, None]
                if K > 64:
                    o_k2 = 64 + o_k1
                    b_gk_last2 = tl.load(p_gk_last + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                    b_h2 *= exp2(b_gk_last2)[:, None]
                if K > 128:
                    o_k3 = 128 + o_k1
                    b_gk_last3 = tl.load(p_gk_last + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                    b_h3 *= exp2(b_gk_last3)[:, None]
                if K > 192:
                    o_k4 = 192 + o_k1
                    b_gk_last4 = tl.load(p_gk_last + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                    b_h4 *= exp2(b_gk_last4)[:, None]
            b_v = b_v.to(k.dtype.element_ty)

            p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            if USE_BG:
                # for DPLR, h += k.T @ v + bg.T @ v2; GDN/KDA below use h += k.T @ v_new
                p_bg = bg + o_k1[:, None] + o_t[None, :] * stride_k
                b_bg = tl.load(p_bg, mask=m_k1[:, None] & m_t[None, :], other=0.0)
                b_h1 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 64:
                    p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                    p_bg = bg + o_k2[:, None] + o_t[None, :] * stride_k
                    b_bg = tl.load(p_bg, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                    b_h2 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 128:
                    p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                    p_bg = bg + o_k3[:, None] + o_t[None, :] * stride_k
                    b_bg = tl.load(p_bg, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                    b_h3 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 192:
                    p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                    p_bg = bg + o_k4[:, None] + o_t[None, :] * stride_k
                    b_bg = tl.load(p_bg, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                    b_h4 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
            else:
                b_h1 = tl.dot(b_k, b_v, b_h1)
                if K > 64:
                    p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                    b_h2 = tl.dot(b_k, b_v, b_h2)
                if K > 128:
                    p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                    b_h3 = tl.dot(b_k, b_v, b_h3)
                if K > 192:
                    p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                    b_h4 = tl.dot(b_k, b_v, b_h4)

        stride_hm_kv = K + V
        p_h1 = hm + o_k1[:, None] * stride_hm_kv + o_vb[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_k1[:, None] & m_vb[None, :])
        if K > 64:
            p_h2 = hm + o_k2[:, None] * stride_hm_kv + o_vb[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_k2[:, None] & m_vb[None, :])
        if K > 128:
            p_h3 = hm + o_k3[:, None] * stride_hm_kv + o_vb[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_k3[:, None] & m_vb[None, :])
        if K > 192:
            p_h4 = hm + o_k4[:, None] * stride_hm_kv + o_vb[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_k4[:, None] & m_vb[None, :])
    else:
        # M starts after the padded h_local grid, not after an unpadded V / BLOCK_SIZE
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        # identity represents an unchanged incoming state before any local chunk is applied
        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            if USE_BG:
                # use bg for the DPLR transition matrix
                p_k = bg + o_t[:, None] * stride_k + row[None, :]
            else:
                p_k = k + o_t[:, None] * stride_k + row[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (row < K)[None, :], other=0.0)
            p_w = w + o_t[:, None] * stride_w + row[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (row < K)[None, :], other=0.0)

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = g + o_t * HV
                b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
                b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_gk_last = tl.load(gk + last_idx * HV * K + row, mask=(row < K), other=0.).to(tl.float32)
                b_gk_last = exp2(b_gk_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)

            # each chunk left-composes its transform: diag + bg.T @ w for DPLR, diag - k.T @ w for GDN/KDA
            if USE_BG:
                b_kw = tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)
                b_m_i = b_diag + b_kw
            else:
                b_kw = tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)
                b_m_i = b_diag - b_kw
            # keep the composed transform in fp32; repeated bf16 casts accumulate error along the chain
            b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32), input_precision=CP_DOT_PRECISION)

        stride_hm_kv = K + V
        p_m = hm + V + row[:, None] * stride_hm_kv + col[None, :]
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), mask=(row < K)[:, None] & (col < K)[None, :])


def chunk_cp_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    hm: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    multi_seqs: bool = False,
) -> None:
    """Write forward affine summaries into hm for one boundary sequence or multiple subsequences."""
    _check_cp_device(k.device)
    _, T, H, K = k.shape
    HV, V = u.shape[2:]
    BK = triton.next_power_of_2(K)
    BLOCK_SIZE = 32 if K <= 64 else 64
    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)
    if multi_seqs:
        grid += (hm.shape[0],)
    # DPLR uses original v and WY-processed u; delta-rule paths use u for both.
    chunk_cp_fwd_kernel_h[grid](
        k=k,
        v=u if v is None else v,
        w=w,
        g=g,
        gk=gk,
        bg=bg,
        u=u,
        hm=hm,
        cu_seqlens=cu_seqlens,
        T=0 if multi_seqs else T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BLOCK_SIZE=BLOCK_SIZE,
        BK1=BK,
        MULTI_SEQS=multi_seqs,
    )
