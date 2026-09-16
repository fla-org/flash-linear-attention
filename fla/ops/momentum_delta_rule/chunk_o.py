# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs, check_shared_mem

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BKV, 'BV': BKV}, num_warps=warps, num_stages=stages)
        for BKV in [32, 64]
        for warps in [2, 4]
        for stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_momentum_delta_rule_fwd_kernel_o(
    q,
    k,
    v,
    gamma_mask_q,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):

    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_q = tl.load(q + o_t[:, None] * (H*K) + o_k[None, :],
                      mask=m_t[:, None] & m_k[None, :], other=0.0)    # [BT, BK]
        b_k = tl.load(k + o_k[:, None] + o_t[None, :] * (H*K),
                      mask=m_k[:, None] & m_t[None, :], other=0.0)    # [BK, BT]

        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k, allow_tf32=False)

    base = (bos * H + i_h) * BT
    b_gamma_mask_q = tl.load(gamma_mask_q + base + o_t[:, None] * (H*BT) + tl.arange(0, BT)[None, :],
                             mask=m_t[:, None], other=0.0)

    # belowing ops could be redundant due to b_gamma_mask_q is already masked with tril
    b_A = b_A * b_gamma_mask_q

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        mask_v = m_t[:, None] & m_v[None, :]
        b_v = tl.load(v + o_t[:, None] * (H*V) + o_v[None, :], mask=mask_v, other=0.0)
        b_o = tl.dot(b_A.to(b_v.dtype), b_v, allow_tf32=False) * scale
        tl.store(o + o_t[:, None] * (H*V) + o_v[None, :],
                 b_o.to(o.dtype.element_ty), mask=mask_v)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4,]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV',
         ],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_momentum_delta_rule_bwd_kernel_dqk(
    q,
    k,
    v,
    dv,
    s,
    m,
    ds,
    dm,
    log_mu_cum,
    log_a_cum,
    bt,
    gamma_mask_q,
    do,
    dq,
    dk,
    d_log_mu_cum,
    d_log_a_cum,
    d_bt,
    d_decay_s,
    d_Attn,
    y,
    z,
    dy,
    dz,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V

    s += (i_tg * H + i_h).to(tl.int64) * K*V
    m += (i_tg * H + i_h).to(tl.int64) * K*V
    ds += (i_tg * H + i_h).to(tl.int64) * K*V
    dm += (i_tg * H + i_h).to(tl.int64) * K*V

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K

    d_log_a_cum += (bos * H + i_h) * K
    d_log_mu_cum += (bos * H + i_h) * K
    d_bt += (bos * H + i_h) * K
    d_decay_s += (bos * H + i_h) * K

    gamma_mask_q += (bos * H + i_h) * BT
    d_Attn += (bos * H + i_h) * BT

    log_a_cum += bos * H + i_h
    log_mu_cum += bos * H + i_h
    bt += bos * H + i_h

    y += (bos * H + i_h) * K
    z += (bos * H + i_h) * K
    dy += (bos * H + i_h) * K
    dz += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V

    last_idx = min((i_t + 1) * BT, T) - 1

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    b_log_mcum_last = tl.load(log_mu_cum + last_idx * H)
    b_log_acum_last = tl.load(log_a_cum + last_idx * H)

    b_gamma_last_row = tl.load(gamma_mask_q + last_idx * (BT * H) + tl.arange(0, BT))            # [BT]
    b_log_acum = tl.load(log_a_cum + o_t * H, mask=m_t, other=0.0)                         # [BT]
    b_log_mcum = tl.load(log_mu_cum + o_t * H, mask=m_t, other=0.0)                        # [BT]
    b_bt = tl.load(bt + o_t * H, mask=m_t, other=0.0)                                # [BT]
    b_q = tl.load(q + o_t[:, None] * (H*K) + o_k[None, :],
                  mask=m_t[:, None] & m_k[None, :], other=0.0) * scale             # [BT, BK]
    b_k = tl.load(k + o_t[:, None] * (H*K) + o_k[None, :],
                  mask=m_t[:, None] & m_k[None, :], other=0.0)                      # [BT, BK]

    b_decay_s = b_gamma_last_row                        # [BT]
    b_decay_m = tl.exp(b_log_mcum_last - b_log_mcum)    # [BT]

    b_a_cum_last = tl.exp(b_log_acum_last)              # [1]
    b_mu_cum_last = tl.exp(b_log_mcum_last)              # [1]
    b_a_cum = tl.exp(b_log_acum)                   # [BT]

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dAttn = tl.zeros([BT, BT], dtype=tl.float32)

    b_dy = tl.zeros([BT, BK], dtype=tl.float32)
    b_dz = tl.zeros([BT, BK], dtype=tl.float32)

    b_dlog_a_cum_last = tl.zeros([BK], dtype=tl.float32)
    b_dlog_mu_cum_last = tl.zeros([BK], dtype=tl.float32)
    b_dbt_last = tl.zeros([BK], dtype=tl.float32)

    b_dlog_a_cum = tl.zeros([BT, BK], dtype=tl.float32)
    b_dlog_mu_cum = tl.zeros([BT, BK], dtype=tl.float32)
    b_dbt = tl.zeros([BT, BK], dtype=tl.float32)

    b_d_decay_s = tl.zeros([BT, BK],     dtype=tl.float32)
    b_d_decay_m = tl.zeros([BT, BK],     dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        mask_v = m_t[:, None] & m_v[None, :]
        # [BV, BK]
        b_v = tl.load(v + o_t[:, None] * (H*V) + o_v[None, :], mask=mask_v, other=0.0)     # [BT, BV]
        b_do = tl.load(do + o_t[:, None] * (H*V) + o_v[None, :], mask=mask_v, other=0.0)   # [BT, BV]
        b_dv = tl.load(dv + o_t[:, None] * (H*V) + o_v[None, :], mask=mask_v, other=0.0)   # [BT, BV]

        b_s = tl.load(s + o_v[:, None] + o_k[None, :] * V,
                      mask=m_v[:, None] & m_k[None, :], other=0.0)                           # [BV, BK]
        b_m = tl.load(m + o_v[:, None] + o_k[None, :] * V,
                      mask=m_v[:, None] & m_k[None, :], other=0.0)                           # [BV, BK]
        b_ds = tl.load(ds + o_v[:, None] + o_k[None, :] * V,
                       mask=m_v[:, None] & m_k[None, :], other=0.0)                          # [BV, BK]
        b_dm = tl.load(dm + o_v[:, None] + o_k[None, :] * V,
                       mask=m_v[:, None] & m_k[None, :], other=0.0)                          # [BV, BK]

        b_do_dot_s = tl.dot(b_do, b_s.to(b_do.dtype), allow_tf32=False)       # [BT, BK]
        b_do_dot_m = tl.dot(b_do, b_m.to(b_do.dtype), allow_tf32=False)       # [BT, BK]
        b_v_dot_ds = tl.dot(b_v, b_ds.to(b_v.dtype), allow_tf32=False)        # [BT, BK]
        b_v_dot_dm = tl.dot(b_v, b_dm.to(b_v.dtype), allow_tf32=False)        # [BT, BK]
        b_dv_dot_s = tl.dot(b_dv, b_s.to(b_dv.dtype), allow_tf32=False)       # [BT, BK]
        b_dv_dot_m = tl.dot(b_dv, b_m.to(b_dv.dtype), allow_tf32=False)       # [BT, BK]
        """
        # dL/dAttn = dL/do * do/dAttn = dL/do * v
        # [BT, BV] @ [BV, BT] -> [BT, BT]
        """
        b_dAttn += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)

        """
        # dL/dq = dL/do * (do/dqSinter * dqSinter/dq) + dL/dattn * dattn/dq
        #       = dL/do * (S_pre * log_a_cum.exp().unsqueeze(-1) - M_pre * bt )
        #         + dL/dAttn * k * gamma_mask_q                                 # this part is add in out of loop
        # dL/dk = dL/ds * ds/dk + dL/dm * dm/dk + dL/dattn * dattn/dk
        #       = dL/ds * (decay_s * v) - dm * (decay_m * v)
        #         + dL/dattn * q * gamma_mask_q                                 # this part is add in out of loop
        """
        b_dq += b_do_dot_s * b_a_cum[:, None]  \
            - b_do_dot_m * b_bt[:, None]                 # [BT, BV] @ [BV, BK] -> [BT, BK]

        b_dk += b_v_dot_ds * b_decay_s[:, None]     \
            - b_v_dot_dm * b_decay_m[:, None]            # [BT, BV] @ [BV, BK] -> [BT, BK]
        """
        # dy = - dv * S_pre
        # dz =   dv * M_pre
        """
        b_dy += -b_dv_dot_s                                                         # [BT, BK]
        b_dz += b_dv_dot_m                                                          # [BT, BK]

        #################################################
        # below for computing the gradient of coefficient
        #################################################
        """
        # dL/d_bt_last = dL/ds * ds/dbt
        #                         = dL/ds * (-M_pre)                                    [ 1]
        # dL/d_bt      = dL/do * do/dbt
        #                        =  dL/do * (-M_pre) * q_i                           [BT, 1]
        """
        b_dbt_last += -tl.sum(b_ds * b_m, axis=0)       # [BV, BK] -> [BK]
        b_dbt += -b_do_dot_m * b_q                 # [BT, BK]

        """
        # dL/d_decay_s = dL/ds * ds/d_decay_s = dL/ds * k * v                           [BT]
        # dL/d_decay_m = dL/dm * dm/d_decay_m = dL/dm * k * v                           [BT]
        """
        b_d_decay_s += b_v_dot_ds * b_k   # [BT, BK]
        b_d_decay_m += -b_v_dot_dm * b_k   # [BT, BK]

        """
        # dL/da_cum = dL/do * do/d_qS_inter * d_qS_inter/d_bar_alpha_t_q * d_bar_alpha_t_q/da_cum
        #                         = dL/do * 1 * S_pre * q_i                             [BT]
        # dL/dm_cum = dL/d_decay_m * d_decay_m/dlog_m_cum
        #                         = - dL/d_decay_m * eta                                [BT]
        # [BT, BV] @ [BV, BK] -> [BT, BK] -> * [BT, BK] -> [BT,]
        """
        b_dlog_a_cum += (b_do_dot_s * b_q) * b_a_cum[:, None]  # [BT, BK]

        """
        # # dL/dgamma_mask_q_last = dL/ddecay_s * ddecay_s/dgamma_mask_q_last      # [BT, 1]
        # d_dgamma_mask_q_last = b_ddecay_s
        # dL/dlog_a_cum_last = dL/ds * ds/da_cum_last * da_cum_last/d_log_a_cum_last
        #                         = dL/ds * S_pre * exp(*)                               [1]
        # dL/dlog_m_cum_last = dL/dm * dm/dm_cum_last * dm_cum_last/d_log_m_cum_last
        #                       + dL/d_decay_m * d_decay_m/dlog_m_cum_last
        #                         = dL/dm * M_pre + dL/d_decay_m * decay_m               [1]
        """
        b_dlog_a_cum_last += tl.sum(b_ds * b_s, axis=0) * b_a_cum_last       # [BV, BK] -> [BK]
        b_dlog_mu_cum_last += tl.sum(b_dm * b_m, axis=0) * b_mu_cum_last   # [BV, BK] -> [BK]

    temp = b_d_decay_m * b_decay_m[:, None]  # [BT, BK] * [BT, BK]
    b_dlog_mu_cum_last += tl.sum(temp, axis=0)
    b_dlog_mu_cum -= temp

    tl.store(dy + o_t[:, None] * (H*K) + o_k[None, :],
             b_dy.to(dy.dtype.element_ty), mask=m_t[:, None] & m_k[None, :])
    tl.store(dz + o_t[:, None] * (H*K) + o_k[None, :],
             b_dz.to(dz.dtype.element_ty), mask=m_t[:, None] & m_k[None, :])

    # the tl.atomic_add exits some bugs under the diffenrent Triton version, the reduction is operated out of this kernel
    not_lastrow = o_t < min(i_t * BT + BT, T) - 1

    mask = not_lastrow[:, None]
    b_dbt = tl.where(mask, b_dbt,         b_dbt + b_dbt_last)
    b_dlog_mu_cum = tl.where(mask, b_dlog_mu_cum, b_dlog_mu_cum + b_dlog_mu_cum_last)
    b_dlog_a_cum = tl.where(mask, b_dlog_a_cum,  b_dlog_a_cum + b_dlog_a_cum_last)

    tl.store(d_log_a_cum + o_t[:, None] * (H*K) + o_k[None, :],
             b_dlog_a_cum.to(d_log_a_cum.dtype.element_ty), mask=m_t[:, None] & m_k[None, :])
    tl.store(d_log_mu_cum + o_t[:, None] * (H*K) + o_k[None, :],
             b_dlog_mu_cum.to(d_log_mu_cum.dtype.element_ty), mask=m_t[:, None] & m_k[None, :])
    tl.store(d_bt + o_t[:, None] * (H*K) + o_k[None, :],
             b_dbt.to(d_bt.dtype.element_ty), mask=m_t[:, None] & m_k[None, :])
    tl.store(d_decay_s + o_t[:, None] * (H*K) + o_k[None, :],
             b_d_decay_s.to(d_decay_s.dtype.element_ty), mask=m_t[:, None] & m_k[None, :])

    b_gamma_mask_q = tl.load(gamma_mask_q + o_t[:, None] * (H*BT) + tl.arange(0, BT)[None, :],
                             mask=m_t[:, None], other=0.0)

    b_dG = b_dAttn * b_gamma_mask_q               # b_gamma_mask_q is triled
    b_dq += tl.dot(b_dG.to(b_k.dtype), b_k, allow_tf32=False)
    b_dk += tl.dot(tl.trans(b_dG).to(b_k.dtype), b_q.to(b_k.dtype), allow_tf32=False)

    tl.store(dq + o_t[:, None] * (H*K) + o_k[None, :],
             (b_dq * scale).to(dq.dtype.element_ty), mask=m_t[:, None] & m_k[None, :])
    tl.store(dk + o_t[:, None] * (H*K) + o_k[None, :],
             b_dk.to(dk.dtype.element_ty), mask=m_t[:, None] & m_k[None, :])
    tl.store(d_Attn + o_t[:, None] * (H*BT) + tl.arange(0, BT)[None, :],
             b_dAttn.to(d_Attn.dtype.element_ty), mask=m_t[:, None])


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4,]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_momentum_delta_rule_bwd_kernel_dbyz(
    q,
    k,
    d_decay_s,
    d_gamma_mask_q,
    d_Attn,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    d_gamma_mask_q += (bos * H + i_h) * BT
    d_Attn += (bos * H + i_h) * BT

    d_decay_s += bos * H + i_h
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_d_decay_s = tl.load(d_decay_s + o_t * H, mask=m_t, other=0.0).to(tl.float32)                # [BT]
    b_dAttn = tl.load(d_Attn + o_t[:, None] * (H*BT) + tl.arange(0, BT)[None, :],
                      mask=m_t[:, None], other=0.0).to(tl.float32)                            # [BT, BT]
    b_G = tl.zeros([BT, BT], dtype=tl.float32)

    for i_kk in range(tl.cdiv(K, BK)):
        o_kk = i_kk * BK + tl.arange(0, BK)
        m_kk = o_kk < K
        b_qt = tl.load(q + o_t[:, None] * (H*K) + o_kk[None, :],
                       mask=m_t[:, None] & m_kk[None, :], other=0.0)                              # [BT, BK]
        b_kt = tl.load(k + o_kk[:, None] + o_t[None, :] * (H*K),
                       mask=m_kk[:, None] & m_t[None, :], other=0.0)                              # [BK, BT]
        b_G += tl.dot(b_qt, b_kt, allow_tf32=False) * scale

    """
    # dL/dgamma_mask_q = dL/dAttn * dAttn/dgamma_mask_q + dL/d_decay_s
    #                         = dL/dAttn * q * k                                            [BT, BT]
    """
    b_d_gamma_mask_q = b_dAttn * b_G                        # [BT, BT]

    rel_last = (min((i_t + 1) * BT, T) - 1) - i_t * BT  # scalar in [0, BT-1]
    rows = tl.arange(0, BT)[:, None]  # [BT, 1]
    b_last_row = tl.where(rows < rel_last, 0.0, b_d_decay_s[None, :])  # [BT, BT]
    b_d_gamma_mask_q += b_last_row   # b_d_gamma_mask_q_last = b_ddecay_s
    tl.store(d_gamma_mask_q + o_t[:, None] * (H*BT) + tl.arange(0, BT)[None, :],
             b_d_gamma_mask_q.to(d_gamma_mask_q.dtype.element_ty), mask=m_t[:, None])


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_momentum_delta_rule_bwd_kernel_dv_local(
    q,
    k,
    gamma_mask_q,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    """
    According to:   o[:, :, i] = qS_inter + attn @ v_i,  attn = (q_i @ k_i.transpose(-1, -2)) * gamma_mask_q[:, :, i]
    dL/dv = dL/do * do/dv = dL/do * Attn.T
    Attn = q * k.T * gamma_mask_q
    """
    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    gamma_mask_q += (bos * H + i_h) * BT

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_k = tl.load(k + o_t[:, None] * (H*K) + o_k[None, :],
                      mask=m_t[:, None] & m_k[None, :], other=0.0)       # [BT, BK]
        b_q = tl.load(q + o_k[:, None] + o_t[None, :] * (H*K),
                      mask=m_k[:, None] & m_t[None, :], other=0.0)       # [BK, BT]
        b_A += tl.dot(b_k, b_q, allow_tf32=False) * scale

    m_A = (o_t[:, None] <= o_t[None, :]) & m_t[:, None] & m_t[None, :]

    b_gamma_mask_q = tl.load(gamma_mask_q + tl.arange(0, BT)[:, None] + o_t[None, :] * (H*BT),
                             mask=m_t[None, :], other=0.0).to(tl.float32)

    b_A = b_A * b_gamma_mask_q
    b_A = tl.where(m_A, b_A, 0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        mask_v = m_t[:, None] & (o_v[None, :] < V)
        b_do = tl.load(do + o_t[:, None] * (H*V) + o_v[None, :], mask=mask_v, other=0.0)
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do, allow_tf32=False)   # Attn^T @ dO
        tl.store(dv + o_t[:, None] * (H*V) + o_v[None, :],
                 b_dv.to(dv.dtype.element_ty), mask=mask_v)


def chunk_momentum_delta_rule_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,  # here is v_new
    o_inter: torch.Tensor,
    gamma_mask_q: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64
) -> torch.Tensor:

    # if use gamma_mask_q, the chunk_size should be equal to previous function
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        NT = triton.cdiv(T, BT)
    else:
        NT = len(chunk_indices)

    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)

    o = torch.empty_like(v)

    def grid(meta): return (NT, B * H)
    chunk_momentum_delta_rule_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        gamma_mask_q=gamma_mask_q,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o


def chunk_momentum_delta_rule_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    gamma_mask_q: torch.Tensor,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64
) -> torch.Tensor:
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = gamma_mask_q.shape[-1]
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem:
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(triton.next_power_of_2(K), CONST_TILING)
    BV = min(triton.next_power_of_2(V), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(do)
    grid = (NT, B * H)
    chunk_momentum_delta_rule_bwd_kernel_dv_local[grid](
        q=q,
        k=k,
        gamma_mask_q=gamma_mask_q,
        do=do,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dv


def chunk_momentum_delta_rule_bwd_dqkyz(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    s: torch.Tensor,
    m: torch.Tensor,
    ds: torch.Tensor,
    dm: torch.Tensor,
    log_mu_cum: torch.Tensor,
    log_a_cum: torch.Tensor,
    bt: torch.Tensor,
    gamma_mask_q: torch.Tensor,
    dv: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None,
           torch.Tensor | None, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:

    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = gamma_mask_q.shape[-1]

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING_K = 32
    CONST_TILING_V = 32

    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING_K)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING_V)

    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q)
    dy = torch.empty_like(y)
    dz = torch.empty_like(z)

    dk = torch.empty_like(k, dtype=torch.float)

    d_log_mu_cum = torch.empty_like(k, dtype=torch.float)
    d_log_a_cum = torch.empty_like(k, dtype=torch.float)
    d_bt = torch.empty_like(k, dtype=torch.float)
    d_decay_s = torch.empty_like(k, dtype=torch.float)

    d_Attn_do_v = torch.empty_like(gamma_mask_q, dtype=torch.float)

    grid = (NK, NT, B * H)
    chunk_momentum_delta_rule_bwd_kernel_dqk[grid](
        q=q,
        k=k,
        v=v,
        dv=dv,
        s=s,
        m=m,
        ds=ds,
        dm=dm,
        log_mu_cum=log_mu_cum,
        log_a_cum=log_a_cum,
        bt=bt,
        gamma_mask_q=gamma_mask_q,
        do=do,
        dq=dq,
        dk=dk,
        d_log_mu_cum=d_log_mu_cum,
        d_log_a_cum=d_log_a_cum,
        d_bt=d_bt,
        d_decay_s=d_decay_s,
        d_Attn=d_Attn_do_v,
        y=y,
        z=z,
        dy=dy,
        dz=dz,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )

    d_log_a_cum2 = d_log_a_cum.sum(dim=-1)
    d_log_mu_cum2 = d_log_mu_cum.sum(dim=-1)
    d_bt2 = d_bt.sum(dim=-1)
    d_decay_s2 = d_decay_s.sum(dim=-1)
    del d_log_a_cum
    del d_log_mu_cum
    del d_bt
    del d_decay_s

    return dq, dk, dy, dz, d_log_mu_cum2, d_log_a_cum2, d_bt2, d_Attn_do_v, d_decay_s2
