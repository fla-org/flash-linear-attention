# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs, check_shared_mem

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.heuristics({
    'USE_INITIAL_S': lambda args: args['s0'] is not None,
    'USE_INITIAL_M': lambda args: args['m0'] is not None,
    'STORE_FINAL_S': lambda args: args['st'] is not None,
    'STORE_FINAL_M': lambda args: args['mt'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=warps, num_stages=stages)
        for BV in [16, 32, 64]
        for warps in [2, 4]
        for stages in [2, 3,]
    ],
    key=['H', 'K', 'V', 'BT'],

    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_momentum_delta_rule_fwd_kernel_inter_qh_blockdim64(
        q,
        k,
        u,  # u y z for recomputing v_new
        y,
        z,
        log_a_cum,
        log_mu_cum,
        bt,
        gamma_mask_q,
        s0,
        m0,
        v_new,
        o_inter,
        st,
        mt,
        scale,
        cu_seqlens,
        chunk_offsets,
        T,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        USE_INITIAL_S: tl.constexpr,
        USE_INITIAL_M: tl.constexpr,
        STORE_FINAL_S: tl.constexpr,
        STORE_FINAL_M: tl.constexpr,
        SAVE_NEW_VALUE: tl.constexpr,
        IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

    # [BK, BV]  zero initialize the hidden state
    b_s1 = tl.zeros([64, BV], dtype=tl.float32)
    b_m1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_s2 = tl.zeros([64, BV], dtype=tl.float32)
        b_m2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_s3 = tl.zeros([64, BV], dtype=tl.float32)
        b_m3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_s4 = tl.zeros([64, BV], dtype=tl.float32)
        b_m4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    u += (bos * H + i_h) * V
    o_inter += (bos * H + i_h) * V

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    y += (bos * H + i_h) * K
    z += (bos * H + i_h) * K

    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h) * V

    stride_v = H * V
    stride_k = H * K
    if USE_INITIAL_S:
        s0 = s0 + i_nh * K * V
    if USE_INITIAL_M:
        m0 = m0 + i_nh * K * V

    if STORE_FINAL_S:
        st = st + i_nh * K * V
    if STORE_FINAL_M:
        mt = mt + i_nh * K * V

    # load initial state
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    if USE_INITIAL_S:
        o_kk = tl.arange(0, 64)
        b_s1 += tl.load(s0 + o_kk[:, None] * V + o_v[None, :],
                        mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            b_s2 += tl.load(s0 + o_kk[:, None] * V + o_v[None, :],
                            mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            b_s3 += tl.load(s0 + o_kk[:, None] * V + o_v[None, :],
                            mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            b_s4 += tl.load(s0 + o_kk[:, None] * V + o_v[None, :],
                            mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)

    if USE_INITIAL_M:
        o_kk = tl.arange(0, 64)
        b_m1 += tl.load(m0 + o_kk[:, None] * V + o_v[None, :],
                        mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            b_m2 += tl.load(m0 + o_kk[:, None] * V + o_v[None, :],
                            mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            b_m3 += tl.load(m0 + o_kk[:, None] * V + o_v[None, :],
                            mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            b_m4 += tl.load(m0 + o_kk[:, None] * V + o_v[None, :],
                            mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)

    # main recurrence, NT is number of chunks, i_t means the i_t-th chunk
    for i_t in range(NT):
        b_s1_pre, b_m1_pre = b_s1, b_m1
        if K > 64:
            b_s2_pre, b_m2_pre = b_s2, b_m2
        if K > 128:
            b_s3_pre, b_m3_pre = b_s3, b_m3
        if K > 192:
            b_s4_pre, b_m4_pre = b_s4, b_m4

        o_t = i_t * BT + tl.arange(0, BT)
        mask_t = o_t < T

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_a_cum = tl.exp(tl.load(log_a_cum + bos * H + i_h + o_t * H, mask=mask_t, other=0.0))[:, None]
        b_bt = tl.load(bt + bos * H + i_h + o_t * H, mask=mask_t, other=0.0)[:, None]

        b_o_inter = tl.zeros([BT, BV], dtype=tl.float32)

        # Computing new (pseudo) value: v_c = u_c[:, :, i] - y_c[:, :, i] @ S_pre + z_c[:, :, i] @ M_pre
        o_kk = tl.arange(0, 64)
        mask_q = mask_t[:, None] & (o_kk[None, :] < K)

        b_v_new = tl.load(u + o_t[:, None] * stride_v + o_v[None, :],
                          mask=mask_t[:, None] & m_v[None, :], other=0.0)
        b_y = tl.load(y + o_t[:, None] * stride_k + o_kk[None, :], mask=mask_q, other=0.0)
        b_z = tl.load(z + o_t[:, None] * stride_k + o_kk[None, :], mask=mask_q, other=0.0)
        b_q = tl.load(q + o_t[:, None] * stride_k + o_kk[None, :], mask=mask_q, other=0.0)    # [BT, BK]

        b_v_new += - tl.dot(b_y, b_s1_pre.to(b_y.dtype), allow_tf32=False) + \
            tl.dot(b_z, b_m1_pre.to(b_z.dtype), allow_tf32=False)

        b_btq = b_bt * b_q                   # [BT, BK]
        b_baraq = b_a_cum * b_q                # [BT, BK]
        b_o_inter += tl.dot(b_baraq.to(b_q.dtype), b_s1_pre.to(b_q.dtype), allow_tf32=False) - \
            tl.dot(b_btq.to(b_q.dtype), b_m1_pre.to(b_q.dtype), allow_tf32=False)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            mask_q = mask_t[:, None] & (o_kk[None, :] < K)
            b_y = tl.load(y + o_t[:, None] * stride_k + o_kk[None, :], mask=mask_q, other=0.0)
            b_z = tl.load(z + o_t[:, None] * stride_k + o_kk[None, :], mask=mask_q, other=0.0)
            b_q = tl.load(q + o_t[:, None] * stride_k + o_kk[None, :], mask=mask_q, other=0.0)

            b_v_new += - tl.dot(b_y, b_s2_pre.to(b_y.dtype), allow_tf32=False) + \
                tl.dot(b_z, b_m2_pre.to(b_z.dtype), allow_tf32=False)

            b_btq = b_bt * b_q                   # [BT, BK]
            b_baraq = b_a_cum * b_q                # [BT, BK]
            b_o_inter += tl.dot(b_baraq.to(b_q.dtype), b_s2_pre.to(b_q.dtype), allow_tf32=False) - \
                tl.dot(b_btq.to(b_q.dtype), b_m2_pre.to(b_q.dtype), allow_tf32=False)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            mask_q = mask_t[:, None] & (o_kk[None, :] < K)
            b_y = tl.load(y + o_t[:, None] * stride_k + o_kk[None, :], mask=mask_q, other=0.0)
            b_z = tl.load(z + o_t[:, None] * stride_k + o_kk[None, :], mask=mask_q, other=0.0)
            b_q = tl.load(q + o_t[:, None] * stride_k + o_kk[None, :], mask=mask_q, other=0.0)

            b_v_new += - tl.dot(b_y, b_s3_pre.to(b_q.dtype), allow_tf32=False) + \
                tl.dot(b_z, b_m3_pre.to(b_z.dtype), allow_tf32=False)

            b_btq = b_bt * b_q                   # [BT, BK]
            b_baraq = b_a_cum * b_q                # [BT, BK]
            b_o_inter += tl.dot(b_baraq.to(b_q.dtype), b_s3_pre.to(b_q.dtype), allow_tf32=False) - \
                tl.dot(b_btq.to(b_q.dtype), b_m3_pre.to(b_q.dtype), allow_tf32=False)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            mask_q = mask_t[:, None] & (o_kk[None, :] < K)
            b_y = tl.load(y + o_t[:, None] * stride_k + o_kk[None, :], mask=mask_q, other=0.0)
            b_z = tl.load(z + o_t[:, None] * stride_k + o_kk[None, :], mask=mask_q, other=0.0)
            b_q = tl.load(q + o_t[:, None] * stride_k + o_kk[None, :], mask=mask_q, other=0.0)

            b_v_new += - tl.dot(b_y, b_s4_pre.to(b_y.dtype), allow_tf32=False) + \
                tl.dot(b_z, b_m4_pre.to(b_z.dtype), allow_tf32=False)

            b_btq = b_bt * b_q                   # [BT, BK]
            b_baraq = b_a_cum * b_q                # [BT, BK]
            b_o_inter += tl.dot(b_baraq.to(b_q.dtype), b_s4_pre.to(b_q.dtype), allow_tf32=False) - \
                tl.dot(b_btq.to(b_q.dtype), b_m4_pre.to(b_q.dtype), allow_tf32=False)

        # Storing new (pseudo) value and b_o_inter
        mask_v = mask_t[:, None] & m_v[None, :]
        if SAVE_NEW_VALUE:
            tl.store(v_new + o_t[:, None] * stride_v + o_v[None, :],
                     b_v_new.to(v_new.dtype.element_ty), mask=mask_v)

        tl.store(o_inter + o_t[:, None] * stride_v + o_v[None, :],
                 (b_o_inter * scale).to(o_inter.dtype.element_ty), mask=mask_v)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_log_mcum_last = tl.load(log_mu_cum + bos * H + last_idx * H + i_h)
        b_log_acum_last = tl.load(log_a_cum + bos * H + last_idx * H + i_h)
        b_bt_last = tl.load(bt + bos * H + last_idx * H + i_h)
        #  access last raw
        base_plane = gamma_mask_q + (bos * H + i_h) * BT  # (T, BT)
        row_stride = BT * H
        row_base = base_plane + last_idx * row_stride
        o_tr = tl.arange(0, BT)
        b_gamma_last_row = tl.load(row_base + o_tr, mask=o_tr < BT, other=0.0)

        b_log_mcum = tl.load(log_mu_cum + bos * H + i_h + o_t * H, mask=mask_t, other=0.0)

        b_log_mcum_last_vec = b_log_mcum_last + tl.zeros([BT], dtype=b_log_mcum_last.dtype)
        b_for_m = tl.exp(b_log_mcum_last_vec[:, None] - b_log_mcum[:, None])

        b_v_new = tl.where(mask_t[:, None], b_v_new, 0.0)
        b_for_s = tl.where(mask_t, b_gamma_last_row, 0.0)
        b_for_m = tl.where(mask_t[:, None], b_for_m, 0.0)

        b_v_new_s = b_v_new * b_for_s[:, None]  # [BT,BV]
        b_v_new_m = b_v_new * b_for_m  # [BT,BV]

        b_mcum_last = tl.exp(b_log_mcum_last)
        b_acum_last = tl.exp(b_log_acum_last)

        # computing H += K @ V
        o_kk = tl.arange(0, 64)
        b_k = tl.load(k + o_kk[:, None] + o_t[None, :] * stride_k,
                      mask=(o_kk[:, None] < K) & mask_t[None, :], other=0.0)
        b_s1 = b_acum_last * b_s1_pre - b_bt_last * b_m1_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype), allow_tf32=False)
        b_m1 = b_mcum_last * b_m1_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype), allow_tf32=False)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            b_k = tl.load(k + o_kk[:, None] + o_t[None, :] * stride_k,
                          mask=(o_kk[:, None] < K) & mask_t[None, :], other=0.0)
            b_s2 = b_acum_last * b_s2_pre - b_bt_last * b_m2_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype), allow_tf32=False)
            b_m2 = b_mcum_last * b_m2_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype), allow_tf32=False)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            b_k = tl.load(k + o_kk[:, None] + o_t[None, :] * stride_k,
                          mask=(o_kk[:, None] < K) & mask_t[None, :], other=0.0)
            b_s3 = b_acum_last * b_s3_pre - b_bt_last * b_m3_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype), allow_tf32=False)
            b_m3 = b_mcum_last * b_m3_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype), allow_tf32=False)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            b_k = tl.load(k + o_kk[:, None] + o_t[None, :] * stride_k,
                          mask=(o_kk[:, None] < K) & mask_t[None, :], other=0.0)
            b_s4 = b_acum_last * b_s4_pre - b_bt_last * b_m4_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype), allow_tf32=False)
            b_m4 = b_mcum_last * b_m4_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype), allow_tf32=False)

    # epilogue
    if STORE_FINAL_S:
        o_kk = tl.arange(0, 64)
        m_kk = (o_kk[:, None] < K) & m_v[None, :]
        tl.store(st + o_kk[:, None] * V + o_v[None, :], b_s1.to(st.dtype.element_ty), mask=m_kk)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(st + o_kk[:, None] * V + o_v[None, :], b_s2.to(st.dtype.element_ty), mask=m_kk)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(st + o_kk[:, None] * V + o_v[None, :], b_s3.to(st.dtype.element_ty), mask=m_kk)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(st + o_kk[:, None] * V + o_v[None, :], b_s4.to(st.dtype.element_ty), mask=m_kk)

    if STORE_FINAL_M:
        o_kk = tl.arange(0, 64)
        m_kk = (o_kk[:, None] < K) & m_v[None, :]
        tl.store(mt + o_kk[:, None] * V + o_v[None, :], b_m1.to(mt.dtype.element_ty), mask=m_kk)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(mt + o_kk[:, None] * V + o_v[None, :], b_m2.to(mt.dtype.element_ty), mask=m_kk)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(mt + o_kk[:, None] * V + o_v[None, :], b_m3.to(mt.dtype.element_ty), mask=m_kk)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(mt + o_kk[:, None] * V + o_v[None, :], b_m4.to(mt.dtype.element_ty), mask=m_kk)


@triton.heuristics({
    # 'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_S': lambda args: args['ds0'] is not None,
    'USE_INITIAL_M': lambda args: args['dm0'] is not None,
    'USE_FINAL_S_GRADIENT': lambda args: args['dst'] is not None,
    'USE_FINAL_M_GRADIENT': lambda args: args['dmt'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=warps, num_stages=stages)
        for BV in [16, 32, 64]
        for warps in [2, 4]
        for stages in [2, 3]
    ],
    key=['H', 'K', 'V', 'BT', 'BV',
         #  'USE_G'
         ],

    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_momentum_delta_rule_bwd_kernel_dhu_blockdim64(
        q,
        k,
        u,
        y,
        z,
        log_mu_cum,
        log_a_cum,
        bt,
        gamma_mask_q,
        dst,
        dmt,
        ds0,
        dm0,
        do,
        ds,
        dm,
        dv,
        dv2,
        cu_seqlens,
        chunk_offsets,
        scale,
        T,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        USE_INITIAL_S: tl.constexpr,
        USE_INITIAL_M: tl.constexpr,
        USE_FINAL_S_GRADIENT: tl.constexpr,
        USE_FINAL_M_GRADIENT: tl.constexpr,
        IS_VARLEN: tl.constexpr
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
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

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    # [BK, BV]  zero initialize the hidden state
    b_ds1 = tl.zeros([64, BV], dtype=tl.float32)
    b_dm1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_ds2 = tl.zeros([64, BV], dtype=tl.float32)
        b_dm2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_ds3 = tl.zeros([64, BV], dtype=tl.float32)
        b_dm3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_ds4 = tl.zeros([64, BV], dtype=tl.float32)
        b_dm4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    ds += (boh * H + i_h) * K * V
    dm += (boh * H + i_h) * K * V

    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V

    u += (bos * H + i_h) * V
    k += (bos * H + i_h) * K
    q += (bos * H + i_h) * K
    y += (bos * H + i_h) * K
    z += (bos * H + i_h) * K
    do += (bos * H + i_h) * V

    gamma_mask_q += (bos * H + i_h) * BT  # (T, BT)

    log_a_cum += bos * H + i_h
    log_mu_cum += bos * H + i_h
    bt += bos * H + i_h

    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K

    if USE_INITIAL_S:
        ds0 += i_nh * K * V
    if USE_INITIAL_M:
        dm0 += i_nh * K * V

    if USE_FINAL_S_GRADIENT:
        dst += i_nh * K * V
        o_kk = tl.arange(0, 64)
        b_ds1 += tl.load(dst + o_kk[:, None] * V + o_v[None, :],
                         mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            b_ds2 += tl.load(dst + o_kk[:, None] * V + o_v[None, :],
                             mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            b_ds3 += tl.load(dst + o_kk[:, None] * V + o_v[None, :],
                             mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            b_ds4 += tl.load(dst + o_kk[:, None] * V + o_v[None, :],
                             mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)

    if USE_FINAL_M_GRADIENT:
        dmt += i_nh * K * V
        o_kk = tl.arange(0, 64)
        b_dm1 += tl.load(dmt + o_kk[:, None] * V + o_v[None, :],
                         mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            b_dm2 += tl.load(dmt + o_kk[:, None] * V + o_v[None, :],
                             mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            b_dm3 += tl.load(dmt + o_kk[:, None] * V + o_v[None, :],
                             mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            b_dm4 += tl.load(dmt + o_kk[:, None] * V + o_v[None, :],
                             mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)

    # main recurrence, NT is number of chunks, i_t means the i_t-th chunk
    for i_t in range(NT - 1, -1, -1):
        o_t = i_t * BT + tl.arange(0, BT)
        mask_t = o_t < T
        # Storing last State gradients
        o_kk = tl.arange(0, 64)
        m_kk = (o_kk[:, None] < K) & m_v[None, :]
        tl.store(ds + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_ds1.to(ds.dtype.element_ty), mask=m_kk)
        tl.store(dm + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_dm1.to(dm.dtype.element_ty), mask=m_kk)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(ds + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_ds2.to(ds.dtype.element_ty), mask=m_kk)
            tl.store(dm + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_dm2.to(dm.dtype.element_ty), mask=m_kk)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(ds + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_ds3.to(ds.dtype.element_ty), mask=m_kk)
            tl.store(dm + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_dm3.to(dm.dtype.element_ty), mask=m_kk)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(ds + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_ds4.to(ds.dtype.element_ty), mask=m_kk)
            tl.store(dm + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_dm4.to(dm.dtype.element_ty), mask=m_kk)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_log_mcum_last = tl.load(log_mu_cum + last_idx * H)
        b_log_acum_last = tl.load(log_a_cum + last_idx * H)
        b_bt_last = tl.load(bt + last_idx * H)
        #  access last raw

        row_base = gamma_mask_q + last_idx * (BT * H)
        o_tr = tl.arange(0, BT)
        b_gamma_last_row = tl.load(row_base + o_tr, mask=o_tr < BT, other=0.0)
        b_bt = tl.load(bt + o_t * H, mask=mask_t, other=0.0)
        b_log_mcum = tl.load(log_mu_cum + o_t * H, mask=mask_t, other=0.0)
        b_log_acum = tl.load(log_a_cum + o_t * H, mask=mask_t, other=0.0)

        b_decay_s = b_gamma_last_row                        # [BT]
        b_decay_m = tl.exp(b_log_mcum_last - b_log_mcum)    # [BT]

        b_mcum_last = tl.exp(b_log_mcum_last)
        b_acum_last = tl.exp(b_log_acum_last)

        b_do = tl.load(do + o_t[:, None] * stride_v + o_v[None, :],
                       mask=mask_t[:, None] & m_v[None, :], other=0.0)

        # Update dv
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)
        o_kk = tl.arange(0, 64)
        b_k = tl.load(k + o_t[:, None] * stride_k + o_kk[None, :],
                      mask=mask_t[:, None] & (o_kk[None, :] < K), other=0.0)
        b_k_decay_s = b_k * b_decay_s[:, None]   # [BT,BK]
        b_k_decay_m = b_k * b_decay_m[:, None]   # [BT,BK]
        b_dv += tl.dot(b_k_decay_s.to(b_k.dtype), b_ds1.to(b_k.dtype), allow_tf32=False) - \
            tl.dot(b_k_decay_m.to(b_k.dtype), b_dm1.to(b_k.dtype), allow_tf32=False)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            b_k = tl.load(k + o_t[:, None] * stride_k + o_kk[None, :],
                          mask=mask_t[:, None] & (o_kk[None, :] < K), other=0.0)
            b_k_decay_s = b_k * b_decay_s[:, None]  # [BT,BK]
            b_k_decay_m = b_k * b_decay_m[:, None]  # [BT,BK]
            b_dv += tl.dot(b_k_decay_s.to(b_k.dtype), b_ds2.to(b_k.dtype), allow_tf32=False) - \
                tl.dot(b_k_decay_m.to(b_k.dtype), b_dm2.to(b_k.dtype), allow_tf32=False)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            b_k = tl.load(k + o_t[:, None] * stride_k + o_kk[None, :],
                          mask=mask_t[:, None] & (o_kk[None, :] < K), other=0.0)
            b_k_decay_s = b_k * b_decay_s[:, None]  # [BT,BK]
            b_k_decay_m = b_k * b_decay_m[:, None]  # [BT,BK]
            b_dv += tl.dot(b_k_decay_s.to(b_k.dtype), b_ds3.to(b_k.dtype), allow_tf32=False) - \
                tl.dot(b_k_decay_m.to(b_k.dtype), b_dm3.to(b_k.dtype), allow_tf32=False)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            b_k = tl.load(k + o_t[:, None] * stride_k + o_kk[None, :],
                          mask=mask_t[:, None] & (o_kk[None, :] < K), other=0.0)
            b_k_decay_s = b_k * b_decay_s[:, None]  # [BT,BK]
            b_k_decay_m = b_k * b_decay_m[:, None]  # [BT,BK]
            b_dv += tl.dot(b_k_decay_s.to(b_k.dtype), b_ds4.to(b_k.dtype), allow_tf32=False) - \
                tl.dot(b_k_decay_m.to(b_k.dtype), b_dm4.to(b_k.dtype), allow_tf32=False)

        # here dL/dv_t is part of dL/dot * dot/dvt
        b_dv += tl.load(dv + o_t[:, None] * stride_v + o_v[None, :],
                        mask=mask_t[:, None] & m_v[None, :], other=0.0)

        # store dv2
        # here complete dL/dvt = dL/dSt * dS_t/dvt + dL/dMt * dM_t/dvt + dL/dot * dot/dvt
        tl.store(dv2 + o_t[:, None] * stride_v + o_v[None, :],
                 b_dv.to(dv2.dtype.element_ty), mask=mask_t[:, None] & m_v[None, :])

        # Update ds, dh, ref name is corresponding to the /fla/test/momentum_delta_net.py
        # dL/d_S{t-1} = dL/d_St * d_St/d_S{t-1} + dL/d_Mt * d_Mt/d_S{t-1}
        #             + dL/d_ot * d_ot/d_S{t-1} + dL/d_vt * d_vt/d_S{t-1}
        # dL/d_M{t-1} = dL/d_St * d_St/d_M{t-1} + dL/d_Mt * d_Mt/d_M{t-1}
        #             + dL/d_ot * d_ot/d_M{t-1} + dL/d_vt * d_vt/d_M{t-1}
        # get
        # dL/d_S_{t-1} = dL/d_St * (last_log_a_cum.exp()) + dL/d_Mt * (0)
        #              + dL/d_ot * (bar_alpha_t_q)        + dL/d_vt * (-y)
        # dL/d_M_{t-1} = dL/d_St * (last_bt)              + dL/d_Mt * (last_log_m_cum.exp())
        #              + dL/d_ot * (- b_t_q)              + dL/d_vt * (+z)
        o_kk = tl.arange(0, 64)
        m_kk = (o_kk[:, None] < K) & mask_t[None, :]
        b_y = tl.load(y + o_kk[:, None] + o_t[None, :] * stride_k, mask=m_kk, other=0.0)
        b_z = tl.load(z + o_kk[:, None] + o_t[None, :] * stride_k, mask=m_kk, other=0.0)
        b_q = tl.load(q + o_kk[:, None] + o_t[None, :] * stride_k, mask=m_kk, other=0.0) * scale

        b_ds1_pre = b_acum_last * b_ds1 \
            + tl.dot((tl.exp(b_log_acum)[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype), allow_tf32=False) \
            - tl.dot(b_y, b_dv.to(b_y.dtype), allow_tf32=False)

        b_dm1_pre = -b_bt_last * b_ds1 + b_mcum_last * b_dm1 \
                    - tl.dot((b_bt[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype), allow_tf32=False) \
            + tl.dot(b_z, b_dv.to(b_z.dtype), allow_tf32=False)

        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & mask_t[None, :]
            b_y = tl.load(y + o_kk[:, None] + o_t[None, :] * stride_k, mask=m_kk, other=0.0)
            b_z = tl.load(z + o_kk[:, None] + o_t[None, :] * stride_k, mask=m_kk, other=0.0)
            b_q = tl.load(q + o_kk[:, None] + o_t[None, :] * stride_k, mask=m_kk, other=0.0) * scale

            b_ds2_pre = b_acum_last * b_ds2 \
                + tl.dot((tl.exp(b_log_acum)[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype), allow_tf32=False) \
                - tl.dot(b_y, b_dv.to(b_y.dtype), allow_tf32=False)

            b_dm2_pre = -b_bt_last * b_ds2 + b_mcum_last * b_dm2 \
                        - tl.dot((b_bt[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype), allow_tf32=False) \
                + tl.dot(b_z, b_dv.to(b_z.dtype), allow_tf32=False)

        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & mask_t[None, :]
            b_y = tl.load(y + o_kk[:, None] + o_t[None, :] * stride_k, mask=m_kk, other=0.0)
            b_z = tl.load(z + o_kk[:, None] + o_t[None, :] * stride_k, mask=m_kk, other=0.0)
            b_q = tl.load(q + o_kk[:, None] + o_t[None, :] * stride_k, mask=m_kk, other=0.0) * scale

            b_ds3_pre = b_acum_last * b_ds3 \
                + tl.dot((tl.exp(b_log_acum)[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype), allow_tf32=False) \
                - tl.dot(b_y, b_dv.to(b_y.dtype), allow_tf32=False)

            b_dm3_pre = -b_bt_last * b_ds3 + b_mcum_last * b_dm3 \
                        - tl.dot((b_bt[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype), allow_tf32=False) \
                + tl.dot(b_z, b_dv.to(b_z.dtype), allow_tf32=False)

        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & mask_t[None, :]
            b_y = tl.load(y + o_kk[:, None] + o_t[None, :] * stride_k, mask=m_kk, other=0.0)
            b_z = tl.load(z + o_kk[:, None] + o_t[None, :] * stride_k, mask=m_kk, other=0.0)
            b_q = tl.load(q + o_kk[:, None] + o_t[None, :] * stride_k, mask=m_kk, other=0.0) * scale

            b_ds4_pre = b_acum_last * b_ds4 \
                + tl.dot((tl.exp(b_log_acum)[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype), allow_tf32=False) \
                - tl.dot(b_y, b_dv.to(b_y.dtype), allow_tf32=False)

            b_dm4_pre = -b_bt_last * b_ds4 + b_mcum_last * b_dm4 \
                        - tl.dot((b_bt[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype), allow_tf32=False) \
                + tl.dot(b_z, b_dv.to(b_z.dtype), allow_tf32=False)

        b_ds1, b_dm1 = b_ds1_pre, b_dm1_pre
        if K > 64:
            b_ds2, b_dm2 = b_ds2_pre, b_dm2_pre
        if K > 128:
            b_ds3, b_dm3 = b_ds3_pre, b_dm3_pre
        if K > 192:
            b_ds4, b_dm4 = b_ds4_pre, b_dm4_pre

    if USE_INITIAL_S:
        o_kk = tl.arange(0, 64)
        m_kk = (o_kk[:, None] < K) & m_v[None, :]
        tl.store(ds0 + o_kk[:, None] * V + o_v[None, :], b_ds1.to(ds0.dtype.element_ty), mask=m_kk)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(ds0 + o_kk[:, None] * V + o_v[None, :], b_ds2.to(ds0.dtype.element_ty), mask=m_kk)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(ds0 + o_kk[:, None] * V + o_v[None, :], b_ds3.to(ds0.dtype.element_ty), mask=m_kk)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(ds0 + o_kk[:, None] * V + o_v[None, :], b_ds4.to(ds0.dtype.element_ty), mask=m_kk)

    if USE_INITIAL_M:
        o_kk = tl.arange(0, 64)
        m_kk = (o_kk[:, None] < K) & m_v[None, :]
        tl.store(dm0 + o_kk[:, None] * V + o_v[None, :], b_dm1.to(dm0.dtype.element_ty), mask=m_kk)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(dm0 + o_kk[:, None] * V + o_v[None, :], b_dm2.to(dm0.dtype.element_ty), mask=m_kk)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(dm0 + o_kk[:, None] * V + o_v[None, :], b_dm3.to(dm0.dtype.element_ty), mask=m_kk)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(dm0 + o_kk[:, None] * V + o_v[None, :], b_dm4.to(dm0.dtype.element_ty), mask=m_kk)


@triton.heuristics({
    'USE_INITIAL_S': lambda args: args['s0'] is not None,
    'USE_INITIAL_M': lambda args: args['m0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=warps, num_stages=stages)
        for BV in [32, 64]
        for warps in [2, 4]
        for stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT'],

    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_momentum_delta_rule_fwd_kernel_h_blockdim64_recompute_by_vnew(
        k,
        v_new,
        log_a_cum,
        log_mu_cum,
        bt,
        gamma_mask_q,
        s0,
        m0,
        hS,
        hM,
        cu_seqlens,
        chunk_offsets,
        T,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        USE_INITIAL_S: tl.constexpr,
        USE_INITIAL_M: tl.constexpr,
        IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
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

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    # [BK, BV]  zero initialize the hidden state
    b_s1 = tl.zeros([64, BV], dtype=tl.float32)
    b_m1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_s2 = tl.zeros([64, BV], dtype=tl.float32)
        b_m2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_s3 = tl.zeros([64, BV], dtype=tl.float32)
        b_m3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_s4 = tl.zeros([64, BV], dtype=tl.float32)
        b_m4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    hS += (boh * H + i_h) * K * V
    hM += (boh * H + i_h) * K * V
    k += (bos * H + i_h) * K

    v_new += (bos * H + i_h) * V

    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K
    if USE_INITIAL_S:
        s0 = s0 + i_nh * K * V
    if USE_INITIAL_M:
        m0 = m0 + i_nh * K * V

    # load initial state
    if USE_INITIAL_S:
        o_kk = tl.arange(0, 64)
        b_s1 += tl.load(s0 + o_kk[:, None] * V + o_v[None, :],
                        mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            b_s2 += tl.load(s0 + o_kk[:, None] * V + o_v[None, :],
                            mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            b_s3 += tl.load(s0 + o_kk[:, None] * V + o_v[None, :],
                            mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            b_s4 += tl.load(s0 + o_kk[:, None] * V + o_v[None, :],
                            mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)

    if USE_INITIAL_M:
        o_kk = tl.arange(0, 64)
        b_m1 += tl.load(m0 + o_kk[:, None] * V + o_v[None, :],
                        mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            b_m2 += tl.load(m0 + o_kk[:, None] * V + o_v[None, :],
                            mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            b_m3 += tl.load(m0 + o_kk[:, None] * V + o_v[None, :],
                            mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            b_m4 += tl.load(m0 + o_kk[:, None] * V + o_v[None, :],
                            mask=(o_kk[:, None] < K) & m_v[None, :], other=0.0).to(tl.float32)

    # main recurrence, NT is number of chunks, i_t means the i_t-th chunk
    for i_t in range(NT):
        b_s1_pre, b_m1_pre = b_s1, b_m1
        if K > 64:
            b_s2_pre, b_m2_pre = b_s2, b_m2
        if K > 128:
            b_s3_pre, b_m3_pre = b_s3, b_m3
        if K > 192:
            b_s4_pre, b_m4_pre = b_s4, b_m4

        o_t = i_t * BT + tl.arange(0, BT)
        mask_t = o_t < T

        # Storing Previous State
        o_kk = tl.arange(0, 64)
        m_kk = (o_kk[:, None] < K) & m_v[None, :]
        tl.store(hS + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_s1_pre.to(hS.dtype.element_ty), mask=m_kk)
        tl.store(hM + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_m1_pre.to(hM.dtype.element_ty), mask=m_kk)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(hS + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_s2_pre.to(hS.dtype.element_ty), mask=m_kk)
            tl.store(hM + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_m2_pre.to(hM.dtype.element_ty), mask=m_kk)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(hS + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_s3_pre.to(hS.dtype.element_ty), mask=m_kk)
            tl.store(hM + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_m3_pre.to(hM.dtype.element_ty), mask=m_kk)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            m_kk = (o_kk[:, None] < K) & m_v[None, :]
            tl.store(hS + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_s4_pre.to(hS.dtype.element_ty), mask=m_kk)
            tl.store(hM + i_t * stride_h + o_kk[:, None] * V + o_v[None, :], b_m4_pre.to(hM.dtype.element_ty), mask=m_kk)

        # Computing new (pseudo) value: v_c = u_c[:, :, i] - y_c[:, :, i] @ S_pre + z_c[:, :, i] @ M_pre
        b_v_new = tl.load(v_new + o_t[:, None] * stride_v + o_v[None, :],
                          mask=mask_t[:, None] & m_v[None, :], other=0.0)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_log_mcum_last = tl.load(log_mu_cum + bos * H + last_idx * H + i_h)
        b_log_acum_last = tl.load(log_a_cum + bos * H + last_idx * H + i_h)
        b_bt_last = tl.load(bt + bos * H + last_idx * H + i_h)

        #  access last raw
        base_plane = gamma_mask_q + (bos * H + i_h) * BT  # (T, BT)
        row_stride = BT * H
        row_base = base_plane + last_idx * row_stride
        o_tr = tl.arange(0, BT)
        b_gamma_last_row = tl.load(row_base + o_tr, mask=o_tr < BT, other=0.0)

        b_log_mcum = tl.load(log_mu_cum + bos * H + i_h + o_t * H, mask=mask_t, other=0.0)

        b_log_mcum_last_vec = b_log_mcum_last + tl.zeros([BT], dtype=b_log_mcum_last.dtype)
        b_for_m = tl.exp(b_log_mcum_last_vec[:, None] - b_log_mcum[:, None])

        b_v_new = tl.where(mask_t[:, None], b_v_new, 0.0)
        b_for_s = tl.where(mask_t, b_gamma_last_row, 0.0)
        b_for_m = tl.where(mask_t[:, None], b_for_m, 0.0)

        b_v_new_s = b_v_new * b_for_s[:, None]  # [BT,BV]
        b_v_new_m = b_v_new * b_for_m           # [BT,BV]

        b_mcum_last = tl.exp(b_log_mcum_last)
        b_acum_last = tl.exp(b_log_acum_last)

        # computing H += K @ V
        o_kk = tl.arange(0, 64)
        b_k = tl.load(k + o_kk[:, None] + o_t[None, :] * stride_k,
                      mask=(o_kk[:, None] < K) & mask_t[None, :], other=0.0)
        b_s1 = b_acum_last * b_s1_pre - b_bt_last * b_m1_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype), allow_tf32=False)
        b_m1 = b_mcum_last * b_m1_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype), allow_tf32=False)
        if K > 64:
            o_kk = 64 + tl.arange(0, 64)
            b_k = tl.load(k + o_kk[:, None] + o_t[None, :] * stride_k,
                          mask=(o_kk[:, None] < K) & mask_t[None, :], other=0.0)
            b_s2 = b_acum_last * b_s2_pre - b_bt_last * b_m2_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype), allow_tf32=False)
            b_m2 = b_mcum_last * b_m2_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype), allow_tf32=False)
        if K > 128:
            o_kk = 128 + tl.arange(0, 64)
            b_k = tl.load(k + o_kk[:, None] + o_t[None, :] * stride_k,
                          mask=(o_kk[:, None] < K) & mask_t[None, :], other=0.0)
            b_s3 = b_acum_last * b_s3_pre - b_bt_last * b_m3_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype), allow_tf32=False)
            b_m3 = b_mcum_last * b_m3_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype), allow_tf32=False)
        if K > 192:
            o_kk = 192 + tl.arange(0, 64)
            b_k = tl.load(k + o_kk[:, None] + o_t[None, :] * stride_k,
                          mask=(o_kk[:, None] < K) & mask_t[None, :], other=0.0)
            b_s4 = b_acum_last * b_s4_pre - b_bt_last * b_m4_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype), allow_tf32=False)
            b_m4 = b_mcum_last * b_m4_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype), allow_tf32=False)


def chunk_momentum_delta_rule_fwd_h_recompute_by_vnew(
        k: torch.Tensor,
        v_new: torch.Tensor,
        log_a_cum: torch.Tensor,
        log_mu_cum: torch.Tensor,
        bt: torch.Tensor,
        gamma_mask_q: torch.Tensor,
        initial_S: torch.Tensor | None = None,
        initial_M: torch.Tensor | None = None,
        chunk_size: int = 64,
        cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    V = v_new.shape[-1]
    BT = gamma_mask_q.shape[-1]

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    hS = k.new_empty(B, NT, H, K, V)
    hM = k.new_empty(B, NT, H, K, V)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N * H)
    chunk_momentum_delta_rule_fwd_kernel_h_blockdim64_recompute_by_vnew[grid](
        k=k,
        v_new=v_new,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        bt=bt,
        gamma_mask_q=gamma_mask_q,
        s0=initial_S,
        m0=initial_M,
        hS=hS,
        hM=hM,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT
    )

    return hS, hM


def chunk_momentum_delta_rule_fwd_inter_qS_qM(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        log_a_cum: torch.Tensor,
        log_mu_cum: torch.Tensor,
        bt: torch.Tensor,
        gamma_mask_q: torch.Tensor,
        scale: float,
        initial_S: torch.Tensor | None = None,
        initial_M: torch.Tensor | None = None,
        output_final_state: bool = False,
        chunk_size: int = 64,
        save_new_value: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor,  torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = gamma_mask_q.shape[-1]

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, _, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, _, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    o_inter = torch.empty_like(v)

    final_S = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    final_M = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(v) if save_new_value else None

    def grid(meta): return (triton.cdiv(V, meta['BV']), N * H)
    chunk_momentum_delta_rule_fwd_kernel_inter_qh_blockdim64[grid](
        q=q,
        k=k,
        u=u,  # u y z for recomputing v_new
        y=y,
        z=z,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        bt=bt,
        gamma_mask_q=gamma_mask_q,
        s0=initial_S,
        m0=initial_M,
        v_new=v_new,
        o_inter=o_inter,
        st=final_S,
        mt=final_M,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT
    )
    return o_inter, v_new, final_S, final_M


def chunk_momentum_delta_rule_bwd_dhu(
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        log_mu_cum: torch.Tensor,
        log_a_cum: torch.Tensor,
        bt: torch.Tensor,
        gamma_mask_q: torch.Tensor,
        s0: torch.Tensor,
        m0: torch.Tensor,
        dst: torch.Tensor | None,
        dmt: torch.Tensor | None,
        do: torch.Tensor,
        dv: torch.Tensor,
        scale: float,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = gamma_mask_q.shape[-1]
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    ds = q.new_empty(B, NT, H, K, V)
    dm = q.new_empty(B, NT, H, K, V)
    ds0 = torch.empty_like(s0, dtype=torch.float32) if s0 is not None else None
    dm0 = torch.empty_like(m0, dtype=torch.float32) if m0 is not None else None

    dv2 = torch.empty_like(dv)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), N * H)

    chunk_momentum_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        u=u,
        y=y,
        z=z,
        log_mu_cum=log_mu_cum,
        log_a_cum=log_a_cum,
        bt=bt,
        gamma_mask_q=gamma_mask_q,
        # eta=eta,
        dst=dst,
        dmt=dmt,
        ds0=ds0,
        dm0=dm0,
        do=do,
        ds=ds,
        dm=dm,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return ds, dm, ds0, dm0, dv2
