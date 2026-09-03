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
        for BV in ([32] if IS_NVIDIA_HOPPER else [16, 32, 64])
        for warps in [2, 4]
        for stages in ([1] if IS_NVIDIA_HOPPER else [2, 3])
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_mode_rule_fwd_kernel_inter_qh_blockdim64(
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
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos = (i_n * T).to(tl.int64)
        eos = bos + T
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
    if USE_INITIAL_S:
        o_k = tl.arange(0, 64)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_s0 = s0 + o_k[:, None] * V + o_v[None, :]
        b_s1 += tl.load(p_s0, mask=mask_kv, other=0.0).to(tl.float32)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kv2 = mask_k2[:, None] & mask_v[None, :]
            p_s0_2 = s0 + o_k2[:, None] * V + o_v[None, :]
            b_s2 += tl.load(p_s0_2, mask=mask_kv2, other=0.0).to(tl.float32)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kv3 = mask_k3[:, None] & mask_v[None, :]
            p_s0_3 = s0 + o_k3[:, None] * V + o_v[None, :]
            b_s3 += tl.load(p_s0_3, mask=mask_kv3, other=0.0).to(tl.float32)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kv4 = mask_k4[:, None] & mask_v[None, :]
            p_s0_4 = s0 + o_k4[:, None] * V + o_v[None, :]
            b_s4 += tl.load(p_s0_4, mask=mask_kv4, other=0.0).to(tl.float32)

    if USE_INITIAL_M:
        o_k = tl.arange(0, 64)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_m0 = m0 + o_k[:, None] * V + o_v[None, :]
        b_m1 += tl.load(p_m0, mask=mask_kv, other=0.0).to(tl.float32)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kv2 = mask_k2[:, None] & mask_v[None, :]
            p_m0_2 = m0 + o_k2[:, None] * V + o_v[None, :]
            b_m2 += tl.load(p_m0_2, mask=mask_kv2, other=0.0).to(tl.float32)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kv3 = mask_k3[:, None] & mask_v[None, :]
            p_m0_3 = m0 + o_k3[:, None] * V + o_v[None, :]
            b_m3 += tl.load(p_m0_3, mask=mask_kv3, other=0.0).to(tl.float32)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kv4 = mask_k4[:, None] & mask_v[None, :]
            p_m0_4 = m0 + o_k4[:, None] * V + o_v[None, :]
            b_m4 += tl.load(p_m0_4, mask=mask_kv4, other=0.0).to(tl.float32)

    # main recurrence, NT is number of chunks, i_t means the i_t-th chunk
    for i_t in range(NT):
        b_s1_pre, b_m1_pre = b_s1, b_m1
        if K > 64:
            b_s2_pre, b_m2_pre = b_s2, b_m2
        if K > 128:
            b_s3_pre, b_m3_pre = b_s3, b_m3
        if K > 192:
            b_s4_pre, b_m4_pre = b_s4, b_m4

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        o_t_1d = i_t * BT + tl.arange(0, BT)
        mask_t_1d = o_t_1d < T
        p_log_a = log_a_cum + bos * H + i_h + o_t_1d * H
        p_bt_1d = bt + bos * H + i_h + o_t_1d * H
        b_a_cum = tl.exp(tl.load(p_log_a, mask=mask_t_1d, other=0.0))[:, None]
        b_bt = tl.load(p_bt_1d, mask=mask_t_1d, other=0.0)[:, None]

        b_o_inter = tl.zeros([BT, BV], dtype=tl.float32)

        # Computing new (pseudo) value: v_c = u_c[:, :, i] - y_c[:, :, i] @ S_pre + z_c[:, :, i] @ M_pre
        o_t = i_t * BT + tl.arange(0, BT)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_t = o_t < T
        mask_v = o_v < V
        mask_tv = mask_t[:, None] & mask_v[None, :]
        p_u = u + o_t[:, None] * stride_v + o_v[None, :]
        b_v_new = tl.load(p_u, mask=mask_tv, other=0.0)

        o_k = tl.arange(0, 64)
        mask_k = o_k < K
        mask_tk = mask_t[:, None] & mask_k[None, :]
        p_y = y + o_t[:, None] * stride_k + o_k[None, :]
        p_z = z + o_t[:, None] * stride_k + o_k[None, :]
        p_q = q + o_t[:, None] * stride_k + o_k[None, :]
        b_y = tl.load(p_y, mask=mask_tk, other=0.0)
        b_z = tl.load(p_z, mask=mask_tk, other=0.0)
        b_q = tl.load(p_q, mask=mask_tk, other=0.0)    # [BT, BK]

        b_v_new += - tl.dot(b_y, b_s1_pre.to(b_y.dtype)) + tl.dot(b_z, b_m1_pre.to(b_z.dtype))

        b_btq = b_bt * b_q                   # [BT, BK]
        b_baraq = b_a_cum * b_q                # [BT, BK]
        b_o_inter += tl.dot(b_baraq.to(b_q.dtype), b_s1_pre.to(b_q.dtype)) - \
            tl.dot(b_btq.to(b_q.dtype), b_m1_pre.to(b_q.dtype))
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_tk2 = mask_t[:, None] & mask_k2[None, :]
            p_y2 = y + o_t[:, None] * stride_k + o_k2[None, :]
            p_z2 = z + o_t[:, None] * stride_k + o_k2[None, :]
            p_q2 = q + o_t[:, None] * stride_k + o_k2[None, :]
            b_y = tl.load(p_y2, mask=mask_tk2, other=0.0)
            b_z = tl.load(p_z2, mask=mask_tk2, other=0.0)
            b_q = tl.load(p_q2, mask=mask_tk2, other=0.0)

            b_v_new += - tl.dot(b_y, b_s2_pre.to(b_y.dtype)) + tl.dot(b_z, b_m2_pre.to(b_z.dtype))

            b_btq = b_bt * b_q                   # [BT, BK]
            b_baraq = b_a_cum * b_q                # [BT, BK]
            b_o_inter += tl.dot(b_baraq.to(b_q.dtype), b_s2_pre.to(b_q.dtype)) - \
                tl.dot(b_btq.to(b_q.dtype), b_m2_pre.to(b_q.dtype))
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_tk3 = mask_t[:, None] & mask_k3[None, :]
            p_y3 = y + o_t[:, None] * stride_k + o_k3[None, :]
            p_z3 = z + o_t[:, None] * stride_k + o_k3[None, :]
            p_q3 = q + o_t[:, None] * stride_k + o_k3[None, :]
            b_y = tl.load(p_y3, mask=mask_tk3, other=0.0)
            b_z = tl.load(p_z3, mask=mask_tk3, other=0.0)
            b_q = tl.load(p_q3, mask=mask_tk3, other=0.0)
            b_v_new += - tl.dot(b_y, b_s3_pre.to(b_q.dtype)) + tl.dot(b_z, b_m3_pre.to(b_z.dtype))

            b_btq = b_bt * b_q                   # [BT, BK]
            b_baraq = b_a_cum * b_q                # [BT, BK]
            b_o_inter += tl.dot(b_baraq.to(b_q.dtype), b_s3_pre.to(b_q.dtype)) - \
                tl.dot(b_btq.to(b_q.dtype), b_m3_pre.to(b_q.dtype))
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_tk4 = mask_t[:, None] & mask_k4[None, :]
            p_y4 = y + o_t[:, None] * stride_k + o_k4[None, :]
            p_z4 = z + o_t[:, None] * stride_k + o_k4[None, :]
            p_q4 = q + o_t[:, None] * stride_k + o_k4[None, :]
            b_y = tl.load(p_y4, mask=mask_tk4, other=0.0)
            b_z = tl.load(p_z4, mask=mask_tk4, other=0.0)
            b_q = tl.load(p_q4, mask=mask_tk4, other=0.0)
            b_v_new += - tl.dot(b_y, b_s4_pre.to(b_y.dtype)) + tl.dot(b_z, b_m4_pre.to(b_z.dtype))

            b_btq = b_bt * b_q                   # [BT, BK]
            b_baraq = b_a_cum * b_q                # [BT, BK]
            b_o_inter += tl.dot(b_baraq.to(b_q.dtype), b_s4_pre.to(b_q.dtype)) - \
                tl.dot(b_btq.to(b_q.dtype), b_m4_pre.to(b_q.dtype))

        # Storing new (pseudo) value and b_o_inter
        if SAVE_NEW_VALUE:
            p_v_new = v_new + o_t[:, None] * stride_v + o_v[None, :]
            tl.store(p_v_new, b_v_new.to(tl.float32), mask=mask_tv)
        p_o_inter = o_inter + o_t[:, None] * stride_v + o_v[None, :]
        tl.store(p_o_inter, (b_o_inter * scale).to(tl.float32), mask=mask_tv)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_log_mcum_last = tl.load(log_mu_cum + bos * H + last_idx * H + i_h)
        b_log_acum_last = tl.load(log_a_cum + bos * H + last_idx * H + i_h)
        b_bt_last = tl.load(bt + bos * H + last_idx * H + i_h)
        #  access last raw
        base_plane = gamma_mask_q + (bos * H + i_h) * BT  # (T, BT)
        row_stride = BT * H
        row_base = base_plane + last_idx * row_stride
        o_bt = tl.arange(0, BT)
        mask_bt = o_bt < BT
        p_last_row = row_base + o_bt
        b_gamma_last_row = tl.load(p_last_row, mask=mask_bt, other=0.0)

        o_t_1d_2 = i_t * BT + tl.arange(0, BT)
        mask_t_1d_2 = o_t_1d_2 < T
        p_log_mcum = log_mu_cum + bos * H + i_h + o_t_1d_2 * H
        b_log_mcum = tl.load(p_log_mcum, mask=mask_t_1d_2, other=0.0)

        mask_t = (i_t * BT + tl.arange(0, BT)) < T

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
        o_k = tl.arange(0, 64)
        o_t_k = i_t * BT + tl.arange(0, BT)
        mask_k = o_k < K
        mask_t_k = o_t_k < T
        mask_kt = mask_k[:, None] & mask_t_k[None, :]
        p_k = k + o_k[:, None] * 1 + o_t_k[None, :] * stride_k
        b_k = tl.load(p_k, mask=mask_kt, other=0.0)
        b_s1 = b_acum_last * b_s1_pre - b_bt_last * b_m1_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype))
        b_m1 = b_mcum_last * b_m1_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype))
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kt2 = mask_k2[:, None] & mask_t_k[None, :]
            p_k2 = k + o_k2[:, None] * 1 + o_t_k[None, :] * stride_k
            b_k = tl.load(p_k2, mask=mask_kt2, other=0.0)
            b_s2 = b_acum_last * b_s2_pre - b_bt_last * b_m2_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype))
            b_m2 = b_mcum_last * b_m2_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype))
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kt3 = mask_k3[:, None] & mask_t_k[None, :]
            p_k3 = k + o_k3[:, None] * 1 + o_t_k[None, :] * stride_k
            b_k = tl.load(p_k3, mask=mask_kt3, other=0.0)
            b_s3 = b_acum_last * b_s3_pre - b_bt_last * b_m3_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype))
            b_m3 = b_mcum_last * b_m3_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype))
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kt4 = mask_k4[:, None] & mask_t_k[None, :]
            p_k4 = k + o_k4[:, None] * 1 + o_t_k[None, :] * stride_k
            b_k = tl.load(p_k4, mask=mask_kt4, other=0.0)
            b_s4 = b_acum_last * b_s4_pre - b_bt_last * b_m4_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype))
            b_m4 = b_mcum_last * b_m4_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype))

    # epilogue
    if STORE_FINAL_S:
        o_k = tl.arange(0, 64)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_st = st + o_k[:, None] * V + o_v[None, :]
        tl.store(p_st, b_s1.to(tl.float32), mask=mask_kv)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kv2 = mask_k2[:, None] & mask_v[None, :]
            p_st2 = st + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_st2, b_s2.to(tl.float32), mask=mask_kv2)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kv3 = mask_k3[:, None] & mask_v[None, :]
            p_st3 = st + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_st3, b_s3.to(tl.float32), mask=mask_kv3)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kv4 = mask_k4[:, None] & mask_v[None, :]
            p_st4 = st + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_st4, b_s4.to(tl.float32), mask=mask_kv4)

    if STORE_FINAL_M:
        o_k = tl.arange(0, 64)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_mt = mt + o_k[:, None] * V + o_v[None, :]
        tl.store(p_mt, b_m1.to(tl.float32), mask=mask_kv)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kv2 = mask_k2[:, None] & mask_v[None, :]
            p_mt2 = mt + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_mt2, b_m2.to(tl.float32), mask=mask_kv2)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kv3 = mask_k3[:, None] & mask_v[None, :]
            p_mt3 = mt + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_mt3, b_m3.to(tl.float32), mask=mask_kv3)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kv4 = mask_k4[:, None] & mask_v[None, :]
            p_mt4 = mt + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_mt4, b_m4.to(tl.float32), mask=mask_kv4)


@triton.heuristics({
    'USE_INITIAL_S': lambda args: args['ds0'] is not None,
    'USE_INITIAL_M': lambda args: args['dm0'] is not None,
    'USE_FINAL_S_GRADIENT': lambda args: args['dst'] is not None,
    'USE_FINAL_M_GRADIENT': lambda args: args['dmt'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=warps, num_stages=stages)
        for BV in ([32] if IS_NVIDIA_HOPPER else [16, 32, 64])
        for warps in [2, 4]
        for stages in ([1] if IS_NVIDIA_HOPPER else [2, 3])
    ],
    key=['H', 'K', 'V', 'BT', 'BV',
         ],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_mode_rule_bwd_kernel_dhu_blockdim64(
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
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos = (i_n * T).to(tl.int64)
        eos = bos + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

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
        o_k = tl.arange(0, 64)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_dst = dst + o_k[:, None] * V + o_v[None, :]
        b_ds1 += tl.load(p_dst, mask=mask_kv, other=0.0).to(tl.float32)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kv2 = mask_k2[:, None] & mask_v[None, :]
            p_dst2 = dst + o_k2[:, None] * V + o_v[None, :]
            b_ds2 += tl.load(p_dst2, mask=mask_kv2, other=0.0).to(tl.float32)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kv3 = mask_k3[:, None] & mask_v[None, :]
            p_dst3 = dst + o_k3[:, None] * V + o_v[None, :]
            b_ds3 += tl.load(p_dst3, mask=mask_kv3, other=0.0).to(tl.float32)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kv4 = mask_k4[:, None] & mask_v[None, :]
            p_dst4 = dst + o_k4[:, None] * V + o_v[None, :]
            b_ds4 += tl.load(p_dst4, mask=mask_kv4, other=0.0).to(tl.float32)

    if USE_FINAL_M_GRADIENT:
        dmt += i_nh * K * V
        o_k = tl.arange(0, 64)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_dmt = dmt + o_k[:, None] * V + o_v[None, :]
        b_dm1 += tl.load(p_dmt, mask=mask_kv, other=0.0).to(tl.float32)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kv2 = mask_k2[:, None] & mask_v[None, :]
            p_dmt2 = dmt + o_k2[:, None] * V + o_v[None, :]
            b_dm2 += tl.load(p_dmt2, mask=mask_kv2, other=0.0).to(tl.float32)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kv3 = mask_k3[:, None] & mask_v[None, :]
            p_dmt3 = dmt + o_k3[:, None] * V + o_v[None, :]
            b_dm3 += tl.load(p_dmt3, mask=mask_kv3, other=0.0).to(tl.float32)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kv4 = mask_k4[:, None] & mask_v[None, :]
            p_dmt4 = dmt + o_k4[:, None] * V + o_v[None, :]
            b_dm4 += tl.load(p_dmt4, mask=mask_kv4, other=0.0).to(tl.float32)

    # main recurrence, NT is number of chunks, i_t means the i_t-th chunk
    for i_t in range(NT - 1, -1, -1):
        # Storing last State gradients
        base_ds = ds + i_t * stride_h
        base_dm = dm + i_t * stride_h
        o_k = tl.arange(0, 64)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_ds1 = base_ds + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ds1, b_ds1.to(tl.float32), mask=mask_kv)
        p_dm1 = base_dm + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dm1, b_dm1.to(tl.float32), mask=mask_kv)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kv2 = mask_k2[:, None] & mask_v[None, :]
            p_ds2 = base_ds + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_ds2, b_ds2.to(tl.float32), mask=mask_kv2)
            p_dm2 = base_dm + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_dm2, b_dm2.to(tl.float32), mask=mask_kv2)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kv3 = mask_k3[:, None] & mask_v[None, :]
            p_ds3 = base_ds + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_ds3, b_ds3.to(tl.float32), mask=mask_kv3)
            p_dm3 = base_dm + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_dm3, b_dm3.to(tl.float32), mask=mask_kv3)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kv4 = mask_k4[:, None] & mask_v[None, :]
            p_ds4 = base_ds + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_ds4, b_ds4.to(tl.float32), mask=mask_kv4)
            p_dm4 = base_dm + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_dm4, b_dm4.to(tl.float32), mask=mask_kv4)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_log_mcum_last = tl.load(log_mu_cum + last_idx * H)
        b_log_acum_last = tl.load(log_a_cum + last_idx * H)
        b_bt_last = tl.load(bt + last_idx * H)
        #  access last raw

        row_base = gamma_mask_q + last_idx * (BT * H)
        o_bt = tl.arange(0, BT)
        mask_bt = o_bt < BT
        p_last_row = row_base + o_bt
        b_gamma_last_row = tl.load(p_last_row, mask=mask_bt, other=0.0)

        o_t_1d = i_t * BT + tl.arange(0, BT)
        mask_t_1d = o_t_1d < T
        p_log_acum = log_a_cum + o_t_1d * H
        p_log_mcum = log_mu_cum + o_t_1d * H
        p_bt_1d = bt + o_t_1d * H

        b_bt = tl.load(p_bt_1d, mask=mask_t_1d, other=0.0)
        b_log_mcum = tl.load(p_log_mcum, mask=mask_t_1d, other=0.0)
        b_log_acum = tl.load(p_log_acum, mask=mask_t_1d, other=0.0)

        b_decay_s = b_gamma_last_row                        # [BT]
        b_decay_m = tl.exp(b_log_mcum_last - b_log_mcum)    # [BT]

        b_mcum_last = tl.exp(b_log_mcum_last)
        b_acum_last = tl.exp(b_log_acum_last)

        o_t = i_t * BT + tl.arange(0, BT)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_t = o_t < T
        mask_v = o_v < V
        mask_tv = mask_t[:, None] & mask_v[None, :]
        p_do = do + o_t[:, None] * stride_v + o_v[None, :]
        b_do = tl.load(p_do, mask=mask_tv, other=0.0)

        # Update dv
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)
        o_t_k = i_t * BT + tl.arange(0, BT)
        o_k = tl.arange(0, 64)
        mask_t_k = o_t_k < T
        mask_k = o_k < K
        mask_tk = mask_t_k[:, None] & mask_k[None, :]
        # for dv we need (T,K) layout: p = k + o_t*stride_k + o_k
        p_k = k + o_t_k[:, None] * stride_k + o_k[None, :]
        b_k = tl.load(p_k, mask=mask_tk, other=0.0)
        b_k_decay_s = b_k * b_decay_s[:, None]   # [BT,BK]
        b_k_decay_m = b_k * b_decay_m[:, None]   # [BT,BK]
        b_dv += tl.dot(b_k_decay_s.to(b_k.dtype), b_ds1.to(b_k.dtype)) - tl.dot(b_k_decay_m.to(b_k.dtype), b_dm1.to(b_k.dtype))
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_tk2 = mask_t_k[:, None] & mask_k2[None, :]
            p_k2 = k + o_t_k[:, None] * stride_k + o_k2[None, :]
            b_k = tl.load(p_k2, mask=mask_tk2, other=0.0)
            b_k_decay_s = b_k * b_decay_s[:, None]  # [BT,BK]
            b_k_decay_m = b_k * b_decay_m[:, None]  # [BT,BK]
            b_dv += tl.dot(b_k_decay_s.to(b_k.dtype), b_ds2.to(b_k.dtype)) - \
                tl.dot(b_k_decay_m.to(b_k.dtype), b_dm2.to(b_k.dtype))
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_tk3 = mask_t_k[:, None] & mask_k3[None, :]
            p_k3 = k + o_t_k[:, None] * stride_k + o_k3[None, :]
            b_k = tl.load(p_k3, mask=mask_tk3, other=0.0)
            b_k_decay_s = b_k * b_decay_s[:, None]  # [BT,BK]
            b_k_decay_m = b_k * b_decay_m[:, None]  # [BT,BK]
            b_dv += tl.dot(b_k_decay_s.to(b_k.dtype), b_ds3.to(b_k.dtype)) - \
                tl.dot(b_k_decay_m.to(b_k.dtype), b_dm3.to(b_k.dtype))
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_tk4 = mask_t_k[:, None] & mask_k4[None, :]
            p_k4 = k + o_t_k[:, None] * stride_k + o_k4[None, :]
            b_k = tl.load(p_k4, mask=mask_tk4, other=0.0)
            b_k_decay_s = b_k * b_decay_s[:, None]  # [BT,BK]
            b_k_decay_m = b_k * b_decay_m[:, None]  # [BT,BK]
            b_dv += tl.dot(b_k_decay_s.to(b_k.dtype), b_ds4.to(b_k.dtype)) - \
                tl.dot(b_k_decay_m.to(b_k.dtype), b_dm4.to(b_k.dtype))

        # here dL/dv_t is part of dL/dot * dot/dvt
        p_dv = dv + o_t[:, None] * stride_v + o_v[None, :]
        b_dv += tl.load(p_dv, mask=mask_tv, other=0.0)

        # store dv2
        # here complete dL/dvt = dL/dSt * dS_t/dvt + dL/dMt * dM_t/dvt + dL/dot * dot/dvt
        p_dv2 = dv2 + o_t[:, None] * stride_v + o_v[None, :]
        tl.store(p_dv2, b_dv.to(tl.float32), mask=mask_tv)

        # Update ds, dh, ref name is corresponding to the /fla/test/momentum_delta_net.py
        # transposed loads (K,T)
        o_k_t = tl.arange(0, 64)
        o_t_t = i_t * BT + tl.arange(0, BT)
        mask_k_t = o_k_t < K
        mask_t_t = o_t_t < T
        mask_kt_t = mask_k_t[:, None] & mask_t_t[None, :]
        p_y = y + o_k_t[:, None] * 1 + o_t_t[None, :] * stride_k
        p_z = z + o_k_t[:, None] * 1 + o_t_t[None, :] * stride_k
        p_q = q + o_k_t[:, None] * 1 + o_t_t[None, :] * stride_k
        b_y = tl.load(p_y, mask=mask_kt_t, other=0.0)
        b_z = tl.load(p_z, mask=mask_kt_t, other=0.0)
        b_q = tl.load(p_q, mask=mask_kt_t, other=0.0) * scale

        b_ds1_pre = b_acum_last * b_ds1 \
            + tl.dot((tl.exp(b_log_acum)[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype)) \
            - tl.dot(b_y, b_dv.to(b_y.dtype))

        b_dm1_pre = -b_bt_last * b_ds1 + b_mcum_last * b_dm1 \
                    - tl.dot((b_bt[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype)) \
            + tl.dot(b_z, b_dv.to(b_z.dtype))

        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kt2 = mask_k2[:, None] & mask_t_t[None, :]
            p_y2 = y + o_k2[:, None] * 1 + o_t_t[None, :] * stride_k
            p_z2 = z + o_k2[:, None] * 1 + o_t_t[None, :] * stride_k
            p_q2 = q + o_k2[:, None] * 1 + o_t_t[None, :] * stride_k
            b_y = tl.load(p_y2, mask=mask_kt2, other=0.0)
            b_z = tl.load(p_z2, mask=mask_kt2, other=0.0)
            b_q = tl.load(p_q2, mask=mask_kt2, other=0.0) * scale

            b_ds2_pre = b_acum_last * b_ds2 \
                + tl.dot((tl.exp(b_log_acum)[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype)) \
                - tl.dot(b_y, b_dv.to(b_y.dtype))

            b_dm2_pre = -b_bt_last * b_ds2 + b_mcum_last * b_dm2 \
                        - tl.dot((b_bt[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype)) \
                + tl.dot(b_z, b_dv.to(b_z.dtype))

        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kt3 = mask_k3[:, None] & mask_t_t[None, :]
            p_y3 = y + o_k3[:, None] * 1 + o_t_t[None, :] * stride_k
            p_z3 = z + o_k3[:, None] * 1 + o_t_t[None, :] * stride_k
            p_q3 = q + o_k3[:, None] * 1 + o_t_t[None, :] * stride_k
            b_y = tl.load(p_y3, mask=mask_kt3, other=0.0)
            b_z = tl.load(p_z3, mask=mask_kt3, other=0.0)
            b_q = tl.load(p_q3, mask=mask_kt3, other=0.0) * scale

            b_ds3_pre = b_acum_last * b_ds3 \
                + tl.dot((tl.exp(b_log_acum)[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype)) \
                - tl.dot(b_y, b_dv.to(b_y.dtype))

            b_dm3_pre = -b_bt_last * b_ds3 + b_mcum_last * b_dm3 \
                        - tl.dot((b_bt[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype)) \
                + tl.dot(b_z, b_dv.to(b_z.dtype))

        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kt4 = mask_k4[:, None] & mask_t_t[None, :]
            p_y4 = y + o_k4[:, None] * 1 + o_t_t[None, :] * stride_k
            p_z4 = z + o_k4[:, None] * 1 + o_t_t[None, :] * stride_k
            p_q4 = q + o_k4[:, None] * 1 + o_t_t[None, :] * stride_k
            b_y = tl.load(p_y4, mask=mask_kt4, other=0.0)
            b_z = tl.load(p_z4, mask=mask_kt4, other=0.0)
            b_q = tl.load(p_q4, mask=mask_kt4, other=0.0) * scale

            b_ds4_pre = b_acum_last * b_ds4 \
                + tl.dot((tl.exp(b_log_acum)[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype)) \
                - tl.dot(b_y, b_dv.to(b_y.dtype))

            b_dm4_pre = -b_bt_last * b_ds4 + b_mcum_last * b_dm4 \
                        - tl.dot((b_bt[None, :] * b_q).to(b_k.dtype), b_do.to(b_k.dtype)) \
                + tl.dot(b_z, b_dv.to(b_z.dtype))

        b_ds1, b_dm1 = b_ds1_pre, b_dm1_pre
        if K > 64:
            b_ds2, b_dm2 = b_ds2_pre, b_dm2_pre
        if K > 128:
            b_ds3, b_dm3 = b_ds3_pre, b_dm3_pre
        if K > 192:
            b_ds4, b_dm4 = b_ds4_pre, b_dm4_pre

    if USE_INITIAL_S:
        o_k = tl.arange(0, 64)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_ds1 = ds0 + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ds1, b_ds1.to(tl.float32), mask=mask_kv)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kv2 = mask_k2[:, None] & mask_v[None, :]
            p_ds2 = ds0 + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_ds2, b_ds2.to(tl.float32), mask=mask_kv2)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kv3 = mask_k3[:, None] & mask_v[None, :]
            p_ds3 = ds0 + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_ds3, b_ds3.to(tl.float32), mask=mask_kv3)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kv4 = mask_k4[:, None] & mask_v[None, :]
            p_ds4 = ds0 + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_ds4, b_ds4.to(tl.float32), mask=mask_kv4)

    if USE_INITIAL_M:
        o_k = tl.arange(0, 64)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_dm1 = dm0 + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dm1, b_dm1.to(tl.float32), mask=mask_kv)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kv2 = mask_k2[:, None] & mask_v[None, :]
            p_dm2 = dm0 + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_dm2, b_dm2.to(tl.float32), mask=mask_kv2)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kv3 = mask_k3[:, None] & mask_v[None, :]
            p_dm3 = dm0 + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_dm3, b_dm3.to(tl.float32), mask=mask_kv3)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kv4 = mask_k4[:, None] & mask_v[None, :]
            p_dm4 = dm0 + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_dm4, b_dm4.to(tl.float32), mask=mask_kv4)


@triton.heuristics({
    'USE_INITIAL_S': lambda args: args['s0'] is not None,
    'USE_INITIAL_M': lambda args: args['m0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=warps, num_stages=stages)
        for BV in ([32] if IS_NVIDIA_HOPPER else [32, 64])
        for warps in [2, 4]
        for stages in ([1] if IS_NVIDIA_HOPPER else [2, 3, 4])
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_mode_rule_fwd_kernel_h_blockdim64_recompute_by_vnew(
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
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos = (i_n * T).to(tl.int64)
        eos = bos + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

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
        o_k = tl.arange(0, 64)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_s0 = s0 + o_k[:, None] * V + o_v[None, :]
        b_s1 += tl.load(p_s0, mask=mask_kv, other=0.0).to(tl.float32)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kv2 = mask_k2[:, None] & mask_v[None, :]
            p_s0_2 = s0 + o_k2[:, None] * V + o_v[None, :]
            b_s2 += tl.load(p_s0_2, mask=mask_kv2, other=0.0).to(tl.float32)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kv3 = mask_k3[:, None] & mask_v[None, :]
            p_s0_3 = s0 + o_k3[:, None] * V + o_v[None, :]
            b_s3 += tl.load(p_s0_3, mask=mask_kv3, other=0.0).to(tl.float32)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kv4 = mask_k4[:, None] & mask_v[None, :]
            p_s0_4 = s0 + o_k4[:, None] * V + o_v[None, :]
            b_s4 += tl.load(p_s0_4, mask=mask_kv4, other=0.0).to(tl.float32)

    if USE_INITIAL_M:
        o_k = tl.arange(0, 64)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_m0 = m0 + o_k[:, None] * V + o_v[None, :]
        b_m1 += tl.load(p_m0, mask=mask_kv, other=0.0).to(tl.float32)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kv2 = mask_k2[:, None] & mask_v[None, :]
            p_m0_2 = m0 + o_k2[:, None] * V + o_v[None, :]
            b_m2 += tl.load(p_m0_2, mask=mask_kv2, other=0.0).to(tl.float32)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kv3 = mask_k3[:, None] & mask_v[None, :]
            p_m0_3 = m0 + o_k3[:, None] * V + o_v[None, :]
            b_m3 += tl.load(p_m0_3, mask=mask_kv3, other=0.0).to(tl.float32)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kv4 = mask_k4[:, None] & mask_v[None, :]
            p_m0_4 = m0 + o_k4[:, None] * V + o_v[None, :]
            b_m4 += tl.load(p_m0_4, mask=mask_kv4, other=0.0).to(tl.float32)

    # main recurrence, NT is number of chunks, i_t means the i_t-th chunk
    for i_t in range(NT):
        b_s1_pre, b_m1_pre = b_s1, b_m1
        if K > 64:
            b_s2_pre, b_m2_pre = b_s2, b_m2
        if K > 128:
            b_s3_pre, b_m3_pre = b_s3, b_m3
        if K > 192:
            b_s4_pre, b_m4_pre = b_s4, b_m4

        # Storing Previous State
        base_hS = hS + i_t * stride_h
        base_hM = hM + i_t * stride_h
        o_k = tl.arange(0, 64)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_hS1 = base_hS + o_k[:, None] * V + o_v[None, :]
        p_hM1 = base_hM + o_k[:, None] * V + o_v[None, :]
        tl.store(p_hS1, b_s1_pre.to(tl.float32), mask=mask_kv)
        tl.store(p_hM1, b_m1_pre.to(tl.float32), mask=mask_kv)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kv2 = mask_k2[:, None] & mask_v[None, :]
            p_hS2 = base_hS + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_hS2, b_s2_pre.to(tl.float32), mask=mask_kv2)
            p_hM2 = base_hM + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_hM2, b_m2_pre.to(tl.float32), mask=mask_kv2)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kv3 = mask_k3[:, None] & mask_v[None, :]
            p_hS3 = base_hS + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_hS3, b_s3_pre.to(tl.float32), mask=mask_kv3)
            p_hM3 = base_hM + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_hM3, b_m3_pre.to(tl.float32), mask=mask_kv3)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kv4 = mask_k4[:, None] & mask_v[None, :]
            p_hS4 = base_hS + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_hS4, b_s4_pre.to(tl.float32), mask=mask_kv4)
            p_hM4 = base_hM + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_hM4, b_m4_pre.to(tl.float32), mask=mask_kv4)

        # Computing new (pseudo) value: v_c = u_c[:, :, i] - y_c[:, :, i] @ S_pre + z_c[:, :, i] @ M_pre
        o_t = i_t * BT + tl.arange(0, BT)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_t = o_t < T
        mask_v = o_v < V
        mask_tv = mask_t[:, None] & mask_v[None, :]
        p_v_new = v_new + o_t[:, None] * stride_v + o_v[None, :]
        b_v_new = tl.load(p_v_new, mask=mask_tv, other=0.0)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_log_mcum_last = tl.load(log_mu_cum + bos * H + last_idx * H + i_h)
        b_log_acum_last = tl.load(log_a_cum + bos * H + last_idx * H + i_h)
        b_bt_last = tl.load(bt + bos * H + last_idx * H + i_h)

        #  access last raw
        base_plane = gamma_mask_q + (bos * H + i_h) * BT  # (T, BT)
        row_stride = BT * H
        row_base = base_plane + last_idx * row_stride
        o_bt = tl.arange(0, BT)
        mask_bt = o_bt < BT
        p_last_row = row_base + o_bt
        b_gamma_last_row = tl.load(p_last_row, mask=mask_bt, other=0.0)

        o_t_1d = i_t * BT + tl.arange(0, BT)
        mask_t_1d = o_t_1d < T
        p_log_mcum = log_mu_cum + bos * H + i_h + o_t_1d * H
        b_log_mcum = tl.load(p_log_mcum, mask=mask_t_1d, other=0.0)

        mask_t = (i_t * BT + tl.arange(0, BT)) < T

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
        o_k = tl.arange(0, 64)
        o_t_k = i_t * BT + tl.arange(0, BT)
        mask_k = o_k < K
        mask_t_k = o_t_k < T
        mask_kt = mask_k[:, None] & mask_t_k[None, :]
        p_k = k + o_k[:, None] * 1 + o_t_k[None, :] * stride_k
        b_k = tl.load(p_k, mask=mask_kt, other=0.0)
        b_s1 = b_acum_last * b_s1_pre - b_bt_last * b_m1_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype))
        b_m1 = b_mcum_last * b_m1_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype))
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            mask_k2 = o_k2 < K
            mask_kt2 = mask_k2[:, None] & mask_t_k[None, :]
            p_k2 = k + o_k2[:, None] * 1 + o_t_k[None, :] * stride_k
            b_k = tl.load(p_k2, mask=mask_kt2, other=0.0)
            b_s2 = b_acum_last * b_s2_pre - b_bt_last * b_m2_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype))
            b_m2 = b_mcum_last * b_m2_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype))
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            mask_k3 = o_k3 < K
            mask_kt3 = mask_k3[:, None] & mask_t_k[None, :]
            p_k3 = k + o_k3[:, None] * 1 + o_t_k[None, :] * stride_k
            b_k = tl.load(p_k3, mask=mask_kt3, other=0.0)
            b_s3 = b_acum_last * b_s3_pre - b_bt_last * b_m3_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype))
            b_m3 = b_mcum_last * b_m3_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype))
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            mask_k4 = o_k4 < K
            mask_kt4 = mask_k4[:, None] & mask_t_k[None, :]
            p_k4 = k + o_k4[:, None] * 1 + o_t_k[None, :] * stride_k
            b_k = tl.load(p_k4, mask=mask_kt4, other=0.0)
            b_s4 = b_acum_last * b_s4_pre - b_bt_last * b_m4_pre + tl.dot(b_k, b_v_new_s.to(b_k.dtype))
            b_m4 = b_mcum_last * b_m4_pre - tl.dot(b_k, b_v_new_m.to(b_k.dtype))


def chunk_mode_rule_fwd_h_recompute_by_vnew(
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
    chunk_mode_rule_fwd_kernel_h_blockdim64_recompute_by_vnew[grid](
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


def chunk_mode_rule_fwd_inter_qS_qM(
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

    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, chunk_offsets = B, None
    else:
        N, chunk_offsets = len(cu_seqlens) - 1, prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    o_inter = torch.empty_like(v)

    final_S = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    final_M = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(v) if save_new_value else None

    def grid(meta): return (triton.cdiv(V, meta['BV']), N * H)
    chunk_mode_rule_fwd_kernel_inter_qh_blockdim64[grid](
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


def chunk_mode_rule_bwd_dhu(
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

    chunk_mode_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        u=u,
        y=y,
        z=z,
        log_mu_cum=log_mu_cum,
        log_a_cum=log_a_cum,
        bt=bt,
        gamma_mask_q=gamma_mask_q,
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
