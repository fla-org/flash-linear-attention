# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Copyright (c) 2026, Pieter-Jan Hoedt

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
    pytorch_matmul_config,
)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs
)
@triton.jit(do_not_specialize=['T'])
def chunk_mlstm_fwd_kernel_state(
    k,
    v,
    log_i,
    log_f,
    c0,
    n0,
    m0,
    c_out,
    n_out,
    m_out,
    cs,
    ns,
    ms,
    cu_seqlens,
    split_offsets,
    qk_scale: float,
    T: int,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    DO_NORMALISE: tl.constexpr,
    FP32_PREC: tl.constexpr,
):
    tl.static_assert(BK >= K, "not yet implemented")
    tl.static_assert(BV >= V, "not yet implemented")

    ACC_DTYPE: tl.constexpr = c0.type.element_ty
    F_MASK = tl.arange(0, BT) != BT - 1

    pid_b = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1).to(tl.int64)

    if IS_VARLEN:
        offsets_t = tl.load(cu_seqlens + pid_b + tl.arange(0, 2)).to(tl.int64)
        start_t, end_t = tl.split(offsets_t)
        T = end_t - start_t
        NT = tl.cdiv(T, BT)
        pid_bt = start_t
        pid_bt_state = tl.load(split_offsets + pid_b).to(tl.int64)
    else:
        start_t, end_t = 0, T
        NT = tl.cdiv(T, BT)
        pid_bt = pid_b * T
        pid_bt_state = pid_b * NT

    # input pointers
    o_1 = tl.arange(0, 1).to(tl.int64)
    o_k = tl.arange(0, BK).to(tl.int64)
    o_t = tl.arange(0, BT).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    kT_ptr = k + (pid_bt * H + pid_h) * K + o_k[:, None] + o_t[None, :] * (H * K)
    v_ptr = v + (pid_bt * H + pid_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    log_i_ptr = log_i + pid_bt * H + pid_h + o_t * H
    log_f_ptr = log_f + pid_bt * H + pid_h + o_t * H
    c_ptr = c0 + (pid_b * H + pid_h) * K * V + o_k[:, None] * V + o_v[None, :]
    m_ptr = m0 + pid_b * H + pid_h

    # output pointers
    cs_ptr = cs + (pid_bt_state * H + pid_h) * V * K + o_k[None, :, None] * V + o_v[None, None, :]
    ms_ptr = ms + pid_bt_state * H + pid_h + o_1
    if STORE_FINAL_STATE:
        c_out_ptr = c_out + (pid_b * H + pid_h) * K * V + o_k[:, None] * V + o_v[None, :]
        m_out_ptr = m_out + pid_b * H + pid_h

    if DO_NORMALISE:
        # additional input pointer
        n_ptr = n0 + (pid_b * H + pid_h) * K + o_k

        # additional output pointers
        ns_ptr = ns + (pid_bt_state * H + pid_h) * K + o_k[None, :]
        if STORE_FINAL_STATE:
            n_out_ptr = n_out + (pid_b * H + pid_h) * K + o_k

    qk_scale = tl.cast(qk_scale, dtype=k.type.element_ty)
    c_vals = tl.load(c_ptr, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0)  # element-wise
    m_vals = tl.load(m_ptr)
    if DO_NORMALISE:
        n_vals = tl.load(n_ptr, mask=(o_k < K), other=0)  # element-wise
    for offset_t in range(start_t, end_t, BT):
        tl.store(cs_ptr, c_vals[None, :, :].to(cs.type.element_ty), mask=(o_k[None, :, None] < K) & (o_v[None, None, :] < V))
        tl.store(ms_ptr, m_vals[None].to(ms.type.element_ty))

        i_vals = tl.load(log_i_ptr, mask=(offset_t - start_t + o_t < T), other=0)  # masked elsewhere
        i_vals = tl.where(tl.arange(0, BT) < end_t - offset_t, i_vals, -float("inf"))
        f_vals_shifted = tl.load(log_f_ptr + H, mask=(offset_t - start_t + 1 + o_t < T), other=0)
        f_vals_shifted = tl.where(F_MASK, f_vals_shifted, 0)
        log_prods_f = tl.cumsum(f_vals_shifted, axis=0, reverse=True)
        log_scale = i_vals + log_prods_f
        mc = tl.max(log_scale).to(log_scale.type.element_ty)

        f_vals = tl.load(log_f_ptr, mask=(offset_t - start_t + o_t < T), other=0)
        log_prod_f = tl.sum(f_vals)
        m_and_f = log_prod_f + m_vals
        m_vals = tl.maximum(mc, m_and_f)
        kt_vals = tl.load(kT_ptr, mask=(o_k[:, None] < K) & ((offset_t - start_t + o_t)[None, :] < T), other=0)
        k_gated = qk_scale * kt_vals * tl.exp(tl.cast(log_scale - m_vals, tl.float32)).to(kt_vals.type.element_ty)
        f_rec = tl.exp(tl.cast(m_and_f - m_vals, tl.float32)).to(c_vals.type.element_ty)

        v_vals = tl.load(v_ptr, mask=((offset_t - start_t + o_t)[:, None] < T) & (o_v[None, :] < V), other=0)
        c_vals *= f_rec
        c_vals = tl.dot(k_gated, v_vals, c_vals.to(ACC_DTYPE), out_dtype=c_vals.type.element_ty, input_precision=FP32_PREC)
        if DO_NORMALISE:
            tl.store(ns_ptr, n_vals[None, :].to(ns.type.element_ty), mask=(o_k[None, :] < K))
            n_vals *= f_rec
            n_vals += tl.sum(k_gated, axis=1).to(n_vals.type.element_ty)
            ns_ptr += H * K

        log_i_ptr += BT * H
        log_f_ptr += BT * H
        kT_ptr += BT * H * K
        v_ptr += BT * H * V
        cs_ptr += H * V * K
        ms_ptr += H

    if STORE_FINAL_STATE:
        tl.store(c_out_ptr, c_vals.to(c_out.type.element_ty), mask=(o_k[:, None] < K) & (o_v[None, :] < V))
        tl.store(m_out_ptr, m_vals.to(m_out.type.element_ty))
        if DO_NORMALISE:
            tl.store(n_out_ptr, n_vals.to(n_out.type.element_ty), mask=(o_k < K))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=3)
        for num_warps in [2, 4, 8]
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_mlstm_fwd_kernel_out(
    q,
    k,
    v,
    log_i,
    log_f,
    cs,
    ns,
    ms,
    h,
    z,
    f_all,
    m_all_out,
    cu_seqlens,
    chunk_indices,
    qk_scale: float,
    eps: float,
    T: int,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MAX_NORMALISATION: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    FP32_PREC: tl.constexpr,
):
    tl.static_assert(BK >= K, "not yet implemented")
    tl.static_assert(BV >= V, "not yet implemented")

    BLK_IDX = tl.arange(0, BT)

    pid_b = tl.program_id(axis=0).to(tl.int64)
    pid_n = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2).to(tl.int64)

    if IS_VARLEN:
        pid_bn = pid_n
        layout_n = tl.load(chunk_indices + 2 * pid_n + tl.arange(0, 2)).to(tl.int64)
        pid_b, pid_n = tl.split(layout_n)
        offsets_t = tl.load(cu_seqlens + pid_b + tl.arange(0, 2)).to(tl.int64)
        start_t, end_t = tl.split(offsets_t)
        T = end_t - start_t
        pid_bt = start_t
    else:
        start_t, end_t = 0, T
        pid_bt = pid_b * T
        pid_bn = pid_b * tl.cdiv(T, BT) + pid_n

    o_k = tl.arange(0, BK).to(tl.int64)
    o_t = tl.arange(0, BT).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    q_ptr = q + (pid_bt * H + pid_h) * K + (pid_n * BT + o_t)[:, None] * (H * K) + o_k[None, :]
    kT_ptr = k + (pid_bt * H + pid_h) * K + o_k[:, None] + (pid_n * BT + o_t)[None, :] * (H * K)
    v_ptr = v + (pid_bt * H + pid_h) * V + (pid_n * BT + o_t)[:, None] * (H * V) + o_v[None, :]
    log_i_ptr = log_i + pid_bt * H + pid_h + (pid_n * BT + o_t) * H
    log_f_ptr = log_f + pid_bt * H + pid_h + (pid_n * BT + o_t) * H
    cs_ptr = cs + (pid_bn * H + pid_h) * K * V + o_k[:, None] * V + o_v[None, :]
    ms_ptr = ms + pid_bn * H + pid_h

    # output pointers
    h_ptr = h + (pid_bt * H + pid_h) * V + (pid_n * BT + o_t)[:, None] * (H * V) + o_v[None, :]
    f_all_ptr = f_all + pid_bt * H + pid_h + (pid_n * BT + o_t) * H
    if MAX_NORMALISATION:
        m_all_out_ptr = m_all_out + pid_bt * H + pid_h + (pid_n * BT + o_t) * H

    if MAX_NORMALISATION is not None:
        # additional input pointer
        ns_ptr = ns + (pid_bn * H + pid_h) * K + o_k

        # additional output pointer
        z_ptr = z + pid_bt * H + pid_h + (pid_n * BT + o_t) * H

    # gating
    f_vals = tl.load(log_f_ptr, mask=(pid_n * BT + o_t < T), other=0)
    f_mat_tmp = tl.where(BLK_IDX[None, :] < BLK_IDX[:, None], f_vals[:, None], 0)
    f_mat = tl.cumsum(f_mat_tmp, axis=0)
    f_mat = tl.where(BLK_IDX[None, :] <= BLK_IDX[:, None], f_mat, float("-inf"))
    i_vals = tl.load(log_i_ptr, mask=(pid_n * BT + o_t < T), other=0)
    log_scale = f_mat + i_vals[None, :]

    log_prod_f = tl.cumsum(f_vals, axis=0)[:, None]
    m_val = tl.load(ms_ptr)
    m_and_f = m_val + log_prod_f
    m_local = tl.max(log_scale, axis=1, keep_dims=True).to(ACC_DTYPE)
    m_all = tl.maximum(m_and_f, m_local).to(tl.float32)

    # attention
    q_vals = tl.load(q_ptr, mask=((pid_n * BT + o_t)[:, None] < T) & (o_k[None, :] < K), other=0)
    kT_vals = tl.load(kT_ptr, mask=(o_k[:, None] < K) & ((pid_n * BT + o_t)[None, :] < T), other=0)
    # Query-key scores determine the normalizer reused by backward and require FP32 accumulation.
    qk = tl.dot(q_vals, kT_vals, out_dtype=tl.float32, input_precision=FP32_PREC)
    e = tl.exp(log_scale - m_all).to(q_vals.type.element_ty) * (qk_scale * qk)

    v_vals = tl.load(v_ptr, mask=((pid_n * BT + o_t)[:, None] < T) & (o_v[None, :] < V), other=0)
    s_vals = tl.dot(
        e.to(v.dtype.element_ty), v_vals,
        out_dtype=ACC_DTYPE, input_precision=FP32_PREC
    )

    f_all_vals = tl.exp(m_and_f - m_all).to(q_vals.type.element_ty)
    q_fgate = f_all_vals * q_vals
    c_vals = tl.load(cs_ptr, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0)
    s_vals = tl.dot(q_fgate, c_vals, s_vals, out_dtype=s_vals.type.element_ty, input_precision=FP32_PREC)

    h_vals = s_vals
    if MAX_NORMALISATION is not None:
        z_vals = tl.sum(e, axis=1, keep_dims=True)
        n_vals = tl.load(ns_ptr, mask=(o_k < K), other=0)
        z_vals += tl.sum(q_fgate * n_vals[None, :], axis=1, keep_dims=True)

        eps = tl.cast(eps, dtype=ACC_DTYPE)
        if MAX_NORMALISATION:
            _z = tl.maximum(tl.abs(z_vals), tl.exp(-m_all).to(ACC_DTYPE) + eps)
            # tl.static_print("m_all", m_all)
            # tl.static_print("_z", _z)
            # _z = tl.maximum(tl.abs(z_vals), 1 + eps)
        else:
            _z = z_vals + eps

        h_vals = s_vals / _z
        tl.store(z_ptr, tl.ravel(z_vals).to(z.dtype.element_ty), mask=(pid_n * BT + o_t < T))

    tl.store(h_ptr, h_vals.to(h.dtype.element_ty), mask=((pid_n * BT + o_t)[:, None] < T) & (o_v[None, :] < V))
    tl.store(f_all_ptr, tl.ravel(f_all_vals).to(f_all.dtype.element_ty), mask=(pid_n * BT + o_t < T))
    if MAX_NORMALISATION:
        tl.store(m_all_out_ptr, tl.ravel(m_all).to(m_all_out.dtype.element_ty), mask=(pid_n * BT + o_t < T))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_mlstm_bwd_kernel_dstate(
    dh,
    dc,
    dn,
    q,
    f_all,
    h,
    z,
    m_all,
    ms,
    dcs,
    dns,
    dc_out,
    dn_out,
    cu_seqlens,
    split_offsets,
    eps: float,
    T: int,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    WITH_FINAL_STATE: tl.constexpr,
    MAX_NORMALISATION: tl.constexpr,
    FP32_PREC: tl.constexpr,
):
    tl.static_assert(BK >= K, "not yet implemented")
    tl.static_assert(BV >= V, "not yet implemented")

    pid_b = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1).to(tl.int64)

    if IS_VARLEN:
        offsets_t = tl.load(cu_seqlens + pid_b + tl.arange(0, 2)).to(tl.int64)
        start_t, end_t = tl.split(offsets_t)
        T = end_t - start_t
        NT = tl.cdiv(T, BT)
        pid_bt = start_t
        pid_bt_state = tl.load(split_offsets + pid_b).to(tl.int64)
    else:
        start_t, end_t = 0, T
        NT = tl.cdiv(T, BT)
        pid_bt = pid_b * T
        pid_bt_state = pid_b * NT
    final_n = (NT - 1) * BT

    # input pointers
    o_1 = tl.arange(0, 1).to(tl.int64)
    o_k = tl.arange(0, BK).to(tl.int64)
    o_t = tl.arange(0, BT).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    dh_ptr = dh + (pid_bt * H + pid_h) * V + (final_n + o_t)[:, None] * (H * V) + o_v[None, :]
    qT_ptr = q + (pid_bt * H + pid_h) * K + o_k[:, None] + (final_n + o_t)[None, :] * (H * K)
    f_all_ptr = f_all + pid_bt * H + pid_h + (final_n + o_t) * H

    # output pointers
    dcs_ptr = (
        dcs
        + (pid_bt_state * H + pid_h) * K * V
        + (NT - 1 + o_1)[:, None, None] * (H * K * V)
        + o_k[None, :, None] * V
        + o_v[None, None, :]
    )

    if WITH_FINAL_STATE:
        dc_ptr = dc + (pid_b * H + pid_h) * K * V + o_k[:, None] * V + o_v[None, :]
        dc_out_ptr = dc_out + (pid_b * H + pid_h) * K * V + o_k[:, None] * V + o_v[None, :]

    if MAX_NORMALISATION is not None:
        # additional input pointers
        h_ptr = h + (pid_bt * H + pid_h) * V + (final_n + o_t)[:, None] * (H * V) + o_v[None, :]
        z_ptr = z + pid_bt * H + pid_h + (final_n + o_t) * H
        if MAX_NORMALISATION:
            m_all_ptr = m_all + pid_bt * H + pid_h + (final_n + o_t) * H

        # additional output pointers
        dns_ptr = dns + (pid_bt_state * H + pid_h) * K + (NT - 1 + o_1)[:, None] * (H * K) + o_k[None, :]

        if WITH_FINAL_STATE:
            dn_ptr = dn + (pid_b * H + pid_h) * K + o_k
            dn_out_ptr = dn_out + (pid_b * H + pid_h) * K + o_k

    if WITH_FINAL_STATE:
        dc_vals = tl.load(dc_ptr, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0)  # element-wise
        if MAX_NORMALISATION is not None:
            dn_vals = tl.load(dn_ptr, mask=(o_k < K), other=0)  # element-wise
    else:
        dc_vals = tl.zeros([BK, BV], dtype=tl.float32)
        if MAX_NORMALISATION is not None:
            dn_vals = tl.zeros([BK], dtype=tl.float32)
    for offset_n in range(final_n, -1, -BT):
        last_val_idx = tl.full([1], tl.minimum(BT, T - offset_n) - 1, dtype=tl.int32)
        tl.store(
            dcs_ptr,
            dc_vals[None, :, :].to(dcs.type.element_ty),
            mask=(o_k[None, :, None] < K) & (o_v[None, None, :] < V),
        )

        f_all_vals = tl.load(f_all_ptr, mask=(offset_n + o_t < T), other=0)
        qT_vals = tl.load(qT_ptr, mask=(o_k[:, None] < K) & ((offset_n + o_t)[None, :] < T), other=0)
        q_gated = f_all_vals[None, :] * qT_vals
        f_all_final_val = tl.gather(f_all_vals, last_val_idx, axis=0)

        dh_vals = tl.load(dh_ptr, mask=((offset_n + o_t)[:, None] < T) & (o_v[None, :] < V), other=0)
        if MAX_NORMALISATION is not None:
            tl.store(dns_ptr, dn_vals[None, :].to(dns.type.element_ty), mask=(o_k[None, :] < K))
            z_vals = tl.load(z_ptr, mask=(offset_n + o_t < T), other=0)
            h_vals = tl.load(h_ptr, mask=((offset_n + o_t)[:, None] < T) & (o_v[None, :] < V), other=0)
            if MAX_NORMALISATION:
                m_all_vals = tl.load(m_all_ptr, mask=(offset_n + o_t < T), other=0).to(tl.float32)
                _z_mask = tl.abs(z_vals) > (tl.exp(-m_all_vals) + eps)
                _z = tl.where(_z_mask, tl.abs(z_vals), tl.exp(-m_all_vals) + eps)
                ds = tl.cast(dh_vals / _z[:, None], dtype=dh_vals.type.element_ty)
                z_sign = tl.where(z_vals < 0, -1, 1)
                dz_neg = _z_mask * z_sign * tl.sum(ds * h_vals, axis=1)
            else:
                _z = z_vals + eps
                ds = tl.cast(dh_vals / _z[:, None], dtype=dh_vals.type.element_ty)
                dz_neg = tl.sum(ds * h_vals, axis=1)

            dn_vals *= f_all_final_val
            dn_vals -= tl.sum(q_gated * dz_neg[None, :], axis=1)

            h_ptr -= BT * H * V
            z_ptr -= BT * H
            if MAX_NORMALISATION:
                m_all_ptr -= BT * H
            dns_ptr -= H * K
        else:
            ds = dh_vals

        dc_vals *= f_all_final_val
        dc_vals = tl.dot(q_gated, ds, dc_vals, out_dtype=dc_vals.type.element_ty, input_precision=FP32_PREC)

        dh_ptr -= BT * H * V
        qT_ptr -= BT * H * K
        f_all_ptr -= BT * H
        dcs_ptr -= H * K * V

    if WITH_FINAL_STATE:
        tl.store(dc_out_ptr, dc_vals.to(dc_out.dtype.element_ty), mask=(o_k[:, None] < K) & (o_v[None, :] < V))
        if MAX_NORMALISATION is not None:
            tl.store(dn_out_ptr, dn_vals.to(dn_out.dtype.element_ty), mask=(o_k < K))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_mlstm_bwd_kernel_dv(
    dh,
    q,
    k,
    i,
    log_f,
    dcs,
    z,
    ms,
    dv,
    cu_seqlens,
    chunk_indices,
    qk_scale: float,
    eps: float,
    T: int,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MAX_NORMALISATION: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    FP32_PREC: tl.constexpr,
):
    tl.static_assert(BK >= K, "not yet implemented")

    pid_bn = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1).to(tl.int64)
    pid_l = tl.program_id(axis=2).to(tl.int64)

    if IS_VARLEN:
        layout_n = tl.load(chunk_indices + 2 * pid_bn + tl.arange(0, 2)).to(tl.int64)
        pid_b, pid_n = tl.split(layout_n)
        offsets_t = tl.load(cu_seqlens + pid_b + tl.arange(0, 2)).to(tl.int64)
        start_t, end_t = tl.split(offsets_t)
        T = end_t - start_t
        pid_bt = start_t
    else:
        NT = tl.cdiv(T, BT)
        pid_b, pid_n = pid_bn // NT, pid_bn % NT
        start_t, end_t = 0, T
        pid_bt = pid_b * T

    offset_n = pid_n * BT
    offset_l = pid_l * BV

    # input pointers
    o_k = tl.arange(0, BK).to(tl.int64)
    o_t = tl.arange(0, BT).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    dh_ptr = dh + (pid_bt * H + pid_h) * V + (offset_n + o_t)[:, None] * (H * V) + (offset_l + o_v)[None, :]
    qT_ptr = q + (pid_bt * H + pid_h) * K + o_k[:, None] + (offset_n + o_t)[None, :] * (H * K)
    k_ptr = k + (pid_bt * H + pid_h) * K + (offset_n + o_t)[:, None] * (H * K) + o_k[None, :]
    i_ptr = i + pid_bt * H + pid_h + (offset_n + o_t)[:, None] * H
    log_f_ptr = log_f + (pid_bt * H + pid_h) + (offset_n + o_t)[None, :] * H
    dcs_ptr = dcs + (pid_bn * H + pid_h) * K * V + o_k[:, None] * V + (offset_l + o_v)[None, :]
    ms_ptr = ms + pid_bn * H + pid_h

    # output pointers
    dv_ptr = dv + (pid_bt * H + pid_h) * V + (offset_n + o_t)[:, None] * (H * V) + (offset_l + o_v)[None, :]

    BLK_IDX = tl.arange(0, BT)
    last_val_idx = tl.full([1], tl.minimum(BT, T - offset_n) - 1, dtype=tl.int32)
    last_row_idx = tl.broadcast_to(last_val_idx[None, :], [BT, 1])

    # recompute gating
    log_f_vals = tl.load(log_f_ptr, mask=((offset_n + o_t)[None, :] < T), other=0)
    f_mat_tmp = tl.where(BLK_IDX[None, :] > BLK_IDX[:, None], log_f_vals, 0)
    f_matT = tl.cumsum(f_mat_tmp, axis=1)
    f_matT = tl.where(BLK_IDX[None, :] >= BLK_IDX[:, None], f_matT, float("-inf"))
    i_vals = tl.load(i_ptr, mask=((offset_n + o_t)[:, None] < T), other=0)
    log_scaleT = f_matT + i_vals

    log_prod_f = tl.cumsum(tl.ravel(log_f_vals), axis=0)[None, :]
    m_vals = tl.load(ms_ptr).to(tl.float32)
    m_and_f = m_vals + log_prod_f
    m_local = tl.max(log_scaleT, axis=0, keep_dims=True).to(ACC_DTYPE)  # triton bug?
    m_all = tl.maximum(m_and_f, m_local)

    # recompute parallel part
    k_vals = tl.load(k_ptr, mask=((offset_n + o_t)[:, None] < T) & (o_k[None, :] < K), other=0)
    qT_vals = tl.load(qT_ptr, mask=(o_k[:, None] < K) & ((offset_n + o_t)[None, :] < T), other=0)
    kq = tl.dot(k_vals, qT_vals, out_dtype=ACC_DTYPE, input_precision=FP32_PREC)
    scaleT = tl.cast(qk_scale * tl.exp(log_scaleT - m_all), dtype=i_vals.type.element_ty)
    scale_last = tl.gather(scaleT, last_row_idx, axis=1).ravel()

    dh_vals = tl.load(dh_ptr, mask=((offset_n + o_t)[:, None] < T) & ((offset_l + o_v)[None, :] < V), other=0)
    if MAX_NORMALISATION is not None:
        # additional input pointer
        z_ptr = z + pid_bt * H + pid_h + (offset_n + o_t) * H

        z_vals = tl.load(z_ptr, mask=(offset_n + o_t < T), other=0)
        if MAX_NORMALISATION:
            m_all_per_token = tl.ravel(m_all)
            _z = tl.maximum(tl.abs(z_vals), tl.exp(-m_all_per_token) + eps)
        else:
            _z = z_vals + eps

        ds = tl.cast(dh_vals / _z[:, None], dtype=dh_vals.type.element_ty)
    else:
        ds = dh_vals

    dc_vals = tl.load(dcs_ptr, mask=(o_k[:, None] < K) & ((offset_l + o_v)[None, :] < V), other=0)
    k_gated = k_vals * scale_last[:, None]
    dv_vals = tl.dot(k_gated, dc_vals, out_dtype=ACC_DTYPE, input_precision=FP32_PREC)
    dv_vals = tl.dot(kq.to(scaleT.type.element_ty) * scaleT, ds, dv_vals, input_precision=FP32_PREC)

    tl.store(dv_ptr, dv_vals.to(dv.type.element_ty), mask=((offset_n + o_t)[:, None] < T) & ((offset_l + o_v)[None, :] < V))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BV', 'BK'],
    restore_value=['di', 'dlog_f'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_mlstm_bwd_kernel_dqk(
    dh,
    q,
    k,
    v,
    i,
    log_f,
    cs,
    ns,
    ms,
    h,
    dcs,
    dns,
    f_all,
    z,
    dq,
    dk,
    di,
    dlog_f,
    cu_seqlens,
    chunk_indices,
    qk_scale: float,
    eps: float,
    T: int,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MAX_NORMALISATION: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    FP32_PREC: tl.constexpr,
):
    tl.static_assert(BV >= V, "not yet implemented")

    pid_bn = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1).to(tl.int64)
    pid_k = tl.program_id(axis=2).to(tl.int64)

    if IS_VARLEN:
        layout_n = tl.load(chunk_indices + 2 * pid_bn + tl.arange(0, 2)).to(tl.int64)
        pid_b, pid_n = tl.split(layout_n)
        offsets_t = tl.load(cu_seqlens + pid_b + tl.arange(0, 2)).to(tl.int64)
        start_t, end_t = tl.split(offsets_t)
        T = end_t - start_t
        pid_bt = start_t
    else:
        NT = tl.cdiv(T, BT)
        pid_b, pid_n = pid_bn // NT, pid_bn % NT
        start_t, end_t = 0, T
        pid_bt = pid_b * T

    offset_k = pid_k * BK
    offset_n = pid_n * BT

    # input pointers
    o_k = tl.arange(0, BK).to(tl.int64)
    o_t = tl.arange(0, BT).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    dh_ptr = dh + (pid_bt * H + pid_h) * V + (offset_n + o_t)[:, None] * (H * V) + o_v[None, :]
    qT_ptr = q + (pid_bt * H + pid_h) * K + (offset_k + o_k)[:, None] + (offset_n + o_t)[None, :] * (H * K)
    k_ptr = k + (pid_bt * H + pid_h) * K + (offset_n + o_t)[:, None] * (H * K) + (offset_k + o_k)[None, :]
    vT_ptr = v + (pid_bt * H + pid_h) * V + o_v[:, None] + (offset_n + o_t)[None, :] * (H * V)
    i_ptr = i + pid_bt * H + pid_h + (offset_n + o_t)[None, :] * H
    log_f_ptr = log_f + (pid_bt * H + pid_h) + (offset_n + o_t)[:, None] * H
    csT_ptr = cs + (pid_bn * H + pid_h) * K * V + o_v[:, None] + (offset_k + o_k)[None, :] * V
    ms_ptr = ms + pid_bn * H + pid_h
    h_ptr = h + (pid_bt * H + pid_h) * V + (offset_n + o_t)[:, None] * (H * V) + o_v[None, :]
    dCs_ptr = dcs + (pid_bn * H + pid_h) * K * V + (offset_k + o_k)[:, None] * V + o_v[None, :]
    f_all_ptr = f_all + pid_bt * H + pid_h + (offset_n + o_t) * H

    # output pointers
    dq_ptr = dq + (pid_bt * H + pid_h) * K + (offset_n + o_t)[:, None] * (H * K) + (offset_k + o_k)[None, :]
    dk_ptr = dk + (pid_bt * H + pid_h) * K + (offset_n + o_t)[:, None] * (H * K) + (offset_k + o_k)[None, :]

    BLK_IDX = tl.arange(0, BT)
    last_val_idx = tl.full([1], tl.minimum(BT, T - offset_n) - 1, dtype=tl.int32)
    last_row_idx = tl.broadcast_to(last_val_idx[:, None], [1, BT])

    # recompute gating
    log_f_vals = tl.load(log_f_ptr, mask=((offset_n + o_t)[:, None] < T), other=0)
    f_mat_tmp = tl.where(BLK_IDX[None, :] < BLK_IDX[:, None], log_f_vals, 0)
    f_mat = tl.cumsum(f_mat_tmp, axis=0)
    f_mat = tl.where(BLK_IDX[None, :] <= BLK_IDX[:, None], f_mat, float("-inf"))
    i_vals = tl.load(i_ptr, mask=((offset_n + o_t)[None, :] < T), other=0)
    log_scale = f_mat + i_vals

    log_prod_f = tl.cumsum(tl.ravel(log_f_vals), axis=0)[:, None]
    m_vals = tl.load(ms_ptr).to(tl.float32)
    m_and_f = m_vals + log_prod_f
    m_local = tl.max(log_scale, axis=1, keep_dims=True).to(ACC_DTYPE)  # triton bug?
    m_all = tl.maximum(m_and_f, m_local)

    # recompute parallel part
    k_vals = tl.load(k_ptr, mask=((offset_n + o_t)[:, None] < T) & ((offset_k + o_k)[None, :] < K), other=0)
    qT_vals = tl.load(qT_ptr, mask=((offset_k + o_k)[:, None] < K) & ((offset_n + o_t)[None, :] < T), other=0)
    kq = tl.dot(k_vals, qT_vals, out_dtype=ACC_DTYPE, input_precision=FP32_PREC)
    scale = tl.cast(qk_scale * tl.exp(log_scale - m_all), dtype=i_vals.type.element_ty)
    scale_last = tl.gather(scale, last_row_idx, axis=0).ravel()

    dh_vals = tl.load(dh_ptr, mask=((offset_n + o_t)[:, None] < T) & (o_v[None, :] < V), other=0)
    f_all_vals = tl.load(f_all_ptr, mask=(offset_n + o_t < T), other=0)
    f_all_final_val = tl.gather(f_all_vals, last_val_idx, axis=0)
    if MAX_NORMALISATION is not None:
        # additional input pointers
        ns_ptr = ns + (pid_bn * H + pid_h) * K + (offset_k + o_k)
        dns_ptr = dns + (pid_bn * H + pid_h) * K + (offset_k + o_k)
        z_ptr = z + pid_bt * H + pid_h + (offset_n + o_t) * H

        z_vals = tl.load(z_ptr, mask=(offset_n + o_t < T), other=0)
        h_vals = tl.load(h_ptr, mask=((offset_n + o_t)[:, None] < T) & (o_v[None, :] < V), other=0)
        if MAX_NORMALISATION:
            m_all_per_token = tl.ravel(m_all)
            _z_mask = tl.abs(z_vals) > tl.exp(-m_all_per_token) + eps
            _z = tl.where(_z_mask, tl.abs(z_vals), tl.exp(-m_all_per_token) + eps)
            ds = tl.cast(dh_vals / _z[:, None], dtype=dh_vals.type.element_ty)
            z_neg_sign = tl.where(z_vals < 0, 1, -1)
            dz = _z_mask * z_neg_sign * tl.sum(ds * h_vals, axis=1)
        else:
            _z = z_vals + eps
            ds = tl.cast(dh_vals / _z[:, None], dtype=dh_vals.type.element_ty)
            dz = -tl.sum(ds * h_vals, axis=1)

        de = tl.broadcast_to(dz[:, None], [BT, BT]).to(ACC_DTYPE)

        n_vals = tl.load(ns_ptr, mask=(offset_k + o_k < K), other=0)
        dq_vals = tl.cast(dz[:, None] * n_vals[None, :], dtype=ACC_DTYPE)

        dn_vals = tl.load(dns_ptr, mask=(offset_k + o_k < K), other=0)
        dkT_vals = tl.broadcast(dn_vals[:, None], qT_vals)[0].to(ACC_DTYPE)

        dlog_f_vals_state = tl.sum(dn_vals * n_vals)
    else:
        ds = dh_vals
        de = tl.zeros([BT, BT], dtype=ACC_DTYPE)
        dq_vals = tl.zeros([BT, BK], dtype=ACC_DTYPE)
        dkT_vals = tl.zeros([BK, BT], dtype=ACC_DTYPE)
        dlog_f_vals_state = 0

    # attn gradients
    vT_vals = tl.load(vT_ptr, mask=(o_v[:, None] < V) & ((offset_n + o_t)[None, :] < T), other=0)
    de = tl.dot(ds, vT_vals, de, input_precision=FP32_PREC)
    dqk = scale * tl.cast(de, dtype=scale.type.element_ty)

    # recurrent q gradient
    cT_vals = tl.load(csT_ptr, mask=(o_v[:, None] < V) & ((offset_k + o_k)[None, :] < K), other=0)
    dq_vals = tl.dot(ds, cT_vals, dq_vals, input_precision=FP32_PREC)
    dq_vals *= f_all_vals[:, None]

    # recurrent k gradient
    dc_vals = tl.load(dCs_ptr, mask=((offset_k + o_k)[:, None] < K) & (o_v[None, :] < V), other=0)
    dkT_vals = tl.dot(dc_vals, vT_vals, dkT_vals, input_precision=FP32_PREC)
    dkT_vals *= scale_last[None, :]

    # forget-gate gradients
    dlog_f_mat = tl.cumsum(dqk.T * kq, axis=1, reverse=True)
    dlog_f_mat += tl.sum(dkT_vals.T * k_vals, axis=1, keep_dims=True)
    dlog_f_mat = tl.where(BLK_IDX[None, :] > BLK_IDX[:, None], dlog_f_mat, 0)
    dlog_f_vals = tl.sum(dq_vals * qT_vals.T, axis=1)
    dlog_f_vals = tl.cumsum(dlog_f_vals, axis=0, reverse=True)
    dlog_f_vals += tl.sum(dlog_f_mat, axis=0)
    dlog_f_vals += f_all_final_val * (dlog_f_vals_state + tl.sum(dc_vals.T * cT_vals))

    # parallel gradients
    dq_vals = tl.dot(dqk, k_vals, dq_vals, input_precision=FP32_PREC)
    dkT_vals = tl.dot(qT_vals, dqk, dkT_vals, input_precision=FP32_PREC)

    # input-gate gradients
    di_vals = tl.sum(dkT_vals.T * k_vals, axis=1)

    tl.store(dq_ptr, dq_vals.to(dq.type.element_ty), mask=((offset_n + o_t)[:, None] < T) & ((offset_k + o_k)[None, :] < K))
    tl.store(dk_ptr, dkT_vals.T.to(dk.type.element_ty), mask=((offset_n + o_t)[:, None] < T) & ((offset_k + o_k)[None, :] < K))
    tl.atomic_add(
        di + pid_bt * H + pid_h + (offset_n + tl.arange(0, BT)) * H,
        di_vals.to(di.type.element_ty),
        mask=offset_n + tl.arange(0, BT) < T,
    )
    tl.atomic_add(
        dlog_f + pid_bt * H + pid_h + (offset_n + tl.arange(0, BT)) * H,
        dlog_f_vals.to(dlog_f.type.element_ty),
        mask=offset_n + tl.arange(0, BT) < T,
    )


def chunk_mlstm_fwd_state(
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    log_f: torch.Tensor,
    scale: float,
    initial_state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    do_normalisation: bool = True,
    chunk_size: int = 64,
) -> tuple[
    torch.Tensor, torch.Tensor | None, torch.Tensor,
    torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    fp32_prec, allow_fp16_acc = pytorch_matmul_config()
    use_fp16_acc = allow_fp16_acc and all(x.dtype == torch.float16 for x in (k, v, i, log_f))

    batch_dim = B
    if cu_seqlens is None:
        NT, split_offsets = triton.cdiv(T, chunk_size), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
        B, NT = len(cu_seqlens) - 1, split_offsets[-1].item()

    if initial_state is None:
        state_dtype = torch.float16 if use_fp16_acc else torch.float32
        c0 = torch.zeros(B, H, K, V, dtype=state_dtype, device=k.device)
        n0 = None
        m0 = torch.full((B, H), float('-inf'), dtype=state_dtype, device=k.device)
        if do_normalisation:
            n0 = torch.zeros(B, H, K, dtype=state_dtype, device=k.device)
    else:
        c0, n0, m0 = initial_state
        c0 = c0.contiguous()
        m0 = m0.contiguous()
        if do_normalisation:
            n0 = n0.contiguous()  # FIX assigned value

    if output_final_state:
        c_out = torch.empty_like(c0)
        n_out = torch.empty_like(n0) if do_normalisation else n0
        m_out = torch.empty_like(m0)
    else:
        c_out, n_out, m_out = None, None, None

    cs = torch.empty(batch_dim, NT, H, K, V, dtype=k.dtype, device=k.device)
    ns = torch.empty(batch_dim, NT, H, K, dtype=k.dtype, device=k.device) if do_normalisation else None
    ms = torch.empty(batch_dim, NT, H, dtype=k.dtype, device=k.device)

    chunk_mlstm_fwd_kernel_state[(B, H)](
        k=k,
        v=v,
        log_i=i,
        log_f=log_f,
        c0=c0,
        n0=n0,
        m0=m0,
        c_out=c_out,
        n_out=n_out,
        m_out=m_out,
        cs=cs,
        ns=ns,
        ms=ms,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        qk_scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=chunk_size,
        BK=max(16, triton.next_power_of_2(K)),
        BV=max(16, triton.next_power_of_2(V)),
        IS_VARLEN=cu_seqlens is not None,
        STORE_FINAL_STATE=c_out is not None,
        DO_NORMALISE=do_normalisation,
        FP32_PREC=fp32_prec,
    )

    return cs, ns, ms, c_out, n_out, m_out


def chunk_mlstm_fwd_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    log_f: torch.Tensor,
    cs: torch.Tensor,
    ns: torch.Tensor | None,
    ms: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    cu_seqlens: torch.LongTensor | None = None,
    max_normalisation: bool | None = True,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    fp32_prec, allow_fp16_acc = pytorch_matmul_config()
    use_fp16_acc = allow_fp16_acc and all(x.dtype == torch.float16 for x in (q, k, v, i, log_f))

    if cu_seqlens is None:
        NT, chunk_indices = triton.cdiv(T, chunk_size), None
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        B, NT = 1, cs.shape[1]

    out = torch.empty_like(v)
    f_all = torch.empty_like(log_f)
    m_all = torch.empty_like(log_f, dtype=torch.float32) if max_normalisation else None
    z = torch.empty_like(log_f) if max_normalisation is not None else None

    chunk_mlstm_fwd_kernel_out[(B, NT, H)](
        q=q,
        k=k,
        v=v,
        log_i=i,
        log_f=log_f,
        cs=cs,
        ns=ns,
        ms=ms,
        h=out,
        z=z,
        f_all=f_all,
        m_all_out=m_all,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        qk_scale=scale,
        eps=eps,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=chunk_size,
        BK=max(16, triton.next_power_of_2(K)),
        BV=max(16, triton.next_power_of_2(V)),
        IS_VARLEN=cu_seqlens is not None,
        MAX_NORMALISATION=max_normalisation,
        ACC_DTYPE=tl.float16 if use_fp16_acc else tl.float32,
        FP32_PREC=fp32_prec,
    )

    return out, z, f_all, m_all


def chunk_mlstm_bwd_dstate(
    q: torch.Tensor,
    f_all: torch.Tensor,
    h: torch.Tensor | None,
    z: torch.Tensor | None,
    m_all: torch.Tensor | None,
    ms: torch.Tensor,
    dh: torch.Tensor,
    dc: torch.Tensor | None,
    dn: torch.Tensor | None,
    eps: float = 1e-6,
    cu_seqlens: torch.LongTensor | None = None,
    max_normalisation: bool | None = True,
    chunk_size: int = 64,
):
    B, T, H, K, V = *q.shape, dh.shape[-1]
    fp32_prec, allow_fp16_acc = pytorch_matmul_config()
    use_fp16_acc = allow_fp16_acc and all(x.dtype == torch.float16 for x in (q, dh, dc) if x is not None)

    batch_dim = B
    if cu_seqlens is None:
        NT, split_offsets = triton.cdiv(T, chunk_size), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
        B, NT = len(cu_seqlens) - 1, split_offsets[-1].item()

    state_dtype = torch.float16 if use_fp16_acc else torch.float32
    dc_out = torch.empty_like(dc, dtype=state_dtype) if dc is not None else None
    dcs = torch.empty(batch_dim, NT, H, K, V, dtype=dh.dtype, device=dh.device)
    if max_normalisation is not None:
        dn_out = torch.empty_like(dn, dtype=state_dtype) if dn is not None else None
        dns = torch.empty(batch_dim, NT, H, K, dtype=dh.dtype, device=dh.device)
    else:
        dn_out, dns = dn, None

    chunk_mlstm_bwd_kernel_dstate[(B, H)](
        dh,
        dc,
        dn,
        q,
        f_all,
        h,
        z,
        m_all,
        ms,
        dcs,
        dns,
        dc_out,
        dn_out,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        eps=eps,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=chunk_size,
        BK=max(16, triton.next_power_of_2(K)),
        BV=max(16, triton.next_power_of_2(V)),
        IS_VARLEN=cu_seqlens is not None,
        WITH_FINAL_STATE=dc is not None,
        MAX_NORMALISATION=max_normalisation,
        FP32_PREC=fp32_prec,
    )

    return dcs, dns, dc_out, dn_out


def chunk_mlstm_bwd_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    i: torch.Tensor,
    log_f: torch.Tensor,
    z: torch.Tensor | None,
    ms: torch.Tensor,
    dh: torch.Tensor,
    dcs: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    cu_seqlens: torch.LongTensor | None = None,
    max_normalisation: bool | None = True,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, H, K, V = *k.shape, dh.shape[-1]
    fp32_prec, allow_fp16_acc = pytorch_matmul_config()
    use_fp16_acc = allow_fp16_acc and all(x.dtype == torch.float16 for x in (q, k, i, log_f, dh))

    if cu_seqlens is None:
        B_NT, chunk_indices = B * triton.cdiv(T, chunk_size), None
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        B_NT = len(chunk_indices)

    dv = torch.empty_like(dh)
    chunk_mlstm_bwd_kernel_dv[lambda meta: (
        B_NT,
        H,
        triton.cdiv(V, meta["BV"]),
    )](
        dh,
        q,
        k,
        i,
        log_f,
        dcs,
        z,
        ms,
        dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        qk_scale=scale,
        eps=eps,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=chunk_size,
        BK=max(16, triton.next_power_of_2(K)),
        BV=min(max(16, triton.next_power_of_2(V)), 64),
        IS_VARLEN=cu_seqlens is not None,
        MAX_NORMALISATION=max_normalisation,
        ACC_DTYPE=tl.float16 if use_fp16_acc else tl.float32,
        FP32_PREC=fp32_prec,
    )

    return dv


def chunk_mlstm_bwd_dqk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    f_all: torch.Tensor,
    h: torch.Tensor | None,
    z: torch.Tensor | None,
    cs: torch.Tensor,
    ns: torch.Tensor | None,
    ms: torch.Tensor,
    dh: torch.Tensor,
    dcs: torch.Tensor,
    dns: torch.Tensor | None,
    scale: float,
    eps: float = 1e-6,
    cu_seqlens: torch.LongTensor | None = None,
    max_normalisation: bool | None = True,
    chunk_size: int = 64,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    fp32_prec, allow_fp16_acc = pytorch_matmul_config()
    use_fp16_acc = allow_fp16_acc and all(x.dtype == torch.float16 for x in (q, k, v, i, f, dh))

    if cu_seqlens is None:
        B_NT, chunk_indices = B * triton.cdiv(T, chunk_size), None
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        B_NT = len(chunk_indices)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    di = torch.zeros_like(i)
    dlog_f = torch.zeros_like(f)
    chunk_mlstm_bwd_kernel_dqk[lambda meta: (
        B_NT,
        H,
        triton.cdiv(K, meta["BK"]),
    )](
        dh,
        q,
        k,
        v,
        i,
        torch.nn.functional.logsigmoid(f),
        cs,
        ns,
        ms,
        h,
        dcs,
        dns,
        f_all,
        z,
        dq,
        dk,
        di,
        dlog_f,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        qk_scale=scale,
        eps=eps,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=chunk_size,
        BK=min(max(16, triton.next_power_of_2(K)), 64),
        BV=max(16, triton.next_power_of_2(V)),
        IS_VARLEN=cu_seqlens is not None,
        MAX_NORMALISATION=max_normalisation,
        ACC_DTYPE=tl.float16 if use_fp16_acc else tl.float32,
        FP32_PREC=fp32_prec,
    )
    df = torch.sigmoid(-f) * dlog_f

    return dq, dk, di, df


def chunk_mlstm_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    max_normalisation: bool | None = True,
    chunk_size: int = 64,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
]:
    log_f = torch.nn.functional.logsigmoid(f)
    cs, ns, ms, c_out, n_out, m_out = chunk_mlstm_fwd_state(
        k, v, i, log_f, scale, initial_state, output_final_state, cu_seqlens,
        do_normalisation=max_normalisation is not None,
        chunk_size=chunk_size,
    )

    out, z, f_all, m_all = chunk_mlstm_fwd_out(
        q, k, v, i, log_f, cs, ns, ms, scale, eps, cu_seqlens,
        max_normalisation=max_normalisation,
        chunk_size=chunk_size
    )

    return out, c_out, n_out, m_out, z, f_all, m_all


def chunk_mlstm_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    f_all: torch.Tensor,
    h: torch.Tensor,
    z: torch.Tensor,
    m_all: torch.Tensor | None,
    cs: torch.Tensor | None,
    ns: torch.Tensor | None,
    ms: torch.Tensor | None,
    dh: torch.Tensor,
    dc: torch.Tensor,
    dn: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    max_normalisation: bool | None = True,
    chunk_size: int = 64,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor | None
]:
    log_f = torch.nn.functional.logsigmoid(f)
    if cs is None:
        cs, ns, ms, _, _, _ = chunk_mlstm_fwd_state(
            k, v, i, log_f, scale, initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            do_normalisation=max_normalisation is not None,
            chunk_size=chunk_size,
        )

    if initial_state is not None:
        # seed absent final-state gradients with zero to propagate gradients to the initial state
        c0, n0, _ = initial_state
        if dc is None:
            dc = torch.zeros_like(c0)
        if max_normalisation is not None and dn is None:
            dn = torch.zeros_like(n0)

    dcs, dns, dc_out, dn_out = chunk_mlstm_bwd_dstate(
        q, f_all, h, z, m_all, ms, dh, dc, dn,
        eps=eps,
        cu_seqlens=cu_seqlens,
        max_normalisation=max_normalisation,
        chunk_size=chunk_size
    )

    dv = chunk_mlstm_bwd_dv(
        q, k, i, log_f, z, ms, dh, dcs,
        scale=scale,
        eps=eps,
        cu_seqlens=cu_seqlens,
        max_normalisation=max_normalisation,
        chunk_size=chunk_size,
    )

    dq, dk, di, df = chunk_mlstm_bwd_dqk(
        q, k, v, i, f, f_all, h, z, cs, ns, ms, dh, dcs, dns,
        scale=scale,
        eps=eps,
        cu_seqlens=cu_seqlens,
        max_normalisation=max_normalisation,
        chunk_size=chunk_size,
    )

    return dq, dk, dv, di, df, dc_out, dn_out


class ChunkMLSTMFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        i,
        f,
        initial_c: torch.Tensor | None,
        initial_n: torch.Tensor | None,
        initial_m: torch.Tensor | None,
        scale: float,
        eps: float,
        output_final_state: bool,
        cu_seqlens,
        max_normalisation: bool | None,
    ):
        T = q.shape[1] if cu_seqlens is None else (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))

        initial_state = None if initial_c is None else (initial_c, initial_n, initial_m)
        h, c, n, m, z, f_all, m_all = chunk_mlstm_fwd(
            q=q,
            k=k,
            v=v,
            i=i,
            f=f,
            scale=scale,
            eps=eps,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            max_normalisation=max_normalisation,
            chunk_size=chunk_size,
        )
        if m is not None:
            ctx.mark_non_differentiable(m)
        ctx.save_for_backward(q, k, v, i, f, f_all, h, z, m_all)
        ctx.initial_state = initial_state
        ctx.scale = scale
        ctx.eps = eps
        ctx.cu_seqlens = cu_seqlens
        ctx.max_normalisation = max_normalisation
        ctx.chunk_size = chunk_size
        return h, c, n, m

    @staticmethod
    @input_guard
    @torch.autograd.function.once_differentiable
    @autocast_custom_bwd
    def backward(ctx, dh, dc, dn, dm):
        q, k, v, i, f, f_all, h, z, m_all = ctx.saved_tensors
        dq, dk, dv, di, df, dc_out, dn_out = chunk_mlstm_bwd(
            q=q,
            k=k,
            v=v,
            i=i,
            f=f,
            f_all=f_all,
            h=h,
            z=z,
            m_all=m_all,
            cs=None,
            ns=None,
            ms=None,
            dh=dh,
            dc=dc,
            dn=dn,
            scale=ctx.scale,
            eps=ctx.eps,
            initial_state=ctx.initial_state,
            cu_seqlens=ctx.cu_seqlens,
            max_normalisation=ctx.max_normalisation,
            chunk_size=ctx.chunk_size,
        )

        if ctx.initial_state is None:
            dc_out, dn_out = None, None

        return dq, dk, dv, di, df, dc_out, dn_out, None, None, None, None, None, None


@torch.compiler.disable
def chunk_mlstm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    scale: float | None = None,
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    max_normalisation: bool | None = True,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        i (torch.Tensor):
            Input gate logits of shape `[B, T, H]`.
        f (torch.Tensor):
            Forget gate logits of shape `[B, T, H]`.
        scale (float, Optional):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        eps (float, Optional):
            Stabiliser for the normalisation denominator. Default: 1e-6.
        initial_state (tuple[torch.Tensor, torch.Tensor | None, torch.Tensor], Optional):
            Initial state `(c0, n0, m0)` for `N` input sequences,
            with shapes `[N, H, K, V]`, `[N, H, K]`, and `[N, H]`, respectively.
            `n0` may be `None` when `max_normalisation` is `None`.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final state tuple `(c, n, m)`. Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.
        max_normalisation (bool | None, Optional):
            Whether to use the maximum-based normalisation of the original paper
            or a regular FLA normalisation inside the kernel.
            If set to `None`, no normalisation will be computed in the kernel
            and the normaliser state will not be updated.
            Default: `True`.

    Returns:
        h (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None):
            Final state with same shapes as `initial_state` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> from fla.ops.mlstm import chunk_mlstm
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> i = torch.randn(B, T, H, device='cuda')
        >>> f = torch.randn(B, T, H, device='cuda')
        >>> c0 = torch.randn(B, H, K, V, device='cuda')
        >>> n0 = torch.randn(B, H, K, device='cuda')
        >>> m0 = torch.randn(B, H, device='cuda')
        >>> h, ct = chunk_mlstm(
            q, k, v, i, f,
            initial_state=(c0, n0, m0),
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v = map(lambda x: x.flatten(end_dim=1).unsqueeze(0), (q, k, v))
        >>> i, f = map(lambda x: x.flatten(end_dim=1).unsqueeze(0), (i, f))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> h_var, ct_var = chunk_mlstm(
            q, k, v, i, f,
            initial_state=(c0, n0, m0),
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
        >>> assert h.allclose(h_var.view(h.shape))
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and any(
            s.shape[0] != len(cu_seqlens) - 1 for s in initial_state if s is not None
        ):
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {tuple(None if s is None else s.shape[0] for s in initial_state)}.",
            )
    if scale is None:
        scale = q.shape[-1] ** -0.5

    if initial_state is None:
        initial_state = (None, None, None)
    initial_c, initial_n, initial_m = initial_state
    assert q.shape == k.shape, "q, k must have the same shape."
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (batch size, seq len, num of head, head dim)."
    h, c, n, m = ChunkMLSTMFunction.apply(
        q, k, v, i, f, initial_c, initial_n, initial_m,
        scale, eps, output_final_state, cu_seqlens, max_normalisation
    )
    return h, (c, n, m) if output_final_state else None
