# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Copyright (c) 2026, Pieter-Jan Hoedt

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
)


def _get_matmul_config() -> tuple[str, bool]:
    """Match PyTorch's matmul precision for mLSTM state accumulation."""
    mm_config = torch.backends.cuda.matmul
    fp16_acc = getattr(mm_config, "allow_fp16_accumulation", False)

    fp32_precision = getattr(mm_config, "fp32_precision", "none")
    if fp32_precision == "none":
        # fallback to generic interface
        fp32_map = {"medium": "tf32", "high": "tf32", "highest": "ieee"}
        fp32_precision = fp32_map[torch.get_float32_matmul_precision()]

    return fp32_precision, fp16_acc


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
    m_f = tl.arange(0, BT) != BT - 1

    i_b = tl.program_id(axis=0).to(tl.int64)
    i_h = tl.program_id(axis=1).to(tl.int64)

    if IS_VARLEN:
        o_seqlens = tl.load(cu_seqlens + i_b + tl.arange(0, 2)).to(tl.int64)
        bos, eos = tl.split(o_seqlens)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        i_bt = bos
        i_bt_state = tl.load(split_offsets + i_b).to(tl.int64)
    else:
        bos, eos = 0, T
        NT = tl.cdiv(T, BT)
        i_bt = i_b * T
        i_bt_state = i_b * NT

    o_1 = tl.arange(0, 1).to(tl.int64)
    o_k = tl.arange(0, BK).to(tl.int64)
    o_t = tl.arange(0, BT).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    p_kt = k + (i_bt * H + i_h) * K + o_k[:, None] + o_t[None, :] * (H * K)
    p_v = v + (i_bt * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    p_log_i = log_i + i_bt * H + i_h + o_t * H
    p_log_f = log_f + i_bt * H + i_h + o_t * H
    p_c = c0 + (i_b * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
    p_m = m0 + i_b * H + i_h

    p_cs = cs + (i_bt_state * H + i_h) * V * K + o_k[None, :, None] * V + o_v[None, None, :]
    p_ms = ms + i_bt_state * H + i_h + o_1
    if STORE_FINAL_STATE:
        p_c_out = c_out + (i_b * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        p_m_out = m_out + i_b * H + i_h

    if DO_NORMALISE:
        p_n = n0 + (i_b * H + i_h) * K + o_k

        p_ns = ns + (i_bt_state * H + i_h) * K + o_k[None, :]
        if STORE_FINAL_STATE:
            p_n_out = n_out + (i_b * H + i_h) * K + o_k

    qk_scale = tl.cast(qk_scale, dtype=k.type.element_ty)
    b_c = tl.load(p_c, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0)
    b_m = tl.load(p_m)
    if DO_NORMALISE:
        b_n = tl.load(p_n, mask=(o_k < K), other=0)
    for i_t in range(bos, eos, BT):
        tl.store(p_cs, b_c[None, :, :].to(cs.type.element_ty), mask=(o_k[None, :, None] < K) & (o_v[None, None, :] < V))
        tl.store(p_ms, b_m[None].to(ms.type.element_ty))

        # invalid input gates are masked to negative infinity below
        b_i = tl.load(p_log_i, mask=(i_t - bos + o_t < T), other=0)
        b_i = tl.where(tl.arange(0, BT) < eos - i_t, b_i, -float("inf"))
        f_vals_shifted = tl.load(p_log_f + H, mask=(i_t - bos + 1 + o_t < T), other=0)
        f_vals_shifted = tl.where(m_f, f_vals_shifted, 0)
        log_prods_f = tl.cumsum(f_vals_shifted, axis=0, reverse=True)
        log_scale = b_i + log_prods_f
        mc = tl.max(log_scale).to(log_scale.type.element_ty)

        b_f = tl.load(p_log_f, mask=(i_t - bos + o_t < T), other=0)
        log_prod_f = tl.sum(b_f)
        m_and_f = log_prod_f + b_m
        b_m = tl.maximum(mc, m_and_f)
        b_kt = tl.load(p_kt, mask=(o_k[:, None] < K) & ((i_t - bos + o_t)[None, :] < T), other=0)
        k_gated = qk_scale * b_kt * tl.exp(tl.cast(log_scale - b_m, tl.float32)).to(b_kt.type.element_ty)
        f_rec = tl.exp(tl.cast(m_and_f - b_m, tl.float32)).to(b_c.type.element_ty)

        b_v = tl.load(p_v, mask=((i_t - bos + o_t)[:, None] < T) & (o_v[None, :] < V), other=0)
        b_c *= f_rec
        b_c = tl.dot(k_gated, b_v, b_c.to(ACC_DTYPE), out_dtype=b_c.type.element_ty, input_precision=FP32_PREC)
        if DO_NORMALISE:
            tl.store(p_ns, b_n[None, :].to(ns.type.element_ty), mask=(o_k[None, :] < K))
            b_n *= f_rec
            b_n += tl.sum(k_gated, axis=1).to(b_n.type.element_ty)
            p_ns += H * K

        p_log_i += BT * H
        p_log_f += BT * H
        p_kt += BT * H * K
        p_v += BT * H * V
        p_cs += H * V * K
        p_ms += H

    if STORE_FINAL_STATE:
        tl.store(p_c_out, b_c.to(c_out.type.element_ty), mask=(o_k[:, None] < K) & (o_v[None, :] < V))
        tl.store(p_m_out, b_m.to(m_out.type.element_ty))
        if DO_NORMALISE:
            tl.store(p_n_out, b_n.to(n_out.type.element_ty), mask=(o_k < K))


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

    o_blk = tl.arange(0, BT)

    i_b = tl.program_id(axis=0).to(tl.int64)
    i_n = tl.program_id(axis=1).to(tl.int64)
    i_h = tl.program_id(axis=2).to(tl.int64)

    if IS_VARLEN:
        i_bn = i_n
        layout_n = tl.load(chunk_indices + 2 * i_n + tl.arange(0, 2)).to(tl.int64)
        i_b, i_n = tl.split(layout_n)
        o_seqlens = tl.load(cu_seqlens + i_b + tl.arange(0, 2)).to(tl.int64)
        bos, eos = tl.split(o_seqlens)
        T = eos - bos
        i_bt = bos
    else:
        bos, eos = 0, T
        i_bt = i_b * T
        i_bn = i_b * tl.cdiv(T, BT) + i_n

    o_k = tl.arange(0, BK).to(tl.int64)
    o_t = tl.arange(0, BT).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    p_q = q + (i_bt * H + i_h) * K + (i_n * BT + o_t)[:, None] * (H * K) + o_k[None, :]
    p_kt = k + (i_bt * H + i_h) * K + o_k[:, None] + (i_n * BT + o_t)[None, :] * (H * K)
    p_v = v + (i_bt * H + i_h) * V + (i_n * BT + o_t)[:, None] * (H * V) + o_v[None, :]
    p_log_i = log_i + i_bt * H + i_h + (i_n * BT + o_t) * H
    p_log_f = log_f + i_bt * H + i_h + (i_n * BT + o_t) * H
    p_cs = cs + (i_bn * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
    p_ms = ms + i_bn * H + i_h

    p_h = h + (i_bt * H + i_h) * V + (i_n * BT + o_t)[:, None] * (H * V) + o_v[None, :]
    p_f_all = f_all + i_bt * H + i_h + (i_n * BT + o_t) * H
    if MAX_NORMALISATION:
        p_m_all_out = m_all_out + i_bt * H + i_h + (i_n * BT + o_t) * H

    if MAX_NORMALISATION is not None:
        p_ns = ns + (i_bn * H + i_h) * K + o_k

        p_z = z + i_bt * H + i_h + (i_n * BT + o_t) * H

    b_f = tl.load(p_log_f, mask=(i_n * BT + o_t < T), other=0)
    f_mat_tmp = tl.where(o_blk[None, :] < o_blk[:, None], b_f[:, None], 0)
    f_mat = tl.cumsum(f_mat_tmp, axis=0)
    f_mat = tl.where(o_blk[None, :] <= o_blk[:, None], f_mat, float("-inf"))
    b_i = tl.load(p_log_i, mask=(i_n * BT + o_t < T), other=0)
    log_scale = f_mat + b_i[None, :]

    log_prod_f = tl.cumsum(b_f, axis=0)[:, None]
    b_m = tl.load(p_ms)
    m_and_f = b_m + log_prod_f
    m_local = tl.max(log_scale, axis=1, keep_dims=True).to(ACC_DTYPE)
    m_all = tl.maximum(m_and_f, m_local).to(tl.float32)

    b_q = tl.load(p_q, mask=((i_n * BT + o_t)[:, None] < T) & (o_k[None, :] < K), other=0)
    b_kt = tl.load(p_kt, mask=(o_k[:, None] < K) & ((i_n * BT + o_t)[None, :] < T), other=0)
    # query-key scores determine the normalizer reused by backward and require FP32 accumulation.
    qk = tl.dot(b_q, b_kt, out_dtype=tl.float32, input_precision=FP32_PREC)
    e = tl.exp(log_scale - m_all).to(b_q.type.element_ty) * (qk_scale * qk)

    b_v = tl.load(p_v, mask=((i_n * BT + o_t)[:, None] < T) & (o_v[None, :] < V), other=0)
    b_s = tl.dot(e.to(v.dtype.element_ty), b_v, out_dtype=ACC_DTYPE, input_precision=FP32_PREC)

    b_f_all = tl.exp(m_and_f - m_all).to(b_q.type.element_ty)
    q_fgate = b_f_all * b_q
    b_c = tl.load(p_cs, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0)
    b_s = tl.dot(q_fgate, b_c, b_s, out_dtype=b_s.type.element_ty, input_precision=FP32_PREC)

    b_h = b_s
    if MAX_NORMALISATION is not None:
        b_z = tl.sum(e, axis=1, keep_dims=True)
        b_n = tl.load(p_ns, mask=(o_k < K), other=0)
        b_z += tl.sum(q_fgate * b_n[None, :], axis=1, keep_dims=True)

        eps = tl.cast(eps, dtype=ACC_DTYPE)
        if MAX_NORMALISATION:
            _z = tl.maximum(tl.abs(b_z), tl.exp(-m_all).to(ACC_DTYPE) + eps)
        else:
            _z = b_z + eps

        b_h = b_s / _z
        tl.store(p_z, tl.ravel(b_z).to(z.dtype.element_ty), mask=(i_n * BT + o_t < T))

    tl.store(p_h, b_h.to(h.dtype.element_ty), mask=((i_n * BT + o_t)[:, None] < T) & (o_v[None, :] < V))
    tl.store(p_f_all, tl.ravel(b_f_all).to(f_all.dtype.element_ty), mask=(i_n * BT + o_t < T))
    if MAX_NORMALISATION:
        tl.store(p_m_all_out, tl.ravel(m_all).to(m_all_out.dtype.element_ty), mask=(i_n * BT + o_t < T))


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

    i_b = tl.program_id(axis=0).to(tl.int64)
    i_h = tl.program_id(axis=1).to(tl.int64)

    if IS_VARLEN:
        o_seqlens = tl.load(cu_seqlens + i_b + tl.arange(0, 2)).to(tl.int64)
        bos, eos = tl.split(o_seqlens)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        i_bt = bos
        i_bt_state = tl.load(split_offsets + i_b).to(tl.int64)
    else:
        bos, eos = 0, T
        NT = tl.cdiv(T, BT)
        i_bt = i_b * T
        i_bt_state = i_b * NT
    final_n = (NT - 1) * BT

    o_1 = tl.arange(0, 1).to(tl.int64)
    o_k = tl.arange(0, BK).to(tl.int64)
    o_t = tl.arange(0, BT).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    p_dh = dh + (i_bt * H + i_h) * V + (final_n + o_t)[:, None] * (H * V) + o_v[None, :]
    p_qt = q + (i_bt * H + i_h) * K + o_k[:, None] + (final_n + o_t)[None, :] * (H * K)
    p_f_all = f_all + i_bt * H + i_h + (final_n + o_t) * H

    p_dcs = (
        dcs
        + (i_bt_state * H + i_h) * K * V
        + (NT - 1 + o_1)[:, None, None] * (H * K * V)
        + o_k[None, :, None] * V
        + o_v[None, None, :]
    )

    if WITH_FINAL_STATE:
        p_dc = dc + (i_b * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        p_dc_out = dc_out + (i_b * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]

    if MAX_NORMALISATION is not None:
        p_h = h + (i_bt * H + i_h) * V + (final_n + o_t)[:, None] * (H * V) + o_v[None, :]
        p_z = z + i_bt * H + i_h + (final_n + o_t) * H
        if MAX_NORMALISATION:
            p_m_all = m_all + i_bt * H + i_h + (final_n + o_t) * H

        p_dns = dns + (i_bt_state * H + i_h) * K + (NT - 1 + o_1)[:, None] * (H * K) + o_k[None, :]

        if WITH_FINAL_STATE:
            p_dn = dn + (i_b * H + i_h) * K + o_k
            p_dn_out = dn_out + (i_b * H + i_h) * K + o_k

    if WITH_FINAL_STATE:
        b_dc = tl.load(p_dc, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0)
        if MAX_NORMALISATION is not None:
            b_dn = tl.load(p_dn, mask=(o_k < K), other=0)
    else:
        b_dc = tl.zeros([BK, BV], dtype=tl.float32)
        if MAX_NORMALISATION is not None:
            b_dn = tl.zeros([BK], dtype=tl.float32)
    for o_chunk in range(final_n, -1, -BT):
        last_val_idx = tl.full([1], tl.minimum(BT, T - o_chunk) - 1, dtype=tl.int32)
        tl.store(
            p_dcs,
            b_dc[None, :, :].to(dcs.type.element_ty),
            mask=(o_k[None, :, None] < K) & (o_v[None, None, :] < V),
        )

        b_f_all = tl.load(p_f_all, mask=(o_chunk + o_t < T), other=0)
        b_qt = tl.load(p_qt, mask=(o_k[:, None] < K) & ((o_chunk + o_t)[None, :] < T), other=0)
        q_gated = b_f_all[None, :] * b_qt
        b_f_all_final = tl.gather(b_f_all, last_val_idx, axis=0)

        b_dh = tl.load(p_dh, mask=((o_chunk + o_t)[:, None] < T) & (o_v[None, :] < V), other=0)
        if MAX_NORMALISATION is not None:
            tl.store(p_dns, b_dn[None, :].to(dns.type.element_ty), mask=(o_k[None, :] < K))
            b_z = tl.load(p_z, mask=(o_chunk + o_t < T), other=0)
            b_h = tl.load(p_h, mask=((o_chunk + o_t)[:, None] < T) & (o_v[None, :] < V), other=0)
            if MAX_NORMALISATION:
                b_m_all = tl.load(p_m_all, mask=(o_chunk + o_t < T), other=0).to(tl.float32)
                _z_mask = tl.abs(b_z) > (tl.exp(-b_m_all) + eps)
                _z = tl.where(_z_mask, tl.abs(b_z), tl.exp(-b_m_all) + eps)
                ds = tl.cast(b_dh / _z[:, None], dtype=b_dh.type.element_ty)
                z_sign = tl.where(b_z < 0, -1, 1)
                dz_neg = _z_mask * z_sign * tl.sum(ds * b_h, axis=1)
            else:
                _z = b_z + eps
                ds = tl.cast(b_dh / _z[:, None], dtype=b_dh.type.element_ty)
                dz_neg = tl.sum(ds * b_h, axis=1)

            b_dn *= b_f_all_final
            b_dn -= tl.sum(q_gated * dz_neg[None, :], axis=1)

            p_h -= BT * H * V
            p_z -= BT * H
            if MAX_NORMALISATION:
                p_m_all -= BT * H
            p_dns -= H * K
        else:
            ds = b_dh

        b_dc *= b_f_all_final
        b_dc = tl.dot(q_gated, ds, b_dc, out_dtype=b_dc.type.element_ty, input_precision=FP32_PREC)

        p_dh -= BT * H * V
        p_qt -= BT * H * K
        p_f_all -= BT * H
        p_dcs -= H * K * V

    if WITH_FINAL_STATE:
        tl.store(p_dc_out, b_dc.to(dc_out.dtype.element_ty), mask=(o_k[:, None] < K) & (o_v[None, :] < V))
        if MAX_NORMALISATION is not None:
            tl.store(p_dn_out, b_dn.to(dn_out.dtype.element_ty), mask=(o_k < K))


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

    i_bn = tl.program_id(axis=0).to(tl.int64)
    i_h = tl.program_id(axis=1).to(tl.int64)
    i_l = tl.program_id(axis=2).to(tl.int64)

    if IS_VARLEN:
        layout_n = tl.load(chunk_indices + 2 * i_bn + tl.arange(0, 2)).to(tl.int64)
        i_b, i_n = tl.split(layout_n)
        o_seqlens = tl.load(cu_seqlens + i_b + tl.arange(0, 2)).to(tl.int64)
        bos, eos = tl.split(o_seqlens)
        T = eos - bos
        i_bt = bos
    else:
        NT = tl.cdiv(T, BT)
        i_b, i_n = i_bn // NT, i_bn % NT
        bos, eos = 0, T
        i_bt = i_b * T

    o_chunk = i_n * BT
    offset_l = i_l * BV

    o_k = tl.arange(0, BK).to(tl.int64)
    o_t = tl.arange(0, BT).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    p_dh = dh + (i_bt * H + i_h) * V + (o_chunk + o_t)[:, None] * (H * V) + (offset_l + o_v)[None, :]
    p_qt = q + (i_bt * H + i_h) * K + o_k[:, None] + (o_chunk + o_t)[None, :] * (H * K)
    p_k = k + (i_bt * H + i_h) * K + (o_chunk + o_t)[:, None] * (H * K) + o_k[None, :]
    p_i = i + i_bt * H + i_h + (o_chunk + o_t)[:, None] * H
    p_log_f = log_f + (i_bt * H + i_h) + (o_chunk + o_t)[None, :] * H
    p_dcs = dcs + (i_bn * H + i_h) * K * V + o_k[:, None] * V + (offset_l + o_v)[None, :]
    p_ms = ms + i_bn * H + i_h

    p_dv = dv + (i_bt * H + i_h) * V + (o_chunk + o_t)[:, None] * (H * V) + (offset_l + o_v)[None, :]

    o_blk = tl.arange(0, BT)
    last_val_idx = tl.full([1], tl.minimum(BT, T - o_chunk) - 1, dtype=tl.int32)
    last_row_idx = tl.broadcast_to(last_val_idx[None, :], [BT, 1])

    b_log_f = tl.load(p_log_f, mask=((o_chunk + o_t)[None, :] < T), other=0)
    f_mat_tmp = tl.where(o_blk[None, :] > o_blk[:, None], b_log_f, 0)
    f_mat_t = tl.cumsum(f_mat_tmp, axis=1)
    f_mat_t = tl.where(o_blk[None, :] >= o_blk[:, None], f_mat_t, float("-inf"))
    b_i = tl.load(p_i, mask=((o_chunk + o_t)[:, None] < T), other=0)
    log_scale_t = f_mat_t + b_i

    log_prod_f = tl.cumsum(tl.ravel(b_log_f), axis=0)[None, :]
    b_m = tl.load(p_ms).to(tl.float32)
    m_and_f = b_m + log_prod_f
    # keep the reduction in the accumulation dtype for Triton type inference
    m_local = tl.max(log_scale_t, axis=0, keep_dims=True).to(ACC_DTYPE)
    m_all = tl.maximum(m_and_f, m_local)

    b_k = tl.load(p_k, mask=((o_chunk + o_t)[:, None] < T) & (o_k[None, :] < K), other=0)
    b_qt = tl.load(p_qt, mask=(o_k[:, None] < K) & ((o_chunk + o_t)[None, :] < T), other=0)
    kq = tl.dot(b_k, b_qt, out_dtype=ACC_DTYPE, input_precision=FP32_PREC)
    scale_t = tl.cast(qk_scale * tl.exp(log_scale_t - m_all), dtype=b_i.type.element_ty)
    scale_last = tl.gather(scale_t, last_row_idx, axis=1).ravel()

    b_dh = tl.load(p_dh, mask=((o_chunk + o_t)[:, None] < T) & ((offset_l + o_v)[None, :] < V), other=0)
    if MAX_NORMALISATION is not None:
        p_z = z + i_bt * H + i_h + (o_chunk + o_t) * H

        b_z = tl.load(p_z, mask=(o_chunk + o_t < T), other=0)
        if MAX_NORMALISATION:
            m_all_per_token = tl.ravel(m_all)
            _z = tl.maximum(tl.abs(b_z), tl.exp(-m_all_per_token) + eps)
        else:
            _z = b_z + eps

        ds = tl.cast(b_dh / _z[:, None], dtype=b_dh.type.element_ty)
    else:
        ds = b_dh

    b_dc = tl.load(p_dcs, mask=(o_k[:, None] < K) & ((offset_l + o_v)[None, :] < V), other=0)
    k_gated = b_k * scale_last[:, None]
    b_dv = tl.dot(k_gated, b_dc, out_dtype=ACC_DTYPE, input_precision=FP32_PREC)
    b_dv = tl.dot(kq.to(scale_t.type.element_ty) * scale_t, ds, b_dv, input_precision=FP32_PREC)

    tl.store(p_dv, b_dv.to(dv.type.element_ty), mask=((o_chunk + o_t)[:, None] < T) & ((offset_l + o_v)[None, :] < V))


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

    i_bn = tl.program_id(axis=0).to(tl.int64)
    i_h = tl.program_id(axis=1).to(tl.int64)
    i_k = tl.program_id(axis=2).to(tl.int64)

    if IS_VARLEN:
        layout_n = tl.load(chunk_indices + 2 * i_bn + tl.arange(0, 2)).to(tl.int64)
        i_b, i_n = tl.split(layout_n)
        o_seqlens = tl.load(cu_seqlens + i_b + tl.arange(0, 2)).to(tl.int64)
        bos, eos = tl.split(o_seqlens)
        T = eos - bos
        i_bt = bos
    else:
        NT = tl.cdiv(T, BT)
        i_b, i_n = i_bn // NT, i_bn % NT
        bos, eos = 0, T
        i_bt = i_b * T

    offset_k = i_k * BK
    o_chunk = i_n * BT

    o_k = tl.arange(0, BK).to(tl.int64)
    o_t = tl.arange(0, BT).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    p_dh = dh + (i_bt * H + i_h) * V + (o_chunk + o_t)[:, None] * (H * V) + o_v[None, :]
    p_qt = q + (i_bt * H + i_h) * K + (offset_k + o_k)[:, None] + (o_chunk + o_t)[None, :] * (H * K)
    p_k = k + (i_bt * H + i_h) * K + (o_chunk + o_t)[:, None] * (H * K) + (offset_k + o_k)[None, :]
    p_vt = v + (i_bt * H + i_h) * V + o_v[:, None] + (o_chunk + o_t)[None, :] * (H * V)
    p_i = i + i_bt * H + i_h + (o_chunk + o_t)[None, :] * H
    p_log_f = log_f + (i_bt * H + i_h) + (o_chunk + o_t)[:, None] * H
    p_cst = cs + (i_bn * H + i_h) * K * V + o_v[:, None] + (offset_k + o_k)[None, :] * V
    p_ms = ms + i_bn * H + i_h
    p_h = h + (i_bt * H + i_h) * V + (o_chunk + o_t)[:, None] * (H * V) + o_v[None, :]
    p_dcs = dcs + (i_bn * H + i_h) * K * V + (offset_k + o_k)[:, None] * V + o_v[None, :]
    p_f_all = f_all + i_bt * H + i_h + (o_chunk + o_t) * H

    p_dq = dq + (i_bt * H + i_h) * K + (o_chunk + o_t)[:, None] * (H * K) + (offset_k + o_k)[None, :]
    p_dk = dk + (i_bt * H + i_h) * K + (o_chunk + o_t)[:, None] * (H * K) + (offset_k + o_k)[None, :]

    o_blk = tl.arange(0, BT)
    last_val_idx = tl.full([1], tl.minimum(BT, T - o_chunk) - 1, dtype=tl.int32)
    last_row_idx = tl.broadcast_to(last_val_idx[:, None], [1, BT])

    b_log_f = tl.load(p_log_f, mask=((o_chunk + o_t)[:, None] < T), other=0)
    f_mat_tmp = tl.where(o_blk[None, :] < o_blk[:, None], b_log_f, 0)
    f_mat = tl.cumsum(f_mat_tmp, axis=0)
    f_mat = tl.where(o_blk[None, :] <= o_blk[:, None], f_mat, float("-inf"))
    b_i = tl.load(p_i, mask=((o_chunk + o_t)[None, :] < T), other=0)
    log_scale = f_mat + b_i

    log_prod_f = tl.cumsum(tl.ravel(b_log_f), axis=0)[:, None]
    b_m = tl.load(p_ms).to(tl.float32)
    m_and_f = b_m + log_prod_f
    # keep the reduction in the accumulation dtype for Triton type inference
    m_local = tl.max(log_scale, axis=1, keep_dims=True).to(ACC_DTYPE)
    m_all = tl.maximum(m_and_f, m_local)

    b_k = tl.load(p_k, mask=((o_chunk + o_t)[:, None] < T) & ((offset_k + o_k)[None, :] < K), other=0)
    b_qt = tl.load(p_qt, mask=((offset_k + o_k)[:, None] < K) & ((o_chunk + o_t)[None, :] < T), other=0)
    kq = tl.dot(b_k, b_qt, out_dtype=ACC_DTYPE, input_precision=FP32_PREC)
    scale = tl.cast(qk_scale * tl.exp(log_scale - m_all), dtype=b_i.type.element_ty)
    scale_last = tl.gather(scale, last_row_idx, axis=0).ravel()

    b_dh = tl.load(p_dh, mask=((o_chunk + o_t)[:, None] < T) & (o_v[None, :] < V), other=0)
    b_f_all = tl.load(p_f_all, mask=(o_chunk + o_t < T), other=0)
    b_f_all_final = tl.gather(b_f_all, last_val_idx, axis=0)
    if MAX_NORMALISATION is not None:
        p_ns = ns + (i_bn * H + i_h) * K + (offset_k + o_k)
        p_dns = dns + (i_bn * H + i_h) * K + (offset_k + o_k)
        p_z = z + i_bt * H + i_h + (o_chunk + o_t) * H

        b_z = tl.load(p_z, mask=(o_chunk + o_t < T), other=0)
        b_h = tl.load(p_h, mask=((o_chunk + o_t)[:, None] < T) & (o_v[None, :] < V), other=0)
        if MAX_NORMALISATION:
            m_all_per_token = tl.ravel(m_all)
            _z_mask = tl.abs(b_z) > tl.exp(-m_all_per_token) + eps
            _z = tl.where(_z_mask, tl.abs(b_z), tl.exp(-m_all_per_token) + eps)
            ds = tl.cast(b_dh / _z[:, None], dtype=b_dh.type.element_ty)
            z_neg_sign = tl.where(b_z < 0, 1, -1)
            dz = _z_mask * z_neg_sign * tl.sum(ds * b_h, axis=1)
        else:
            _z = b_z + eps
            ds = tl.cast(b_dh / _z[:, None], dtype=b_dh.type.element_ty)
            dz = -tl.sum(ds * b_h, axis=1)

        de = tl.broadcast_to(dz[:, None], [BT, BT]).to(ACC_DTYPE)

        b_n = tl.load(p_ns, mask=(offset_k + o_k < K), other=0)
        b_dq = tl.cast(dz[:, None] * b_n[None, :], dtype=ACC_DTYPE)

        b_dn = tl.load(p_dns, mask=(offset_k + o_k < K), other=0)
        b_dkt = tl.broadcast(b_dn[:, None], b_qt)[0].to(ACC_DTYPE)

        dlog_f_vals_state = tl.sum(b_dn * b_n)
    else:
        ds = b_dh
        de = tl.zeros([BT, BT], dtype=ACC_DTYPE)
        b_dq = tl.zeros([BT, BK], dtype=ACC_DTYPE)
        b_dkt = tl.zeros([BK, BT], dtype=ACC_DTYPE)
        dlog_f_vals_state = 0

    b_vt = tl.load(p_vt, mask=(o_v[:, None] < V) & ((o_chunk + o_t)[None, :] < T), other=0)
    de = tl.dot(ds, b_vt, de, input_precision=FP32_PREC)
    dqk = scale * tl.cast(de, dtype=scale.type.element_ty)

    b_ct = tl.load(p_cst, mask=(o_v[:, None] < V) & ((offset_k + o_k)[None, :] < K), other=0)
    b_dq = tl.dot(ds, b_ct, b_dq, input_precision=FP32_PREC)
    b_dq *= b_f_all[:, None]

    b_dc = tl.load(p_dcs, mask=((offset_k + o_k)[:, None] < K) & (o_v[None, :] < V), other=0)
    b_dkt = tl.dot(b_dc, b_vt, b_dkt, input_precision=FP32_PREC)
    b_dkt *= scale_last[None, :]

    dlog_f_mat = tl.cumsum(dqk.T * kq, axis=1, reverse=True)
    dlog_f_mat += tl.sum(b_dkt.T * b_k, axis=1, keep_dims=True)
    dlog_f_mat = tl.where(o_blk[None, :] > o_blk[:, None], dlog_f_mat, 0)
    b_dlog_f = tl.sum(b_dq * b_qt.T, axis=1)
    b_dlog_f = tl.cumsum(b_dlog_f, axis=0, reverse=True)
    b_dlog_f += tl.sum(dlog_f_mat, axis=0)
    b_dlog_f += b_f_all_final * (dlog_f_vals_state + tl.sum(b_dc.T * b_ct))

    b_dq = tl.dot(dqk, b_k, b_dq, input_precision=FP32_PREC)
    b_dkt = tl.dot(b_qt, dqk, b_dkt, input_precision=FP32_PREC)

    b_di = tl.sum(b_dkt.T * b_k, axis=1)

    tl.store(p_dq, b_dq.to(dq.type.element_ty), mask=((o_chunk + o_t)[:, None] < T) & ((offset_k + o_k)[None, :] < K))
    tl.store(p_dk, b_dkt.T.to(dk.type.element_ty), mask=((o_chunk + o_t)[:, None] < T) & ((offset_k + o_k)[None, :] < K))
    tl.atomic_add(
        di + i_bt * H + i_h + (o_chunk + tl.arange(0, BT)) * H,
        b_di.to(di.type.element_ty),
        mask=o_chunk + tl.arange(0, BT) < T,
    )
    tl.atomic_add(
        dlog_f + i_bt * H + i_h + (o_chunk + tl.arange(0, BT)) * H,
        b_dlog_f.to(dlog_f.type.element_ty),
        mask=o_chunk + tl.arange(0, BT) < T,
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
    fp32_prec, allow_fp16_acc = _get_matmul_config()
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
            n0 = n0.contiguous()

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
    fp32_prec, allow_fp16_acc = _get_matmul_config()
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
    fp32_prec, allow_fp16_acc = _get_matmul_config()
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
    fp32_prec, allow_fp16_acc = _get_matmul_config()
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
    fp32_prec, allow_fp16_acc = _get_matmul_config()
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
        F.logsigmoid(f),
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
    log_f = F.logsigmoid(f)
    cs, ns, ms, c_out, n_out, m_out = chunk_mlstm_fwd_state(
        k=k,
        v=v,
        i=i,
        log_f=log_f,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        do_normalisation=max_normalisation is not None,
        chunk_size=chunk_size,
    )

    out, z, f_all, m_all = chunk_mlstm_fwd_out(
        q=q,
        k=k,
        v=v,
        i=i,
        log_f=log_f,
        cs=cs,
        ns=ns,
        ms=ms,
        scale=scale,
        eps=eps,
        cu_seqlens=cu_seqlens,
        max_normalisation=max_normalisation,
        chunk_size=chunk_size,
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
    log_f = F.logsigmoid(f)
    if cs is None:
        cs, ns, ms, _, _, _ = chunk_mlstm_fwd_state(
            k=k,
            v=v,
            i=i,
            log_f=log_f,
            scale=scale,
            initial_state=initial_state,
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
        q=q,
        f_all=f_all,
        h=h,
        z=z,
        m_all=m_all,
        ms=ms,
        dh=dh,
        dc=dc,
        dn=dn,
        eps=eps,
        cu_seqlens=cu_seqlens,
        max_normalisation=max_normalisation,
        chunk_size=chunk_size,
    )

    dv = chunk_mlstm_bwd_dv(
        q=q,
        k=k,
        i=i,
        log_f=log_f,
        z=z,
        ms=ms,
        dh=dh,
        dcs=dcs,
        scale=scale,
        eps=eps,
        cu_seqlens=cu_seqlens,
        max_normalisation=max_normalisation,
        chunk_size=chunk_size,
    )

    dq, dk, di, df = chunk_mlstm_bwd_dqk(
        q=q,
        k=k,
        v=v,
        i=i,
        f=f,
        f_all=f_all,
        h=h,
        z=z,
        cs=cs,
        ns=ns,
        ms=ms,
        dh=dh,
        dcs=dcs,
        dns=dns,
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
    r"""Compute mLSTM with chunkwise parallelism.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
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
            Gradients are supported for `c` and `n`, but not the stabilizer `m`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.
        max_normalisation (bool | None, Optional):
            Whether to use the maximum-based normalisation of the original paper
            or division by the signed normaliser plus `eps` when `False`.
            If set to `None`, no normalisation will be computed in the kernel
            and the normaliser state will not be updated.
            Default: `True`.

    Returns:
        h (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None):
            Final state with same shapes as `initial_state` if `output_final_state=True` else `None`.
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
