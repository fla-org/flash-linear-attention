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
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA_BLACKWELL, IS_TF32_SUPPORTED, autotune_cache_kwargs, check_shared_mem

if IS_TF32_SUPPORTED:
    DOT_PRECISION = tl.constexpr('tf32x3')
    DOT_PRECISION_FAST = tl.constexpr('tf32')
else:
    DOT_PRECISION = tl.constexpr('ieee')
    DOT_PRECISION_FAST = tl.constexpr('ieee')

# the same Blackwell tl.dot recurrence race as `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in ([2, 3, 4] if check_shared_mem('ampere') else [2, 1])
        for BV in ([32, 64] if check_shared_mem('ada') else [32])
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_diag_kdn_fwd_kernel_h(
    k,
    v,
    w,
    v_new,
    gk,
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
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v, i_nh = pid % NV, (pid // NV).to(tl.int64)
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

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h) * K*V
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    gk += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    if USE_INITIAL_STATE:
        h0 += i_nh * K*V
    if STORE_FINAL_STATE:
        ht += i_nh * K*V

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    if USE_INITIAL_STATE:
        b_h1 += tl.load(h0 + o_k1[:, None] * V + o_v[None, :], mask=m_k1[:, None] & m_v[None, :], other=0.).to(tl.float32)
        if K > 64:
            b_h2 += tl.load(h0 + o_k2[:, None] * V + o_v[None, :], mask=m_k2[:, None] & m_v[None, :], other=0.).to(tl.float32)
        if K > 128:
            b_h3 += tl.load(h0 + o_k3[:, None] * V + o_v[None, :], mask=m_k3[:, None] & m_v[None, :], other=0.).to(tl.float32)
        if K > 192:
            b_h4 += tl.load(h0 + o_k4[:, None] * V + o_v[None, :], mask=m_k4[:, None] & m_v[None, :], other=0.).to(tl.float32)

    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t_int64 * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tv = m_t[:, None] & m_v[None, :]
        p_h = h + i_t_int64 * H*K*V
        tl.store(p_h + o_k1[:, None] * V + o_v[None, :], b_h1, mask=m_k1[:, None] & m_v[None, :])
        if K > 64:
            tl.store(p_h + o_k2[:, None] * V + o_v[None, :], b_h2, mask=m_k2[:, None] & m_v[None, :])
        if K > 128:
            tl.store(p_h + o_k3[:, None] * V + o_v[None, :], b_h3, mask=m_k3[:, None] & m_v[None, :])
        if K > 192:
            tl.store(p_h + o_k4[:, None] * V + o_v[None, :], b_h4, mask=m_k4[:, None] & m_v[None, :])

        # the residual v - w h is formed at tf32 and the state update at tf32x3, both on fp32 operands
        b_w = tl.load(w + o_t[:, None] * (H*K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.)
        b_v = tl.dot(b_w, b_h1, input_precision=DOT_PRECISION_FAST)
        if K > 64:
            b_w = tl.load(w + o_t[:, None] * (H*K) + o_k2[None, :], mask=m_t[:, None] & m_k2[None, :], other=0.)
            b_v += tl.dot(b_w, b_h2, input_precision=DOT_PRECISION_FAST)
        if K > 128:
            b_w = tl.load(w + o_t[:, None] * (H*K) + o_k3[None, :], mask=m_t[:, None] & m_k3[None, :], other=0.)
            b_v += tl.dot(b_w, b_h3, input_precision=DOT_PRECISION_FAST)
        if K > 192:
            b_w = tl.load(w + o_t[:, None] * (H*K) + o_k4[None, :], mask=m_t[:, None] & m_k4[None, :], other=0.)
            b_v += tl.dot(b_w, b_h4, input_precision=DOT_PRECISION_FAST)
        b_v = tl.load(v + o_t[:, None] * (H*V) + o_v[None, :], mask=m_tv, other=0.).to(tl.float32) - b_v
        tl.store(v_new + o_t[:, None] * (H*V) + o_v[None, :], b_v, mask=m_tv)

        last_idx = min((i_t_int64 + 1) * BT, T) - 1
        b_gk_last1 = tl.load(gk + last_idx * H*K + o_k1, mask=m_k1, other=0.).to(tl.float32)
        b_h1 *= exp2(b_gk_last1)[:, None]
        if K > 64:
            b_gk_last2 = tl.load(gk + last_idx * H*K + o_k2, mask=m_k2, other=0.).to(tl.float32)
            b_h2 *= exp2(b_gk_last2)[:, None]
        if K > 128:
            b_gk_last3 = tl.load(gk + last_idx * H*K + o_k3, mask=m_k3, other=0.).to(tl.float32)
            b_h3 *= exp2(b_gk_last3)[:, None]
        if K > 192:
            b_gk_last4 = tl.load(gk + last_idx * H*K + o_k4, mask=m_k4, other=0.).to(tl.float32)
            b_h4 *= exp2(b_gk_last4)[:, None]

        b_k = tl.load(k + o_k1[:, None] + o_t[None, :] * (H*K), mask=m_k1[:, None] & m_t[None, :], other=0.)
        b_h1 += tl.dot(b_k, b_v, input_precision=DOT_PRECISION)
        if K > 64:
            b_k = tl.load(k + o_k2[:, None] + o_t[None, :] * (H*K), mask=m_k2[:, None] & m_t[None, :], other=0.)
            b_h2 += tl.dot(b_k, b_v, input_precision=DOT_PRECISION)
        if K > 128:
            b_k = tl.load(k + o_k3[:, None] + o_t[None, :] * (H*K), mask=m_k3[:, None] & m_t[None, :], other=0.)
            b_h3 += tl.dot(b_k, b_v, input_precision=DOT_PRECISION)
        if K > 192:
            b_k = tl.load(k + o_k4[:, None] + o_t[None, :] * (H*K), mask=m_k4[:, None] & m_t[None, :], other=0.)
            b_h4 += tl.dot(b_k, b_v, input_precision=DOT_PRECISION)

    if STORE_FINAL_STATE:
        tl.store(ht + o_k1[:, None] * V + o_v[None, :], b_h1, mask=m_k1[:, None] & m_v[None, :])
        if K > 64:
            tl.store(ht + o_k2[:, None] * V + o_v[None, :], b_h2, mask=m_k2[:, None] & m_v[None, :])
        if K > 128:
            tl.store(ht + o_k3[:, None] * V + o_v[None, :], b_h3, mask=m_k3[:, None] & m_v[None, :])
        if K > 192:
            tl.store(ht + o_k4[:, None] * V + o_v[None, :], b_h4, mask=m_k4[:, None] & m_v[None, :])


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([2, 3, 4] if check_shared_mem('ampere') else [1])
        for BV in ([32, 64] if check_shared_mem('ada') else [32])
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_diag_kdn_bwd_kernel_dhu(
    q,
    k,
    w,
    gk,
    dht,
    dh0,
    do,
    dh,
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
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v, i_nh = pid % NV, (pid // NV).to(tl.int64)
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

    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    gk += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    dh += (boh * H + i_h) * K*V
    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    if USE_FINAL_STATE_GRADIENT:
        b_dh1 += tl.load(dht + o_k1[:, None] * V + o_v[None, :], mask=m_k1[:, None] & m_v[None, :], other=0.)
        if K > 64:
            b_dh2 += tl.load(dht + o_k2[:, None] * V + o_v[None, :], mask=m_k2[:, None] & m_v[None, :], other=0.)
        if K > 128:
            b_dh3 += tl.load(dht + o_k3[:, None] * V + o_v[None, :], mask=m_k3[:, None] & m_v[None, :], other=0.)
        if K > 192:
            b_dh4 += tl.load(dht + o_k4[:, None] * V + o_v[None, :], mask=m_k4[:, None] & m_v[None, :], other=0.)

    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t_int64 * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tv = m_t[:, None] & m_v[None, :]
        p_dh = dh + i_t_int64 * H*K*V
        tl.store(p_dh + o_k1[:, None] * V + o_v[None, :], b_dh1, mask=m_k1[:, None] & m_v[None, :])
        if K > 64:
            tl.store(p_dh + o_k2[:, None] * V + o_v[None, :], b_dh2, mask=m_k2[:, None] & m_v[None, :])
        if K > 128:
            tl.store(p_dh + o_k3[:, None] * V + o_v[None, :], b_dh3, mask=m_k3[:, None] & m_v[None, :])
        if K > 192:
            tl.store(p_dh + o_k4[:, None] * V + o_v[None, :], b_dh4, mask=m_k4[:, None] & m_v[None, :])

        last_idx = min((i_t_int64 + 1) * BT, T) - 1
        # [BT, BV]
        b_do = tl.load(do + o_t[:, None] * (H*V) + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
        b_k = tl.load(k + o_t[:, None] * (H*K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.)
        b_gk_last1 = tl.load(gk + last_idx * H*K + o_k1, mask=m_k1, other=0.).to(tl.float32)
        b_dv = tl.dot(b_k, b_dh1, input_precision=DOT_PRECISION)
        if K > 64:
            b_k = tl.load(k + o_t[:, None] * (H*K) + o_k2[None, :], mask=m_t[:, None] & m_k2[None, :], other=0.)
            b_gk_last2 = tl.load(gk + last_idx * H*K + o_k2, mask=m_k2, other=0.).to(tl.float32)
            b_dv += tl.dot(b_k, b_dh2, input_precision=DOT_PRECISION)
        if K > 128:
            b_k = tl.load(k + o_t[:, None] * (H*K) + o_k3[None, :], mask=m_t[:, None] & m_k3[None, :], other=0.)
            b_gk_last3 = tl.load(gk + last_idx * H*K + o_k3, mask=m_k3, other=0.).to(tl.float32)
            b_dv += tl.dot(b_k, b_dh3, input_precision=DOT_PRECISION)
        if K > 192:
            b_k = tl.load(k + o_t[:, None] * (H*K) + o_k4[None, :], mask=m_t[:, None] & m_k4[None, :], other=0.)
            b_gk_last4 = tl.load(gk + last_idx * H*K + o_k4, mask=m_k4, other=0.).to(tl.float32)
            b_dv += tl.dot(b_k, b_dh4, input_precision=DOT_PRECISION)
        b_dv += tl.load(dv + o_t[:, None] * (H*V) + o_v[None, :], mask=m_tv, other=0.)
        tl.store(dv2 + o_t[:, None] * (H*V) + o_v[None, :], b_dv.to(dv2.dtype.element_ty), mask=m_tv)
        # the gated query q is fp32 and multiplies do at tf32x3, w multiplies dv on operands in the input dtype
        b_dv = b_dv.to(do.dtype.element_ty)

        # [64, BT]
        b_q = tl.load(q + o_k1[:, None] + o_t[None, :] * (H*K), mask=m_k1[:, None] & m_t[None, :], other=0.)
        b_w = tl.load(w + o_k1[:, None] + o_t[None, :] * (H*K), mask=m_k1[:, None] & m_t[None, :], other=0.)
        b_dh1 *= exp2(b_gk_last1)[:, None]
        b_dh1 += tl.dot(b_q, b_do, input_precision=DOT_PRECISION) * scale - tl.dot(b_w.to(b_dv.dtype), b_dv)
        if K > 64:
            b_q = tl.load(q + o_k2[:, None] + o_t[None, :] * (H*K), mask=m_k2[:, None] & m_t[None, :], other=0.)
            b_w = tl.load(w + o_k2[:, None] + o_t[None, :] * (H*K), mask=m_k2[:, None] & m_t[None, :], other=0.)
            b_dh2 *= exp2(b_gk_last2)[:, None]
            b_dh2 += tl.dot(b_q, b_do, input_precision=DOT_PRECISION) * scale - tl.dot(b_w.to(b_dv.dtype), b_dv)
        if K > 128:
            b_q = tl.load(q + o_k3[:, None] + o_t[None, :] * (H*K), mask=m_k3[:, None] & m_t[None, :], other=0.)
            b_w = tl.load(w + o_k3[:, None] + o_t[None, :] * (H*K), mask=m_k3[:, None] & m_t[None, :], other=0.)
            b_dh3 *= exp2(b_gk_last3)[:, None]
            b_dh3 += tl.dot(b_q, b_do, input_precision=DOT_PRECISION) * scale - tl.dot(b_w.to(b_dv.dtype), b_dv)
        if K > 192:
            b_q = tl.load(q + o_k4[:, None] + o_t[None, :] * (H*K), mask=m_k4[:, None] & m_t[None, :], other=0.)
            b_w = tl.load(w + o_k4[:, None] + o_t[None, :] * (H*K), mask=m_k4[:, None] & m_t[None, :], other=0.)
            b_dh4 *= exp2(b_gk_last4)[:, None]
            b_dh4 += tl.dot(b_q, b_do, input_precision=DOT_PRECISION) * scale - tl.dot(b_w.to(b_dv.dtype), b_dv)

    if USE_INITIAL_STATE:
        tl.store(dh0 + o_k1[:, None] * V + o_v[None, :], b_dh1, mask=m_k1[:, None] & m_v[None, :])
        if K > 64:
            tl.store(dh0 + o_k2[:, None] * V + o_v[None, :], b_dh2, mask=m_k2[:, None] & m_v[None, :])
        if K > 128:
            tl.store(dh0 + o_k3[:, None] * V + o_v[None, :], b_dh3, mask=m_k3[:, None] & m_v[None, :])
        if K > 192:
            tl.store(dh0 + o_k4[:, None] * V + o_v[None, :], b_dh4, mask=m_k4[:, None] & m_v[None, :])


def chunk_diag_kdn_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 32,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V, dtype=torch.float)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    v_new = torch.empty_like(u, dtype=torch.float)

    def grid(meta): return (triton.cdiv(V, meta['BV']) * N * H,)
    chunk_diag_kdn_fwd_kernel_h[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        gk=gk,
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
    )
    return h, v_new, final_state


def chunk_diag_kdn_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    gk: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = 32,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]
    BT = chunk_size
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    dh = q.new_empty(B, NT, H, K, V, dtype=torch.float)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None
    dv2 = torch.empty_like(dv, dtype=torch.float)

    def grid(meta): return (triton.cdiv(V, meta['BV']) * N * H,)
    chunk_diag_kdn_bwd_kernel_dhu[grid](
        q=q,
        k=k,
        w=w,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
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
    return dh, dh0, dv2
