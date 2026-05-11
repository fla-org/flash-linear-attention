# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl
import torch.nn.functional as F

from fla.utils import input_guard
from fla.ops.utils.softmax import softmax_bwd


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['BN'],
)
@triton.jit
def _fused_softmax_topk_fwd_kernel(
    e,
    e_o,
    mw,
    mr,
    stride_e_b,
    stride_e_l,
    B,
    T,
    N,
    NUM_WRITER: tl.constexpr,
    NUM_READER: tl.constexpr,
    BN: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)

    offsets_n = tl.arange(0, BN)
    mask_n = offsets_n < N
    p_e = e + i_b * stride_e_b + i_t * stride_e_l + offsets_n
    p_e_o = e_o + i_b * stride_e_b + i_t * stride_e_l + offsets_n
    p_mw = mw + i_b * stride_e_b + i_t * stride_e_l + offsets_n
    p_mr = mr + i_b * stride_e_b + i_t * stride_e_l + offsets_n

    ### stable softmax and topk ###
    b_e = tl.load(p_e, mask=mask_n, other=-float('inf')).to(tl.float32)
    b_m = tl.max(b_e, axis=0)
    b_e = tl.exp(b_e - b_m)
    b_p = b_e / tl.sum(b_e, axis=0)
    b_p = tl.where(mask_n, b_p.to(p_e.dtype.element_ty), -float('inf'))
    b_ps = tl.sort(b_p, descending=True)
    tl.store(p_e_o, b_p.to(p_e_o.dtype.element_ty), mask=mask_n)

    mask_w = tl.full((BN,), 1, dtype=b_p.dtype)
    if NUM_WRITER < N:
        threshold_w = tl.sum(b_ps * (offsets_n == NUM_WRITER - 1))        
        mask_w_gr = b_p > threshold_w
        need = NUM_WRITER - tl.sum(mask_w_gr.to(tl.int32))
        mask_w_eq = b_p == threshold_w
        mask_w_eq_need = mask_w_eq & (tl.cumsum(mask_w_eq.to(tl.int32), axis=0) <= need)
        mask_w = mask_w_gr | mask_w_eq_need
        mask_w = mask_w.to(b_p.dtype)
    tl.store(p_mw, mask_w.to(p_mw.dtype.element_ty), mask=mask_n)

    mask_r = tl.full((BN,), 1, dtype=b_p.dtype)
    if NUM_READER < N:
        threshold_r = tl.sum(b_ps * (offsets_n == NUM_READER - 1))
        mask_r_gr = b_p > threshold_r
        need = NUM_READER - tl.sum(mask_r_gr.to(tl.int32))
        mask_r_eq = b_p == threshold_r
        mask_r_eq_need = mask_r_eq & (tl.cumsum(mask_r_eq.to(tl.int32), axis=0) <= need)
        mask_r = mask_r_gr | mask_r_eq_need
        mask_r = mask_r.to(b_p.dtype)
    tl.store(p_mr, mask_r.to(p_mr.dtype.element_ty), mask=mask_n)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['BN', 'BK', 'BV'],
)
@triton.jit
def _fused_mask_fwd_kernel(
    q, k, v, g, e, mw, mr,
    q_o, k_o, v_o, g_o,
    stride_k_b, stride_k_l, stride_k_h,
    stride_v_b, stride_v_l, stride_v_h,
    stride_e_b, stride_e_l,
    B, T, N, H, K, V,
    BN: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    i_b, i_t, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    offsets_n = tl.arange(0, BN)
    offsets_k = tl.arange(0, BK)
    offsets_v = tl.arange(0, BV)
    mask_n = offsets_n < N
    mask_k = offsets_k < K
    mask_v = offsets_v < V

    p_e = e + i_b * stride_e_b + i_t * stride_e_l + offsets_n
    p_mw = mw + i_b * stride_e_b + i_t * stride_e_l + offsets_n
    p_mr = mr + i_b * stride_e_b + i_t * stride_e_l + offsets_n

    p_q = q + i_b * stride_k_b + i_t * stride_k_l + i_h * stride_k_h + offsets_k
    p_k = k + i_b * stride_k_b + i_t * stride_k_l + i_h * stride_k_h + offsets_k
    p_g = g + i_b * stride_k_b + i_t * stride_k_l + i_h * stride_k_h + offsets_k
    p_v = v + i_b * stride_v_b + i_t * stride_v_l + i_h * stride_v_h + offsets_v
    p_q_o = q_o + i_b * stride_k_b * N + i_t * stride_k_l * N + i_h * stride_k_h \
        + offsets_n[:, None] * H * K + offsets_k[None, :]
    p_k_o = k_o + i_b * stride_k_b * N + i_t * stride_k_l * N + i_h * stride_k_h \
        + offsets_n[:, None] * H * K + offsets_k[None, :]
    p_g_o = g_o + i_b * stride_k_b * N + i_t * stride_k_l * N + i_h * stride_k_h \
        + offsets_n[:, None] * H * K + offsets_k[None, :]
    p_v_o = v_o + i_b * stride_v_b * N + i_t * stride_v_l * N + i_h * stride_v_h \
        + offsets_n[:, None] * H * V + offsets_v[None, :]

    b_e = tl.load(p_e, mask=mask_n, other=0.)
    mask_w = tl.load(p_mw, mask=mask_n, other=0.).to(b_e.dtype)
    mask_r = tl.load(p_mr, mask=mask_n, other=0.).to(b_e.dtype)
    b_e_topk_w = b_e * mask_w
    b_e_topk_r = b_e * mask_r

    ### mask qkvg ###
    b_q = tl.load(p_q, mask=mask_k, other=0.)
    b_q = b_q[None, :] * b_e_topk_r[:, None]

    b_k = tl.load(p_k, mask=mask_k, other=0.)
    b_k = b_k[None, :] * b_e_topk_w[:, None]

    b_g = tl.load(p_g, mask=mask_k, other=0.)
    b_g = b_g[None, :] * mask_w[:, None]
    
    b_v = tl.load(p_v, mask=mask_v, other=0.)
    b_v = b_v[None, :] * mask_w[:, None]

    mask_nk = mask_n[:, None] & mask_k[None, :]
    mask_nv = mask_n[:, None] & mask_v[None, :]
    tl.store(p_q_o, b_q.to(p_q_o.dtype.element_ty), mask=mask_nk)
    tl.store(p_k_o, b_k.to(p_k_o.dtype.element_ty), mask=mask_nk)
    tl.store(p_g_o, b_g.to(p_g_o.dtype.element_ty), mask=mask_nk)
    tl.store(p_v_o, b_v.to(p_v_o.dtype.element_ty), mask=mask_nv)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['BN', 'BK', 'BV'],
)
@triton.jit
def _fused_mask_bwd_kernel(
    q, k, e, mw, mr,
    dq_o, dk_o, dv_o, dg_o,
    dq, dk, dv, dg, de,
    stride_k_b, stride_k_l, stride_k_h,
    stride_v_b, stride_v_l, stride_v_h,
    stride_e_b, stride_e_l,
    B, T, N, H, K, V,
    BN: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    i_b, i_t, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    offsets_n = tl.arange(0, BN)
    offsets_k = tl.arange(0, BK)
    offsets_v = tl.arange(0, BV)
    mask_n = offsets_n < N
    mask_k = offsets_k < K
    mask_v = offsets_v < V

    p_e = e + i_b * stride_e_b + i_t * stride_e_l + offsets_n
    p_de = de + (i_b * stride_e_b + i_t * stride_e_l + offsets_n) * H + i_h
    p_mw = mw + i_b * stride_e_b + i_t * stride_e_l + offsets_n
    p_mr = mr + i_b * stride_e_b + i_t * stride_e_l + offsets_n

    p_q = q + i_b * stride_k_b + i_t * stride_k_l + i_h * stride_k_h + offsets_k
    p_k = k + i_b * stride_k_b + i_t * stride_k_l + i_h * stride_k_h + offsets_k
    p_dq = dq + i_b * stride_k_b + i_t * stride_k_l + i_h * stride_k_h + offsets_k
    p_dk = dk + i_b * stride_k_b + i_t * stride_k_l + i_h * stride_k_h + offsets_k
    p_dg = dg + i_b * stride_k_b + i_t * stride_k_l + i_h * stride_k_h + offsets_k
    p_dv = dv + i_b * stride_v_b + i_t * stride_v_l + i_h * stride_v_h + offsets_v
    p_dq_o = dq_o + i_b * stride_k_b * N + i_t * stride_k_l * N + i_h * stride_k_h \
        + offsets_n[:, None] * H * K + offsets_k[None, :]
    p_dk_o = dk_o + i_b * stride_k_b * N + i_t * stride_k_l * N + i_h * stride_k_h \
        + offsets_n[:, None] * H * K + offsets_k[None, :]
    p_dg_o = dg_o + i_b * stride_k_b * N + i_t * stride_k_l * N + i_h * stride_k_h \
        + offsets_n[:, None] * H * K + offsets_k[None, :]
    p_dv_o = dv_o + i_b * stride_v_b * N + i_t * stride_v_l * N + i_h * stride_v_h \
        + offsets_n[:, None] * H * V + offsets_v[None, :]

    b_e = tl.load(p_e, mask=mask_n, other=0.)
    mask_w = tl.load(p_mw, mask=mask_n, other=0.).to(b_e.dtype)
    mask_r = tl.load(p_mr, mask=mask_n, other=0.).to(b_e.dtype)
    b_e_topk_w = b_e * mask_w
    b_e_topk_r = b_e * mask_r

    mask_nk = mask_n[:, None] & mask_k[None, :]
    mask_nv = mask_n[:, None] & mask_v[None, :]
    b_dq_o = tl.load(p_dq_o, mask=mask_nk, other=0.)
    b_dk_o = tl.load(p_dk_o, mask=mask_nk, other=0.)
    b_dg_o = tl.load(p_dg_o, mask=mask_nk, other=0.)
    b_dv_o = tl.load(p_dv_o, mask=mask_nv, other=0.)
    b_dq = tl.sum((b_dq_o * b_e_topk_r[:, None]).to(tl.float32), axis=0).to(b_dq_o.dtype)
    b_dk = tl.sum((b_dk_o * b_e_topk_w[:, None]).to(tl.float32), axis=0).to(b_dk_o.dtype)
    b_dg = tl.sum((b_dg_o * mask_w[:, None]).to(tl.float32), axis=0).to(b_dg_o.dtype)
    b_dv = tl.sum((b_dv_o * mask_w[:, None]).to(tl.float32), axis=0).to(b_dv_o.dtype)

    b_q = tl.load(p_q, mask=mask_k, other=0.)
    b_k = tl.load(p_k, mask=mask_k, other=0.)
    b_de = b_dq_o * b_q[None, :] * mask_r[:, None] + b_dk_o * b_k[None, :] * mask_w[:, None]
    b_de = tl.sum(b_de.to(tl.float32), axis=1).to(b_de.dtype)  

    tl.store(p_de, b_de.to(p_de.dtype.element_ty), mask=mask_n)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=mask_k)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)
        

class SoftmaxAndMask(torch.autograd.Function):
    r"""
    Applies softmax to router weights, repeats and masks inputs,
    scales queries and keys with the router weights, and generates reader/writer masks.

    Notation:
        B: batch size
        T: sequence length
        H: number of attention heads
        K: key/query head dimension
        V: value head dimension
        N: number of state partitions

    Args:
        q (torch.Tensor):
            Queries of shape `(B, T, H, K)`.
        k (torch.Tensor):
            Keys of shape `(B, T, H, K)`.
        v (torch.Tensor):
            Values of shape `(B, T, H, V)`.
        g (torch.Tensor):
            Gates of shape `(B, T, H, V)`.
        e (torch.Tensor):
            Router weights before softmax of shape `(B, T, N)`.
        num_writer (int):
            Number of state partitions to write.
        num_reader (int):
            Number of state partitions to read.

    Returns:
        q_out (torch.Tensor):
            Repeated and masked queries of shape `(B, T, N * H, K)`.
        k_out (torch.Tensor):
            Repeated and masked keys of shape `(B, T, N * H, K)`.
        v_out (torch.Tensor):
            Repeated and masked values of shape `(B, T, N * H, V)`.
        g_out (torch.Tensor):
            Repeated and masked gates of shape `(B, T, N * H, V)`.
        e_out (torch.Tensor):
            Router weights after softmax of shape `(B, T, N)`.
        mask_w (torch.Tensor):
            Writer mask of shape `(B, T, N)`.
        mask_r (torch.Tensor):
            Reader mask of shape `(B, T, N)`.
    """

    @staticmethod
    @input_guard
    def forward(ctx, q, k, v, g, e, num_writer, num_reader):
        B, T, H, K, V, N = *k.shape, v.shape[-1], e.shape[-1]
        BN = triton.next_power_of_2(N)
        BK = triton.next_power_of_2(K)
        BV = triton.next_power_of_2(V)

        q_out = q.new_empty(B, T, N * H, K)
        k_out = k.new_empty(B, T, N * H, K)
        v_out = v.new_empty(B, T, N * H, V)
        g_out = g.new_empty(B, T, N * H, K)
        e_out = torch.empty_like(e)
        mask_w = torch.empty_like(e, dtype=torch.int32)
        mask_r = torch.empty_like(e, dtype=torch.int32)

        _fused_softmax_topk_fwd_kernel[(B, T)](
            e,
            e_out,
            mask_w,
            mask_r,
            e.stride(0),
            e.stride(1),
            B,
            T,
            N,
            NUM_WRITER=num_writer,
            NUM_READER=num_reader,
            BN=BN,
        )

        _fused_mask_fwd_kernel[(B, T, H)](
            q, k, v, g, e_out, mask_w, mask_r,
            q_out, k_out, v_out, g_out,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            e.stride(0), e.stride(1),
            B, T, N, H, K, V,
            BN=BN,
            BK=BK,
            BV=BV,
        )
        
        ctx.save_for_backward(q, k, v, g, e_out, mask_w, mask_r)
        ctx.num_writer = num_writer
        ctx.num_reader = num_reader
        return q_out, k_out, v_out, g_out, e_out, mask_w, mask_r

    @staticmethod
    @input_guard
    def backward(ctx, dq_out, dk_out, dv_out, dg_out, de_out, dmask_w, dmask_r):
        q, k, v, g, e_out, mask_w, mask_r = ctx.saved_tensors

        B, T, H, K, V, N = *k.shape, v.shape[-1], e_out.shape[-1]
        BN = triton.next_power_of_2(N)
        BK = triton.next_power_of_2(K)
        BV = triton.next_power_of_2(V)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dg = torch.empty_like(g)
        de = g.new_empty(B, T, N, H)

        grid = (B, T, H)
        
        _fused_mask_bwd_kernel[grid](
            q, k, e_out, mask_w, mask_r,
            dq_out, dk_out, dv_out, dg_out,
            dq, dk, dv, dg, de,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            e_out.stride(0), e_out.stride(1),
            B, T, N, H, K, V,
            BN=BN,
            BK=BK,
            BV=BV,
        )

        de = de.sum(dim=-1).add_(de_out)
        de = softmax_bwd(e_out, de, dtype=de.dtype)
        
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), de.to(e_out), None, None

softmax_and_mask = SoftmaxAndMask.apply
