
# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous

# Rebased: Linear Transformers with Learnable Kernel Functions are Better In-Context Models
# https://github.com/corl-team/rebased/blob/main/flash_linear_attention/fla/ops/triton/rebased_fast/parallel.py


@triton.jit
def parallel_rebased_fwd_kernel(
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_V]
    v,  # value [B, H, L, D_head_V]
    o,  # output [B, H, L, D_head_V]
    z,  # normalizer [B, H, L]
    scale,  # D_head_K ** -0.5
    B,  # batch size
    H,  # H
    T,  # T
    K: tl.constexpr,  # D_head_K
    V: tl.constexpr,  # D_head_V
    BTL: tl.constexpr,  # BLOCK SIZE along the sequence dimension for Q
    BTS: tl.constexpr,  # BLOCK SIZE along the sequence dimension for K/V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
):
    # i_c: chunk index. used for sequence parallelism
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)

    p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k*BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (0, i_v*BV), (BTS, BV), (1, 0))

    # [BQ, BD] block Q, in the shared memory throughout the whole kernel
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)

    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for _ in range(0, i_c*BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1))

        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BTL, BTS]
        b_s = tl.dot(b_q, (b_k), allow_tf32=False)
        b_s = b_s * b_s
        b_z += tl.sum(b_s, axis=1)

        # [BQ, BD]
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    # # rescale interchunk output
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    # # sync threads, easy for compiler to optimize
    # tl.debug_barrier()

    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k*BK, i_c*BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTS, BV), (1, 0))
    # Q block and K block have overlap. masks required
    for _ in range(i_c*BTL, (i_c + 1) * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        # [BTL, BV]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c*BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_z, b_z.to(p_z.dtype.element_ty), mask=((i_c*BTL + tl.arange(0, BTL)) < T))


@triton.jit
def _parallel_rebased_bwd_dq(
    i_bh,
    i_c,
    i_k,
    i_v,
    i_h,
    q,
    k,
    v,
    do,
    dz,
    dq,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + (i_bh) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
    b_q = (b_q * scale).to(b_q.dtype)
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (0, i_k*BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v*BV, 0), (BV, BTS), (0, 1))
    p_dz = dz + i_bh * T + i_c*BTL + tl.arange(0, BTL)
    b_dz = tl.load(p_dz, mask=(i_c*BTL + tl.arange(0, BTL)) < T)

    for _ in range(0, i_c*BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BTL, BTS]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        # [BQ, BD]
        b_dq += tl.dot((2 * b_ds * b_s).to(b_v.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))

    b_dq *= scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v*BV, i_c*BTL), (BV, BTS), (0, 1))
    # Q block and K block have overlap. masks required
    for _ in range(i_c*BTL, (i_c + 1) * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        # [BTL, BK]
        b_dq += tl.dot((2 * b_ds * b_s).to(b_k.dtype),
                       b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    return


@triton.jit
def _parallel_rebased_bwd_dkv(
    i_bh,
    i_c,
    i_k,
    i_v,
    i_h,
    q,
    k,
    v,
    do,
    dz,
    dk,
    dv,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # compute dk dv
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1))
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros(
        [BTL, BV], dtype=tl.float32)

    for i in range((tl.cdiv(T, BTS) * BTS)-BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k*BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (V, T), (1, V), (i_v*BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        # [BK, BTS]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BV, BTS]
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T)
        # [BTL, BTS]
        b_s = tl.dot(b_k.to(b_q.dtype), b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * scale
        if i_v == 0:
            b_ds += b_dz[None, :] * scale
        else:
            b_ds = b_ds
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=False)

    tl.debug_barrier()
    o_q, o_k = tl.arange(0, BTS), tl.arange(0, BTL)
    for i in range(i_c*BTL, (i_c+1)*BTL, BTS):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k*BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (V, T), (1, V), (i_v*BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1))  # [BD, BQ]
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T)
        # [BK, BQ]
        m_s = o_k[:, None] <= o_q[None, :]
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)

        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        # [BK, BD]
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=False)
        o_q += BTS

    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    return


@triton.jit
def parallel_rebased_bwd_kernel(
    q,
    k,
    v,
    do,
    dz,
    dq,
    dk,
    dv,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    _parallel_rebased_bwd_dq(
        i_bh,
        i_c,
        i_k,
        i_v,
        i_h,
        q,
        k,
        v,
        do,
        dz,
        dq,
        scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BTL=BTL,
        BTS=BTS,
        BK=BK,
        BV=BV
    )
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(
        i_bh,
        i_c,
        i_k,
        i_v,
        i_h,
        q,
        k,
        v,
        do,
        dz,
        dk,
        dv,
        scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BTL=BTL,
        BTS=BTS,
        BK=BK,
        BV=BV
    )


class ParallelBasedFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale):
        BTL, BTS = 128, 32
        assert BTL % BTS == 0
        # assert q.shape[-1] % 16 == 0
        BK = min(128, triton.next_power_of_2(k.shape[-1]))
        BV = min(128, triton.next_power_of_2(v.shape[-1]))
        BK, BV = max(BK, 16), max(BV, 16)
        B, H, T, K, V = *k.shape, v.shape[-1]
        num_stages = 2
        num_warps = 4
        NK = triton.cdiv(K, BK)
        NV = triton.cdiv(V, BV)
        grid = (NK * NV, triton.cdiv(T, BTL), B * H)

        assert NK == 1, "will encounter some synchronization issue if not."

        o = torch.empty(NK, B, H, T, V, device=q.device)
        z = torch.empty(NK, B, H, T, device=q.device)
        parallel_rebased_fwd_kernel[grid](
            q, k, v, o, z,
            scale,
            B=B, H=H, T=T, K=K, V=V,
            BTL=BTL, BTS=BTS, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        return o.sum(0).to(q.dtype), z.sum(0).to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dz):
        q, k, v = ctx.saved_tensors
        scale = ctx.scale
        BTL, BTS = 64, 32
        assert BTL % BTS == 0
        BK = min(128, triton.next_power_of_2(k.shape[-1]))
        BV = min(128, triton.next_power_of_2(v.shape[-1]))
        BK, BV = max(BK, 16), max(BV, 16)
        B, H, T, K, V = *k.shape, v.shape[-1]
        num_stages = 2
        num_warps = 4
        NK = triton.cdiv(K, BK)
        NV = triton.cdiv(V, BV)
        grid = (NK * NV, triton.cdiv(T, BTL), B * H)

        assert NK == 1, "will encounter some synchronization issue if not"

        dq = torch.empty(NV, B, H, T, K, dtype=q.dtype, device=q.device)
        dk = torch.empty(NV, B, H, T, K, dtype=q.dtype, device=q.device)
        dv = torch.empty(NK, B, H, T, V, dtype=q.dtype, device=q.device)

        parallel_rebased_bwd_kernel[grid](
            q, k, v, do, dz, dq, dk, dv,
            scale,
            B=B, H=H, T=T, K=K, V=V,
            BTL=BTL, BTS=BTS, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )

        return dq.sum(0).to(q.dtype), dk.sum(0).to(k.dtype), dv.sum(0).to(v.dtype), None


triton_parallel_based = ParallelBasedFunction.apply


def parallel_rebased(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-5,
    use_scale: bool = True,
    use_normalize: bool = True,
    return_both: bool = False,
    head_first: bool = True
):
    assert q.shape[-1] <= 128, "only support feature dim up to 128"
    if use_scale:
        scale = q.shape[-1] ** -0.5
    else:
        scale = 1
    if not head_first:
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
    o, z = triton_parallel_based(q, k, v, scale)
    if return_both:
        return o, z
    if use_normalize:
        o = o / (z[..., None] + eps)
    if not head_first:
        o = o.transpose(1, 2)
    return o.to(q.dtype)
