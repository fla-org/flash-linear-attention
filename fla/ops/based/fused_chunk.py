# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.jit(do_not_specialize=['T'])
def fused_chunk_based_fwd_kernel(
    q,
    k,
    v,
    o,
    z,
    scale,  # K ** -0.5
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)

    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BV], zero-order taylor expansion
    b_h_0o = tl.zeros([BV], dtype=tl.float32)
    # [BK, BV], first-order taylor expansion
    b_h_1o = tl.zeros([BK, BV], dtype=tl.float32)
    # [BK, BK, BV] second-order taylor expansion
    b_h_2o = tl.zeros([BK*BK, BV], dtype=tl.float32)

    # make block pointers
    p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k*B*H) * T*V, (T, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0))

    p_z = z + (i_bh + i_k * B * H) * T + tl.arange(0, BT)
    k_2o = tl.zeros([1, BK * BK], dtype=tl.float32)
    k_1o = tl.zeros([1, BK], dtype=tl.float32)
    k_0o = 0

    for i in range(0, tl.cdiv(T, BT)):
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK*BK, BT]
        b_k_2o = b_k[:, None, :] * b_k[None, :, :]
        b_k_2o = tl.reshape(b_k_2o, [BK * BK, BT]).to(b_k.dtype)
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BK]
        b_q = (tl.load(p_q, boundary_check=(0, 1)) * scale).to(b_k.dtype)
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_z = tl.zeros([BT], dtype=tl.float32)

        # interchunk
        # zero-order
        b_o += b_h_0o
        b_z += k_0o
        # first-order
        b_o += tl.dot(b_q, b_h_1o.to(b_q.dtype), allow_tf32=False)
        b_z += tl.sum(b_q * k_1o, axis=1)
        # second-order
        b_q_2o = b_q[:, :, None] * b_q[:, None, :]
        b_q_2o = tl.reshape(b_q_2o, [BT, BK * BK]).to(b_k.dtype)
        b_o += tl.dot(b_q_2o, b_h_2o.to(b_q_2o.dtype), allow_tf32=False) * 0.5
        b_z += tl.sum(b_q_2o * k_2o, axis=1) * 0.5

        # update running statistics
        k_1o += tl.sum(b_k, axis=1)[None, :]
        k_2o += tl.sum(b_k_2o, axis=1)[None, :]
        k_0o += BT

        # intrachunk
        # [BT, BT]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        # [TB, BV]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), mask=(i * BT + tl.arange(0, BT)) < T)

        # update hidden state
        # [BK, BV]
        b_h_2o = b_h_2o + tl.dot(b_k_2o.to(b_v.dtype), b_v, allow_tf32=False)
        b_h_1o = b_h_1o + tl.dot(b_k, b_v, allow_tf32=False)
        b_h_0o = b_h_0o + tl.sum(b_v, axis=0)

        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_z += BT


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_chunk_based_bwd_kernel(
    # NV: number of split in the V dimension. NK: number of split in the K dimension
    q,
    k,
    v,
    do,
    dz,
    dq,
    dk,
    dv,
    scale,  # K ** -0.5
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    # [BV], zero-order taylor expansion
    # b_h_0o = tl.zeros([BV], dtype=tl.float32)
    # [BK, BV], first-order taylor expansion
    b_h_1o = tl.zeros([BV, BK], dtype=tl.float32)
    # [BK, BK, BV] second-order taylor expansion
    b_h_2o = tl.zeros([BV, BK*BK], dtype=tl.float32)

    k_1o = tl.zeros([1, BK], dtype=tl.float32)
    k_2o = tl.zeros([1, BK * BK], dtype=tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v*B*H) * T*K, (T, K), (K, 1), (i*BT, i_k*BK), (BT, BK), (1, 0))
        p_dz = dz + (i_bh) * T + tl.arange(0, BT) + i * BT
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)

        # load tensors
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(tl.arange(0, BT) + i * BT) < T)
        # [BV, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # inter-chunk
        b_dq += tl.dot(b_do, (b_h_1o).to(b_do.dtype), allow_tf32=False)
        if i_v == 0:
            b_dq += b_dz[:, None] * k_1o
        b_dq_2o = tl.dot(b_do, (b_h_2o).to(b_do.dtype), allow_tf32=False) * 0.5
        if i_v == 0:
            b_dq_2o += (b_dz[:, None] * k_2o) * 0.5
        b_dq_2o = tl.reshape(b_dq_2o, [BT, BK, BK])
        b_dq += tl.sum(b_dq_2o * b_q[:, :, None], axis=1)
        b_dq += tl.sum(b_dq_2o * b_q[:, None, :], axis=2)
        b_dq *= scale

        # intra-chunk
        # [BT, BT]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dq += tl.dot((b_ds * (1 + b_s)).to(b_q.dtype), b_k, allow_tf32=False)

        # store
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

        # update hidden state
        # [BT, BK*BK]
        b_k_2o = b_k[:, :, None] * b_k[:, None, :]
        b_k_2o = tl.reshape(b_k_2o, [BT, BK * BK]).to(b_k.dtype)
        # [BV, BK*BK]
        b_h_2o = b_h_2o + tl.dot(b_v, b_k_2o.to(b_v.dtype), allow_tf32=False)
        # [BV, BK]
        b_h_1o = b_h_1o + tl.dot(b_v, b_k, allow_tf32=False)

        if i_v == 0:
            # update running statistics
            k_1o += tl.sum(b_k, axis=0)[None, :]
            k_2o += tl.sum(b_k_2o, axis=0)[None, :]

    tl.debug_barrier()
    b_h_1o = None
    b_h_2o = None

    # [BK, BV], first-order taylor expansion
    b_dh_1o = tl.zeros([BK, BV], dtype=tl.float32)
    # [BK, BK, BV] second-order taylor expansion
    b_dh_2o = tl.zeros([BK*BK, BV], dtype=tl.float32)
    b_dh_0o = tl.zeros([BV], dtype=tl.float32)
    m_s = tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]

    dq_1o = tl.zeros([1, BK], dtype=tl.float32)
    dq_2o = tl.zeros([BK * BK, 1], dtype=tl.float32)

    for i in range(tl.cdiv(T, BT) * BT - BT, -BT, -BT):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k * BK, i), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh+i_v*B*H) * T*K, (T, K), (K, 1), (i, i_k*BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh+i_k*B*H) * T*V, (T, V), (V, 1), (i, i_v*BV), (BT, BV), (1, 0))
        p_dz = dz + (i_bh) * T + tl.arange(0, BT) + i

        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(tl.arange(0, BT)+i) < T)
        b_q = (b_q * scale).to(b_k.dtype)

        # intra chunk
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        b_ds = tl.where(m_s, b_ds, 0)
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s2 = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)
        b_ds *= (1+b_s)

        b_dk += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_q), allow_tf32=False)
        b_dv += tl.dot(b_s2.to(b_do.dtype), b_do, allow_tf32=False)

        # inter chunk
        b_k_2o = b_k[:, :, None] * b_k[:, None, :]
        b_k_2o = tl.reshape(b_k_2o, [BT, BK * BK]).to(b_k.dtype)

        b_dv += tl.dot(b_k, b_dh_1o.to(b_k.dtype), allow_tf32=False)
        b_dv += tl.dot(b_k_2o, b_dh_2o.to(b_k.dtype), allow_tf32=False)
        b_dv += b_dh_0o

        b_dk += tl.dot(b_v, tl.trans(b_dh_1o).to(b_k.dtype), allow_tf32=False)

        if i_v == 0:
            b_dk += dq_1o

        b_dk_2o = tl.dot(b_dh_2o.to(b_k.dtype), tl.trans(b_v), allow_tf32=False)
        if i_v == 0:
            b_dk_2o += dq_2o
        b_dk_2o = tl.reshape(b_dk_2o, [BK, BK, BT])
        b_k_fp32 = tl.trans(b_k.to(tl.float32))
        b_dk2 = tl.sum(b_dk_2o * b_k_fp32[:, None, :], axis=0)
        b_dk2 += tl.sum(b_dk_2o * b_k_fp32[None, :, :], axis=1)
        b_dk += tl.trans(b_dk2)

        # hidden state update
        b_dh_0o += tl.sum(b_do, axis=0)
        b_dh_1o = b_dh_1o + tl.dot(b_q, b_do, allow_tf32=False)
        b_q_2o = b_q[None, :, :] * b_q[:, None, :]
        b_q_2o = tl.reshape(b_q_2o, [BK * BK, BT]).to(b_k.dtype)
        b_dh_2o = b_dh_2o + tl.dot(b_q_2o, b_do, allow_tf32=False) * 0.5

        if i_v == 0:
            dq_1o += (tl.sum(b_dz[None, :] * b_q, axis=1))[None, :]
            dq_2o += (tl.sum(b_dz[None, :] * b_q_2o, axis=1) * 0.5)[:, None]

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class FusedChunkBasedFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale=1):
        B, H, T, K, V = *k.shape, v.shape[-1]

        scale = scale
        BT = 16
        BK, BV = min(K, 16), min(V, 32)
        BK, BV = max(BK, 16), max(BV, 16)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

        num_warps = 4

        # the norm of o might explode, so we need to use float32 here
        o = q.new_empty(NK, B, H, T, V, dtype=torch.float32)
        z = q.new_empty(NK, B, H, T, dtype=torch.float32)

        grid = (NV, NK, B * H)
        fused_chunk_based_fwd_kernel[grid](
            q, k, v, o, z,
            scale,
            T=T, B=B, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
            num_warps=num_warps,
        )
        o = o.sum(0)
        z = z.sum(0)
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        return o.to(q.dtype), z.to(z.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dz):
        q, k, v = ctx.saved_tensors
        B, H, T, K, V = *k.shape, v.shape[-1]
        scale = ctx.scale

        BT = 16
        BK, BV = min(K, 16), min(V, 32)
        BK, BV = max(BK, 16), max(BV, 16)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 4

        dq = q.new_empty(NV, B, H, T, K)
        dk = q.new_empty(NV, B, H, T, K)
        dv = q.new_empty(NK, B, H, T, V)
        grid = (NV, NK, B * H)

        fused_chunk_based_bwd_kernel[grid](
            q, k, v, do, dz, dq, dk, dv,
            scale,
            T=T, B=B, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None


def fused_chunk_based(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    use_norm: bool = True,
    head_first: bool = False
):
    assert q.shape[-1] <= 16, 'only support feature dimension up to 16.'
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if not head_first:
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
    o, z = FusedChunkBasedFunction.apply(q, k, v, scale)
    if use_norm:
        o = o / (z[..., None] + 1e-6)
    if not head_first:
        o = o.transpose(1, 2)
    return o.to(q.dtype)
