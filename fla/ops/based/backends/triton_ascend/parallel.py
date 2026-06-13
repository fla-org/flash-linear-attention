# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Parallel Based kernels adapted for triton-ascend on NPU."""

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

BTL_FWD = 64
BTL_BWD = 64
BTS = 32


def _npu_flat_grids(g0: int, g1: int, g2: int) -> list[tuple[int, int]]:
    total = g0 * g1 * g2
    grids: list[tuple[int, int]] = []
    for start in range(0, total, 8192):
        grids.append((min(8192, total - start), start))
    return grids


def _npu_bv(v: int) -> int:
    return min(32, max(triton.next_power_of_2(v), 16))


@triton.jit(do_not_specialize=['T'])
def parallel_based_fwd_kernel_npu(
    q,
    k,
    v,
    o,
    z,
    scale,
    T,
    BLOCK_START,
    G0: tl.constexpr,
    G1: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0) + BLOCK_START
    i_bh = pid // (G0 * G1)
    rem = pid % (G0 * G1)
    i_c = rem // G0
    i_kv = rem % G0
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV

    o_k = tl.arange(0, BK) + i_k * BK
    o_v = tl.arange(0, BV) + i_v * BV
    o_q = tl.arange(0, BTL)
    t_q = i_c * BTL + o_q

    m_k = o_k < K
    m_v = o_v < V
    m_q = t_q < T

    b_q = tl.load(
        q + i_bh * T * K + t_q[:, None] * K + o_k[None, :],
        mask=m_q[:, None] & m_k[None, :],
        other=0.0,
    ).to(tl.float32)
    b_q = b_q * scale

    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)

    chunk_start = i_c * BTL

    for start in range(0, chunk_start, BTS):
        o_t = start + tl.arange(0, BTS)
        m_t = o_t < T

        b_k = tl.load(
            k + i_bh * T * K + o_t[None, :] * K + o_k[:, None],
            mask=m_t[None, :] & m_k[:, None],
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            v + i_bh * T * V + o_t[:, None] * V + o_v[None, :],
            mask=m_t[:, None] & m_v[None, :],
            other=0.0,
        ).to(tl.float32)

        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = 1 + b_s + 0.5 * b_s * b_s
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s, b_v, allow_tf32=False)

    for start in range(chunk_start, (i_c + 1) * BTL, BTS):
        o_t = start + tl.arange(0, BTS)
        o_k_rel = start - chunk_start + tl.arange(0, BTS)
        m_t = o_t < T
        m_s = o_q[:, None] >= o_k_rel[None, :]

        b_k = tl.load(
            k + i_bh * T * K + o_t[None, :] * K + o_k[:, None],
            mask=m_t[None, :] & m_k[:, None],
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            v + i_bh * T * V + o_t[:, None] * V + o_v[None, :],
            mask=m_t[:, None] & m_v[None, :],
            other=0.0,
        ).to(tl.float32)

        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s, b_v, allow_tf32=False)

    o_base = o + (i_bh + B * H * i_k) * T * V
    z_base = z + (i_bh + B * H * i_k) * T + i_c * BTL
    tl.store(
        o_base + t_q[:, None] * V + o_v[None, :],
        tl.cast(b_o, o.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_q[:, None] & m_v[None, :],
    )
    tl.store(
        z_base + o_q,
        tl.cast(b_z, z.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_q,
    )


@triton.jit(do_not_specialize=['T'])
def parallel_based_bwd_dq_kernel_npu(
    q,
    k,
    v,
    do,
    dz,
    dq,
    scale,
    T,
    BLOCK_START,
    G0: tl.constexpr,
    G1: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0) + BLOCK_START
    i_bh = pid // (G0 * G1)
    rem = pid % (G0 * G1)
    i_c = rem // G0
    i_kv = rem % G0
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    w_v0 = tl.where(i_v == 0, 1.0, 0.0)

    o_k = tl.arange(0, BK) + i_k * BK
    o_v = tl.arange(0, BV) + i_v * BV
    o_q = tl.arange(0, BTL)
    t_q = i_c * BTL + o_q

    m_k = o_k < K
    m_v = o_v < V
    m_q = t_q < T

    b_q = tl.load(
        q + i_bh * T * K + t_q[:, None] * K + o_k[None, :],
        mask=m_q[:, None] & m_k[None, :],
        other=0.0,
    ).to(tl.float32)
    b_q = b_q * scale

    b_do = tl.load(
        do + i_bh * T * V + t_q[:, None] * V + o_v[None, :],
        mask=m_q[:, None] & m_v[None, :],
        other=0.0,
    ).to(tl.float32)

    b_dz = tl.load(dz + i_bh * T + t_q, mask=m_q, other=0.0).to(tl.float32)
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)

    chunk_start = i_c * BTL

    for start in range(0, chunk_start, BTS):
        o_t = start + tl.arange(0, BTS)
        m_t = o_t < T

        b_k = tl.load(
            k + i_bh * T * K + o_t[:, None] * K + o_k[None, :],
            mask=m_t[:, None] & m_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            v + i_bh * T * V + o_v[:, None] + o_t[None, :] * V,
            mask=m_v[:, None] & m_t[None, :],
            other=0.0,
        ).to(tl.float32)

        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = b_ds + b_dz[:, None] * w_v0
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_dq += tl.dot(b_ds * (1 + b_s), b_k, allow_tf32=False)

    for start in range(chunk_start, (i_c + 1) * BTL, BTS):
        o_t = start + tl.arange(0, BTS)
        o_k_rel = start - chunk_start + tl.arange(0, BTS)
        m_t = o_t < T
        m_s = o_q[:, None] >= o_k_rel[None, :]

        b_k = tl.load(
            k + i_bh * T * K + o_t[:, None] * K + o_k[None, :],
            mask=m_t[:, None] & m_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            v + i_bh * T * V + o_v[:, None] + o_t[None, :] * V,
            mask=m_v[:, None] & m_t[None, :],
            other=0.0,
        ).to(tl.float32)

        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = b_ds + b_dz[:, None] * w_v0
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dq += tl.dot(b_ds + b_ds * b_s, b_k, allow_tf32=False)

    b_dq *= scale
    dq_base = dq + (i_bh + B * H * i_v) * T * K
    tl.store(
        dq_base + t_q[:, None] * K + o_k[None, :],
        tl.cast(b_dq, dq.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_q[:, None] & m_k[None, :],
    )


@triton.jit(do_not_specialize=['T'])
def parallel_based_bwd_dkv_kernel_npu(
    q,
    k,
    v,
    do,
    dz,
    dk,
    dv,
    scale,
    T,
    BLOCK_START,
    G0: tl.constexpr,
    G1: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0) + BLOCK_START
    i_bh = pid // (G0 * G1)
    rem = pid % (G0 * G1)
    i_c = rem // G0
    i_kv = rem % G0
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    w_v0 = tl.where(i_v == 0, 1.0, 0.0)

    o_k = tl.arange(0, BK) + i_k * BK
    o_v = tl.arange(0, BV) + i_v * BV
    o_t_blk = tl.arange(0, BTL)
    t_k = i_c * BTL + o_t_blk

    m_k = o_k < K
    m_v = o_v < V
    m_tk = t_k < T

    b_k = tl.load(
        k + i_bh * T * K + t_k[:, None] * K + o_k[None, :],
        mask=m_tk[:, None] & m_k[None, :],
        other=0.0,
    ).to(tl.float32)
    b_v = tl.load(
        v + i_bh * T * V + t_k[:, None] * V + o_v[None, :],
        mask=m_tk[:, None] & m_v[None, :],
        other=0.0,
    ).to(tl.float32)
    b_dk = tl.zeros([BTL, BK], dtype=tl.float32)
    b_dv = tl.zeros([BTL, BV], dtype=tl.float32)

    chunk_start = i_c * BTL
    chunk_end = (i_c + 1) * BTL
    t_pad = tl.cdiv(T, BTS) * BTS
    stop = chunk_end - BTS

    for j in range(0, tl.maximum((t_pad - BTS - stop) // BTS, 0)):
        i = t_pad - BTS - j * BTS
        o_t = i + tl.arange(0, BTS)
        m_t = o_t < T

        b_q = tl.load(
            q + i_bh * T * K + o_k[:, None] * T + o_t[None, :],
            mask=m_k[:, None] & m_t[None, :],
            other=0.0,
        ).to(tl.float32)
        b_do = tl.load(
            do + i_bh * T * V + o_v[:, None] + o_t[None, :] * V,
            mask=m_v[:, None] & m_t[None, :],
            other=0.0,
        ).to(tl.float32)
        b_dz = tl.load(dz + i_bh * T + o_t, mask=m_t, other=0.0).to(tl.float32)

        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = 1 + b_s + 0.5 * b_s * b_s
        b_dv += tl.dot(b_s2, tl.trans(b_do), allow_tf32=False)
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * scale
        b_ds = b_ds + b_dz[None, :] * scale * w_v0
        b_dk += tl.dot(b_ds + b_ds * b_s, tl.trans(b_q), allow_tf32=False)

    o_q_rel = tl.arange(0, BTS)
    for start in range(chunk_start, chunk_end, BTS):
        o_t = start + o_q_rel
        o_k_rel = start - chunk_start + o_q_rel
        m_t = o_t < T
        m_s = o_t_blk[:, None] <= o_k_rel[None, :]

        b_q = tl.load(
            q + i_bh * T * K + o_k[:, None] * T + o_t[None, :],
            mask=m_k[:, None] & m_t[None, :],
            other=0.0,
        ).to(tl.float32)
        b_do = tl.load(
            do + i_bh * T * V + o_v[:, None] + o_t[None, :] * V,
            mask=m_v[:, None] & m_t[None, :],
            other=0.0,
        ).to(tl.float32)
        b_dz = tl.load(dz + i_bh * T + o_t, mask=m_t, other=0.0).to(tl.float32)

        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)

        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        b_ds = b_ds + b_dz[None, :] * w_v0
        b_ds = tl.where(m_s, b_ds, 0) * scale

        b_dv += tl.dot(b_s2, tl.trans(b_do), allow_tf32=False)
        b_dk += tl.dot(b_ds + b_ds * b_s, tl.trans(b_q), allow_tf32=False)

    dk_base = dk + (i_bh + B * H * i_v) * T * K
    dv_base = dv + (i_bh + B * H * i_k) * T * V
    tl.store(
        dk_base + t_k[:, None] * K + o_k[None, :],
        tl.cast(b_dk, dk.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_tk[:, None] & m_k[None, :],
    )
    tl.store(
        dv_base + t_k[:, None] * V + o_v[None, :],
        tl.cast(b_dv, dv.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_tk[:, None] & m_v[None, :],
    )


def _launch_parallel_kernel(kernel, g0, g1, g2, args, kwargs):
    for nblocks, bstart in _npu_flat_grids(g0, g1, g2):
        kernel[(nblocks,)](*args, BLOCK_START=bstart, **kwargs)
        torch.npu.synchronize()


class ParallelBasedFunctionNPU(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale):
        BTL, BTS_ = BTL_FWD, BTS
        BK = min(128, max(triton.next_power_of_2(k.shape[-1]), 16))
        BV = _npu_bv(v.shape[-1])
        B, H, T, K, V = *k.shape, v.shape[-1]
        NK = triton.cdiv(K, BK)
        NV = triton.cdiv(V, BV)
        g0, g1, g2 = NK * NV, triton.cdiv(T, BTL), B * H

        assert NK == 1, "will encounter some synchronization issue if not."

        o = torch.empty(NK, B, H, T, V, device=q.device)
        z = torch.empty(NK, B, H, T, device=q.device)
        kw = dict(G0=g0, G1=g1, B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS_, BK=BK, BV=BV)
        _launch_parallel_kernel(
            parallel_based_fwd_kernel_npu,
            g0, g1, g2,
            (q, k, v, o, z, scale),
            kw,
        )
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.sizes = (B, H, T, K, V, BTS_)
        return o.sum(0).to(q.dtype), z.sum(0).to(q.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dz):
        q, k, v = ctx.saved_tensors
        scale = ctx.scale
        B, H, T, K, V, BTS_ = ctx.sizes
        BTL = BTL_BWD
        BK = min(128, max(triton.next_power_of_2(k.shape[-1]), 16))
        BV = _npu_bv(v.shape[-1])
        g0, g1, g2 = triton.cdiv(K, BK) * triton.cdiv(V, BV), triton.cdiv(T, BTL), B * H

        dq = torch.empty(triton.cdiv(V, BV), B, H, T, K, dtype=q.dtype, device=q.device)
        dk = torch.empty(triton.cdiv(V, BV), B, H, T, K, dtype=q.dtype, device=q.device)
        dv = torch.empty(triton.cdiv(K, BK), B, H, T, V, dtype=q.dtype, device=q.device)

        kw = dict(G0=g0, G1=g1, B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS_, BK=BK, BV=BV)
        _launch_parallel_kernel(
            parallel_based_bwd_dq_kernel_npu,
            g0, g1, g2,
            (q, k, v, do, dz, dq, scale),
            kw,
        )
        _launch_parallel_kernel(
            parallel_based_bwd_dkv_kernel_npu,
            g0, g1, g2,
            (q, k, v, do, dz, dk, dv, scale),
            kw,
        )

        return dq.sum(0).to(q.dtype), dk.sum(0).to(k.dtype), dv.sum(0).to(v.dtype), None


triton_parallel_based_npu = ParallelBasedFunctionNPU.apply


def parallel_based_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    use_norm: bool = True,
    head_first: bool = False,
):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if not head_first:
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
    o, z = triton_parallel_based_npu(q, k, v, scale)
    if use_norm:
        o = o / (z[..., None] + 1e-6)
    if not head_first:
        o = o.transpose(1, 2)
    return o.to(q.dtype)
