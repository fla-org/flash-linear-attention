# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused-chunk Based kernels adapted for triton-ascend on NPU."""

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

BT = 16
_NPU_MAX_TRITON_GRID = 128


def _npu_flat_grids(g0: int, g1: int, g2: int) -> list[tuple[int, int]]:
    total = g0 * g1 * g2
    grids: list[tuple[int, int]] = []
    for start in range(0, total, _NPU_MAX_TRITON_GRID):
        grids.append((min(_NPU_MAX_TRITON_GRID, total - start), start))
    return grids


def _npu_bk_bv(k: int, v: int) -> tuple[int, int]:
    bk = min(k, 16)
    bv = min(v, 32)
    return max(bk, 16), max(bv, 16)


@triton.jit(do_not_specialize=['T'])
def fused_chunk_based_fwd_kernel_npu(
    q,
    k,
    v,
    o,
    z,
    ws_h0,
    ws_h1,
    ws_h2,
    ws_k0,
    ws_k1,
    ws_k2,
    scale,
    T,
    BLOCK_START,
    G0: tl.constexpr,
    G1: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0) + BLOCK_START
    i_bh = pid // (G0 * G1)
    rem = pid % (G0 * G1)
    i_k = rem // G0
    i_v = rem % G0

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    o_k = tl.arange(0, BK) + i_k * BK
    o_v = tl.arange(0, BV) + i_v * BV
    o_kk = tl.arange(0, BK * BK)
    m_k = o_k < K
    m_v = o_v < V
    m_kk = o_kk < BK * BK

    q_base = q + i_bh * T * K
    k_base = k + i_bh * T * K
    v_base = v + i_bh * T * V
    o_base = o + (i_bh + i_k * B * H) * T * V
    z_base = z + (i_bh + i_k * B * H) * T

    h0_ws = ws_h0 + pid * BV
    h1_ws = ws_h1 + pid * BK * BV
    h2_ws = ws_h2 + pid * BK * BK * BV
    k0_ws = ws_k0 + pid
    k1_ws = ws_k1 + pid * BK
    k2_ws = ws_k2 + pid * BK * BK

    for i in range(0, tl.cdiv(T, BT)):
        b_h_0o = tl.load(h0_ws + o_v, mask=m_v, other=0.0)
        b_h_1o = tl.load(
            h1_ws + o_k[:, None] * BV + o_v[None, :],
            mask=m_k[:, None] & m_v[None, :],
            other=0.0,
        )
        b_h_2o = tl.load(
            h2_ws + o_kk[:, None] * BV + o_v[None, :],
            mask=m_kk[:, None] & m_v[None, :],
            other=0.0,
        )
        k_0o = tl.load(k0_ws)
        k_1o = tl.load(k1_ws + o_k, mask=m_k, other=0.0)
        k_2o = tl.load(k2_ws + o_kk, mask=m_kk, other=0.0)

        t = i * BT + o_i
        m_t = t < T

        b_k = tl.load(
            k_base + t[None, :] * K + o_k[:, None],
            mask=m_k[:, None] & m_t[None, :],
            other=0.0,
        ).to(tl.float32)
        b_k_2o = b_k[:, None, :] * b_k[None, :, :]
        b_k_2o = tl.reshape(b_k_2o, [BK * BK, BT])

        b_v = tl.load(
            v_base + t[:, None] * V + o_v[None, :],
            mask=m_t[:, None] & m_v[None, :],
            other=0.0,
        ).to(tl.float32)
        b_q = tl.load(
            q_base + t[:, None] * K + o_k[None, :],
            mask=m_t[:, None] & m_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b_q = b_q * scale

        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_z = tl.zeros([BT], dtype=tl.float32)
        b_o += b_h_0o
        b_z += k_0o
        b_o += tl.dot(b_q, b_h_1o, allow_tf32=False)
        b_z += tl.sum(b_q * k_1o, axis=1)
        b_q_2o = b_q[:, :, None] * b_q[:, None, :]
        b_q_2o = tl.reshape(b_q_2o, [BT, BK * BK])
        b_o += tl.dot(b_q_2o, b_h_2o, allow_tf32=False) * 0.5
        b_z += tl.sum(b_q_2o * k_2o, axis=1) * 0.5

        k_1o = k_1o + tl.sum(b_k, axis=1)
        k_2o = k_2o + tl.sum(b_k_2o, axis=1)
        k_0o = k_0o + BT

        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s, b_v, allow_tf32=False)

        tl.store(
            o_base + t[:, None] * V + o_v[None, :],
            tl.cast(b_o, o.dtype.element_ty, fp_downcast_rounding='rtne'),
            mask=m_t[:, None] & m_v[None, :],
        )
        tl.store(
            z_base + t,
            tl.cast(b_z, z.dtype.element_ty, fp_downcast_rounding='rtne'),
            mask=m_t,
        )

        b_h_2o = b_h_2o + tl.dot(b_k_2o, b_v, allow_tf32=False)
        b_h_1o = b_h_1o + tl.dot(b_k, b_v, allow_tf32=False)
        b_h_0o = b_h_0o + tl.sum(b_v, axis=0)

        tl.store(h0_ws + o_v, b_h_0o, mask=m_v)
        tl.store(
            h1_ws + o_k[:, None] * BV + o_v[None, :],
            b_h_1o,
            mask=m_k[:, None] & m_v[None, :],
        )
        tl.store(
            h2_ws + o_kk[:, None] * BV + o_v[None, :],
            b_h_2o,
            mask=m_kk[:, None] & m_v[None, :],
        )
        tl.store(k0_ws, k_0o)
        tl.store(k1_ws + o_k, k_1o, mask=m_k)
        tl.store(k2_ws + o_kk, k_2o, mask=m_kk)


@triton.jit(do_not_specialize=['T'])
def fused_chunk_based_bwd_dq_kernel_npu(
    q,
    k,
    v,
    do,
    dz,
    dq,
    ws_bh1,
    ws_bh2,
    ws_k1,
    ws_k2,
    scale,
    T,
    BLOCK_START,
    G0: tl.constexpr,
    G1: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0) + BLOCK_START
    i_bh = pid // (G0 * G1)
    rem = pid % (G0 * G1)
    i_k = rem // G0
    i_v = rem % G0
    w_v0 = tl.where(i_v == 0, 1.0, 0.0)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    o_k = tl.arange(0, BK) + i_k * BK
    o_v = tl.arange(0, BV) + i_v * BV
    o_kk = tl.arange(0, BK * BK)
    m_k = o_k < K
    m_v = o_v < V
    m_kk = o_kk < BK * BK

    q_base = q + i_bh * T * K
    k_base = k + i_bh * T * K
    v_base = v + i_bh * T * V
    do_base = do + i_bh * T * V
    dq_base = dq + (i_bh + i_v * B * H) * T * K
    dz_base = dz + i_bh * T

    bh1_ws = ws_bh1 + pid * BV * BK
    bh2_ws = ws_bh2 + pid * BV * BK * BK
    k1_ws = ws_k1 + pid * BK
    k2_ws = ws_k2 + pid * BK * BK

    for i in range(0, tl.cdiv(T, BT)):
        b_h_1o = tl.load(
            bh1_ws + o_v[:, None] * BK + o_k[None, :],
            mask=m_v[:, None] & m_k[None, :],
            other=0.0,
        )
        b_h_2o = tl.load(
            bh2_ws + o_v[:, None] * (BK * BK) + o_kk[None, :],
            mask=m_v[:, None] & m_kk[None, :],
            other=0.0,
        )
        k_1o = tl.load(k1_ws + o_k, mask=m_k, other=0.0)
        k_2o = tl.load(k2_ws + o_kk, mask=m_kk, other=0.0)

        t = i * BT + o_i
        m_t = t < T

        b_q = tl.load(
            q_base + t[:, None] * K + o_k[None, :],
            mask=m_t[:, None] & m_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b_q = b_q * scale
        b_k = tl.load(
            k_base + t[:, None] * K + o_k[None, :],
            mask=m_t[:, None] & m_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b_do = tl.load(
            do_base + t[:, None] * V + o_v[None, :],
            mask=m_t[:, None] & m_v[None, :],
            other=0.0,
        ).to(tl.float32)
        b_dz = tl.load(dz_base + t, mask=m_t, other=0.0).to(tl.float32)
        b_v = tl.load(
            v_base + o_v[:, None] + t[None, :] * V,
            mask=m_v[:, None] & m_t[None, :],
            other=0.0,
        ).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)

        b_dq += tl.dot(b_do, b_h_1o, allow_tf32=False)
        b_dq += b_dz[:, None] * k_1o * w_v0
        b_dq_2o = tl.dot(b_do, b_h_2o, allow_tf32=False) * 0.5
        b_dq_2o += (b_dz[:, None] * k_2o) * 0.5 * w_v0
        b_dq_2o = tl.reshape(b_dq_2o, [BT, BK, BK])
        b_dq += tl.sum(b_dq_2o * b_q[:, :, None], axis=1)
        b_dq += tl.sum(b_dq_2o * b_q[:, None, :], axis=2)
        b_dq *= scale

        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = b_ds + b_dz[:, None] * w_v0
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dq += tl.dot(b_ds * (1 + b_s), b_k, allow_tf32=False)

        tl.store(
            dq_base + t[:, None] * K + o_k[None, :],
            tl.cast(b_dq, dq.dtype.element_ty, fp_downcast_rounding='rtne'),
            mask=m_t[:, None] & m_k[None, :],
        )

        b_k_2o = b_k[:, :, None] * b_k[:, None, :]
        b_k_2o = tl.reshape(b_k_2o, [BT, BK * BK])
        b_h_2o = b_h_2o + tl.dot(b_v, b_k_2o, allow_tf32=False)
        b_h_1o = b_h_1o + tl.dot(b_v, b_k, allow_tf32=False)

        k_1o = k_1o + tl.sum(b_k, axis=0) * w_v0
        k_2o = k_2o + tl.sum(b_k_2o, axis=0) * w_v0

        tl.store(
            bh1_ws + o_v[:, None] * BK + o_k[None, :],
            b_h_1o,
            mask=m_v[:, None] & m_k[None, :],
        )
        tl.store(
            bh2_ws + o_v[:, None] * (BK * BK) + o_kk[None, :],
            b_h_2o,
            mask=m_v[:, None] & m_kk[None, :],
        )
        tl.store(k1_ws + o_k, k_1o, mask=m_k)
        tl.store(k2_ws + o_kk, k_2o, mask=m_kk)


@triton.jit(do_not_specialize=['T'])
def fused_chunk_based_bwd_dkv_kernel_npu(
    q,
    k,
    v,
    do,
    dz,
    dk,
    dv,
    ws_dh0,
    ws_dh1,
    ws_dh2,
    ws_dq1,
    ws_dq2,
    scale,
    T,
    BLOCK_START,
    G0: tl.constexpr,
    G1: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0) + BLOCK_START
    i_bh = pid // (G0 * G1)
    rem = pid % (G0 * G1)
    i_k = rem // G0
    i_v = rem % G0
    w_v0 = tl.where(i_v == 0, 1.0, 0.0)

    o_k = tl.arange(0, BK) + i_k * BK
    o_v = tl.arange(0, BV) + i_v * BV
    o_kk = tl.arange(0, BK * BK)
    m_k = o_k < K
    m_v = o_v < V
    m_kk = o_kk < BK * BK

    q_base = q + i_bh * T * K
    k_base = k + i_bh * T * K
    v_base = v + i_bh * T * V
    do_base = do + i_bh * T * V
    dk_base = dk + (i_bh + i_v * B * H) * T * K
    dv_base = dv + (i_bh + i_k * B * H) * T * V
    dz_base = dz + i_bh * T

    dh0_ws = ws_dh0 + pid * BV
    dh1_ws = ws_dh1 + pid * BK * BV
    dh2_ws = ws_dh2 + pid * BK * BK * BV
    dq1_ws = ws_dq1 + pid * BK
    dq2_ws = ws_dq2 + pid * BK * BK

    n_chunks = tl.cdiv(T, BT)
    for j in range(0, n_chunks):
        b_dh_0o = tl.load(dh0_ws + o_v, mask=m_v, other=0.0)
        b_dh_1o = tl.load(
            dh1_ws + o_k[:, None] * BV + o_v[None, :],
            mask=m_k[:, None] & m_v[None, :],
            other=0.0,
        )
        b_dh_2o = tl.load(
            dh2_ws + o_kk[:, None] * BV + o_v[None, :],
            mask=m_kk[:, None] & m_v[None, :],
            other=0.0,
        )
        dq_1o = tl.load(dq1_ws + o_k, mask=m_k, other=0.0)
        dq_2o = tl.load(dq2_ws + o_kk, mask=m_kk, other=0.0)

        i = (n_chunks - 1 - j) * BT
        t = i + tl.arange(0, BT)
        m_t = t < T
        m_s = tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]

        b_q = tl.load(
            q_base + o_k[:, None] + t[None, :] * K,
            mask=m_k[:, None] & m_t[None, :],
            other=0.0,
        ).to(tl.float32)
        b_k = tl.load(
            k_base + t[:, None] * K + o_k[None, :],
            mask=m_t[:, None] & m_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            v_base + t[:, None] * V + o_v[None, :],
            mask=m_t[:, None] & m_v[None, :],
            other=0.0,
        ).to(tl.float32)
        b_do = tl.load(
            do_base + t[:, None] * V + o_v[None, :],
            mask=m_t[:, None] & m_v[None, :],
            other=0.0,
        ).to(tl.float32)
        b_dz = tl.load(dz_base + t, mask=m_t, other=0.0).to(tl.float32)
        b_q = b_q * scale

        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)

        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = b_ds + b_dz[None, :] * w_v0
        b_ds = tl.where(m_s, b_ds, 0)
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s2 = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)
        b_ds = b_ds * (1 + b_s)

        b_dk += tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        b_dv += tl.dot(b_s2, b_do, allow_tf32=False)

        b_k_2o = b_k[:, :, None] * b_k[:, None, :]
        b_k_2o = tl.reshape(b_k_2o, [BT, BK * BK])

        b_dv += tl.dot(b_k, b_dh_1o, allow_tf32=False)
        b_dv += tl.dot(b_k_2o, b_dh_2o, allow_tf32=False)
        b_dv += b_dh_0o

        b_dk += tl.dot(b_v, tl.trans(b_dh_1o), allow_tf32=False)
        b_dk += dq_1o * w_v0

        b_dk_2o = tl.dot(b_dh_2o, tl.trans(b_v), allow_tf32=False)
        b_dk_2o += dq_2o[:, None] * w_v0
        b_dk_2o = tl.reshape(b_dk_2o, [BK, BK, BT])
        b_k_fp32 = tl.trans(b_k)
        b_dk2 = tl.sum(b_dk_2o * b_k_fp32[:, None, :], axis=0)
        b_dk2 += tl.sum(b_dk_2o * b_k_fp32[None, :, :], axis=1)
        b_dk += tl.trans(b_dk2)

        b_dh_0o = b_dh_0o + tl.sum(b_do, axis=0)
        b_dh_1o = b_dh_1o + tl.dot(b_q, b_do, allow_tf32=False)
        b_q_2o = b_q[None, :, :] * b_q[:, None, :]
        b_q_2o = tl.reshape(b_q_2o, [BK * BK, BT])
        b_dh_2o = b_dh_2o + tl.dot(b_q_2o, b_do, allow_tf32=False) * 0.5

        dq_1o = dq_1o + tl.sum(b_dz[None, :] * b_q, axis=1) * w_v0
        dq_2o = dq_2o + tl.sum(b_dz[None, :] * b_q_2o, axis=1) * 0.5 * w_v0

        tl.store(
            dk_base + t[:, None] * K + o_k[None, :],
            tl.cast(b_dk, dk.dtype.element_ty, fp_downcast_rounding='rtne'),
            mask=m_t[:, None] & m_k[None, :],
        )
        tl.store(
            dv_base + t[:, None] * V + o_v[None, :],
            tl.cast(b_dv, dv.dtype.element_ty, fp_downcast_rounding='rtne'),
            mask=m_t[:, None] & m_v[None, :],
        )

        tl.store(dh0_ws + o_v, b_dh_0o, mask=m_v)
        tl.store(
            dh1_ws + o_k[:, None] * BV + o_v[None, :],
            b_dh_1o,
            mask=m_k[:, None] & m_v[None, :],
        )
        tl.store(
            dh2_ws + o_kk[:, None] * BV + o_v[None, :],
            b_dh_2o,
            mask=m_kk[:, None] & m_v[None, :],
        )
        tl.store(dq1_ws + o_k, dq_1o, mask=m_k)
        tl.store(dq2_ws + o_kk, dq_2o, mask=m_kk)


def _alloc_bwd_dq_workspace(nblocks: int, bk: int, bv: int, device: torch.device):
    return (
        torch.zeros(nblocks, bv, bk, dtype=torch.float32, device=device),
        torch.zeros(nblocks, bv, bk * bk, dtype=torch.float32, device=device),
        torch.zeros(nblocks, bk, dtype=torch.float32, device=device),
        torch.zeros(nblocks, bk * bk, dtype=torch.float32, device=device),
    )


def _alloc_bwd_dkv_workspace(nblocks: int, bk: int, bv: int, device: torch.device):
    return (
        torch.zeros(nblocks, bv, dtype=torch.float32, device=device),
        torch.zeros(nblocks, bk, bv, dtype=torch.float32, device=device),
        torch.zeros(nblocks, bk * bk, bv, dtype=torch.float32, device=device),
        torch.zeros(nblocks, bk, dtype=torch.float32, device=device),
        torch.zeros(nblocks, bk * bk, dtype=torch.float32, device=device),
    )


def _alloc_fwd_workspace(nblocks: int, bk: int, bv: int, device: torch.device):
    return (
        torch.zeros(nblocks, bv, dtype=torch.float32, device=device),
        torch.zeros(nblocks, bk, bv, dtype=torch.float32, device=device),
        torch.zeros(nblocks, bk * bk, bv, dtype=torch.float32, device=device),
        torch.zeros(nblocks, dtype=torch.float32, device=device),
        torch.zeros(nblocks, bk, dtype=torch.float32, device=device),
        torch.zeros(nblocks, bk * bk, dtype=torch.float32, device=device),
    )


def _launch_fused_chunk_kernel(kernel, g0, g1, g2, args, kwargs):
    for nblocks, bstart in _npu_flat_grids(g0, g1, g2):
        kernel[(nblocks,)](*args, BLOCK_START=bstart, **kwargs)


class FusedChunkBasedFunctionNPU(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale=1):
        B, H, T, K, V = *k.shape, v.shape[-1]
        BT_ = BT
        BK, BV = _npu_bk_bv(K, V)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        g0, g1, g2 = NV, NK, B * H
        nblocks = g0 * g1 * g2

        o = q.new_empty(NK, B, H, T, V, dtype=torch.float32)
        z = q.new_empty(NK, B, H, T, dtype=torch.float32)
        ws_h0, ws_h1, ws_h2, ws_k0, ws_k1, ws_k2 = _alloc_fwd_workspace(nblocks, BK, BV, q.device)

        kw = dict(G0=g0, G1=g1, B=B, H=H, T=T, K=K, V=V, BT=BT_, BK=BK, BV=BV)
        _launch_fused_chunk_kernel(
            fused_chunk_based_fwd_kernel_npu,
            g0, g1, g2,
            (q, k, v, o, z, ws_h0, ws_h1, ws_h2, ws_k0, ws_k1, ws_k2, scale),
            kw,
        )
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.sizes = (B, H, T, K, V, BK, BV, NK, NV)
        return o.sum(0).to(q.dtype), z.sum(0).to(z.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dz):
        q, k, v = ctx.saved_tensors
        B, H, T, K, V, BK, BV, NK, NV = ctx.sizes
        scale = ctx.scale
        g0, g1, g2 = NV, NK, B * H
        nblocks = g0 * g1 * g2

        dq = q.new_empty(NV, B, H, T, K)
        dk = q.new_empty(NV, B, H, T, K)
        dv = q.new_empty(NK, B, H, T, V)
        ws_bh1, ws_bh2, ws_k1, ws_k2 = _alloc_bwd_dq_workspace(nblocks, BK, BV, q.device)
        ws_dh0, ws_dh1, ws_dh2, ws_dq1, ws_dq2 = _alloc_bwd_dkv_workspace(nblocks, BK, BV, q.device)

        kw = dict(G0=g0, G1=g1, B=B, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV)
        _launch_fused_chunk_kernel(
            fused_chunk_based_bwd_dq_kernel_npu,
            g0, g1, g2,
            (q, k, v, do, dz, dq, ws_bh1, ws_bh2, ws_k1, ws_k2, scale),
            kw,
        )
        _launch_fused_chunk_kernel(
            fused_chunk_based_bwd_dkv_kernel_npu,
            g0, g1, g2,
            (q, k, v, do, dz, dk, dv, ws_dh0, ws_dh1, ws_dh2, ws_dq1, ws_dq2, scale),
            kw,
        )
        return dq.sum(0).to(q.dtype), dk.sum(0).to(k.dtype), dv.sum(0).to(v.dtype), None


def fused_chunk_based_npu(
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
    o, z = FusedChunkBasedFunctionNPU.apply(q, k, v, scale)
    if use_norm:
        o = o / (z[..., None] + 1e-6)
    if not head_first:
        o = o.transpose(1, 2)
    return o.to(q.dtype)
