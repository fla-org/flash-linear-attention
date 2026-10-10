# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend kernels for the momentum delta rule chunk forward.

Decomposition mirrors the torch base in ``fla/ops/common/chunk_momentum_delta.py``:
kkt (masked p@k^T) -> fused inverse + three RHS -> two-state inter recurrence -> output.
All math is fp32 with ``allow_tf32=False`` (the state recurrence amplifies bf16 noise
past the parity tolerance); every ``exp`` becomes ``exp2`` on pre-scaled log values
(the wrapper multiplies by log2(e), the repo-wide gate convention).

Layouts:
  - gate vectors `[B, H, NT, BT]` fp32 contiguous, one `[BT]` vector per (b, h, n).
  - the kkt ``A`` tensor is `[B, T, H, BT]` (chunk row ``t`` stores its ``BT`` columns).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.utils import input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size

from .chunk_delta_h import _launch_core_grid

LOG2E: float = 1.4426950408889634
_SAFETY_MARGIN = 1.2
_K_MEM_MULT = 16.0   # kkt: fp32 [BT, BT] acc + two fp32 [BT, BK] operands
_H_MEM_MULT = 16.0   # fwd_h/o: both states live in fp32
_MAX_BK = 64


def _get_kkt_bk(BT: int, K: int) -> int:
    return compute_row_tile_block_size(
        BT,
        K,
        _K_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=32,
        min_block=16,
        max_block=min(_MAX_BK, triton.next_power_of_2(K)),
    )


def _select_state_tiles(K: int, V: int) -> tuple[int, int]:
    """(BK, BV) for the two-state fwd kernels; both states stay live in fp32."""
    BK = min(_MAX_BK, triton.next_power_of_2(max(K, 16)))
    BV = compute_row_tile_block_size(
        min(K, BK),
        V,
        _H_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=16,
        min_block=16,
        max_block=min(64, triton.next_power_of_2(V)),
    )
    return BK, BV


def _exp2_vec(x: torch.Tensor) -> torch.Tensor:
    """Host helper: convert natural-log values to exp2 units (x * log2(e))."""
    return (x * LOG2E).contiguous()


# ---------------------------------------------------------------- K1: kkt


@triton.jit(do_not_specialize=['T', 'B', 'task_num', 'num_core'])
def chunk_momentum_delta_kkt_fwd_kernel_npu(
    p_eff,
    k_eta,
    cfac,
    lct1,
    b_tm1,
    A,
    T,
    task_num,
    num_core,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    """attn = (p_eff @ k_eta^T) * gamma_mask, packed row-wise into [B, T, H, BT]."""
    core_id = tl.program_id(0)
    NT = tl.cdiv(T, BT)
    for task in tl.range(core_id, task_num, num_core):
        i_t = task % NT
        i_nh = task // NT
        i_n = i_nh // H
        i_h = i_nh % H

        o_t = i_t * BT + tl.arange(0, BT)
        o_i = tl.arange(0, BT)
        m_t = o_t < T
        b_r = tl.arange(0, BT)   # local row index within the chunk

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_p = tl.make_block_ptr(
                p_eff + i_n * T * H * K + i_h * K, (T, K), (H * K, 1),
                (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            p_k = tl.make_block_ptr(
                k_eta + i_n * T * H * K + i_h * K, (T, K), (H * K, 1),
                (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            b_p = tl.load(p_p, boundary_check=(0, 1)).to(tl.float32)
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            # ascend tl.dot clobbers the lhs; copy first. in-place dot into the
            # loop-carried b_A crashes bisheng (ConvertLinalgRToBinary); keep the
            # add separate with a non-const zero acc
            b_A = tl.dot(b_p + 0.0, tl.trans(b_k + 0.0), b_A * 0.0, allow_tf32=False) + b_A

        b_tm1_v = tl.load(b_tm1 + (i_n * H + i_h) * NT * BT + i_t * BT + o_i)
        b_cfac = tl.load(cfac + (i_n * H + i_h) * NT * BT + i_t * BT + o_i)
        b_lct1 = tl.load(lct1 + (i_n * H + i_h) * NT * BT + i_t * BT + o_i)
        # gamma_mask[i, j] = b_tm1[i] * cfac[j] * (1 - exp2(lct1[j] - lct1[i])), strictly lower.
        # mask the exponent first (gdn kkt convention): lct1 slot 0 is -inf, and an
        # unmasked inf diff would poison the product with NaN even under tl.where.
        b_diff = b_lct1[None, :] - b_lct1[:, None]
        b_diff = tl.where(b_r[:, None] > b_r[None, :], b_diff, 0.0)
        b_A *= b_tm1_v[:, None] * b_cfac[None, :] * (1.0 - tl.exp2(b_diff))

        p_A = A + (i_n * NT * BT) * H * BT + o_t[:, None] * (H * BT) + i_h * BT + o_i[None, :]
        tl.store(p_A, b_A, mask=m_t[:, None])


@input_guard
def chunk_momentum_delta_kkt_fwd_npu(
    p_eff: torch.Tensor,
    k_eta: torch.Tensor,
    cfac: torch.Tensor,
    lct1: torch.Tensor,
    b_tm1: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    B, T, H, K = p_eff.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    BK = _get_kkt_bk(BT, K)
    # same contract as the torch base: padded rows so the wy stage can reshape [B, pT, H, BT]
    A = torch.zeros(B, NT * BT, H, BT, device=p_eff.device, dtype=torch.float32)
    _launch_core_grid(
        chunk_momentum_delta_kkt_fwd_kernel_npu,
        task_num=NT * B * H,
        kernel_kwargs=dict(
            p_eff=p_eff,
            k_eta=k_eta,
            cfac=cfac,
            lct1=_exp2_vec(lct1),
            b_tm1=b_tm1,
            A=A,
            T=T,
        ),
        H=H, K=K, BT=BT, BK=BK,
    )
    return A


# ------------------------------------------------------- K2: inverse + 3 RHS


@triton.jit(do_not_specialize=['T', 'B', 'task_num', 'num_core'])
def chunk_momentum_delta_wy_fwd_kernel_npu(
    A,
    v,
    p_eff,
    bar_a_tm1,
    b_tm1,
    u_c,
    y_c,
    z_c,
    T,
    task_num,
    num_core,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Ai = (I + attn)^{-1} via the reference's substitution loop, then its three RHS."""
    core_id = tl.program_id(0)
    NT = tl.cdiv(T, BT)
    for task in tl.range(core_id, task_num, num_core):
        i_t = task % NT
        i_nh = task // NT
        i_n = i_nh // H
        i_h = i_nh % H

        o_t = i_t * BT + tl.arange(0, BT)
        o_i = tl.arange(0, BT)
        m_t = o_t < T
        b_r = tl.arange(0, BT)

        p_A = A + (i_n * NT * BT) * H * BT + o_t[:, None] * (H * BT) + i_h * BT + o_i[None, :]
        b_A = tl.load(p_A, mask=m_t[:, None], other=0.0).to(tl.float32)

        # substitution loop, same update order as the torch reference:
        # inv = -A; for i: inv[i, :i] += sum_j inv[i, j] * inv[j, :i]; inv += I
        # tl.range (not static_range): static unrolling BT=64 dots blows up the
        # bisheng compile time (looks like a hang)
        b_inv = -b_A
        for i in tl.range(1, BT):
            # row i: inv[i, :i] += sum_j inv[i, j] * inv[j, :i] — a matmul, not elementwise
            b_row_i = tl.where(b_r[:, None] == i, b_inv, 0.0)
            b_cols = tl.where(b_r[None, :] < i, b_inv, 0.0)
            b_upd = tl.dot(b_row_i, b_cols, allow_tf32=False)
            b_inv += tl.where((b_r[:, None] == i) & (b_r[None, :] < i), b_upd, 0.0)
        b_inv = tl.where(b_r[:, None] == b_r[None, :], b_inv + 1.0, b_inv)

        p_bar_a = bar_a_tm1 + (i_n * H + i_h) * NT * BT + o_t
        p_b_tm1 = b_tm1 + (i_n * H + i_h) * NT * BT + o_t
        b_bar_a = tl.load(p_bar_a, mask=m_t, other=0.0)
        b_btm1 = tl.load(p_b_tm1, mask=m_t, other=0.0)

        for i_v in range(tl.cdiv(V, BV)):
            p_v = tl.make_block_ptr(
                v + i_n * T * H * V + i_h * V, (T, V), (H * V, 1),
                (i_t * BT, i_v * BV), (BT, BV), (1, 0),
            )
            p_u = tl.make_block_ptr(
                u_c + (i_n * NT * BT * H + i_h) * V, (T, V), (H * V, 1),
                (i_t * BT, i_v * BV), (BT, BV), (1, 0),
            )
            b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
            b_u = tl.dot(b_inv, b_v, allow_tf32=False)
            tl.store(p_u, b_u, boundary_check=(0, 1))

        for i_k in range(tl.cdiv(K, BK)):
            p_p = tl.make_block_ptr(
                p_eff + i_n * T * H * K + i_h * K, (T, K), (H * K, 1),
                (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            p_y = tl.make_block_ptr(
                y_c + (i_n * NT * BT * H + i_h) * K, (T, K), (H * K, 1),
                (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            p_z = tl.make_block_ptr(
                z_c + (i_n * NT * BT * H + i_h) * K, (T, K), (H * K, 1),
                (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            b_p = tl.load(p_p, boundary_check=(0, 1)).to(tl.float32)
            b_y = tl.dot(b_inv, b_p * b_bar_a[:, None], allow_tf32=False)
            b_z = tl.dot(b_inv, b_p * b_btm1[:, None], allow_tf32=False)
            tl.store(p_y, b_y, boundary_check=(0, 1))
            tl.store(p_z, b_z, boundary_check=(0, 1))


@input_guard
def chunk_momentum_delta_wy_fwd_npu(
    A: torch.Tensor,
    v: torch.Tensor,
    p_eff: torch.Tensor,
    bar_a_tm1: torch.Tensor,
    b_tm1: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = p_eff.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    BK = _get_kkt_bk(BT, K)
    BV = min(_get_kkt_bk(BT, V), triton.next_power_of_2(V))
    # padded rows like the torch base (the fwd_h stage consumes [B, pT, H, ·])
    u_c = torch.zeros(B, NT * BT, H, V, device=p_eff.device, dtype=torch.float32)
    y_c = torch.zeros(B, NT * BT, H, K, device=p_eff.device, dtype=torch.float32)
    z_c = torch.zeros(B, NT * BT, H, K, device=p_eff.device, dtype=torch.float32)
    _launch_core_grid(
        chunk_momentum_delta_wy_fwd_kernel_npu,
        task_num=NT * B * H,
        kernel_kwargs=dict(
            A=A,
            v=v,
            p_eff=p_eff,
            bar_a_tm1=bar_a_tm1,
            b_tm1=b_tm1,
            u_c=u_c,
            y_c=y_c,
            z_c=z_c,
            T=T,
        ),
        H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
    )
    return u_c, y_c, z_c


# ------------------------------------------------------ K3: two-state fwd_h


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h_s0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['h_sT'] is not None,
})
@triton.jit(do_not_specialize=['T', 'B', 'task_num', 'num_core'])
def chunk_momentum_delta_fwd_h_kernel_npu(
    k_eta,
    u_c,
    y_c,
    z_c,
    cfac,
    lct1,
    lm_cum,
    a_last,
    b_last,
    m_last,
    ct_last,
    h_s,
    h_m,
    v_i,
    h_s0,
    h_m0,
    h_sT,
    h_mT,
    T,
    task_num,
    num_core,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """Two-state S/M inter-chunk recurrence; stores the state entering each chunk."""
    core_id = tl.program_id(0)
    NT = tl.cdiv(T, BT)
    NV = tl.cdiv(V, BV)
    for task in tl.range(core_id, task_num, num_core):
        i_v = task % NV
        i_nh = task // NV
        i_n = i_nh // H
        i_h = i_nh % H
        v_start = i_v * BV

        o_t = tl.arange(0, BT)
        o_v = tl.arange(0, BV)
        o_k1 = tl.arange(0, BK)
        o_k2 = BK + o_k1
        o_k3 = BK * 2 + o_k1
        o_k4 = BK * 3 + o_k1
        # state rows k >= K (BK > K) are out of bounds; mask every store/load on them
        m_st1 = (o_k1[:, None] < K) & (o_v[None, :] < V)
        m_st2 = (o_k2[:, None] < K) & (o_v[None, :] < V)
        m_st3 = (o_k3[:, None] < K) & (o_v[None, :] < V)
        m_st4 = (o_k4[:, None] < K) & (o_v[None, :] < V)

        # K-segmented state tiles: both S and M live across the chunk loop
        b_h_s1 = tl.zeros([BK, BV], dtype=tl.float32)
        b_h_m1 = tl.zeros([BK, BV], dtype=tl.float32)
        if K > BK:
            b_h_s2 = tl.zeros([BK, BV], dtype=tl.float32)
            b_h_m2 = tl.zeros([BK, BV], dtype=tl.float32)
        if K > BK * 2:
            b_h_s3 = tl.zeros([BK, BV], dtype=tl.float32)
            b_h_m3 = tl.zeros([BK, BV], dtype=tl.float32)
        if K > BK * 3:
            b_h_s4 = tl.zeros([BK, BV], dtype=tl.float32)
            b_h_m4 = tl.zeros([BK, BV], dtype=tl.float32)

        if USE_INITIAL_STATE:
            p_h0_s = h_s0 + i_nh * K * V + o_k1[:, None] * V + o_v[None, :]
            m_h0 = (o_k1[:, None] < K) & (o_v[None, :] < V)
            b_h_s1 = tl.load(p_h0_s, mask=m_h0, other=0.0).to(tl.float32)
            p_h0_m = h_m0 + i_nh * K * V + o_k1[:, None] * V + o_v[None, :]
            b_h_m1 = tl.load(p_h0_m, mask=m_h0, other=0.0).to(tl.float32)
            if K > BK:
                o_k2 = BK + o_k1
                p_h0_s2 = h_s0 + i_nh * K * V + o_k2[:, None] * V + o_v[None, :]
                m_h02 = (o_k2[:, None] < K) & (o_v[None, :] < V)
                b_h_s2 = tl.load(p_h0_s2, mask=m_h02, other=0.0).to(tl.float32)
                p_h0_m2 = h_m0 + i_nh * K * V + o_k2[:, None] * V + o_v[None, :]
                b_h_m2 = tl.load(p_h0_m2, mask=m_h02, other=0.0).to(tl.float32)
            if K > BK * 2:
                o_k3 = BK * 2 + o_k1
                p_h0_s3 = h_s0 + i_nh * K * V + o_k3[:, None] * V + o_v[None, :]
                m_h03 = (o_k3[:, None] < K) & (o_v[None, :] < V)
                b_h_s3 = tl.load(p_h0_s3, mask=m_h03, other=0.0).to(tl.float32)
                p_h0_m3 = h_m0 + i_nh * K * V + o_k3[:, None] * V + o_v[None, :]
                b_h_m3 = tl.load(p_h0_m3, mask=m_h03, other=0.0).to(tl.float32)
            if K > BK * 3:
                o_k4 = BK * 3 + o_k1
                p_h0_s4 = h_s0 + i_nh * K * V + o_k4[:, None] * V + o_v[None, :]
                m_h04 = (o_k4[:, None] < K) & (o_v[None, :] < V)
                b_h_s4 = tl.load(p_h0_s4, mask=m_h04, other=0.0).to(tl.float32)
                p_h0_m4 = h_m0 + i_nh * K * V + o_k4[:, None] * V + o_v[None, :]
                b_h_m4 = tl.load(p_h0_m4, mask=m_h04, other=0.0).to(tl.float32)

        # rebind GM bases per chunk; do not ptr += across i_t (Ascend MTE OOB).
        # u_c/y_c/z_c/v_i hold NT*BT padded rows per batch.
        h_s_nh = h_s + (i_n * NT * H + i_h) * K * V
        h_m_nh = h_m + (i_n * NT * H + i_h) * K * V
        u_base = u_c + (i_n * NT * BT * H + i_h) * V
        y_base = y_c + (i_n * NT * BT * H + i_h) * K
        z_base = z_c + (i_n * NT * BT * H + i_h) * K
        k_base = k_eta + (i_n * T * H + i_h) * K

        for i_t in range(NT):
            # store the state entering chunk i_t (K-segmented)
            p_hs = h_s_nh + i_t * H * K * V + o_k1[:, None] * V + o_v[None, :]
            tl.store(p_hs, b_h_s1, mask=m_st1)
            p_hm = h_m_nh + i_t * H * K * V + o_k1[:, None] * V + o_v[None, :]
            tl.store(p_hm, b_h_m1, mask=m_st1)
            if K > BK:
                p_hs2 = h_s_nh + i_t * H * K * V + o_k2[:, None] * V + o_v[None, :]
                tl.store(p_hs2, b_h_s2, mask=m_st2)
                p_hm2 = h_m_nh + i_t * H * K * V + o_k2[:, None] * V + o_v[None, :]
                tl.store(p_hm2, b_h_m2, mask=m_st2)
            if K > BK * 2:
                p_hs3 = h_s_nh + i_t * H * K * V + o_k3[:, None] * V + o_v[None, :]
                tl.store(p_hs3, b_h_s3, mask=m_st3)
                p_hm3 = h_m_nh + i_t * H * K * V + o_k3[:, None] * V + o_v[None, :]
                tl.store(p_hm3, b_h_m3, mask=m_st3)
            if K > BK * 3:
                p_hs4 = h_s_nh + i_t * H * K * V + o_k4[:, None] * V + o_v[None, :]
                tl.store(p_hs4, b_h_s4, mask=m_st4)
                p_hm4 = h_m_nh + i_t * H * K * V + o_k4[:, None] * V + o_v[None, :]
                tl.store(p_hm4, b_h_m4, mask=m_st4)

            o_t2 = i_t * BT + o_t
            # v_i = u_c - sum_k y_c_k @ S_k + sum_k z_c_k @ M_k
            p_u = tl.make_block_ptr(u_base, (T, V), (H * V, 1), (i_t * BT, v_start), (BT, BV), (1, 0))
            b_v = tl.load(p_u, boundary_check=(0, 1)).to(tl.float32)
            p_y1 = tl.make_block_ptr(y_base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            b_y = tl.load(p_y1, boundary_check=(0, 1)).to(tl.float32)
            p_z1 = tl.make_block_ptr(z_base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            b_z = tl.load(p_z1, boundary_check=(0, 1)).to(tl.float32)
            b_v = b_v - tl.dot(b_y, b_h_s1, allow_tf32=False) + tl.dot(b_z, b_h_m1, allow_tf32=False)
            if K > BK:
                p_y2 = tl.make_block_ptr(y_base, (T, K), (H * K, 1), (i_t * BT, BK), (BT, BK), (1, 0))
                b_y = tl.load(p_y2, boundary_check=(0, 1)).to(tl.float32)
                p_z2 = tl.make_block_ptr(z_base, (T, K), (H * K, 1), (i_t * BT, BK), (BT, BK), (1, 0))
                b_z = tl.load(p_z2, boundary_check=(0, 1)).to(tl.float32)
                b_v = b_v - tl.dot(b_y, b_h_s2, allow_tf32=False) + tl.dot(b_z, b_h_m2, allow_tf32=False)
            if K > BK * 2:
                p_y3 = tl.make_block_ptr(y_base, (T, K), (H * K, 1), (i_t * BT, BK * 2), (BT, BK), (1, 0))
                b_y = tl.load(p_y3, boundary_check=(0, 1)).to(tl.float32)
                p_z3 = tl.make_block_ptr(z_base, (T, K), (H * K, 1), (i_t * BT, BK * 2), (BT, BK), (1, 0))
                b_z = tl.load(p_z3, boundary_check=(0, 1)).to(tl.float32)
                b_v = b_v - tl.dot(b_y, b_h_s3, allow_tf32=False) + tl.dot(b_z, b_h_m3, allow_tf32=False)
            if K > BK * 3:
                p_y4 = tl.make_block_ptr(y_base, (T, K), (H * K, 1), (i_t * BT, BK * 3), (BT, BK), (1, 0))
                b_y = tl.load(p_y4, boundary_check=(0, 1)).to(tl.float32)
                p_z4 = tl.make_block_ptr(z_base, (T, K), (H * K, 1), (i_t * BT, BK * 3), (BT, BK), (1, 0))
                b_z = tl.load(p_z4, boundary_check=(0, 1)).to(tl.float32)
                b_v = b_v - tl.dot(b_y, b_h_s4, allow_tf32=False) + tl.dot(b_z, b_h_m4, allow_tf32=False)

            p_vi = tl.make_block_ptr(
                v_i + (i_n * NT * BT * H + i_h) * V, (T, V), (H * V, 1),
                (i_t * BT, v_start), (BT, BV), (1, 0),
            )
            tl.store(p_vi, b_v, boundary_check=(0, 1))

            # per-chunk gate scalars/vectors (exp2-scaled logs)
            g_off = (i_n * H + i_h) * NT
            b_a_last = tl.load(a_last + g_off + i_t)
            b_b_last = tl.load(b_last + g_off + i_t)
            b_m_last = tl.load(m_last + g_off + i_t)
            b_ct_last = tl.load(ct_last + g_off + i_t)
            b_cfac = tl.load(cfac + g_off * BT + o_t2)
            b_lct1 = tl.load(lct1 + g_off * BT + o_t2)
            b_lm_cum = tl.load(lm_cum + g_off * BT + o_t2)
            # decay_s[r] = b_last * cfac[r] * (1 - exp2(lct1[r] - ct_last))
            b_ds = b_b_last * b_cfac * (1.0 - tl.exp2(b_lct1 - b_ct_last))
            # decay_m[r] = exp2(lm_last - lm_cum[r])
            b_dm = tl.exp2(b_m_last - b_lm_cum)

            # state updates per K-slab (bisheng-safe: acc is a product, never the carried tile)
            p_k1 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            b_k = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_h_s1 = tl.dot(tl.trans(b_k * b_ds[:, None]), b_v, b_h_s1 * b_a_last, allow_tf32=False)
            b_h_s1 = b_h_s1 - b_h_m1 * b_b_last
            # m_last is an exp2-scaled log; the M decay is exp(lm_last) = exp2(m_last)
            b_h_m1 = tl.dot(tl.trans(-(b_k * b_dm[:, None])), b_v, b_h_m1 * tl.exp2(b_m_last), allow_tf32=False)
            if K > BK:
                p_k2 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT, BK), (BT, BK), (1, 0))
                b_k = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                b_h_s2 = tl.dot(tl.trans(b_k * b_ds[:, None]), b_v, b_h_s2 * b_a_last, allow_tf32=False)
                b_h_s2 = b_h_s2 - b_h_m2 * b_b_last
                b_h_m2 = tl.dot(tl.trans(-(b_k * b_dm[:, None])), b_v, b_h_m2 * tl.exp2(b_m_last), allow_tf32=False)
            if K > BK * 2:
                p_k3 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT, BK * 2), (BT, BK), (1, 0))
                b_k = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                b_h_s3 = tl.dot(tl.trans(b_k * b_ds[:, None]), b_v, b_h_s3 * b_a_last, allow_tf32=False)
                b_h_s3 = b_h_s3 - b_h_m3 * b_b_last
                b_h_m3 = tl.dot(tl.trans(-(b_k * b_dm[:, None])), b_v, b_h_m3 * tl.exp2(b_m_last), allow_tf32=False)
            if K > BK * 3:
                p_k4 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT, BK * 3), (BT, BK), (1, 0))
                b_k = tl.load(p_k4, boundary_check=(0, 1)).to(tl.float32)
                b_h_s4 = tl.dot(tl.trans(b_k * b_ds[:, None]), b_v, b_h_s4 * b_a_last, allow_tf32=False)
                b_h_s4 = b_h_s4 - b_h_m4 * b_b_last
                b_h_m4 = tl.dot(tl.trans(-(b_k * b_dm[:, None])), b_v, b_h_m4 * tl.exp2(b_m_last), allow_tf32=False)

        if STORE_FINAL_STATE:
            p_hsT = h_sT + i_nh * K * V + o_k1[:, None] * V + o_v[None, :]
            tl.store(p_hsT, b_h_s1, mask=m_st1)
            p_hmT = h_mT + i_nh * K * V + o_k1[:, None] * V + o_v[None, :]
            tl.store(p_hmT, b_h_m1, mask=m_st1)
            if K > BK:
                p_hsT2 = h_sT + i_nh * K * V + o_k2[:, None] * V + o_v[None, :]
                tl.store(p_hsT2, b_h_s2, mask=m_st2)
                p_hmT2 = h_mT + i_nh * K * V + o_k2[:, None] * V + o_v[None, :]
                tl.store(p_hmT2, b_h_m2, mask=m_st2)
            if K > BK * 2:
                p_hsT3 = h_sT + i_nh * K * V + o_k3[:, None] * V + o_v[None, :]
                tl.store(p_hsT3, b_h_s3, mask=m_st3)
                p_hmT3 = h_mT + i_nh * K * V + o_k3[:, None] * V + o_v[None, :]
                tl.store(p_hmT3, b_h_m3, mask=m_st3)
            if K > BK * 3:
                p_hsT4 = h_sT + i_nh * K * V + o_k4[:, None] * V + o_v[None, :]
                tl.store(p_hsT4, b_h_s4, mask=m_st4)
                p_hmT4 = h_mT + i_nh * K * V + o_k4[:, None] * V + o_v[None, :]
                tl.store(p_hmT4, b_h_m4, mask=m_st4)


@input_guard
def chunk_momentum_delta_fwd_h_npu(
    k_eta: torch.Tensor,
    u_c: torch.Tensor,
    y_c: torch.Tensor,
    z_c: torch.Tensor,
    cfac: torch.Tensor,
    lct1: torch.Tensor,
    lm_cum: torch.Tensor,
    a_last: torch.Tensor,
    b_last: torch.Tensor,
    ct_last: torch.Tensor,
    lm_last: torch.Tensor,
    initial_S: torch.Tensor | None,
    initial_M: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K = k_eta.shape
    V = u_c.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    pT = NT * BT
    BK, BV = _select_state_tiles(K, V)
    h_s = torch.empty(B, NT, H, K, V, device=k_eta.device, dtype=torch.float32)
    h_m = torch.empty(B, NT, H, K, V, device=k_eta.device, dtype=torch.float32)
    v_i = torch.empty(B, pT, H, V, device=k_eta.device, dtype=torch.float32)
    h_sT = torch.empty(B, H, K, V, device=k_eta.device, dtype=torch.float32) if output_final_state else None
    h_mT = torch.empty(B, H, K, V, device=k_eta.device, dtype=torch.float32) if output_final_state else None
    # all logs passed in exp2 units; the kernel computes exp2(m_last - lm_cum) = exp(lm_last - lm_cum)
    m_last = _exp2_vec(lm_last)
    _launch_core_grid(
        chunk_momentum_delta_fwd_h_kernel_npu,
        task_num=triton.cdiv(V, BV) * B * H,
        kernel_kwargs=dict(
            k_eta=k_eta,
            u_c=u_c,
            y_c=y_c,
            z_c=z_c,
            cfac=cfac,
            lct1=_exp2_vec(lct1),
            lm_cum=_exp2_vec(lm_cum),
            a_last=a_last,
            b_last=b_last,
            m_last=m_last,
            ct_last=_exp2_vec(ct_last),
            h_s=h_s,
            h_m=h_m,
            v_i=v_i,
            h_s0=initial_S,
            h_m0=initial_M,
            h_sT=h_sT,
            h_mT=h_mT,
            T=T,
        ),
        H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
    )
    return h_s, h_m, v_i, h_sT, h_mT


# -------------------------------------------------------- K4: two-state fwd_o


@triton.jit(do_not_specialize=['T', 'B', 'task_num', 'num_core'])
def chunk_momentum_delta_fwd_o_kernel_npu(
    q,
    k_eta,
    v_i,
    h_s,
    h_m,
    a_cum,
    b_t,
    cfac,
    lct,
    lct1,
    o,
    scale,
    T,
    task_num,
    num_core,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """o = (q*a_cum) @ S - (q*b_t) @ M + ((q @ k^T) * gamma_mask_q) @ v_i."""
    core_id = tl.program_id(0)
    NT = tl.cdiv(T, BT)
    NV = tl.cdiv(V, BV)
    for task in tl.range(core_id, task_num, num_core):
        i_v = task % NV
        i_nh = task // NV
        i_n = i_nh // H
        i_h = i_nh % H
        v_start = i_v * BV

        o_t = tl.arange(0, BT)
        o_v = tl.arange(0, BV)
        o_k1 = tl.arange(0, BK)
        b_r = tl.arange(0, BT)

        h_s_nh = h_s + (i_n * NT * H + i_h) * K * V
        h_m_nh = h_m + (i_n * NT * H + i_h) * K * V
        q_base = q + (i_n * T * H + i_h) * K
        k_base = k_eta + (i_n * T * H + i_h) * K
        vi_base = v_i + (i_n * NT * BT * H + i_h) * V
        g_off = (i_n * H + i_h) * NT

        for i_t in range(NT):
            o_t2 = i_t * BT + o_t
            m_t = o_t2 < T
            p_q1 = tl.make_block_ptr(q_base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            b_q = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32) * scale
            p_k1 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            b_k = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)

            b_a_cum = tl.load(a_cum + g_off * BT + o_t2, mask=m_t, other=0.0)
            b_bt = tl.load(b_t + g_off * BT + o_t2, mask=m_t, other=0.0)
            b_cfac = tl.load(cfac + g_off * BT + o_t2, mask=m_t, other=0.0)
            b_lct = tl.load(lct + g_off * BT + o_t2, mask=m_t, other=0.0)
            b_lct1 = tl.load(lct1 + g_off * BT + o_t2, mask=m_t, other=0.0)

            p_hs1 = h_s_nh + i_t * H * K * V + o_k1[:, None] * V + o_v[None, :]
            b_hs = tl.load(p_hs1, mask=(o_k1[:, None] < K) & (o_v[None, :] < V), other=0.0)
            p_hm1 = h_m_nh + i_t * H * K * V + o_k1[:, None] * V + o_v[None, :]
            b_hm = tl.load(p_hm1, mask=(o_k1[:, None] < K) & (o_v[None, :] < V), other=0.0)
            p_vi = tl.make_block_ptr(vi_base, (T, V), (H * V, 1), (i_t * BT, v_start), (BT, BV), (1, 0))
            b_v = tl.load(p_vi, boundary_check=(0, 1)).to(tl.float32)

            b_o = tl.dot(b_q * b_a_cum[:, None], b_hs, allow_tf32=False)
            b_o = b_o - tl.dot(b_q * b_bt[:, None], b_hm, allow_tf32=False)
            # intra: qk masked by gamma_mask_q[i, j] = b_t[i] * cfac[j] * (1 - exp2(lct1[j] - lct[i]))
            b_qk = tl.dot(b_q + 0.0, tl.trans(b_k + 0.0), allow_tf32=False)
            b_mask = b_bt[:, None] * b_cfac[None, :] * (1.0 - tl.exp2(b_lct1[None, :] - b_lct[:, None]))
            b_qk = tl.where(b_r[:, None] >= b_r[None, :], b_qk * b_mask, 0.0)
            b_o = b_o + tl.dot(b_qk, b_v, allow_tf32=False)

            if K > BK:
                p_q2 = tl.make_block_ptr(q_base, (T, K), (H * K, 1), (i_t * BT, BK), (BT, BK), (1, 0))
                b_q = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32) * scale
                p_k2 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT, BK), (BT, BK), (1, 0))
                b_k = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                o_k2 = BK + o_k1
                p_hs2 = h_s_nh + i_t * H * K * V + o_k2[:, None] * V + o_v[None, :]
                b_hs = tl.load(p_hs2, mask=(o_k2[:, None] < K) & (o_v[None, :] < V), other=0.0)
                p_hm2 = h_m_nh + i_t * H * K * V + o_k2[:, None] * V + o_v[None, :]
                b_hm = tl.load(p_hm2, mask=(o_k2[:, None] < K) & (o_v[None, :] < V), other=0.0)
                b_o = b_o + tl.dot(b_q * b_a_cum[:, None], b_hs, allow_tf32=False)
                b_o = b_o - tl.dot(b_q * b_bt[:, None], b_hm, allow_tf32=False)
                b_qk = tl.dot(b_q + 0.0, tl.trans(b_k + 0.0), allow_tf32=False)
                b_o = b_o + tl.dot(tl.where(b_r[:, None] >= b_r[None, :], b_qk * b_mask, 0.0), b_v, allow_tf32=False)
            if K > BK * 2:
                p_q3 = tl.make_block_ptr(q_base, (T, K), (H * K, 1), (i_t * BT, BK * 2), (BT, BK), (1, 0))
                b_q = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32) * scale
                p_k3 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT, BK * 2), (BT, BK), (1, 0))
                b_k = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                o_k3 = BK * 2 + o_k1
                p_hs3 = h_s_nh + i_t * H * K * V + o_k3[:, None] * V + o_v[None, :]
                b_hs = tl.load(p_hs3, mask=(o_k3[:, None] < K) & (o_v[None, :] < V), other=0.0)
                p_hm3 = h_m_nh + i_t * H * K * V + o_k3[:, None] * V + o_v[None, :]
                b_hm = tl.load(p_hm3, mask=(o_k3[:, None] < K) & (o_v[None, :] < V), other=0.0)
                b_o = b_o + tl.dot(b_q * b_a_cum[:, None], b_hs, allow_tf32=False)
                b_o = b_o - tl.dot(b_q * b_bt[:, None], b_hm, allow_tf32=False)
                b_qk = tl.dot(b_q + 0.0, tl.trans(b_k + 0.0), allow_tf32=False)
                b_o = b_o + tl.dot(tl.where(b_r[:, None] >= b_r[None, :], b_qk * b_mask, 0.0), b_v, allow_tf32=False)
            if K > BK * 3:
                p_q4 = tl.make_block_ptr(q_base, (T, K), (H * K, 1), (i_t * BT, BK * 3), (BT, BK), (1, 0))
                b_q = tl.load(p_q4, boundary_check=(0, 1)).to(tl.float32) * scale
                p_k4 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT, BK * 3), (BT, BK), (1, 0))
                b_k = tl.load(p_k4, boundary_check=(0, 1)).to(tl.float32)
                o_k4 = BK * 3 + o_k1
                p_hs4 = h_s_nh + i_t * H * K * V + o_k4[:, None] * V + o_v[None, :]
                b_hs = tl.load(p_hs4, mask=(o_k4[:, None] < K) & (o_v[None, :] < V), other=0.0)
                p_hm4 = h_m_nh + i_t * H * K * V + o_k4[:, None] * V + o_v[None, :]
                b_hm = tl.load(p_hm4, mask=(o_k4[:, None] < K) & (o_v[None, :] < V), other=0.0)
                b_o = b_o + tl.dot(b_q * b_a_cum[:, None], b_hs, allow_tf32=False)
                b_o = b_o - tl.dot(b_q * b_bt[:, None], b_hm, allow_tf32=False)
                b_qk = tl.dot(b_q + 0.0, tl.trans(b_k + 0.0), allow_tf32=False)
                b_o = b_o + tl.dot(tl.where(b_r[:, None] >= b_r[None, :], b_qk * b_mask, 0.0), b_v, allow_tf32=False)

            p_o = tl.make_block_ptr(
                o + (i_n * T * H + i_h) * V, (T, V), (H * V, 1),
                (i_t * BT, v_start), (BT, BV), (1, 0),
            )
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def chunk_momentum_delta_fwd_o_npu(
    q: torch.Tensor,
    k_eta: torch.Tensor,
    v_i: torch.Tensor,
    h_s: torch.Tensor,
    h_m: torch.Tensor,
    a_cum: torch.Tensor,
    b_t: torch.Tensor,
    cfac: torch.Tensor,
    lct: torch.Tensor,
    lct1: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> torch.Tensor:
    B, T, H, K = q.shape
    V = v_i.shape[-1]
    BT = chunk_size
    BK, BV = _select_state_tiles(K, V)
    o = torch.empty(B, T, H, V, device=q.device, dtype=q.dtype)
    _launch_core_grid(
        chunk_momentum_delta_fwd_o_kernel_npu,
        task_num=triton.cdiv(V, BV) * B * H,
        kernel_kwargs=dict(
            q=q,
            k_eta=k_eta,
            v_i=v_i,
            h_s=h_s,
            h_m=h_m,
            a_cum=a_cum,
            b_t=b_t,
            cfac=cfac,
            lct=_exp2_vec(lct),
            lct1=_exp2_vec(lct1),
            o=o,
            scale=scale,
            T=T,
        ),
        H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
    )
    return o
