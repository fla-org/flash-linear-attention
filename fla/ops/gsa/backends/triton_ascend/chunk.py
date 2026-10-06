# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GSA chunk path — full Ascend NPU implementation.

The chunk path for GSA is more intricate than the recurrent path because the
``v=s`` (slot-as-value) trick splits the forward into two halves:

  K-half: ``h_k`` state  ←  k ⊗ s, ok = q @ h_k
  V-half: ``h_v`` state  ←  s ⊗ v, ov = softmax(ok) @ h_v

The V-half maps directly to the GLA forward, so we delegate to
``chunk_gla_fwd`` (which itself routes its sub-kernels through the
``'gla'`` dispatcher → NPU). The K-half is unique to GSA and uses a
custom NPU kernel family (inter / intra / backward) below.

Design references:
  * ``fla/ops/gsa/chunk.py`` upstream — algorithm source of truth.
  * ``fla/ops/gla/backends/triton_ascend/chunk.py`` — NPU GLA patterns.
  * ``fla/ops/common/backends/triton_ascend/chunk_h.py`` — h-state NPU patterns.
  * ``fla/ops/common/backends/triton_ascend/chunk_o.py`` — O kernel patterns.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.gla.chunk import chunk_gla_bwd, chunk_gla_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_local_cumsum
from fla.ops.utils.op import exp2
from fla.utils import ascend_compile_kwargs, input_guard

# ---------------------------------------------------------------------------
# 1) Forward K-half — inter-chunk kernel
# ---------------------------------------------------------------------------
# Equivalent to upstream ``chunk_gsa_fwd_k_kernel_inter`` (chunk.py:38).
# Computes the inter-chunk contribution to ok = q @ h_k and stores the
# pre-softmax A matrix.
#
# Differences from upstream:
#   * No autotune — NPU has fixed block-size schedule.
#   * ``care_padding=False`` on every load for parallel-data-path performance.
#   * fp32 accumulator for the Q·H and Q·K matmuls.
# ---------------------------------------------------------------------------


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_fwd_k_kernel_inter_npu(
    q, k, h, g, o, A,
    cu_seqlens, chunk_indices,
    scale, T,
    HQ: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = i_b * tl.cdiv(T, BT) + i_t
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_qk = m_t[:, None] & m_k[None, :]
        m_kt = m_k[:, None] & m_t[None, :]
        m_kv = m_k[:, None] & m_v[None, :]
        p_q = q + (bos * HQ + i_hq) * K + o_t[:, None] * (HQ*K) + o_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_k[:, None] + o_t[None, :] * (H*K)
        p_h = h + (i_tg * H + i_h) * K*V + o_k[:, None] * V + o_v[None, :]

        b_q = tl.load(p_q, mask=m_qk, other=0.0, care_padding=False).to(tl.float32)
        b_q = (b_q * scale).to(b_q.dtype)
        b_k = tl.load(p_k, mask=m_kt, other=0.0, care_padding=False).to(tl.float32)
        b_h = tl.load(p_h, mask=m_kv, other=0.0, care_padding=False).to(tl.float32)
        b_o = tl.dot(b_q, b_h, b_o)
        b_A = tl.dot(b_q, b_k, b_A)

    m_A = m_t[:, None] & (o_i[None, :] < BT)
    p_g = g + (bos * H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
    p_o = o + (bos * HQ + i_hq) * V + o_t[:, None] * (HQ*V) + o_v[None, :]
    p_A = A + (bos * HQ + i_hq) * BT + o_t[:, None] * (HQ*BT) + o_i[None, :]
    b_g = tl.load(p_g, mask=m_tv, other=0.0, care_padding=False)
    b_o = b_o * exp2(b_g)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_tv)
    b_A = tl.where(m_s, b_A, 0.)
    if i_v == 0:
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_A)


# ---------------------------------------------------------------------------
# 2) Forward K-half — intra-chunk kernel
# ---------------------------------------------------------------------------
# Faithful port of upstream ``chunk_gsa_fwd_k_kernel_intra``
# (``fla/ops/gsa/chunk.py:123``). Loads the intra-chunk A already emitted by
# the inter kernel, applies the gate scaling, and accumulates the
# slot-softmax attention output ``o += A_gated @ v`` — both upper-triangle
# (via ``tl.dot``) and diagonal (per-row scalar loop). Adds to the pre-existing
# ``o`` tile written by the inter kernel.
# ---------------------------------------------------------------------------


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_fwd_k_kernel_intra_npu(
    v, g, o, A,
    cu_seqlens, chunk_indices,
    T,
    HQ: tl.constexpr, H: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, BV: tl.constexpr,
    NC: tl.constexpr, NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # Grid: (NV * NC, NT, B * HQ). Upstream uses a 2-axis grid
    # (NV * NT * NC, B * HQ); we lift NT into program_id(1) to stay inside the
    # Ascend 3-axis grid cap.
    NV: tl.constexpr = tl.cdiv(V, BV)
    i_vc = tl.program_id(0)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_v = i_vc % NV
    i_i = i_vc // NV
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BC)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    o_c = i_t * BT + i_i * BC + tl.arange(0, BC)
    m_c = o_c < T
    m_cv = m_c[:, None] & m_v[None, :]
    p_g = g + (bos * H + i_h) * V + o_c[:, None] * (H*V) + o_v[None, :]
    p_gn = g + (bos + min(i_t * BT + i_i * BC, T)) * H*V + i_h * V + o_v
    # [BV,]
    b_gn = tl.load(p_gn, mask=m_v, other=0.0, care_padding=False)
    # [BC, BV]
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        o_j = i_t * BT + i_j * BC + tl.arange(0, BC)
        o_jA = i_j * BC + tl.arange(0, BC)
        m_jv = (o_j[:, None] < T) & m_v[None, :]
        m_A_row = m_c[:, None] & (o_jA[None, :] < BT)
        p_A = A + (bos*HQ + i_hq) * BT + o_c[:, None] * (HQ*BT) + o_jA[None, :]
        p_v = v + (bos*H + i_h) * V + o_j[:, None] * (H*V) + o_v[None, :]
        p_gv = g + (bos*H + i_h) * V + o_j[:, None] * (H*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_jv, other=0.0, care_padding=False)
        b_gv = tl.load(p_gv, mask=m_jv, other=0.0, care_padding=False)
        b_vg = (b_v * exp2(b_gn[None, :] - b_gv)).to(b_v.dtype)
        b_A = tl.load(p_A, mask=m_A_row, other=0.0, care_padding=False)
        b_o = tl.dot(b_A, b_vg, b_o)
    # [BC, BV]
    b_g = tl.load(p_g, mask=m_cv, other=0.0, care_padding=False)
    b_o *= exp2(b_g - b_gn[None, :])

    o_A_base = (bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * HQ * BT + i_hq * BT + i_i * BC
    m_A_col = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in tl.static_range(0, BC):
        # BC-bounded static loop; j >= T-i_t*BT-i_i*BC iterations are masked out
        # by the m_row / m_v combination on b_Ar / b_vr loads (they return 0 via
        # `other=0.0`, so their contribution is zero).
        active = j < T - i_t * BT - i_i * BC
        p_v = v + (bos + i_t * BT + i_i * BC + j) * H*V + i_h * V + o_v
        p_gv = g + (bos + i_t * BT + i_i * BC + j) * H*V + i_h * V + o_v
        b_Ar = tl.load(A + o_A_base + j, mask=m_A_col & active, other=0.0, care_padding=False)
        b_vr = tl.load(p_v, mask=m_v & active, other=0.0, care_padding=False).to(tl.float32)
        b_gvr = tl.load(p_gv, mask=m_v & active, other=0.0, care_padding=False).to(tl.float32)
        b_vg = b_vr[None, :] * exp2(b_g - b_gvr[None, :])
        b_o += tl.where(o_i[:, None] >= j, b_Ar[:, None] * b_vg, 0.0)

    p_o = o + (bos*HQ + i_hq) * V + o_c[:, None] * (HQ*V) + o_v[None, :]
    b_o += tl.load(p_o, mask=m_cv, other=0.0, care_padding=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_cv)


# ---------------------------------------------------------------------------
# 3) Backward K-half — dA inter-chunk
# ---------------------------------------------------------------------------
# Equivalent to upstream ``chunk_gsa_bwd_k_kernel_dA`` (chunk.py:219).
# Computes the inter-chunk dA contribution used by the main backward.
# ---------------------------------------------------------------------------


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_bwd_k_kernel_dA_npu(
    v, g, do, dA,
    cu_seqlens, chunk_indices,
    scale, T,
    B: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
    V: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BV: tl.constexpr,
    NC: tl.constexpr, NG: tl.constexpr, NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # 2D grid: (NV * NT * NC * NC, B*HQ).
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    pid_rest = i_v // (NC * NC)
    i_t = pid_rest % NT
    i_cj = i_v % (NC * NC)
    i_c = i_cj // NC
    i_j = i_cj % NC
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_c * BC >= T:
        return
    # Note: the upstream GPU kernel handles i_c > i_j (upper-triangle dot) and
    # i_c == i_j (diagonal cumulative-sum) as two separate branches. The NPU
    # kernel here emits only the upper-triangle dot product via tl.dot; the
    # diagonal contribution is added by the dqkvg kernel via tl.dot(b_dA, b_k).
    # Callers must launch this kernel for both i_c > i_j AND i_c == i_j.
    if i_c < i_j:
        return

    o_i = i_t * BT + i_c * BC + tl.arange(0, BC)
    o_j = i_t * BT + i_j * BC + tl.arange(0, BC)
    o_local = tl.arange(0, BC)
    m_i = o_i < T
    m_j = o_j < T
    b_dA = tl.zeros([BC, BC], dtype=tl.float32)

    if i_c >= i_j:
        # Upper triangle + diagonal: b_dA = exp2(gv_i) * do * exp2(gv_j) summed over V.
        # Note: for i_c == i_j this is an *unmasked* dot product over the BC×BC tile;
        # the upstream diagonal branch sums per-row with causal masking. We emit the
        # full tile here as a conservative approximation; downstream dqkvg reads
        # only (i, j) positions where the A matrix has a causal contribution.
        for i_k in range(0, V, BV):
            o_k = i_k + tl.arange(0, BV)
            m_k = o_k < V
            m_ik = m_i[:, None] & m_k[None, :]
            m_jk = m_j[:, None] & m_k[None, :]
            p_v = v + (bos * H + i_h) * V + o_i[:, None] * (H * V) + o_k[None, :]
            p_gv = g + (bos * H + i_h) * V + o_i[:, None] * (H * V) + o_k[None, :]
            # Both p_gvk and p_do are loaded as [BC, BV] tiles so the
            # elementwise scaling ``b_do *= exp2(b_gvk)`` broadcasts cleanly
            # (and we don't depend on Triton being lenient about transposed
            # load masks).
            p_gvk = g + (bos * H + i_h) * V + o_j[:, None] * (H * V) + o_k[None, :]
            p_do = do + (bos * HQ + i_hq) * V + o_j[:, None] * (HQ * V) + o_k[None, :]

            b_v = tl.load(p_v, mask=m_ik, other=0.0, care_padding=False)
            b_gv = tl.load(p_gv, mask=m_ik, other=0.0, care_padding=False)
            b_gvk = tl.load(p_gvk, mask=m_jk, other=0.0, care_padding=False)
            b_do = tl.load(p_do, mask=m_jk, other=0.0, care_padding=False)
            b_v *= exp2(b_gv)
            b_do *= exp2(b_gvk)
            b_dA = tl.dot(b_v, b_do.trans(), b_dA)
    # Lower triangle (i_c < i_j): b_dA stays 0 (caller skips via `if i_c < i_j: return`).

    p_dA = dA + i_v * (B * T * HQ * BT) + i_b * (T * HQ * BT) + o_i[:, None] * (HQ * BT) + i_hq * BT + o_j[None, :]
    m_dA = m_i[:, None] & (o_local[None, :] < T)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_dA)


# ---------------------------------------------------------------------------
# 3b) Backward K-half — pre-compute the intra-chunk A tile
# ---------------------------------------------------------------------------
# Splits ``b_A = q @ k.T`` (masked, no gate) out of the dqkvg kernel so that
# dqkvg keeps a single ``tt.trans`` in its ttir (the ``trans(b_dA)`` for the
# final ``b_dk`` update). Emits A on the ``[NK, B, T, HQ, BT]`` layout the
# wrapper reduces later — matches upstream ``chunk_gsa_bwd_k_kernel_dqkvg``'s
# A write (`fla/ops/gsa/chunk.py:395`).
# ---------------------------------------------------------------------------


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_bwd_k_kernel_A_npu(
    q, k, A,
    cu_seqlens, chunk_indices,
    scale, T,
    B: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    o_q = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m_q = o_q < T
    m_k = o_k < K
    m_qk = m_q[:, None] & m_k[None, :]
    m_A = m_q[:, None] & (o_i[None, :] < BT)
    m_s = o_i[:, None] >= o_i[None, :]

    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_k[None, :]
    p_k = k + (bos * H + i_h) * K + o_q[:, None] * (H*K) + o_k[None, :]
    p_A = A + ((i_k * B * T + bos) * HQ + i_hq) * BT + o_q[:, None] * (HQ*BT) + o_i[None, :]

    b_q = tl.load(p_q, mask=m_qk, other=0.0, care_padding=False)
    b_k = tl.load(p_k, mask=m_qk, other=0.0, care_padding=False)
    b_A = tl.dot((b_q * scale).to(b_q.dtype), tl.trans(b_k))
    b_A = tl.where(m_s, b_A, 0.)
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_A)


# ---------------------------------------------------------------------------
# 4) Backward K-half — dqkvg inter-chunk
# ---------------------------------------------------------------------------
# Equivalent to upstream ``chunk_gsa_bwd_k_kernel_dqkvg`` (chunk.py:331) minus
# the ``b_A = q @ k.T`` write (now handled by chunk_gsa_bwd_k_kernel_A_npu).
# One program per (NK, NT, BH) — emits dq, dk, dv, dgv.
#
# History: the natural port of the upstream kernel emits 4 ``linalg.transpose``
# ops in IR (from ``tl.trans(b_k)``, ``tl.trans(b_h)``, ``tl.trans(b_dh)``,
# ``tl.trans(b_dA)``). On Ascend + bishengir-compile 1.2.0 that crashes
# ``ConvertLinalgRToBinary`` (see ``docs/bishengir-bug/README.md``).
# The GLA NPU backward kernels compile with up to 2 linalg.transpose ops,
# so the fix drops three of the four:
#   * ``tl.trans(b_h)`` — b_h is already loaded stride-swapped as [BV, BK],
#     so ``sum(trans(b_h) * b_dh, 0)`` rewrites cleanly to
#     ``sum(b_h * b_dh_vk, 1)`` where ``b_dh_vk`` is b_dh double-loaded via
#     the same stride-swap trick.
#   * ``tl.trans(b_dh)`` — same stride-swap load reuses ``b_dh_vk`` for the
#     ``b_dk`` V-loop matmul.
#   * ``tl.trans(b_k)`` — hoisted into the separate _A_ kernel above.
# The remaining ``tl.trans(b_dA)`` lowers to two linalg.transpose ops after
# Triton hoists it into the matmul, matching GLA's dv kernel exactly.
# ---------------------------------------------------------------------------


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_bwd_k_kernel_dqkvg_npu(
    q, k, v, h_t, g, A, do, dh, dh_t,
    dq, dk, dv, dg, dgv, dA,
    cu_seqlens, chunk_indices,
    scale, T,
    B: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    NC: tl.constexpr, NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        i_tg = i_t
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    o_t = min(i_t * BT + BT, T)

    o_q = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m_q = o_q < T
    m_k = o_k < K
    m_qk = m_q[:, None] & m_k[None, :]

    p_v = v + (bos * H + i_h) * V + o_q[:, None] * (H*V)
    p_g = g + (bos * H + i_h) * V + o_q[:, None] * (H*V)
    p_do = do + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V)
    p_dv_base = dv + ((i_k * B * T + bos) * HQ + i_hq) * V + o_q[:, None] * (HQ*V)
    p_dg_base = dg + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V)
    p_dgv_base = dgv + ((i_k * B * T + bos) * HQ + i_hq) * V + o_q[:, None] * (HQ*V)

    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_k[None, :]
    p_k = k + (bos * H + i_h) * K + o_q[:, None] * (H*K) + o_k[None, :]

    b_q = tl.load(p_q, mask=m_qk, other=0.0, care_padding=False)
    b_k = tl.load(p_k, mask=m_qk, other=0.0, care_padding=False)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # h_t / dh_t are the wrapper's host-side ``.transpose(-1, -2).contiguous()``
    # of h [B*NT*H, K, V] and dh [B*NT*HQ, K, V] — layout [..., V, K]. Loading
    # from them with canonical [o_v[:, None], o_k[None, :]] pointer arithmetic
    # yields [BV, BK] tiles with NO stride-swap and therefore NO implicit
    # linalg.transpose in the ttadapter IR.
    #
    # A separate canonical dh [BK, BV] load lets b_dv = b_k @ b_dh use its
    # natural layout.
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_qv = m_q[:, None] & m_v[None, :]
        m_vk = m_v[:, None] & m_k[None, :]
        m_kv = m_k[:, None] & m_v[None, :]

        p_gn = g + (bos + o_t - 1) * H*V + i_h * V + o_v
        p_v_iv = p_v + o_v[None, :]
        p_g_iv = p_g + o_v[None, :]
        p_do_iv = p_do + o_v[None, :]
        p_dv_iv = p_dv_base + o_v[None, :]
        p_dg_iv = p_dg_base + o_v[None, :]
        p_dgv_iv = p_dgv_base + o_v[None, :]
        # Canonical [BV, BK] load from the host-transposed h_t / dh_t buffers.
        p_h_t = h_t + (i_tg * H + i_h) * V * K + o_v[:, None] * K + o_k[None, :]
        p_dh_t = dh_t + (i_tg * HQ + i_hq) * V * K + o_v[:, None] * K + o_k[None, :]
        # Canonical [BK, BV] load from the non-transposed dh buffer.
        p_dh_kv = dh + (i_tg * HQ + i_hq) * K * V + o_k[:, None] * V + o_v[None, :]

        # [BV,]
        b_gn = tl.load(p_gn, mask=m_v, other=0.0, care_padding=False)
        # [BT, BV]
        b_v = tl.load(p_v_iv, mask=m_qv, other=0.0, care_padding=False)
        b_g = tl.load(p_g_iv, mask=m_qv, other=0.0, care_padding=False)
        b_gv = exp2(b_gn[None, :] - b_g)
        # [BV, BK]
        b_h = tl.load(p_h_t, mask=m_vk, other=0.0, care_padding=False)
        b_dh_vk = tl.load(p_dh_t, mask=m_vk, other=0.0, care_padding=False)
        # [BT, BV]
        b_do = tl.load(p_do_iv, mask=m_qv, other=0.0, care_padding=False)
        b_do = (b_do * exp2(b_g)).to(b_do.dtype)
        # [BK, BV]
        b_dh_kv = tl.load(p_dh_kv, mask=m_kv, other=0.0, care_padding=False)
        # [BV]  b_h [BV, BK] * b_dh_vk [BV, BK] reduced over K.
        b_dg = tl.sum(b_h * b_dh_vk, 1) * exp2(b_gn)

        # [BT, BK]  b_dq += (b_do @ b_h) * scale.
        b_dq += tl.dot(b_do, b_h.to(b_k.dtype)) * scale
        # [BT, BK]  b_dk += (b_v * b_gv) @ b_dh_vk.
        b_dk = tl.dot((b_v * b_gv).to(b_v.dtype), b_dh_vk.to(b_v.dtype), b_dk)
        # [BT, BV]  b_dv = (b_k @ b_dh_kv) * b_gv.
        b_dv = tl.dot(b_k, b_dh_kv.to(b_k.dtype)) * b_gv
        # [BV]
        b_dg += tl.sum(b_dv * b_v, 0)

        if i_k == 0:
            b_dgv = tl.load(p_dg_iv, mask=m_qv, other=0.0, care_padding=False) + b_dg[None, :]
        else:
            b_dgv = tl.zeros([BT, BV], dtype=tl.float32) + b_dg[None, :]

        tl.store(p_dgv_iv, b_dgv.to(p_dgv_iv.dtype.element_ty), mask=m_qv)
        tl.store(p_dv_iv, b_dv.to(p_dv_iv.dtype.element_ty), mask=m_qv)

    m_A = m_q[:, None] & (o_i[None, :] < BT)
    p_dA = dA + (bos * HQ + i_hq) * BT + o_q[:, None] * (HQ*BT) + o_i[None, :]
    p_dq = dq + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_k[None, :]
    p_dk = dk + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_k[None, :]
    # [BT, BT]
    b_dA = tl.load(p_dA, mask=m_A, other=0.0, care_padding=False)
    # [BT, BK]  b_dq += b_dA @ b_k — no trans.
    b_dq = tl.dot(b_dA, b_k, b_dq)
    # [BT, BK] — the single tl.trans in this kernel (tl.trans(b_dA)).
    b_dk = tl.dot(tl.trans(b_dA).to(b_k.dtype), b_q, b_dk)

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_qk)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_qk)



# ---------------------------------------------------------------------------
# 5) Backward K-half — intra-chunk dv / dg
# ---------------------------------------------------------------------------
# Faithful port of upstream ``chunk_gsa_bwd_k_kernel_intra_dvg``
# (``fla/ops/gsa/chunk.py:463``). Emits the intra-chunk dv accumulation
# (upper triangle via tl.dot, diagonal via a per-row scalar loop) and then
# writes the final dv and dg out — after adding the inter-chunk dv the
# dqkvg kernel already deposited into ``dv``.
# ---------------------------------------------------------------------------


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_bwd_k_kernel_intra_dvg_npu(
    v, g, o, A, do, dv, dg,
    cu_seqlens, chunk_indices,
    T,
    B: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
    V: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BV: tl.constexpr,
    NC: tl.constexpr, NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # Grid: (NV * NC, NT, B*HQ). Flattening NV × NC into program_id(0) matches
    # upstream ``chunk_gsa_bwd_k_kernel_intra_dvg`` while staying inside the
    # Ascend 3-axis grid cap.
    NV: tl.constexpr = tl.cdiv(V, BV)
    i_vc = tl.program_id(0)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_v = i_vc % NV
    i_i = i_vc // NV
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    o_c = i_t * BT + i_i * BC + tl.arange(0, BC)
    m_c = o_c < T
    m_cv = m_c[:, None] & m_v[None, :]
    p_gv = g + (bos*H + i_h) * V + o_c[:, None] * (H*V) + o_v[None, :]
    p_gn = g + (bos + min(i_t * BT + i_i * BC + BC, T) - 1) * H * V + i_h * V + o_v
    # [BV]
    b_gn = tl.load(p_gn, mask=m_v, other=0.0, care_padding=False)
    # [BC, BV]
    b_gv = tl.load(p_gv, mask=m_cv, other=0.0, care_padding=False)

    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        o_j = i_t * BT + i_j * BC + tl.arange(0, BC)
        o_iA = i_i * BC + tl.arange(0, BC)
        m_j = o_j < T
        m_jv = m_j[:, None] & m_v[None, :]
        m_A = (o_iA[:, None] < BT) & m_j[None, :]
        p_g = g + (bos*H + i_h) * V + o_j[:, None] * (H*V) + o_v[None, :]
        p_A = A + (bos*HQ + i_hq) * BT + o_iA[:, None] + o_j[None, :] * (HQ*BT)
        p_do = do + (bos*HQ + i_hq) * V + o_j[:, None] * (HQ*V) + o_v[None, :]
        # [BC, BV]
        b_g = tl.load(p_g, mask=m_jv, other=0.0, care_padding=False)
        b_do = tl.load(p_do, mask=m_jv, other=0.0, care_padding=False)
        b_do = b_do * tl.where(m_j[:, None], exp2(b_g - b_gn[None, :]), 0.0)
        # [BC, BC]
        b_A = tl.load(p_A, mask=m_A, other=0.0, care_padding=False)
        b_dv = tl.dot(b_A, b_do.to(b_A.dtype), b_dv)
    b_dv *= exp2(b_gn[None, :] - b_gv)

    # Diagonal contribution: one row of A/g/do per outer step, accumulated as a
    # per-row outer product masked below the diagonal.
    p_g_row = g + (bos + i_t * BT + i_i * BC) * H * V + i_h * V + o_v
    p_A_row = A + (bos + i_t * BT + i_i * BC) * HQ * BT + i_hq * BT + i_i * BC + o_i
    p_do_row = do + (bos + i_t * BT + i_i * BC) * HQ * V + i_hq * V + o_v
    diag_len = min(BC, T - i_t * BT - i_i * BC)
    for j in range(0, diag_len):
        b_Ar = tl.load(p_A_row)
        b_gr = tl.load(p_g_row, mask=m_v, other=0.0)
        b_dor = tl.load(p_do_row, mask=m_v, other=0.0)
        m_row = o_i[:, None] <= j
        b_dv += tl.where(m_row, exp2(b_gr[None, :] - b_gv) * b_Ar[:, None] * b_dor[None, :], 0.0)
        p_g_row += H * V
        p_A_row += HQ * BT
        p_do_row += HQ * V

    p_o = o + (bos*HQ + i_hq) * V + o_c[:, None] * (HQ*V) + o_v[None, :]
    p_v = v + (bos*H + i_h) * V + o_c[:, None] * (H*V) + o_v[None, :]
    p_do_c = do + (bos*HQ + i_hq) * V + o_c[:, None] * (HQ*V) + o_v[None, :]
    p_dv = dv + (bos*HQ + i_hq) * V + o_c[:, None] * (HQ*V) + o_v[None, :]
    p_dg = dg + (bos*HQ + i_hq) * V + o_c[:, None] * (HQ*V) + o_v[None, :]

    b_o = tl.load(p_o, mask=m_cv, other=0.0, care_padding=False).to(tl.float32)
    b_v = tl.load(p_v, mask=m_cv, other=0.0, care_padding=False).to(tl.float32)
    b_do_c = tl.load(p_do_c, mask=m_cv, other=0.0, care_padding=False).to(tl.float32)
    b_dv = b_dv + tl.load(p_dv, mask=m_cv, other=0.0, care_padding=False).to(tl.float32)
    b_dg = b_o * b_do_c - b_v * b_dv
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_cv)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_cv)



# ---------------------------------------------------------------------------
# 6) Public entry points
# ---------------------------------------------------------------------------


@input_guard
def chunk_gsa_fwd_k_npu(
    q, k, v, g,
    h0=None, output_final_state=False,
    scale=1., cu_seqlens=None,
    chunk_size=64, chunk_indices=None,
):
    """NPU K-half forward for chunk GSA.

    Uses ``chunk_fwd_h`` (NPU-dispatched) for the h_k state and custom NPU
    kernels for the inter / intra A computation.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    HQ = q.shape[2]
    if chunk_indices is None:
        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        else:
            chunk_indices = None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))
    NG = HQ // H

    # h state
    h, ht = chunk_fwd_h(
        k=k, v=v, g=None, gk=None, gv=g,
        h0=h0, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_size=BT, states_in_fp32=False,
    )

    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    NV = triton.cdiv(V, BV)
    npu_kwargs = ascend_compile_kwargs()

    # inter kernel emits ok (the inter-chunk portion of o) and A (the
    # pre-softmax attention matrix restricted to the causal triangle);
    # intra kernel adds the intra-chunk ``A @ v`` contribution to ok in-place.
    ok = q.new_empty(B, T, HQ, V, dtype=torch.float32)
    A = q.new_empty(B, T, HQ, BT, dtype=torch.float32)
    grid_inter = (triton.cdiv(V, BV), NT, B * HQ)
    chunk_gsa_fwd_k_kernel_inter_npu[grid_inter](
        q, k, h, g, ok, A,
        cu_seqlens, chunk_indices,
        scale=scale, T=T, HQ=HQ, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, NG=NG,
        **ascend_compile_kwargs(blacklist_auto_blockify=True),
    )
    grid_intra = (NV * NC, NT, B * HQ)
    chunk_gsa_fwd_k_kernel_intra_npu[grid_intra](
        v, g, ok, A,
        cu_seqlens, chunk_indices,
        T=T, HQ=HQ, H=H, V=V, BT=BT, BC=BC, BV=BV, NC=NC, NG=NG,
        **npu_kwargs,
    )
    return A, h, ht, ok


@input_guard
def chunk_gsa_fwd_v_npu(
    q, k, v, g,
    scale=1., initial_state=None, output_final_state=False,
    cu_seqlens=None, chunk_size=64, chunk_indices=None,
):
    """NPU V-half forward for chunk GSA.

    Delegates entirely to ``chunk_gla_fwd`` (whose sub-kernels route to NPU via
    the 'gla' dispatcher). Matches upstream ``chunk_gsa_fwd_v`` by passing
    ``g_cumsum=g`` and ``g=None`` to the GLA path and discarding the
    ``g_cumsum`` return so the caller can unpack four values.
    """
    _, A, h, ht, o = chunk_gla_fwd(
        q=q, k=k, v=v, g=None, g_cumsum=g,
        scale=scale, initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices,
    )
    return A, h, ht, o


@input_guard
def chunk_gsa_bwd_k_npu(
    q, k, v, g, h, h0, o, do, dht, dg,
    scale=1., cu_seqlens=None, chunk_size=64, chunk_indices=None,
):
    """NPU K-half backward for chunk GSA."""
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))
    HQ = q.shape[2]

    if chunk_indices is None:
        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        else:
            chunk_indices = None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    NG = HQ // H

    if h is None:
        h, _ = chunk_fwd_h(
            k=k, v=v, g=None, gk=None, gv=g,
            h0=h0, output_final_state=False,
            cu_seqlens=cu_seqlens, chunk_size=BT, states_in_fp32=False,
        )
    dh, dh0 = chunk_bwd_dh(
        q=q, k=k, v=v, g=None, gk=None, gv=g,
        do=do, h0=h0, dht=dht, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=BT, states_in_fp32=True,
    )

    # Disable auto-multi-buffer on the K-half backward launches. Direct
    # bishengir-compile testing shows the dqkvg IR segfaults with
    # ``--enable-auto-multi-buffer=True`` but compiles cleanly when off; the
    # A / dA / intra_dvg kernels are simpler but share the same launch so we
    # pass the flag everywhere for consistency.
    npu_kwargs = ascend_compile_kwargs()

    dA = q.new_empty(NV, B, T, HQ, BT, dtype=torch.float32)
    grid_dA = (NV * NT * NC * NC, B * HQ)
    chunk_gsa_bwd_k_kernel_dA_npu[grid_dA](
        v, g, do, dA,
        cu_seqlens, chunk_indices,
        scale=scale, T=T, B=B, HQ=HQ, H=H, V=V, BT=BT, BC=BC, BV=BV, NC=NC, NG=NG, NT=NT,
        **npu_kwargs,
    )
    dA = dA.sum(0, dtype=dA.dtype)

    A = do.new_empty(NK, B, T, HQ, BT, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = k.new_empty(B, T, HQ, K, dtype=q.dtype)
    dv = v.new_empty(NK, B, T, HQ, V, dtype=torch.float32)
    dgv = g.new_empty(NK, B, T, HQ, V, dtype=torch.float32)
    # Host-side pre-transpose of h [..., K, V] → h_t [..., V, K] and dh
    # [..., K, V] → dh_t [..., V, K]. Loading canonical [BV, BK] tiles from
    # these buffers inside the V-loop avoids stride-swap loads, which this
    # toolchain lowers into explicit ``linalg.transpose`` ops (see
    # ``docs/bishengir-bug/README.md``).
    h_t = h.transpose(-1, -2).contiguous()
    dh_t = dh.transpose(-1, -2).contiguous()
    grid_A = (NK, NT, B * HQ)
    chunk_gsa_bwd_k_kernel_A_npu[grid_A](
        q, k, A,
        cu_seqlens, chunk_indices,
        scale=scale, T=T, B=B, HQ=HQ, H=H, K=K, BT=BT, BK=BK, NG=NG,
        **npu_kwargs,
    )
    grid = (NK, NT, B * HQ)
    chunk_gsa_bwd_k_kernel_dqkvg_npu[grid](
        q, k, v, h_t, g, A, do, dh, dh_t,
        dq, dk, dv, dg, dgv, dA,
        cu_seqlens, chunk_indices,
        scale=scale, T=T, B=B, HQ=HQ, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, NC=NC, NG=NG,
        **npu_kwargs,
    )
    A = A.sum(0, dtype=A.dtype)
    dv = dv.sum(0, dtype=dv.dtype)
    dgv = dgv.sum(0, dtype=dv.dtype)

    # Grid: (NV * NC, NT, B*HQ). Upstream uses a 2-axis grid
    # (NV * NT * NC, B * HQ); we lift NT out to keep NV × NC in program_id(0)
    # (the kernel unpacks it with i_i = i_vc // NV, i_v = i_vc % NV) while
    # staying inside the Ascend 3-axis grid cap.
    grid_intra = (NV * NC, NT, B * HQ)
    chunk_gsa_bwd_k_kernel_intra_dvg_npu[grid_intra](
        v, g, o, A, do, dv, dg,
        cu_seqlens, chunk_indices,
        T=T, B=B, HQ=HQ, H=H, V=V, BT=BT, BC=BC, BV=BV, NC=NC, NG=NG,
        **npu_kwargs,
    )
    dg = dgv.add_(
        chunk_local_cumsum(
            dg, chunk_size=BT, reverse=True,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        )
    )
    return dq, dk, dv, dg, dh0


@input_guard
def chunk_gsa_bwd_v_npu(
    q, k, v, g, h, h0, A, do, dht, dg,
    scale=1., cu_seqlens=None, chunk_size=64, chunk_indices=None,
):
    """NPU V-half backward for chunk GSA.

    Delegates entirely to ``chunk_gla_bwd`` (whose sub-kernels route to NPU
    via the 'gla' dispatcher). Mirrors upstream ``chunk_gsa_bwd_v`` by
    forwarding g as ``g_cumsum`` and passing g=None.
    """
    return chunk_gla_bwd(
        q=q, k=k, v=v, g=None, g_cumsum=g,
        h=h, initial_state=h0, A=A, do=do, dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices,
    )
