# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.cp.context import _check_cp_device
from fla.utils import IS_NPU, autotune_cache_kwargs

CP_DOT_PRECISION = tl.constexpr('ieee' if IS_NPU else 'tf32x3')


@triton.jit
def chunk_cp_merge_state(
    hm,
    b_h,
    o_k,
    o_v,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    """Apply one [h_local | M] summary to a boundary state.

    K-first states use M @ h + h_local; V-first states use h @ M.T + h_local.T.
    The summary itself always keeps the [K, V+K] layout.
    """
    m_k, m_v = o_k < K, o_v < V
    p_h_local = hm + o_k[:, None] * (V + K) + o_v[None, :]
    b_h_local = tl.load(p_h_local, mask=m_k[:, None] & m_v[None, :], other=0.).to(tl.float32)
    if BK <= 128:
        p_m = hm + V + o_k[:, None] * (V + K) + o_k[None, :]
        b_m = tl.load(p_m, mask=m_k[:, None] & m_k[None, :], other=0.).to(tl.float32)
        if STATE_V_FIRST:
            b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_m), input_precision=CP_DOT_PRECISION)
        else:
            b_h = tl.dot(b_m, b_h.to(tl.float32), input_precision=CP_DOT_PRECISION)
    else:
        # tile wide reductions to keep TF32x3 operands within shared memory
        b_acc = tl.full(b_h.shape, 0, tl.float32)
        for i_k in tl.static_range(BK // 128):
            # pair M's reduction columns with the same key rows of the incoming state
            o_r = i_k * 128 + tl.arange(0, 128)
            p_m = hm + V + o_k[:, None] * (V + K) + o_r[None, :]
            b_m = tl.load(p_m, mask=m_k[:, None] & (o_r < K)[None, :], other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_hk = tl.gather(b_h, tl.broadcast_to(o_r[None, :], (BV, 128)), axis=1)
                b_acc = tl.dot(b_hk.to(tl.float32), tl.trans(b_m), b_acc, input_precision=CP_DOT_PRECISION)
            else:
                b_hk = tl.gather(b_h, tl.broadcast_to(o_r[:, None], (128, BV)), axis=0)
                b_acc = tl.dot(b_m, b_hk.to(tl.float32), b_acc, input_precision=CP_DOT_PRECISION)
        b_h = b_acc
    return b_h + (tl.trans(b_h_local) if STATE_V_FIRST else b_h_local)


def _prune_cp_configs(configs, nargs, **kwargs):
    args = {**nargs, **kwargs}
    # wide TF32x3 states need single-stage candidates to fit shared memory
    if args['K'] <= 128 and any(config.num_stages > 1 for config in configs):
        return [config for config in configs if config.num_stages > 1]
    return configs


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_H_SEQ_IDX': lambda args: args['h_seq_idx'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [1, 2, 3, 4]
        for BV in [32, 64]
    ],
    key=['HV', 'K', 'V', 'STATE_V_FIRST', 'IS_INTRACARD'],
    prune_configs_by={'early_config_prune': _prune_cp_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['num_ranks', 'rank', 'N'])
def chunk_cp_fwd_bwd_kernel_merge(
    h,                   # [HV, K, V] or [num_non_first, HV, K, V]; K/V swapped for STATE_V_FIRST
    ag_hm,               # [world_size, HV, K, V+K] or [num_subseqs, HV, K, V+K]
    num_ranks,
    rank,
    seq_offsets,         # [num_split_seqs + 1] or None
    init_offsets,        # [num_split_seqs + 1] or None
    h0_seq_ids,          # [num_split_seqs] or None
    h0,                  # [N_orig, HV, K, V] or None; K/V swapped for STATE_V_FIRST
    h_seq_idx,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    FORWARD: tl.constexpr,
    IS_INTRACARD: tl.constexpr,
    N,
    USE_INITIAL_STATE: tl.constexpr,
    USE_H_SEQ_IDX: tl.constexpr,
    STATE_V_FIRST: tl.constexpr = False,
    NUM_RANKS_ON_DEVICE: tl.constexpr = False,
):
    """Merge affine summaries across ranks or intra-card subsequences.

    Summaries use [K, V+K]; STATE_V_FIRST selects [V, K] input and output states.
    Cross-rank mode writes one boundary state per head after composing the contributing ranks.
    Intra-card mode starts from each original sequence's h0 and writes the state entering each later subsequence.
    """
    i_v = tl.program_id(0).to(tl.int64)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    if IS_INTRACARD:
        i_n = tl.program_id(1).to(tl.int64)
        i_h = tl.program_id(2).to(tl.int64)

        if i_n >= N:
            return

        # seq_offsets indexes summaries; init_offsets indexes outputs with the first subsequence omitted
        bos = tl.load(seq_offsets + i_n).to(tl.int32)
        eos = tl.load(seq_offsets + i_n + 1).to(tl.int32)
        boh = tl.load(init_offsets + i_n).to(tl.int32)
        NS = eos - bos

        stride_hm_n = HV * K * (V + K)
        stride_hm_h = K * (V + K)

        if USE_INITIAL_STATE:
            i_n0 = tl.load(h0_seq_ids + i_n).to(tl.int32)
            if STATE_V_FIRST:
                p_h0 = h0 + (i_n0 * HV + i_h) * V * K + o_v[:, None] * K + o_k[None, :]
                b_h = tl.load(p_h0, mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            else:
                p_h0 = h0 + (i_n0 * HV + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
                b_h = tl.load(p_h0, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        else:
            if STATE_V_FIRST:
                b_h = tl.zeros([BV, BK], dtype=tl.float32)
            else:
                b_h = tl.zeros([BK, BV], dtype=tl.float32)

        for i_s in range(NS):
            i_ns = bos + i_s
            o_hm = i_ns * stride_hm_n + i_h * stride_hm_h

            b_h = chunk_cp_merge_state(
                hm=ag_hm + o_hm,
                b_h=b_h,
                o_k=o_k,
                o_v=o_v,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                STATE_V_FIRST=STATE_V_FIRST,
            )

            # this prefix's final state initializes the next subsequence; the last prefix has no successor
            if i_s < NS - 1:
                i_h0 = boh + i_s
                stride_h = HV * K * V
                if STATE_V_FIRST:
                    p_h = h + i_h0 * stride_h + i_h * V * K + o_v[:, None] * K + o_k[None, :]
                    m_h = m_v[:, None] & m_k[None, :]
                else:
                    p_h = h + i_h0 * stride_h + i_h * K * V + o_k[:, None] * V + o_v[None, :]
                    m_h = m_k[:, None] & m_v[None, :]
                tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=m_h)
    else:
        i_h = tl.program_id(1).to(tl.int64)
        if USE_H_SEQ_IDX:
            h += tl.load(h_seq_idx).to(tl.int64) * HV * K * V
        if NUM_RANKS_ON_DEVICE:
            # graph replay must read refreshed rank counts instead of capture-time Python scalars
            NR = tl.load(num_ranks).to(tl.int32)
        else:
            NR = num_ranks.to(tl.int32)
        h += i_h * K * V
        ag_hm += i_h * K * (K + V)
        stride_hm = HV * K * (K + V)
        if STATE_V_FIRST:
            b_h = tl.zeros([BV, BK], dtype=tl.float32)
        else:
            b_h = tl.zeros([BK, BV], dtype=tl.float32)
        # forward composes preceding ranks left to right; backward composes following ranks right to left
        for i_s in range(NR):
            if FORWARD:
                i_r = rank - NR + i_s
            else:
                i_r = rank + NR - i_s
            b_h = chunk_cp_merge_state(
                hm=ag_hm + i_r * stride_hm,
                b_h=b_h,
                o_k=o_k,
                o_v=o_v,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                STATE_V_FIRST=STATE_V_FIRST,
            )
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            m_h = m_v[:, None] & m_k[None, :]
        else:
            p_h = h + o_k[:, None] * V + o_v[None, :]
            m_h = m_k[:, None] & m_v[None, :]
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=m_h)


def chunk_cp_merge(
    h: torch.Tensor,
    ag_hm: torch.Tensor,
    num_ranks: int | torch.Tensor,
    rank: int,
    state_v_first: bool = False,
    forward: bool = True,
    h_seq_idx: torch.Tensor | None = None,
    use_graph: bool = False,
) -> None:
    """Merge gathered affine summaries into a rank's boundary state."""
    _check_cp_device(h.device)
    _, HV, K, VK = ag_hm.shape
    V = VK - K
    BK = triton.next_power_of_2(K)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), HV)

    chunk_cp_fwd_bwd_kernel_merge[grid](
        h=h,
        ag_hm=ag_hm,
        num_ranks=num_ranks,
        rank=rank,
        seq_offsets=None,
        init_offsets=None,
        h0_seq_ids=None,
        h0=None,
        h_seq_idx=h_seq_idx,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        FORWARD=forward,
        IS_INTRACARD=False,
        N=0,
        STATE_V_FIRST=state_v_first,
        NUM_RANKS_ON_DEVICE=use_graph,
    )
