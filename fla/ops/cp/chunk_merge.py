# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'HAS_H0': lambda args: args['h0'] is not None,
    'HAS_H_SEQ_IDX': lambda args: args['h_seq_idx'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['HV', 'K', 'V'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['pre_or_post_num_ranks', 'rank', 'NUM_SEQ_ENTRIES'])
def chunk_cp_fwd_bwd_kernel_merge(
    h,
    ag_hm,
    pre_or_post_num_ranks,
    rank,
    seq_offsets,
    init_offsets,
    h0_seq_ids,
    h0,
    h_seq_idx,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    FORWARD: tl.constexpr,
    INTRACARD_MODE: tl.constexpr,
    NUM_SEQ_ENTRIES,
    HAS_H0: tl.constexpr,
    HAS_H_SEQ_IDX: tl.constexpr,
    STATE_V_FIRST: tl.constexpr = False,
    AFFINE_CHAIN_PRECISION: tl.constexpr = None,
    NUM_RANKS_ON_DEVICE: tl.constexpr = False,
):
    """Compose affine summaries across ranks or local subsequences.

    Summaries store [K, V+K] regardless of the state layout.
    Intracard mode writes each non-first subsequence's initial state; CP mode writes one boundary state.
    """
    i_v = tl.program_id(0).to(tl.int64)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    if INTRACARD_MODE:
        i_seq = tl.program_id(1).to(tl.int64)
        i_h = tl.program_id(2).to(tl.int64)

        if i_seq >= NUM_SEQ_ENTRIES:
            return

        ss_start = tl.load(seq_offsets + i_seq).to(tl.int32)
        ss_end = tl.load(seq_offsets + i_seq + 1).to(tl.int32)
        init_base = tl.load(init_offsets + i_seq).to(tl.int32)
        num_subseqs = ss_end - ss_start

        stride_hm_s = HV * K * (V + K)
        stride_hm_h = K * (V + K)

        if HAS_H0:
            orig_seq_id = tl.load(h0_seq_ids + i_seq).to(tl.int32)
            if STATE_V_FIRST:
                p_h0 = h0 + (orig_seq_id * HV + i_h) * V * K + o_v[:, None] * K + o_k[None, :]
                b_h = tl.load(p_h0, mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            else:
                p_h0 = h0 + (orig_seq_id * HV + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
                b_h = tl.load(p_h0, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        else:
            if STATE_V_FIRST:
                b_h = tl.zeros([BV, BK], dtype=tl.float32)
            else:
                b_h = tl.zeros([BK, BV], dtype=tl.float32)

        for idx in range(num_subseqs):
            i_ss = ss_start + idx
            base = i_ss * stride_hm_s + i_h * stride_hm_h

            # he and m are always in [K, V+K] layout from pre_scan
            p_he = ag_hm + base + o_k[:, None] * (V + K) + o_v[None, :]
            b_he = tl.load(p_he, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
            p_m = ag_hm + base + V + o_k[:, None] * (V + K) + o_k[None, :]
            b_m = tl.load(p_m, mask=m_k[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            if STATE_V_FIRST:
                # h_T' = h_T @ M^T + he^T
                b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_m), input_precision=AFFINE_CHAIN_PRECISION) + tl.trans(b_he)
            else:
                b_h = tl.dot(b_m.to(tl.float32), b_h.to(tl.float32),
                             input_precision=AFFINE_CHAIN_PRECISION) + b_he.to(tl.float32)

            if idx < num_subseqs - 1:
                init_idx = init_base + idx
                stride_init = HV * K * V
                if STATE_V_FIRST:
                    p_out = h + init_idx * stride_init + i_h * V * K + o_v[:, None] * K + o_k[None, :]
                    m_out = m_v[:, None] & m_k[None, :]
                else:
                    p_out = h + init_idx * stride_init + i_h * K * V + o_k[:, None] * V + o_v[None, :]
                    m_out = m_k[:, None] & m_v[None, :]
                tl.store(p_out, b_h.to(p_out.dtype.element_ty), mask=m_out)
    else:
        i_h = tl.program_id(1).to(tl.int64)
        if HAS_H_SEQ_IDX:
            h += tl.load(h_seq_idx).to(tl.int64) * HV * K * V
        if NUM_RANKS_ON_DEVICE:
            num_ranks = tl.load(pre_or_post_num_ranks).to(tl.int32)
        else:
            num_ranks = pre_or_post_num_ranks.to(tl.int32)
        h += i_h * K * V
        ag_hm += i_h * K * (K + V)
        stride = HV * K * (K + V)
        if STATE_V_FIRST:
            b_h = tl.zeros([BV, BK], dtype=tl.float32)
        else:
            b_h = tl.zeros([BK, BV], dtype=tl.float32)
        for idx in range(num_ranks):
            if FORWARD:
                cur_rank = rank - num_ranks + idx
            else:
                cur_rank = rank + num_ranks - idx
            p_ag_h = ag_hm + cur_rank * stride + o_k[:, None] * (K + V) + o_v[None, :]
            b_ag_h = tl.load(p_ag_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
            p_ag_m = ag_hm + cur_rank * stride + V + o_k[:, None] * (K + V) + o_k[None, :]
            b_ag_m = tl.load(p_ag_m, mask=m_k[:, None] & m_k[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_ag_m).to(tl.float32),
                             input_precision=AFFINE_CHAIN_PRECISION) + tl.trans(b_ag_h).to(tl.float32)
            else:
                b_h = tl.dot(b_ag_m.to(tl.float32), b_h.to(tl.float32),
                             input_precision=AFFINE_CHAIN_PRECISION) + b_ag_h.to(tl.float32)
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
    affine_chain_precision: str | None = None,
) -> None:
    """Merge gathered affine summaries into a rank's boundary state."""
    _, HV, K, VK = ag_hm.shape
    V = VK - K
    BK = triton.next_power_of_2(K)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), HV)

    chunk_cp_fwd_bwd_kernel_merge[grid](
        h=h,
        ag_hm=ag_hm,
        pre_or_post_num_ranks=num_ranks,
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
        INTRACARD_MODE=False,
        NUM_SEQ_ENTRIES=0,
        STATE_V_FIRST=state_v_first,
        AFFINE_CHAIN_PRECISION=affine_chain_precision,
        NUM_RANKS_ON_DEVICE=use_graph,
    )
