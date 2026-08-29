# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import get_max_num_splits, prepare_chunk_indices


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def transform_q_fwd_kernel(
    q,
    q_new,
    w1,
    w2,
    cu_seqlens,
    indices,
    T,
    S: tl.constexpr,
    G: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        # boh = i_n * tl.cdiv(T, BS)
    o_q = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    m_q = (o_q[:, None] < T) & (o_d[None, :] < K)
    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, mask=m_q, other=0.0)

    if BS == BT:
        if (i_t * BT) % S == 0:
            p_q_new = q_new + ((bos.to(tl.int64) * NUM_BLOCKS + (i_t * BT // S)) * HQ + i_hq) * \
                K + o_q[:, None] * (HQ*K*NUM_BLOCKS) + o_d[None, :]
            tl.store(p_q_new, b_q.to(q_new.dtype.element_ty), mask=m_q)

    for offset in range((i_t + 1) * BT - 2 * BS, S-BS, -BS):
        o_w = offset + tl.arange(0, BS)
        m_w = o_w < T
        p_w1 = w1 + (bos * H + i_h) * K + o_d[:, None] + o_w[None, :] * (K*H)
        p_w2 = w2 + (bos * H + i_h) * K + o_w[:, None] * (K*H) + o_d[None, :]
        b_w1 = tl.load(p_w1, mask=(o_d[:, None] < K) & m_w[None, :], other=0.0)
        b_w2 = tl.load(p_w2, mask=m_w[:, None] & (o_d[None, :] < K), other=0.0)
        m_s = o_q >= (offset + BS)
        b_s2 = tl.dot(b_q.to(b_w1.dtype), b_w1)
        b_s2 = tl.where(m_s[:, None], b_s2, 0)
        b_q -= tl.dot(b_s2.to(b_w2.dtype), b_w2)

        if offset % S == 0:
            p_q_new = q_new + ((bos.to(tl.int64) * NUM_BLOCKS + (offset // S)) * HQ + i_hq) * \
                K + o_q[:, None] * (HQ*K*NUM_BLOCKS) + o_d[None, :]
            tl.store(p_q_new, b_q.to(q_new.dtype.element_ty), mask=m_q)


def transform_q_fwd_fn(
    q,
    w1,
    w2,
    cu_seqlens,
    BT,
    BS,
    S,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, HQ, K = q.shape
    H = w1.shape[-2]
    G = HQ // H
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    indices = chunk_indices
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)

    num_blocks = triton.cdiv(T, S) if cu_seqlens is None else get_max_num_splits(cu_seqlens, S)
    q_new = torch.zeros(B, T, num_blocks, HQ, K, dtype=q.dtype, device=q.device)
    transform_q_fwd_kernel[(NT, B * HQ)](
        q=q,
        q_new=q_new,
        w1=w1,
        w2=w2,
        cu_seqlens=cu_seqlens,
        indices=indices,
        T=T,
        K=K,
        BK=triton.next_power_of_2(K),
        G=G,
        HQ=HQ,
        H=H,
        BS=BS,
        BT=BT,
        S=S,
        NUM_BLOCKS=num_blocks,
        num_warps=8 if (BT == 128 and K == 128) else 4,
    )
    return q_new
