# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices


@triton.heuristics({
    'IS_VARLEN': lambda args: args['offsets'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def transform_q_fwd_kernel(
    q, q_new, w1, w2,
    offsets, indices,
    T,
    S: tl.constexpr,
    G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
        # boh = i_n * tl.cdiv(T, BS)
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, boundary_check=(0, 1))

    if BS == BT:
        if (i_t * BT) % S == 0:
            p_q_new = tl.make_block_ptr(q_new + ((bos * NUM_BLOCKS + (i_t * BT // S)) * HQ + i_hq) * K,
                                        (T, K), (HQ*K*NUM_BLOCKS, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            tl.store(p_q_new, b_q.to(q_new.dtype.element_ty), boundary_check=(0, 1))

    for offset in range((i_t + 1) * BT - 2 * BS, S-BS, -BS):
        p_w1 = tl.make_block_ptr(w1 + (bos * H + i_h) * K, (K, T), (1, K*H), (0, offset), (BK, BS), (0, 1))
        p_w2 = tl.make_block_ptr(w2 + (bos * H + i_h) * K, (T, K), (K*H, 1), (offset, 0), (BS, BK), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        m_s = i_t * BT + tl.arange(0, BT) >= (offset + BS)
        b_s2 = tl.dot(b_q.to(b_w1.dtype), b_w1)
        b_s2 = tl.where(m_s[:, None], b_s2, 0)
        b_q -= tl.dot(b_s2.to(b_w2.dtype), b_w2)

        if offset % S == 0:
            p_q_new = tl.make_block_ptr(q_new + ((bos * NUM_BLOCKS + (offset // S)) * HQ + i_hq) * K,
                                        (T, K), (HQ*K*NUM_BLOCKS, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            tl.store(p_q_new, b_q.to(q_new.dtype.element_ty), boundary_check=(0, 1))


def transform_q_fwd_fn(
    q, w1, w2,
    cu_seqlens, BT, BS, S
):
    B, T, HQ, K = q.shape
    H = w1.shape[-2]
    G = HQ // H
    indices_BT = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices_BT)
    grid = (NT, B * HQ)
    num_blocks = triton.cdiv(T, S) if cu_seqlens is None else len(cu_seqlens) - 1
    q_new = torch.empty(B, T, num_blocks, HQ, K, dtype=q.dtype, device=q.device)
    transform_q_fwd_kernel[grid](
        q=q,
        q_new=q_new,
        w1=w1,
        w2=w2,
        offsets=cu_seqlens,
        indices=indices_BT,
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
        num_warps=8 if (BT == 128 and K == 128) else 4
    )
    return q_new
