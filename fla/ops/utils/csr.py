# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_offsets


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.jit(do_not_specialize=['T', 'N'])
def prepare_block_csr_kernel(
    block_indices,
    block_counts,
    cu_seqlens,
    chunk_offsets,
    cursor,
    csr_indices,
    csr_offsets,
    N,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    TC: tl.constexpr,
    COUNT_ONLY: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_s = tl.arange(0, S)
    m_t = o_t < T
    # [BT] flattened (batch, query, kv-head) index; int64 to keep address arithmetic safe at large T
    i_qh = ((i_b * T).to(tl.int64) + o_t) * H + i_h

    # [BT, S] selected blocks, masked to each query's valid causal in-range slots
    b_i = tl.load(block_indices + i_qh[:, None] * S + o_s[None, :], mask=m_t[:, None], other=-1).to(tl.int64)
    if USE_BLOCK_COUNTS:
        b_m = m_t[:, None] & (o_s[None, :] < tl.load(block_counts + i_qh, mask=m_t, other=0)[:, None])
    else:
        b_m = m_t[:, None] & (o_s[None, :] < block_counts)
    b_m = b_m & (b_i >= 0) & (b_i < TC) & (b_i * BS <= o_t[:, None])

    if IS_VARLEN:
        # vectorized binary search for the sequence holding each query (32 steps cover any num_seq)
        lo, hi = tl.zeros([BT], dtype=tl.int32), tl.full([BT], N, dtype=tl.int32)
        for _ in range(32):
            mid = (lo + hi) // 2
            go = tl.load(cu_seqlens + mid + 1, mask=m_t, other=0) <= o_t
            lo, hi = tl.where(go, mid + 1, lo), tl.where(go, hi, mid)
        block_base = tl.load(chunk_offsets + lo, mask=m_t, other=0).to(tl.int64)
        block_id = (block_base[:, None] + b_i) * H + i_h
    else:
        block_id = (i_b * H + i_h) * TC + b_i

    if COUNT_ONLY:
        tl.atomic_add(csr_offsets + block_id + 1, 1, mask=b_m)
    else:
        dst = tl.load(csr_offsets + block_id, mask=b_m, other=0).to(tl.int64) + tl.atomic_add(cursor + block_id, 1, mask=b_m)
        b_q = tl.broadcast_to((i_b * T + o_t)[:, None], (BT, S))
        tl.store(csr_indices + dst, b_q.to(csr_indices.dtype.element_ty), mask=b_m)


def prepare_block_csr(
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None,
    num_blocks: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Invert a per-query block selection into CSR (compressed sparse row) form.

    `block_indices[b, t, h, :]` lists the blocks query `t` (kv-head `h`) selects.
    The inverse maps each block to the queries that selected it,
    which a block-parallel backward (e.g. NSA `bwd_dkv`) needs.
    The result is CSR over a `[block, query]` matrix: `csr_indices` holds the selecting query positions grouped by block,
    and `csr_offsets` holds the per-block row offsets,
    so block `i` owns `csr_indices[csr_offsets[i]:csr_offsets[i + 1]]`.

    Block ids follow the launching kernel's program-id layout:
    `(b * H + h) * num_blocks + s` for dense, `(global_block + s) * H + h` for varlen,
    where the varlen block base comes from an in-kernel binary search over `cu_seqlens` into `chunk_offsets`.
    Short problems use a counting-then-scatter sort (`csr_indices` over-allocated to its upper bound, so its length
    need not be read back to the host); long ones bucket the pairs with a radix sort. Both yield the same CSR.

    Example (dense, B = H = 1, block_size = 1 so block b covers token b; causal needs b <= t):

        # input: each query lists the blocks it selects, -1 is padding
        block_indices[0, :, 0, :] =
            [[ 0, -1],   # query 0 selects block 0
             [ 0,  1],   # query 1 selects blocks 0, 1
             [ 1,  2],   # query 2 selects blocks 1, 2
             [ 0,  3]]   # query 3 selects blocks 0, 3

        # invert -> which queries selected each block:
        #   block 0: queries 0, 1, 3
        #   block 1: queries 1, 2
        #   block 2: query 2
        #   block 3: query 3

        csr_indices = [0, 1, 3,  1, 2,  2,  3]   # 7 selections, grouped by block (order within a block is arbitrary)
        csr_offsets = [0, 3, 5, 6, 7]            # block i's queries = csr_indices[csr_offsets[i]:csr_offsets[i+1]]

    Args:
        block_indices (torch.LongTensor):
            Selected block ids of shape `[B, T, H, S]`, padded with `-1`.
        block_counts (torch.LongTensor or int):
            Number of valid slots per query, a `[B, T, H]` tensor or an int.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths for variable-length packing. Default: `None` (dense).
        chunk_indices (torch.LongTensor):
            Per-chunk `(sequence, local-block)` index pairs; read only to size the varlen block-id space.
        num_blocks (int):
            Number of blocks per `(batch, head)`, i.e. the dense kernel's `TC`.
        block_size (int):
            Selected block size, used for the causal check and varlen block ids.

    Returns:
        csr_indices (torch.Tensor):
            `int32` selecting query positions, grouped by block; absolute (`b * T + t`).
        csr_offsets (torch.Tensor):
            `int32` CSR row offsets of shape `[NB + 1]`, one per block plus a final end offset.
    """
    B, T, H, S = block_indices.shape
    N = 0 if cu_seqlens is None else cu_seqlens.numel() - 1
    NB = B * H * num_blocks if cu_seqlens is None else chunk_indices.shape[0] * H
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, block_size) if cu_seqlens is not None else None

    cursor = block_indices.new_zeros(NB, dtype=torch.int32)
    csr_offsets = block_indices.new_zeros(NB + 1, dtype=torch.int32)
    csr_indices = block_indices.new_empty(B * T * H * S, dtype=torch.int32)

    BT = max(1, min(128, triton.next_power_of_2(max(1, 2048 // S))))
    grid = (triton.cdiv(T, BT), B * H)

    # counting sort: tally per-block counts, prefix-sum them into start offsets, then scatter.
    # the two kernel passes can't merge -- the scatter position needs csr_offsets ready (global prefix sum).
    kwargs = dict(
        block_indices=block_indices,
        block_counts=block_counts,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        cursor=cursor,
        csr_indices=csr_indices,
        csr_offsets=csr_offsets,
        N=N,
        T=T,
        H=H,
        S=S,
        BT=BT,
        BS=block_size,
        TC=num_blocks,
    )
    prepare_block_csr_kernel[grid](**kwargs, COUNT_ONLY=True)
    csr_offsets.cumsum_(0)
    prepare_block_csr_kernel[grid](**kwargs, COUNT_ONLY=False)
    return csr_indices, csr_offsets
