# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_offsets, prepare_token_indices


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.jit(do_not_specialize=['T'])
def prepare_block_csr_kernel(
    block_indices,
    block_counts,
    csr_indices,
    csr_offsets,
    cursor,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    TC: tl.constexpr,
    COUNT_ONLY: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    # one program per (batch, kv-head, query); scan its selected blocks and, for each valid one,
    # count it (pass 1) or scatter it (pass 2) into the selected block's CSR segment
    i_q, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    i_qh = ((i_b * T).to(tl.int64) + i_q) * H + i_h
    if IS_VARLEN:
        # block_base = global block base of this query's sequence (as in the fwd/topk kernels)
        i_n = tl.load(token_indices + i_q * 2).to(tl.int32)
        block_base = tl.load(chunk_offsets + i_n).to(tl.int64)

    block_indices += i_qh * S
    NS = tl.load(block_counts + i_qh) if USE_BLOCK_COUNTS else block_counts
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int64)
        if (i_s >= 0) and (i_s < TC) and (i_s * BS <= i_q):
            block_id = (block_base + i_s) * H + i_h if IS_VARLEN else (i_b * H + i_h) * TC + i_s
            if COUNT_ONLY:
                tl.atomic_add(csr_offsets + block_id + 1, 1)
            else:
                dst = tl.load(csr_offsets + block_id).to(tl.int64) + tl.atomic_add(cursor + block_id, 1)
                tl.store(csr_indices + dst, i_b * T + i_q)


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
    `(b * H + h) * num_blocks + s` for dense, `(global_block + s) * H + h` for varlen.
    The varlen block base is read from the standard `token_indices` / `chunk_offsets` (as in the fwd/topk kernels).
    Counting then scattering avoids a comparison sort,
    and `csr_indices` is over-allocated to its upper bound, so its length need not be read back to the host.

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
    device = block_indices.device
    NB = B * H * num_blocks if cu_seqlens is None else chunk_indices.shape[0] * H
    token_indices = prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, block_size) if cu_seqlens is not None else None

    cursor = torch.zeros(NB, dtype=torch.int32, device=device)
    csr_offsets = torch.zeros(NB + 1, dtype=torch.int32, device=device)
    csr_indices = torch.empty(B * T * H * S, dtype=torch.int32, device=device)

    grid = (T, B * H)

    # counting sort over blocks. it takes two kernel launches, not one: the scatter places each
    # query at csr_offsets[block] + cursor, and csr_offsets is the prefix sum over ALL blocks'
    # counts -- a global dependency one grid launch can't satisfy (no cross-program barrier).
    def launch(count_only):
        prepare_block_csr_kernel[grid](
            block_indices=block_indices,
            block_counts=block_counts,
            csr_indices=csr_indices,
            csr_offsets=csr_offsets,
            cursor=cursor,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
            chunk_offsets=chunk_offsets,
            T=T,
            H=H,
            S=S,
            BS=block_size,
            TC=num_blocks,
            COUNT_ONLY=count_only,
        )

    # pass 1: tally each block's query count into csr_offsets[block + 1]
    launch(count_only=True)
    # prefix sum in place: per-block tallies -> CSR start offsets
    csr_offsets.cumsum_(0)
    # pass 2: scatter each query into its block's segment
    launch(count_only=False)
    return csr_indices, csr_offsets
