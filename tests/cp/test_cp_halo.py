#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for halo_exchange_and_extend utility.

Run (CPU, gloo):
  torchrun --nproc_per_node=2 tests/cp/test_cp_halo.py --cp_size 2

These tests validate:
- Basic halo passing (right tail -> left halo) across ranks
- VarLen safety: zero halo when shard starts at a true sequence boundary
- cu_seqlens extension is correctly shifted by +h
"""

import argparse
import os
from typing import Tuple

import torch
import torch.distributed as dist

from fla.ops.gated_delta_rule.cp_halo import halo_exchange_and_extend


def _init_dist(backend: str = 'gloo') -> Tuple[int, int]:
    """Initialize a simple process group for CPU tests."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size()


def test_basic_halo(cp_size: int, h: int = 2):
    """Test that rank r receives the tail from rank r-1 as its left halo."""
    rank, world_size = _init_dist('gloo')
    assert world_size == cp_size, f"world_size({world_size}) != cp_size({cp_size})"

    # Shapes
    B = 1
    T_local = 6
    Dq, Dk, Dv = 3, 4, 5

    # Create deterministic inputs per rank
    torch.manual_seed(2025 + rank)
    q_in = torch.randn(B, T_local, Dq)
    k_in = torch.randn(B, T_local, Dk)
    v_in = torch.randn(B, T_local, Dv)

    # No varlen here
    cu_seqlens = None

    q_ext, k_ext, v_ext, cu_ext = halo_exchange_and_extend(
        q_in, k_in, v_in, h,
        cp_rank=rank, cp_size=cp_size, cp_group=dist.group.WORLD,
        cu_seqlens=cu_seqlens, cp_shard_start_idx=None,
    )

    # Rank 0 should have zero halo
    if rank == 0:
        assert torch.allclose(q_ext[:, :h], torch.zeros_like(q_ext[:, :h]))
        assert cu_ext is None
    # Rank > 0 should have halo equal to last h tokens of previous rank
    else:
        # Receive prev rank's tail via a helper all_gather for verification
        # We need the previous rank's q_in to compare. Gather small tensors.
        q_tails = [torch.zeros(B, h, Dq) for _ in range(world_size)]
        local_tail = q_in[:, -h:, :].contiguous()
        dist.all_gather(q_tails, local_tail)
        prev_tail = q_tails[rank - 1]
        assert torch.allclose(q_ext[:, :h, :], prev_tail, atol=1e-6), (
            f"Rank {rank}: halo mismatch vs prev rank tail"
        )

    if dist.is_initialized():
        dist.barrier()


def test_varlen_zero_halo(cp_size: int, h: int = 2):
    """
    When a shard begins at a true sequence boundary (per cu_seqlens), its received halo must be zero.
    Construct two equal sequences in flattened layout with boundary at global index T_local.
    """
    rank, world_size = _init_dist('gloo')
    assert world_size == cp_size

    B = 1
    T_local = 5
    Dq, Dk, Dv = 2, 3, 4

    torch.manual_seed(7 + rank)
    q_in = torch.randn(B, T_local, Dq)
    k_in = torch.randn(B, T_local, Dk)
    v_in = torch.randn(B, T_local, Dv)

    # Two sequences, each of length T_local
    # Global flattened boundaries: [0, T_local, 2*T_local]
    cu_seqlens = torch.tensor([0, T_local, 2 * T_local], dtype=torch.long)

    # This shard's global start index
    shard_start = rank * T_local

    q_ext, k_ext, v_ext, cu_ext = halo_exchange_and_extend(
        q_in, k_in, v_in, h,
        cp_rank=rank, cp_size=cp_size, cp_group=dist.group.WORLD,
        cu_seqlens=cu_seqlens, cp_shard_start_idx=shard_start,
    )

    # cu_seqlens must be shifted by +h (except the first 0)
    assert cu_ext is not None
    assert torch.equal(cu_ext, torch.tensor([0, T_local + h, 2 * T_local + h]))

    # For rank 1, shard_start == T_local is a true boundary; halo must be zeros
    if rank > 0 and shard_start == T_local:
        assert torch.allclose(q_ext[:, :h], torch.zeros_like(q_ext[:, :h]))
        assert torch.allclose(k_ext[:, :h], torch.zeros_like(k_ext[:, :h]))
        assert torch.allclose(v_ext[:, :h], torch.zeros_like(v_ext[:, :h]))

    if dist.is_initialized():
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description='Unit tests for CP halo utility')
    parser.add_argument('--cp_size', type=int, default=2)
    args = parser.parse_args()

    # Avoid accidental NCCL on CPU-only runs; use gloo
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29577')

    test_basic_halo(args.cp_size)
    test_varlen_zero_halo(args.cp_size)

    if dist.is_initialized():
        dist.destroy_process_group()

    if int(os.environ.get('RANK', '0')) == 0:
        print('✓ CP halo unit tests passed')


if __name__ == '__main__':
    main()
