# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Test for Context Parallel (CP) Token Shift

Context Parallel Principle for Token Shift:
===========================================

Token shift has a 1-token dependency on previous tokens:
    y[t] = x[t-1] - x[t]  (for t > 0)
    y[0] = cache - x[0]   (cache is the last token from previous rank)

With Context Parallel:
1. Sequence Partitioning: input sequence split across ranks
2. Forward: non-first ranks need the last token from previous rank as cache
3. Backward: non-last ranks send the last token's gradient to previous rank

Test Scenarios:
===============
1. CP2 with sequence cut in the middle
2. CP2 with sequence boundary aligned
3. CP4 with complex sequence distribution
4. Single long sequence spanning all ranks
5. Many short sequences
6. Edge case: short local segments (T_local < 2 for token shift)
"""

import logging
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fla.modules.token_shift import token_shift_ref
from fla.modules.token_shift_cp import token_shift_cp
from fla.ops.cp import build_cp_context
from fla.utils import assert_close

logging.basicConfig(level=logging.INFO, format='%(message)s')


def init_distributed(rank, world_size, port):
    """Initialize distributed environment for a single process."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_cp_token_shift_test_worker(
    rank: int,
    world_size: int,
    test_name: str,
    T: int,
    D: int,
    lengths: list[int],
    dtype,
    port: int,
):
    """Worker function for CP token shift test."""
    try:
        init_distributed(rank, world_size, port)
        device = torch.device(f'cuda:{rank}')

        assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
        assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Test: {test_name}")
            print(f"Config: T={T}, D={D}, world_size={world_size}")
            print(f"Sequence lengths: {lengths}")
            print(f"{'='*60}")

        # Step 1: Prepare Global Data
        torch.manual_seed(42)
        B = 1

        x_global = torch.randn(B, T, D, device=device, dtype=dtype) * 10
        dy_global = torch.randn(B, T, D, device=device, dtype=dtype)

        cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
        cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.int32)

        # Step 2: Reference Run
        ref_out, ref_dx = None, None

        if rank == 0:
            x_ref = x_global.clone().detach().requires_grad_(True)
            y_ref = token_shift_ref(x_ref, cu_seqlens=cu_seqlens_global)
            y_ref.backward(dy_global)
            ref_out = y_ref.detach()
            ref_dx = x_ref.grad.detach()

        # Step 3: Context Parallel Run
        dist.barrier()

        context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)

        chunk_size = T // world_size
        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size

        x_local = x_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        dy_local = dy_global[:, start_idx:end_idx, :].clone()

        print(f"[Rank {rank}] chunk: [{start_idx}, {end_idx}), "
              f"cu_seqlens: {context.cu_seqlens.tolist()}, "
              f"pre_num_ranks: {context.pre_num_ranks}")
        dist.barrier()

        # CP Forward
        y_local = token_shift_cp(
            x=x_local,
            cp_context=context,
            cu_seqlens=context.cu_seqlens,
        )

        # CP Backward
        y_local.backward(dy_local)

        # Step 4: Result Aggregation and Verification
        y_gathered = [torch.zeros_like(y_local) for _ in range(world_size)]
        dist.all_gather(y_gathered, y_local)
        y_cp_global = torch.cat(y_gathered, dim=1)

        dx_gathered = [torch.zeros_like(x_local.grad) for _ in range(world_size)]
        dist.all_gather(dx_gathered, x_local.grad)
        dx_cp_global = torch.cat(dx_gathered, dim=1)

        test_passed = True
        if rank == 0:
            print(f"\n[{test_name}] Verification Results:")
            try:
                assert_close("Output", ref_out, y_cp_global, ratio=0.001)
                assert_close("dx", ref_dx, dx_cp_global, ratio=0.001)
                print(f"✅ [{test_name}] Test Passed!\n")
            except AssertionError as e:
                print(f"❌ [{test_name}] Test Failed: {e}\n")
                test_passed = False

        dist.barrier()
        cleanup_distributed()

        if not test_passed:
            raise AssertionError(f"Test {test_name} failed on rank {rank}")

    except Exception as e:
        cleanup_distributed()
        raise e


def run_cp_test_with_spawn(
    world_size: int,
    test_name: str,
    T: int,
    D: int,
    lengths: list[int],
    dtype=torch.float32,
):
    """Run CP test using torch.multiprocessing.spawn."""
    port = 29510  # Different port from other CP tests
    mp.start_processes(
        run_cp_token_shift_test_worker,
        args=(world_size, test_name, T, D, lengths, dtype, port),
        nprocs=world_size,
        join=True,
        start_method='spawn',
    )


# ============================================================
# Test Scenario Definitions
# ============================================================

def test_cp2_sequence_cut():
    """CP2: sequences cut across rank boundary."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_SequenceCut",
        T=1024, D=128,
        lengths=[300, 400, 324],
        dtype=torch.float32,
    )


def test_cp2_boundary_aligned():
    """CP2: sequence boundaries aligned with rank boundaries."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_BoundaryAligned",
        T=1024, D=128,
        lengths=[512, 512],
        dtype=torch.float32,
    )


def test_cp4_complex():
    """CP4: complex sequence distribution."""
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_Complex",
        T=1024, D=128,
        lengths=[700, 324],
        dtype=torch.float32,
    )


def test_cp4_single_sequence():
    """CP4: single long sequence spanning all ranks."""
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_SingleSequence",
        T=1024, D=128,
        lengths=[1024],
        dtype=torch.float32,
    )


def test_cp2_many_short_sequences():
    """CP2: many short sequences."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_ManyShortSequences",
        T=1024, D=128,
        lengths=[100, 150, 200, 250, 124, 100, 100],
        dtype=torch.float32,
    )


# Edge case: short local segments (T_local = 1 for token shift)
def test_cp2_short_tail_len1():
    """CP2: rank 1 gets a length-1 tail."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_ShortTail_Len1",
        T=1024, D=128,
        lengths=[513, 511],
        dtype=torch.float32,
    )


def test_cp2_short_tail_len2():
    """CP2: rank 1 gets a length-2 tail."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_ShortTail_Len2",
        T=1024, D=128,
        lengths=[514, 510],
        dtype=torch.float32,
    )
