"""
Test for Context Parallel (CP) Causal Convolution 1D

Context Parallel Principle for Causal Conv1d:
=============================================

Causal convolution has a dependency on previous tokens due to the sliding window.
In a standard implementation, each rank processes the full sequence sequentially.

With Context Parallel:
1. Sequence Partitioning: The input sequence is split across ranks along the sequence dimension.
   - Rank 0: tokens [0, T/N)
   - Rank 1: tokens [T/N, 2T/N)
   - Rank 2: tokens [2T/N, 3T/N)
   - ...

2. Forward Pass:
   - Each rank processes its local chunk independently
   - Non-first ranks need the last (W-1) tokens from the previous rank as initial_state
   - Communication: Previous rank sends its tail tokens (last W-1 tokens) to current rank
   - Current rank receives and constructs initial_state from previous rank's tail
   - This allows parallel computation while maintaining causal dependencies

3. Backward Pass:
   - Gradients need to be corrected because tokens used as initial_state by next rank
     also contribute to gradients
   - Communication: Current rank sends d_initial_state to previous rank
   - Previous rank adds received gradients to its tail tokens (last W-1 tokens)
   - This ensures gradient correctness across rank boundaries

Key Insight:
- The last (W-1) tokens of each rank are used as initial_state by the next rank
- These tokens need gradient contributions from both local computation and next rank
- Communication overhead is minimal: only (W-1) tokens per rank boundary

Test Scenarios:
===============
1. CP2 with sequence cut in the middle (sequences span across rank boundary)
2. CP2 with sequence boundary aligned (no sequence is cut)
3. CP4 with one sequence spanning 3 ranks, another sequence also cut
4. Single long sequence spanning all ranks
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fla.modules.convolution import causal_conv1d
from fla.ops.cp import build_cp_context


def init_distributed(rank, world_size):
    """Initialize distributed environment for a single process."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def assert_close(name, ref, tri, atol=1e-3, rtol=1e-3):
    """Helper function for assertion with detailed diff output."""
    diff = (ref - tri).abs().max().item()
    print(f"  Checking {name:<10} | Max Diff: {diff:.6f}")
    assert diff < atol, f"{name} mismatch. Max diff: {diff}"


def run_cp_conv_test_worker(
    rank: int,
    world_size: int,
    test_name: str,
    T: int,
    D: int,
    W: int,
    lengths: list[int],
    dtype,
):
    """
    Worker function for CP convolution test.
    Runs in a spawned process with the given rank.
    """
    try:
        init_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')

        assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
        assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Test: {test_name}")
            print(f"Config: T={T}, D={D}, W={W}, world_size={world_size}")
            print(f"Sequence lengths: {lengths}")
            print(f"{'='*60}")

        # Step 1: Prepare Global Data
        torch.manual_seed(42)
        B = 1

        x_global = torch.randn(B, T, D, device=device, dtype=dtype)
        dy_global = torch.randn(B, T, D, device=device, dtype=dtype)

        weight = torch.randn(D, W, device=device, dtype=dtype)
        bias = torch.randn(D, device=device, dtype=dtype)

        dist.broadcast(weight, src=0)
        dist.broadcast(bias, src=0)

        cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
        cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.int32)

        activation = 'swish'

        # Step 2: Reference Run
        ref_out, ref_dx, ref_dw, ref_db = None, None, None, None

        if rank == 0:
            x_ref = x_global.clone().detach().requires_grad_(True)
            weight_ref = weight.clone().detach().requires_grad_(True)
            bias_ref = bias.clone().detach().requires_grad_(True)

            y_ref, _ = causal_conv1d(
                x=x_ref,
                weight=weight_ref,
                bias=bias_ref,
                activation=activation,
                backend='triton',
                cu_seqlens=cu_seqlens_global,
            )

            y_ref.backward(dy_global)

            ref_out = y_ref.detach()
            ref_dx = x_ref.grad.detach()
            ref_dw = weight_ref.grad.detach()
            ref_db = bias_ref.grad.detach()

        # Step 3: Context Parallel Run
        dist.barrier()

        context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD, kernel_size=W)

        chunk_size = T // world_size
        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size

        x_local = x_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        dy_local = dy_global[:, start_idx:end_idx, :].clone()
        weight_local = weight.clone().detach().requires_grad_(True)
        bias_local = bias.clone().detach().requires_grad_(True)

        print(f"[Rank {rank}] chunk: [{start_idx}, {end_idx}), "
              f"cu_seqlens: {context.cu_seqlens.tolist()}, "
              f"pre_num_ranks: {context.pre_num_ranks}, "
              f"pre_num_conv_tokens: {context.pre_num_conv_tokens}")
        dist.barrier()

        # CP Forward
        y_local, _ = causal_conv1d(
            x=x_local,
            weight=weight_local,
            bias=bias_local,
            activation=activation,
            cu_seqlens=context.cu_seqlens,
            cp_context=context,
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

        dw_cp = weight_local.grad.clone()
        db_cp = bias_local.grad.clone()
        dist.all_reduce(dw_cp, op=dist.ReduceOp.SUM)
        dist.all_reduce(db_cp, op=dist.ReduceOp.SUM)

        test_passed = True
        if rank == 0:
            print(f"\n[{test_name}] Verification Results:")
            try:
                assert_close("Output", ref_out, y_cp_global, atol=1e-3)
                assert_close("dx", ref_dx, dx_cp_global, atol=1e-3)
                assert_close("dw", ref_dw, dw_cp, atol=1e-3)
                assert_close("db", ref_db, db_cp, atol=1e-3)
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
    W: int,
    lengths: list[int],
    dtype=torch.float32,
):
    """
    Run CP test using torch.multiprocessing.spawn.
    This allows running the test directly with pytest.
    """
    # Use start_processes with spawn to avoid fork/spawn conflicts
    mp.start_processes(
        run_cp_conv_test_worker,
        args=(world_size, test_name, T, D, W, lengths, dtype),
        nprocs=world_size,
        join=True,
        start_method='spawn',
    )


# ============================================================
# Test Scenario Definitions
# ============================================================

def test_cp2_sequence_cut():
    """
    Test Case 1: CP2 with sequences cut in the middle.

    Scenario:
    - world_size=2, T=1024, chunk_size=512
    - lengths=[300, 400, 324] -> sequences span across rank boundary
    - Rank 0: tokens [0, 512) contains seq0 (300) + part of seq1 (212)
    - Rank 1: tokens [512, 1024) contains rest of seq1 (188) + seq2 (324)
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_SequenceCut",
        T=1024,
        D=128,
        W=4,
        lengths=[300, 400, 324],
        dtype=torch.float32,
    )


def test_cp2_boundary_aligned():
    """
    Test Case 2: CP2 with sequence boundaries aligned with rank boundaries.

    Scenario:
    - world_size=2, T=1024, chunk_size=512
    - lengths=[512, 512] -> sequence boundary exactly at rank boundary
    - Rank 0: tokens [0, 512) contains exactly seq0
    - Rank 1: tokens [512, 1024) contains exactly seq1
    - No sequence is split across ranks
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_BoundaryAligned",
        T=1024,
        D=128,
        W=4,
        lengths=[512, 512],
        dtype=torch.float32,
    )


def test_cp4_complex():
    """
    Test Case 3: CP4 with complex sequence distribution.

    Scenario:
    - world_size=4, T=1024, chunk_size=256
    - lengths=[700, 324] -> first sequence spans 3 ranks
    - Rank 0: [0, 256) all seq0
    - Rank 1: [256, 512) all seq0
    - Rank 2: [512, 768) - 188 tokens of seq0 + 68 tokens of seq1
    - Rank 3: [768, 1024) - 256 tokens of seq1
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_Complex",
        T=1024,
        D=128,
        W=4,
        lengths=[700, 324],
        dtype=torch.float32,
    )


def test_cp4_single_sequence():
    """
    Test Case 4: CP4 with a single long sequence spanning all ranks.

    Scenario:
    - world_size=4, T=1024, chunk_size=256
    - lengths=[1024] -> single sequence spans all 4 ranks
    - Each rank processes 256 tokens of the same sequence
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_SingleSequence",
        T=1024,
        D=128,
        W=4,
        lengths=[1024],
        dtype=torch.float32,
    )


def test_cp2_many_short_sequences():
    """
    Test Case 5: CP2 with many short sequences.

    Scenario:
    - world_size=2, T=1024, chunk_size=512
    - lengths=[100, 150, 200, 250, 124, 100, 100] -> many short sequences
    - Some sequences are entirely in one rank, some span across
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_ManyShortSequences",
        T=1024,
        D=128,
        W=4,
        lengths=[100, 150, 200, 250, 124, 100, 100],
        dtype=torch.float32,
    )


# ============================================================
# Main Entry Point (for torchrun)
# ============================================================

def setup_distributed_torchrun():
    """Initialize distributed environment for torchrun."""
    if 'RANK' not in os.environ:
        return False

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True


if __name__ == "__main__":
    # Check if running with torchrun
    if 'RANK' in os.environ:
        # Running with torchrun
        if setup_distributed_torchrun():
            from fla.utils import device as fla_device

            world_size = dist.get_world_size()
            rank = dist.get_rank()

            try:
                if rank == 0:
                    print("=" * 60)
                    print("Running CP Conv1d Tests (torchrun mode)")
                    print("=" * 60)

                # Define test configs based on world_size
                if world_size == 2:
                    test_configs = [
                        ("CP2_SequenceCut", 1024, 128, 4, [300, 400, 324]),
                        ("CP2_BoundaryAligned", 1024, 128, 4, [512, 512]),
                        ("CP2_ManyShortSequences", 1024, 128, 4, [100, 150, 200, 250, 124, 100, 100]),
                    ]
                elif world_size == 4:
                    test_configs = [
                        ("CP4_Complex", 1024, 128, 4, [700, 324]),
                        ("CP4_SingleSequence", 1024, 128, 4, [1024]),
                    ]
                else:
                    test_configs = [
                        (f"CP{world_size}_SingleSequence", 1024, 128, 4, [1024]),
                    ]

                for test_name, T, D, W, lengths in test_configs:
                    # Run test inline (already in distributed context)
                    torch.manual_seed(42)
                    B = 1

                    x_global = torch.randn(B, T, D, device=fla_device, dtype=torch.float32)
                    dy_global = torch.randn(B, T, D, device=fla_device, dtype=torch.float32)
                    weight = torch.randn(D, W, device=fla_device, dtype=torch.float32)
                    bias = torch.randn(D, device=fla_device, dtype=torch.float32)

                    dist.broadcast(weight, src=0)
                    dist.broadcast(bias, src=0)

                    cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
                    cu_seqlens_global = torch.tensor(cu_seqlens_list, device=fla_device, dtype=torch.int32)

                    # Reference
                    ref_out, ref_dx, ref_dw, ref_db = None, None, None, None
                    if rank == 0:
                        x_ref = x_global.clone().detach().requires_grad_(True)
                        weight_ref = weight.clone().detach().requires_grad_(True)
                        bias_ref = bias.clone().detach().requires_grad_(True)
                        y_ref, _ = causal_conv1d(x=x_ref, weight=weight_ref, bias=bias_ref,
                                                 activation='swish', backend='triton', cu_seqlens=cu_seqlens_global)
                        y_ref.backward(dy_global)
                        ref_out, ref_dx = y_ref.detach(), x_ref.grad.detach()
                        ref_dw, ref_db = weight_ref.grad.detach(), bias_ref.grad.detach()

                    dist.barrier()
                    context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD, kernel_size=W)

                    chunk_size = T // world_size
                    start_idx, end_idx = rank * chunk_size, (rank + 1) * chunk_size
                    x_local = x_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
                    dy_local = dy_global[:, start_idx:end_idx, :].clone()
                    weight_local = weight.clone().detach().requires_grad_(True)
                    bias_local = bias.clone().detach().requires_grad_(True)

                    y_local, _ = causal_conv1d(x=x_local, weight=weight_local, bias=bias_local,
                                               activation='swish', cu_seqlens=context.cu_seqlens, cp_context=context)
                    y_local.backward(dy_local)

                    y_gathered = [torch.zeros_like(y_local) for _ in range(world_size)]
                    dist.all_gather(y_gathered, y_local)
                    y_cp_global = torch.cat(y_gathered, dim=1)

                    dx_gathered = [torch.zeros_like(x_local.grad) for _ in range(world_size)]
                    dist.all_gather(dx_gathered, x_local.grad)
                    dx_cp_global = torch.cat(dx_gathered, dim=1)

                    dw_cp, db_cp = weight_local.grad.clone(), bias_local.grad.clone()
                    dist.all_reduce(dw_cp, op=dist.ReduceOp.SUM)
                    dist.all_reduce(db_cp, op=dist.ReduceOp.SUM)

                    if rank == 0:
                        print(f"\n[{test_name}] Verification:")
                        assert_close("Output", ref_out, y_cp_global, atol=1e-3)
                        assert_close("dx", ref_dx, dx_cp_global, atol=1e-3)
                        assert_close("dw", ref_dw, dw_cp, atol=1e-3)
                        assert_close("db", ref_db, db_cp, atol=1e-3)
                        print(f"✅ [{test_name}] Passed!")

                    dist.barrier()

                if rank == 0:
                    print("\n" + "=" * 60)
                    print("All tests passed!")
                    print("=" * 60)

            finally:
                cleanup_distributed()
    else:
        # Not running with torchrun, show usage
        print("Run tests with pytest or torchrun:")
        print("  pytest tests/context_parallel/test_cp_conv.py -v")
        print("  torchrun --nproc_per_node=2 tests/context_parallel/test_cp_conv.py")
        print("  torchrun --nproc_per_node=4 tests/context_parallel/test_cp_conv.py")
