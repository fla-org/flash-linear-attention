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

import logging
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fla.modules.convolution import causal_conv1d
from fla.ops.cp import build_cp_context
from fla.utils import assert_close

# Configure logging to see assert_close messages
logging.basicConfig(level=logging.INFO, format='%(message)s')


def init_distributed(rank, world_size):
    """Initialize distributed environment for a single process."""
    # Configure logging in worker process
    logging.basicConfig(level=logging.INFO, format='%(message)s')

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

        x_global = torch.randn(B, T, D, device=device, dtype=dtype) * 100
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

        context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD, conv1d_kernel_size=W)

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
                assert_close("Output", ref_out, y_cp_global, ratio=0.001)
                assert_close("dx", ref_dx, dx_cp_global, ratio=0.001)
                assert_close("dw", ref_dw, dw_cp, ratio=0.001)
                assert_close("db", ref_db, db_cp, ratio=0.001)
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
# Extreme Edge Cases: Short Local Sequences (T_local < W)
#
# These tests target a specific bug in causal_conv1d_bwd_kernel:
# when CP splits a sequence so that a rank gets a very short local
# segment (T_local < W), the backward kernel's mask_head_rows was
# missing the `& (o_t < T)` bound check, causing out-of-bounds
# reads from dy and producing NaN in dw.
# ============================================================

@pytest.mark.parametrize("backend", ["triton"])
def test_cp2_short_tail_len1(backend):
    """
    CP2: seq0 has 513 tokens, so rank 1 gets a length-1 tail (T=1 < W=4).

    Rank 0: [0, 512)  → 512 tokens of seq0
    Rank 1: [512, 1024) → 1 token of seq0 (T_local=1!) + 511 tokens of seq1
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name=f"CP2_ShortTail_Len1_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[513, 511],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("backend", ["triton"])
def test_cp2_short_tail_len2(backend):
    """
    CP2: seq0 has 514 tokens, so rank 1 gets a length-2 tail (T=2 < W=4).

    Rank 1: [512, 1024) → 2 tokens of seq0 (T_local=2!) + 510 tokens of seq1
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name=f"CP2_ShortTail_Len2_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[514, 510],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("backend", ["triton"])
def test_cp4_every_rank_gets_short_tail(backend):
    """
    CP4: every non-first rank gets a length-1 local sequence tail.

    lengths=[257, 255, 257, 255], T=1024, chunk_size=256
    Rank 0: [0, 256)   → 256 tokens of seq0
    Rank 1: [256, 512)  → 1 token of seq0 (T=1!) + 255 tokens of seq1
    Rank 2: [512, 768)  → 256 tokens of seq1(rest) + start of seq2
                          actually seq1=[257,512) so all 255 on rank1, then
                          seq2=[512,769) → rank2 gets 256, rank3 gets 1
    Rank 3: [768, 1024) → 1 token of seq2 (T=1!) + 255 tokens of seq3
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name=f"CP4_EveryRankShortTail_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[257, 255, 257, 255],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("backend", ["triton"])
def test_cp2_multiple_short_tails(backend):
    """
    CP2: multiple sequences each end 1 token into rank 1.

    lengths=[200, 313, 511] → seq0 ends at 200 (rank 0), seq1 ends at 513 (rank 1 gets 1 token)
    Rank 0: [0, 512)   → seq0(200) + 312 tokens of seq1
    Rank 1: [512, 1024) → 1 token of seq1 (T=1!) + 511 tokens of seq2
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name=f"CP2_MultipleShortTails_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[200, 313, 511],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("backend", ["triton"])
def test_cp2_global_len1_sequence(backend):
    """
    CP2: a globally length-1 sequence sits right at the rank boundary.

    lengths=[512, 1, 511] → seq1 is globally length 1, entirely on rank 1
    Rank 0: [0, 512)   → seq0(512)
    Rank 1: [512, 1024) → seq1(1, T=1!) + seq2(511)
    seq1 has no initial_state from prev rank (it starts fresh), but the
    initial_state tensor is still allocated for all seqs on non-first ranks.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name=f"CP2_GlobalLen1Sequence_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[512, 1, 511],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("backend", ["triton"])
def test_cp4_multiple_short_tails(backend):
    """
    CP4: multiple sequences each end 1-2 tokens past a rank boundary,
    creating short local segments on multiple ranks.

    lengths=[257, 253, 257, 257], T=1024, chunk_size=256
    Rank 0: [0, 256)   → seq0[0:256]  (256 tokens)
    Rank 1: [256, 512)  → seq0 tail (1 tok, T=1!) + seq1 (253 tok) + seq2 start (2 tok, T=2!)
    Rank 2: [512, 768)  → seq2 cont (255 tok) + seq3 start (1 tok, T=1!)
    Rank 3: [768, 1024) → seq3 cont (256 tok)

    Ranks 1 and 2 each have local sequences with T < W=4.
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name=f"CP4_MultipleShortTails_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[257, 253, 257, 257],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("backend", ["triton"])
def test_cp4_worst_case_many_len1(backend):
    """
    CP4: globally length-1 sequences + short tails across multiple ranks.
    Stress test for cross-sequence gradient isolation in CP backward.

    lengths=[257, 1, 253, 257, 1, 253, 2], T=1024, chunk_size=256
    Rank 1: cu_seqlens=[0,1,2,255,256] → 4 local seqs, lengths [1,1,253,1]
    Rank 2: cu_seqlens=[0,256], pre_num_conv_tokens=1 → gradient must be
            masked to only 1 valid position, not all W-1=3.
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name=f"CP4_WorstCaseManyLen1_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[257, 1, 253, 257, 1, 253, 2],
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
