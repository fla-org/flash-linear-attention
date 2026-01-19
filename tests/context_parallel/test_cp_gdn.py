"""
Test for Context Parallel (CP) Gated Delta Rule (GDN)

Context Parallel Principle for GDN:
===================================

GDN has a recurrent state dependency across tokens. The hidden state h evolves
as tokens are processed, creating dependencies that span the sequence.

With Context Parallel:
1. Sequence Partitioning: The input sequence is split across ranks along the sequence dimension.
   - Rank 0: tokens [0, T/N)
   - Rank 1: tokens [T/N, 2T/N)
   - ...

2. Forward Pass:
   - Each rank computes its local chunk
   - Non-first ranks need the final state from previous rank as initial_state
   - Communication: All-reduce style state passing between ranks

3. Backward Pass:
   - Gradients flow back through the recurrent state
   - Communication: Gradient synchronization across ranks

Test Scenarios:
===============
1. CP2 with sequence cut in the middle
2. CP2 with sequence boundary aligned
3. CP4 with complex sequence distribution
4. CP4 with single long sequence
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def init_distributed(rank, world_size):
    """Initialize distributed environment for a single process."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29502'  # Different port from other tests
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


def run_cp_gdn_test_worker(
    rank: int,
    world_size: int,
    test_name: str,
    T: int,
    H: int,
    D: int,
    lengths: list[int],
    dtype,
):
    """
    Worker function for CP GDN test.
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
            print(f"Config: T={T}, H={H}, D={D}, world_size={world_size}")
            print(f"Sequence lengths: {lengths}")
            print(f"{'='*60}")

        # Step 1: Prepare Global Data
        torch.manual_seed(42)
        B = 1

        # Generate inputs - note: g is [B, T, H] for gated_delta_rule, not [B, T, H, D]
        q_global = torch.randn(B, T, H, D, device=device, dtype=dtype)
        k_global = F.normalize(torch.randn(B, T, H, D, device=device, dtype=torch.float32), p=2, dim=-1).to(dtype)
        v_global = torch.randn(B, T, H, D, device=device, dtype=dtype)
        g_global = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
        beta_global = torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid()

        # Broadcast to ensure all ranks have same data
        dist.broadcast(q_global, src=0)
        dist.broadcast(k_global, src=0)
        dist.broadcast(v_global, src=0)
        dist.broadcast(g_global, src=0)
        dist.broadcast(beta_global, src=0)

        # Prepare cu_seqlens
        cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
        cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.long)

        # Prepare gradients
        do_global = torch.randn(B, T, H, D, device=device, dtype=dtype)
        dist.broadcast(do_global, src=0)

        # Step 2: Reference Run (single GPU, varlen)
        ref_out = None
        ref_dq, ref_dk, ref_dv, ref_dg, ref_db = None, None, None, None, None

        if rank == 0:
            q_ref = q_global.clone().detach().requires_grad_(True)
            k_ref = k_global.clone().detach().requires_grad_(True)
            v_ref = v_global.clone().detach().requires_grad_(True)
            g_ref = g_global.clone().detach().requires_grad_(True)
            beta_ref = beta_global.clone().detach().requires_grad_(True)

            # Use chunk_gated_delta_rule with varlen (no CP)
            o_ref, _ = chunk_gated_delta_rule(
                q=q_ref,
                k=k_ref,
                v=v_ref,
                g=g_ref,
                beta=beta_ref,
                cu_seqlens=cu_seqlens_global,
            )

            o_ref.backward(do_global)

            ref_out = o_ref.detach()
            ref_dq = q_ref.grad.detach()
            ref_dk = k_ref.grad.detach()
            ref_dv = v_ref.grad.detach()
            ref_dg = g_ref.grad.detach()
            ref_db = beta_ref.grad.detach()

        # Step 3: Context Parallel Run
        dist.barrier()

        # Build CP context
        context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)

        chunk_size = T // world_size
        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size

        # Get local slices - note: g is [B, T, H]
        q_local = q_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        k_local = k_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        v_local = v_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        g_local = g_global[:, start_idx:end_idx].clone().detach().requires_grad_(True)
        beta_local = beta_global[:, start_idx:end_idx].clone().detach().requires_grad_(True)
        do_local = do_global[:, start_idx:end_idx, :].clone()

        print(f"[Rank {rank}] chunk: [{start_idx}, {end_idx}), "
              f"cu_seqlens: {context.cu_seqlens.tolist()}, "
              f"pre_num_ranks: {context.pre_num_ranks}")
        dist.barrier()

        # CP Forward
        o_local, _ = chunk_gated_delta_rule(
            q=q_local,
            k=k_local,
            v=v_local,
            g=g_local,
            beta=beta_local,
            cu_seqlens=context.cu_seqlens,
            cp_context=context,
        )

        # CP Backward
        o_local.backward(do_local)

        # Step 4: Result Aggregation and Verification
        o_gathered = [torch.zeros_like(o_local) for _ in range(world_size)]
        dist.all_gather(o_gathered, o_local)
        o_cp_global = torch.cat(o_gathered, dim=1)

        dq_gathered = [torch.zeros_like(q_local.grad) for _ in range(world_size)]
        dist.all_gather(dq_gathered, q_local.grad)
        dq_cp_global = torch.cat(dq_gathered, dim=1)

        dk_gathered = [torch.zeros_like(k_local.grad) for _ in range(world_size)]
        dist.all_gather(dk_gathered, k_local.grad)
        dk_cp_global = torch.cat(dk_gathered, dim=1)

        dv_gathered = [torch.zeros_like(v_local.grad) for _ in range(world_size)]
        dist.all_gather(dv_gathered, v_local.grad)
        dv_cp_global = torch.cat(dv_gathered, dim=1)

        dg_gathered = [torch.zeros_like(g_local.grad) for _ in range(world_size)]
        dist.all_gather(dg_gathered, g_local.grad)
        dg_cp_global = torch.cat(dg_gathered, dim=1)

        db_gathered = [torch.zeros_like(beta_local.grad) for _ in range(world_size)]
        dist.all_gather(db_gathered, beta_local.grad)
        db_cp_global = torch.cat(db_gathered, dim=1)

        test_passed = True
        if rank == 0:
            print(f"\n[{test_name}] Verification Results:")
            try:
                assert_close("Output", ref_out, o_cp_global, atol=5e-3)
                assert_close("dq", ref_dq, dq_cp_global, atol=8e-3)
                assert_close("dk", ref_dk, dk_cp_global, atol=8e-3)
                assert_close("dv", ref_dv, dv_cp_global, atol=8e-3)
                assert_close("dg", ref_dg, dg_cp_global, atol=2e-2)
                assert_close("db", ref_db, db_cp_global, atol=2e-2)
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
    H: int,
    D: int,
    lengths: list[int],
    dtype=torch.float16,
):
    """
    Run CP test using torch.multiprocessing.spawn.
    This allows running the test directly with pytest.
    """
    mp.start_processes(
        run_cp_gdn_test_worker,
        args=(world_size, test_name, T, H, D, lengths, dtype),
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
        H=4,
        D=64,
        lengths=[300, 400, 324],
        dtype=torch.float16,
    )


def test_cp2_boundary_aligned():
    """
    Test Case 2: CP2 with sequence boundaries aligned with rank boundaries.

    Scenario:
    - world_size=2, T=1024, chunk_size=512
    - lengths=[512, 512] -> sequence boundary exactly at rank boundary
    - Rank 0: tokens [0, 512) contains exactly seq0
    - Rank 1: tokens [512, 1024) contains exactly seq1
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_BoundaryAligned",
        T=1024,
        H=4,
        D=64,
        lengths=[512, 512],
        dtype=torch.float16,
    )


def test_cp4_complex():
    """
    Test Case 3: CP4 with complex sequence distribution.

    Scenario:
    - world_size=4, T=1024, chunk_size=256
    - lengths=[700, 324] -> first sequence spans 3 ranks
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_Complex",
        T=1024,
        H=4,
        D=64,
        lengths=[700, 324],
        dtype=torch.float16,
    )


def test_cp4_single_sequence():
    """
    Test Case 4: CP4 with a single long sequence spanning all ranks.

    Scenario:
    - world_size=4, T=1024, chunk_size=256
    - lengths=[1024] -> single sequence spans all 4 ranks
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_SingleSequence",
        T=1024,
        H=4,
        D=64,
        lengths=[1024],
        dtype=torch.float16,
    )


def test_cp2_many_short_sequences():
    """
    Test Case 5: CP2 with many short sequences.

    Scenario:
    - world_size=2, T=1024, chunk_size=512
    - lengths=[100, 150, 200, 250, 124, 100, 100] -> many short sequences
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_ManyShortSequences",
        T=1024,
        H=4,
        D=64,
        lengths=[100, 150, 200, 250, 124, 100, 100],
        dtype=torch.float16,
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
                    print("Running CP GDN Tests (torchrun mode)")
                    print("=" * 60)

                # Define test configs based on world_size
                if world_size == 2:
                    test_configs = [
                        ("CP2_SequenceCut", 1024, 4, 64, [300, 400, 324]),
                        ("CP2_BoundaryAligned", 1024, 4, 64, [512, 512]),
                        ("CP2_ManyShortSequences", 1024, 4, 64, [100, 150, 200, 250, 124, 100, 100]),
                    ]
                elif world_size == 4:
                    test_configs = [
                        ("CP4_Complex", 1024, 4, 64, [700, 324]),
                        ("CP4_SingleSequence", 1024, 4, 64, [1024]),
                    ]
                else:
                    test_configs = [
                        (f"CP{world_size}_SingleSequence", 1024, 4, 64, [1024]),
                    ]

                for test_name, T, H, D, lengths in test_configs:
                    torch.manual_seed(42)
                    B = 1

                    # Generate global data - note: g is [B, T, H] for gated_delta_rule
                    q_global = torch.randn(B, T, H, D, device=fla_device, dtype=torch.float16)
                    k_global = F.normalize(torch.randn(B, T, H, D, device=fla_device,
                                           dtype=torch.float32), p=2, dim=-1).to(torch.float16)
                    v_global = torch.randn(B, T, H, D, device=fla_device, dtype=torch.float16)
                    g_global = F.logsigmoid(torch.randn(B, T, H, device=fla_device, dtype=torch.float16))
                    beta_global = torch.randn(B, T, H, device=fla_device, dtype=torch.float32).sigmoid()
                    do_global = torch.randn(B, T, H, D, device=fla_device, dtype=torch.float16)

                    # Broadcast
                    dist.broadcast(q_global, src=0)
                    dist.broadcast(k_global, src=0)
                    dist.broadcast(v_global, src=0)
                    dist.broadcast(g_global, src=0)
                    dist.broadcast(beta_global, src=0)
                    dist.broadcast(do_global, src=0)

                    cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
                    cu_seqlens_global = torch.tensor(cu_seqlens_list, device=fla_device, dtype=torch.long)

                    # Reference
                    ref_out, ref_dq, ref_dk, ref_dv, ref_dg, ref_db = None, None, None, None, None, None
                    if rank == 0:
                        q_ref = q_global.clone().detach().requires_grad_(True)
                        k_ref = k_global.clone().detach().requires_grad_(True)
                        v_ref = v_global.clone().detach().requires_grad_(True)
                        g_ref = g_global.clone().detach().requires_grad_(True)
                        beta_ref = beta_global.clone().detach().requires_grad_(True)

                        o_ref, _ = chunk_gated_delta_rule(
                            q=q_ref,
                            k=k_ref,
                            v=v_ref,
                            g=g_ref,
                            beta=beta_ref,
                            cu_seqlens=cu_seqlens_global,
                        )
                        o_ref.backward(do_global)
                        ref_out = o_ref.detach()
                        ref_dq = q_ref.grad.detach()
                        ref_dk = k_ref.grad.detach()
                        ref_dv = v_ref.grad.detach()
                        ref_dg = g_ref.grad.detach()
                        ref_db = beta_ref.grad.detach()

                    dist.barrier()

                    # CP run
                    context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)

                    chunk_size = T // world_size
                    start_idx, end_idx = rank * chunk_size, (rank + 1) * chunk_size

                    # note: g is [B, T, H]
                    q_local = q_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
                    k_local = k_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
                    v_local = v_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
                    g_local = g_global[:, start_idx:end_idx].clone().detach().requires_grad_(True)
                    beta_local = beta_global[:, start_idx:end_idx].clone().detach().requires_grad_(True)
                    do_local = do_global[:, start_idx:end_idx, :].clone()

                    o_local, _ = chunk_gated_delta_rule(
                        q=q_local,
                        k=k_local,
                        v=v_local,
                        g=g_local,
                        beta=beta_local,
                        cu_seqlens=context.cu_seqlens,
                        cp_context=context,
                    )
                    o_local.backward(do_local)

                    # Gather output and gradients
                    o_gathered = [torch.zeros_like(o_local) for _ in range(world_size)]
                    dist.all_gather(o_gathered, o_local)
                    o_cp_global = torch.cat(o_gathered, dim=1)

                    dq_gathered = [torch.zeros_like(q_local.grad) for _ in range(world_size)]
                    dist.all_gather(dq_gathered, q_local.grad)
                    dq_cp_global = torch.cat(dq_gathered, dim=1)

                    dk_gathered = [torch.zeros_like(k_local.grad) for _ in range(world_size)]
                    dist.all_gather(dk_gathered, k_local.grad)
                    dk_cp_global = torch.cat(dk_gathered, dim=1)

                    dv_gathered = [torch.zeros_like(v_local.grad) for _ in range(world_size)]
                    dist.all_gather(dv_gathered, v_local.grad)
                    dv_cp_global = torch.cat(dv_gathered, dim=1)

                    dg_gathered = [torch.zeros_like(g_local.grad) for _ in range(world_size)]
                    dist.all_gather(dg_gathered, g_local.grad)
                    dg_cp_global = torch.cat(dg_gathered, dim=1)

                    db_gathered = [torch.zeros_like(beta_local.grad) for _ in range(world_size)]
                    dist.all_gather(db_gathered, beta_local.grad)
                    db_cp_global = torch.cat(db_gathered, dim=1)

                    if rank == 0:
                        print(f"\n[{test_name}] Verification:")
                        assert_close("Output", ref_out, o_cp_global, atol=5e-3)
                        assert_close("dq", ref_dq, dq_cp_global, atol=8e-3)
                        assert_close("dk", ref_dk, dk_cp_global, atol=8e-3)
                        assert_close("dv", ref_dv, dv_cp_global, atol=8e-3)
                        assert_close("dg", ref_dg, dg_cp_global, atol=2e-2)
                        assert_close("db", ref_db, db_cp_global, atol=2e-2)
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
        print("  pytest tests/context_parallel/test_cp_gdn.py -v")
        print("  torchrun --nproc_per_node=2 tests/context_parallel/test_cp_gdn.py")
        print("  torchrun --nproc_per_node=4 tests/context_parallel/test_cp_gdn.py")
