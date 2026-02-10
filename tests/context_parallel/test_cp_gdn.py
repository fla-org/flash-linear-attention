"""
Test for Context Parallel (CP) Gated Delta Rule (GDN)

Implementation Hierarchy and Relationships:
==========================================

1. chunk_gated_delta_rule (fla/ops/gated_delta_rule/chunk.py):
   - Production Triton kernel for GDN
   - Input g is per-token log-space decay, shape [B, T, H] (scalar per head, NOT per-dim)
   - Internally does chunk_local_cumsum on g
   - Recurrence (delta rule with gating):
       S_t = S_{t-1} * exp(g_t) + beta_t * k_t (x) (v_t - S_{t-1} @ k_t)
       o_t = q_t^T @ S_t
     where (x) denotes outer product, S is [K, V] state matrix
   - No fused gate activation (unlike KDA), no lowerbound
   - Supports variable-length sequences via cu_seqlens
   - Supports context parallel via cp_context

2. Context Parallel (CP) chunk_gated_delta_rule:
   - Extension for multi-GPU distributed training
   - Sequence is partitioned across ranks, with state communication between ranks
   - Uses build_cp_context() to manage cross-rank dependencies
   - Forward: Non-first ranks receive initial_state from previous rank
   - Backward: Gradient dht flows back across rank boundaries

Test Architecture Notes:
========================
- Reference: single-GPU chunk_gated_delta_rule with cu_seqlens (varlen, no CP)
- CP path: same function with cp_context, sequence split across ranks
- Both should produce identical results

Differences from KDA test:
- No use_gate_in_kernel / safe_gate / lower_bound / A_log / dt_bias
- No L2 normalization (q/k are pre-normalized before input)
- g shape is [B, T, H] not [B, T, H, D]

Context Parallel Principle:
===========================

With Context Parallel:
1. Sequence Partitioning: input sequence split across ranks along sequence dim
   - Rank i: tokens [i*T/N, (i+1)*T/N)

2. Forward: each rank computes local chunk; non-first ranks receive state from prev rank

3. Backward: gradients flow back through recurrent state across ranks

Test Scenarios:
===============
1. CP2 with sequence cut in the middle
2. CP2 with sequence boundary aligned
3. CP4 with complex sequence distribution
4. CP4 with single long sequence
"""

import logging
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.utils import assert_close

# Configure logging to see assert_close messages
logging.basicConfig(level=logging.INFO, format='%(message)s')


def init_distributed(rank, world_size):
    """Initialize distributed environment for a single process."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29502'  # Different port from KDA test
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


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

        # Step 1: Prepare Global Data (all generated on rank 0, broadcast to all)
        B = 1
        q_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
        k_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
        v_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
        g_global = torch.empty(B, T, H, device=device, dtype=dtype)
        beta_global = torch.empty(B, T, H, device=device, dtype=torch.float32)
        do_global = torch.empty(B, T, H, D, device=device, dtype=dtype)

        if rank == 0:
            torch.manual_seed(42)
            q_global.copy_(F.normalize(torch.randn(B, T, H, D, device=device, dtype=torch.float32), p=2, dim=-1).to(dtype))
            k_global.copy_(F.normalize(torch.randn(B, T, H, D, device=device, dtype=torch.float32), p=2, dim=-1).to(dtype))
            v_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
            g_global.copy_(F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype)))
            beta_global.copy_(torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid())
            do_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))

        # Broadcast to ensure all ranks have same data
        dist.broadcast(q_global, src=0)
        dist.broadcast(k_global, src=0)
        dist.broadcast(v_global, src=0)
        dist.broadcast(g_global, src=0)
        dist.broadcast(beta_global, src=0)
        dist.broadcast(do_global, src=0)

        # Prepare cu_seqlens
        cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
        cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.long)

        # Step 2: Reference Run (single GPU, varlen, no CP)
        ref_out = None
        ref_dq, ref_dk, ref_dv, ref_dg, ref_db = None, None, None, None, None

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

        # Step 3: Context Parallel Run
        dist.barrier()

        context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)

        chunk_size = T // world_size
        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size

        # Get local slices - note: g is [B, T, H], beta is [B, T, H]
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
            print(f"\n[{test_name}] Verifying results...")

            tensors_to_verify = [
                ("Output", ref_out, o_cp_global),
                ("dq", ref_dq, dq_cp_global),
                ("dk", ref_dk, dk_cp_global),
                ("dv", ref_dv, dv_cp_global),
                ("dg", ref_dg, dg_cp_global),
                ("db", ref_db, db_cp_global),
            ]

            try:
                for name, ref, cp in tensors_to_verify:
                    assert_close(name, ref, cp, ratio=2e-3, warning=False)
                print(f"[{test_name}] Test Passed!\n")
            except AssertionError as e:
                print(f"[{test_name}] Test Failed: {e}\n")
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
    dtype=torch.bfloat16,
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
    """CP2: sequences cut across rank boundary."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_SequenceCut",
        T=10240, H=4, D=64,
        lengths=[3000, 4000, 3240],
        dtype=torch.bfloat16,
    )


def test_cp2_boundary_aligned():
    """CP2: sequence boundaries aligned with rank boundaries."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_BoundaryAligned",
        T=10240, H=4, D=128,
        lengths=[5120, 5120],
        dtype=torch.bfloat16,
    )


def test_cp4_complex():
    """CP4: complex sequence distribution, first sequence spans 3 ranks."""
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_Complex",
        T=10240, H=4, D=128,
        lengths=[7000, 3240],
        dtype=torch.bfloat16,
    )


def test_cp4_single_sequence():
    """CP4: single long sequence spanning all ranks."""
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_SingleSequence",
        T=10240, H=4, D=128,
        lengths=[10240],
        dtype=torch.bfloat16,
    )


def test_cp8_single_sequence():
    """CP8: single long sequence spanning all ranks."""
    if torch.cuda.device_count() < 8:
        pytest.skip("At least 8 GPUs required")

    run_cp_test_with_spawn(
        world_size=8,
        test_name="CP8_SingleSequence",
        T=65536, H=4, D=128,
        lengths=[65536],
        dtype=torch.bfloat16,
    )


def test_cp2_many_short_sequences():
    """CP2: many short sequences."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_ManyShortSequences",
        T=10240, H=4, D=128,
        lengths=[1000, 1500, 2000, 2500, 1240, 1000, 1000],
        dtype=torch.bfloat16,
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
