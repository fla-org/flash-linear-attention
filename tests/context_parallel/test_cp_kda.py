"""
Test for Context Parallel (CP) KDA (Kimi Delta Attention)

Implementation Hierarchy and Relationships:
==========================================

1. naive_recurrent_kda (fla/ops/kda/naive.py):
   - The mathematical gold standard - sequential token-by-token computation
   - Input g is per-token log-space decay (NOT cumulative)
   - Each g_t is used independently as: S_t = S_{t-1} * exp(g_t)
   - Recurrence:
       S_t = S_{t-1} * exp(g_t) + beta_t * k_t (x) (v_t - S_{t-1} @ k_t)
       o_t = q_t^T @ S_t
     where (x) denotes outer product, S is [K, V] state matrix
   - No internal gate computation or L2 normalization
   - Used as the reference baseline for correctness verification

2. naive_chunk_kda (fla/ops/kda/naive.py):
   - Chunk-parallel version mathematically equivalent to naive_recurrent_kda
   - Input g is also per-token (same as naive_recurrent_kda)
   - Performs cumsum on g internally: g = g.cumsum(-2) at line 60
   - Still a PyTorch implementation, but exploits chunk-level parallelism
   - The chunk algorithm uses the "w-y representation" for efficient computation

3. chunk_kda (fla/ops/kda/chunk.py ChunkKDAFunction):
   - Production Triton kernel implementation with optimizations:
     a) Fused L2 normalization (when use_qk_l2norm_in_kernel=True)
     b) Fused gate computation (when use_gate_in_kernel=True)
     c) Variable-length sequence support (cu_seqlens)
   - Gate processing flow when use_gate_in_kernel=True:
     - Raw g -> kda_gate_chunk_cumsum() -> applies gate formula + cumsum + scale(RCP_LN2)
     - For safe_gate=True: gate_t = lower_bound * sigmoid(exp(A_log) * (g_t + dt_bias))
     - For safe_gate=False: gate_t = -exp(A_log) * softplus(g_t + dt_bias)
     - kda_gate_chunk_cumsum outputs CUMSUM'd + SCALED values (gate + cumsum in one fused op)
   - L2 normalization is applied AFTER gate computation

4. Context Parallel (CP) chunk_kda:
   - Extension of chunk_kda for multi-GPU distributed training
   - Sequence is partitioned across ranks, with state communication between ranks
   - Uses build_cp_context() to manage cross-rank dependencies
   - Forward: Non-first ranks receive initial_state from previous rank
   - Backward: Gradient dht flows back across rank boundaries

Gate Functions (fla/ops/kda/gate.py):
=====================================

- naive_kda_gate / naive_kda_lowerbound_gate:
  Pure PyTorch reference. Output is per-token gate values (NOT cumsum'd).
  Used in this test's reference path to compute g for naive_recurrent_kda.

- kda_gate_chunk_cumsum (Triton kernel):
  Fused gate + chunk-local cumsum + optional scale. Output IS cumsum'd.
  Used internally by chunk_kda when use_gate_in_kernel=True.

Test Architecture Notes:
========================
This test uses naive_recurrent_kda as the reference baseline because:
- It represents the exact mathematical definition without chunking approximations
- It expects per-token g (non-cumsum'd) and pre-normalized q/k

When use_gate_in_kernel=True, the reference path must manually:
1. Apply L2 normalization to q/k before passing to naive_recurrent_kda
2. Apply the gate formula (naive_kda_lowerbound_gate or naive_kda_gate)
3. The output g from these naive gate functions is per-token (non-cumsum'd),
   which is exactly what naive_recurrent_kda expects

When use_gate_in_kernel=False:
- For chunk_kda: input g must be pre-processed (gate applied + cumsum'd)
- For naive_recurrent_kda reference: g should be gate-applied but NOT cumsum'd

Variable-Length Sequence Handling:
==================================
- All sequences are flattened to batch size 1 with cu_seqlens markers
- Each sequence is processed independently with h0=0 (no cross-sequence state)
- The reference computation loops over sequences and concatenates results
- CP path uses the same cu_seqlens for correct chunk boundary handling

Context Parallel Principle for KDA:
===================================

KDA has a recurrent state dependency across tokens. The hidden state h evolves
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

import logging
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from fla.ops.cp import build_cp_context
from fla.ops.kda import chunk_kda
from fla.ops.kda.gate import naive_kda_lowerbound_gate
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import assert_close

# Configure logging to see assert_close messages
logging.basicConfig(level=logging.INFO, format='%(message)s')


def init_distributed(rank, world_size):
    """Initialize distributed environment for a single process."""
    # Configure logging in worker process
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'  # Different port from conv test
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_cp_kda_test_worker(
    rank: int,
    world_size: int,
    test_name: str,
    T: int,
    H: int,
    D: int,
    lengths: list[int],
    dtype,
    disable_recompute: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
):
    """
    Worker function for CP KDA test.
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
            print(f"Config: T={T}, H={H}, D={D}, world_size={world_size}, disable_recompute={disable_recompute}")
            print(f"use_gate_in_kernel={use_gate_in_kernel}, safe_gate={safe_gate}, lower_bound={lower_bound}")
            print(f"Sequence lengths: {lengths}")
            print(f"{'='*60}")

        # Step 1: Prepare Global Data (all generated on rank 0, broadcast to other ranks)
        B = 1
        q_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
        k_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
        v_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
        g_global = torch.empty(B, T, H, D, device=device, dtype=dtype if use_gate_in_kernel else torch.float)
        beta_global = torch.empty(B, T, H, device=device, dtype=dtype)
        do_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
        A_log_global = None
        dt_bias_global = None

        if rank == 0:
            torch.manual_seed(42)
            # Generate inputs
            # Asymmetric initialization for q and k to test L2 norm
            # q: small positive values around 1.0
            # k: large negative values around -50.0
            q_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype) + 1.0)
            k_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype) * 10.0 - 50.0)
            v_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
            if use_gate_in_kernel:
                g_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
            else:
                g_global.copy_(F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=torch.float)))
                g_global.clamp_(min=-5.0)
            beta_global.copy_(torch.randn(B, T, H, device=device, dtype=dtype).sigmoid())
            do_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))

        # Broadcast to ensure all ranks have same data
        dist.broadcast(q_global, src=0)
        dist.broadcast(k_global, src=0)
        dist.broadcast(v_global, src=0)
        dist.broadcast(g_global, src=0)
        dist.broadcast(beta_global, src=0)
        dist.broadcast(do_global, src=0)

        # Prepare and broadcast A_log and dt_bias for gate computation
        # Always generate these for reference, even when use_gate_in_kernel=False
        import triton
        num_even_heads = triton.next_power_of_2(H)
        projection_size = H * D

        # All ranks create tensors with same shape first
        A_log_global = torch.empty(num_even_heads, dtype=torch.float32, device=device)
        dt_bias_global = torch.empty(projection_size, dtype=torch.float32, device=device)

        # Rank 0 fills in the actual data
        if rank == 0:
            import math
            if safe_gate and lower_bound is not None:
                A_log_global.copy_(torch.log(torch.ones(num_even_heads, dtype=torch.float32, device=device)))
            else:
                A_log_global.copy_(torch.log(torch.empty(num_even_heads, dtype=torch.float32, device=device).uniform_(1, 16)))
            # dt initialization from real training
            dt = torch.exp(torch.rand(projection_size, device=device) *
                           (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp_(min=1e-4)
            dt_bias_global.copy_(dt + torch.log(-torch.expm1(-dt)))

        # Broadcast from rank 0 to all ranks
        dist.broadcast(A_log_global, src=0)
        dist.broadcast(dt_bias_global, src=0)

        # Prepare cu_seqlens
        cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
        cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.long)

        # Step 2: Reference Run using naive_recurrent_kda (ground truth)
        # Each sequence is computed independently with h0=0 for simplicity
        ref_out = None
        ref_dq, ref_dk, ref_dv, ref_dg, ref_db = None, None, None, None, None

        if rank == 0:
            N = len(lengths)
            ref_outputs = []

            for i in range(N):
                seq_start = cu_seqlens_list[i]
                seq_end = cu_seqlens_list[i + 1]

                # Extract sequence data and create leaf tensors
                q_seq = q_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                k_seq = k_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                v_seq = v_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                g_seq = g_global[:, seq_start:seq_end].clone().detach()
                beta_seq = beta_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                do_seq = do_global[:, seq_start:seq_end].clone()

                # Apply L2 normalization to q (naive implementation expects normalized q)
                q_seq_norm = F.normalize(q_seq, p=2, dim=-1)
                k_seq_norm = F.normalize(k_seq, p=2, dim=-1)

                # Compute gate using naive implementation
                # When use_gate_in_kernel=True, reference should compute g_raw's grad
                # When use_gate_in_kernel=False, reference should compute g_processed's grad
                if use_gate_in_kernel:
                    # For kernel mode: compare g_raw gradients
                    g_seq_input = g_seq.requires_grad_(True)
                    if safe_gate and lower_bound is not None:
                        g_processed = naive_kda_lowerbound_gate(
                            g_seq_input.to(torch.float),
                            A_log_global[:H].contiguous(),
                            dt_bias_global,
                            lower_bound=lower_bound
                        )
                    else:
                        from fla.ops.kda.gate import naive_kda_gate
                        g_processed = naive_kda_gate(
                            g_seq_input.to(torch.float),
                            A_log_global[:H].contiguous(),
                            dt_bias_global
                        )
                    g_for_grad = g_seq_input
                else:
                    # For non-kernel mode: compare g_processed gradients
                    g_seq_input = g_seq  # No grad needed for raw input
                    if safe_gate and lower_bound is not None:
                        g_processed = naive_kda_lowerbound_gate(
                            g_seq_input.to(torch.float),
                            A_log_global[:H].contiguous(),
                            dt_bias_global,
                            lower_bound=lower_bound
                        ).requires_grad_(True)
                    else:
                        from fla.ops.kda.gate import naive_kda_gate
                        g_processed = naive_kda_gate(
                            g_seq_input.to(torch.float),
                            A_log_global[:H].contiguous(),
                            dt_bias_global
                        ).requires_grad_(True)
                    g_for_grad = g_processed

                # Run naive forward with h0=0 (independent sequences)
                o_seq, _ = naive_recurrent_kda(
                    q=q_seq_norm,
                    k=k_seq_norm,
                    v=v_seq,
                    g=g_processed,
                    beta=beta_seq,
                    initial_state=None,
                    output_final_state=False,
                )

                ref_outputs.append({
                    'o': o_seq,
                    'q': q_seq,
                    'k': k_seq,
                    'v': v_seq,
                    'g': g_for_grad,
                    'beta': beta_seq,
                    'do': do_seq,
                })

            # Backward pass for each sequence
            all_dq = []
            all_dk = []
            all_dv = []
            all_dg = []
            all_db = []

            for item in ref_outputs:
                (item['o'] * item['do']).sum().backward()
                all_dq.append(item['q'].grad.detach())
                all_dk.append(item['k'].grad.detach())
                all_dv.append(item['v'].grad.detach())
                all_dg.append(item['g'].grad.detach())
                all_db.append(item['beta'].grad.detach())

            # Concatenate outputs and gradients
            ref_out = torch.cat([item['o'].detach() for item in ref_outputs], dim=1)
            ref_dq = torch.cat(all_dq, dim=1)
            ref_dk = torch.cat(all_dk, dim=1)
            ref_dv = torch.cat(all_dv, dim=1)
            ref_dg = torch.cat(all_dg, dim=1)
            ref_db = torch.cat(all_db, dim=1)

        # Step 3: Context Parallel Run
        dist.barrier()

        # Build CP context
        context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)

        chunk_size = T // world_size
        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size

        # Get local slices
        q_local = q_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        k_local = k_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        v_local = v_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        g_local = g_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        beta_local = beta_global[:, start_idx:end_idx].clone().detach().requires_grad_(True)
        do_local = do_global[:, start_idx:end_idx, :].clone()

        print(f"[Rank {rank}] chunk: [{start_idx}, {end_idx}), "
              f"cu_seqlens: {context.cu_seqlens.tolist()}, "
              f"pre_num_ranks: {context.pre_num_ranks}")
        dist.barrier()

        # CP Forward
        o_local, _ = chunk_kda(
            q=q_local,
            k=k_local,
            v=v_local,
            g=g_local,
            beta=beta_local,
            cp_context=context,
            disable_recompute=disable_recompute,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            A_log=A_log_global[:H].contiguous() if use_gate_in_kernel else None,
            dt_bias=dt_bias_global if use_gate_in_kernel else None,
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
                    assert_close(name, ref, cp, ratio=5e-2, warning=False)
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
    dtype=torch.bfloat16,
    disable_recompute: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
):
    """
    Run CP test using torch.multiprocessing.spawn.
    This allows running the test directly with pytest.
    """
    mp.start_processes(
        run_cp_kda_test_worker,
        args=(world_size, test_name, T, H, D, lengths, dtype, disable_recompute, use_gate_in_kernel, safe_gate, lower_bound),
        nprocs=world_size,
        join=True,
        start_method='spawn',
    )


# ============================================================
# Test Scenario Definitions
# All tests use: use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0
# ============================================================

GATE_KWARGS = dict(use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0)


def test_cp2_sequence_cut():
    """CP2: sequences cut across rank boundary."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_SequenceCut",
        T=10240, H=4, D=128,
        lengths=[3000, 4000, 3240],
        dtype=torch.bfloat16,
        **GATE_KWARGS,
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
        **GATE_KWARGS,
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
        **GATE_KWARGS,
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
        **GATE_KWARGS,
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
        **GATE_KWARGS,
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
        **GATE_KWARGS,
    )


def test_cp2_disable_recompute():
    """CP2: disable_recompute=True."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_DisableRecompute",
        T=10240, H=4, D=128,
        lengths=[3000, 4000, 3240],
        dtype=torch.bfloat16,
        disable_recompute=True,
        **GATE_KWARGS,
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
