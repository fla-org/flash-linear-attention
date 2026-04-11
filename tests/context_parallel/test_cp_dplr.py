# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Test for Context Parallel (CP) DPLR (Delta Product Linear RNN)

Test Architecture Notes:
========================
- Reference: dplr_recurrence from naive.py (exact mathematical definition)
- CP path: chunk_dplr_delta_rule with cp_context, sequence split across ranks
- Both should produce identical results

DPLR uses per-dim gate gk of shape [B, T, H, K], similar to KDA.
The CP pre-process uses USE_GK=True and USE_BG=True for the DPLR-specific
transition matrix computation.

Context Parallel Principle:
===========================
With Context Parallel:
1. Sequence Partitioning: input sequence split across ranks along sequence dim
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
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule
from fla.ops.generalized_delta_rule.dplr.naive import dplr_recurrence
from fla.utils import assert_close

# Configure logging to see assert_close messages
logging.basicConfig(level=logging.INFO, format='%(message)s')


def init_distributed(rank, world_size):
    """Initialize distributed environment for a single process."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29503'  # Different port from other CP tests
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_cp_dplr_test_worker(
    rank: int,
    world_size: int,
    test_name: str,
    T: int,
    H: int,
    D: int,
    lengths: list[int],
    dtype,
    safe_gate: bool = False,
):
    """
    Worker function for CP DPLR test.
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
            print(f"Config: T={T}, H={H}, D={D}, world_size={world_size}, safe_gate={safe_gate}")
            print(f"Sequence lengths: {lengths}")
            print(f"{'='*60}")

        # Step 1: Prepare Global Data (all generated on rank 0, broadcast to all)
        B = 1
        q_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
        k_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
        v_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
        a_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
        b_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
        gk_global = torch.empty(B, T, H, D, device=device, dtype=torch.float)
        do_global = torch.empty(B, T, H, D, device=device, dtype=dtype)

        if rank == 0:
            torch.manual_seed(42)
            # Mirror fla/layers/rwkv7.py:
            #   q/k are l2-normalized over head_dim,
            #   a = -kk where kk = l2norm(...),
            #   b = kk * sigmoid(...),
            #   gk (w in rwkv7) = -0.6065306597126334 * sigmoid(...)  → range [-0.6065, 0]
            q_global.copy_(F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), dim=-1, p=2.0))
            k_global.copy_(F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), dim=-1, p=2.0))
            v_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
            kk = F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), dim=-1, p=2.0)
            a_sigmoid = torch.randn(B, T, H, D, device=device, dtype=dtype).sigmoid()
            a_global.copy_(-kk)
            b_global.copy_(kk * a_sigmoid)
            gk_global.copy_(-0.6065306597126334 * torch.randn(B, T, H, D, device=device, dtype=torch.float).sigmoid())
            do_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))

        # Broadcast to ensure all ranks have same data
        dist.broadcast(q_global, src=0)
        dist.broadcast(k_global, src=0)
        dist.broadcast(v_global, src=0)
        dist.broadcast(a_global, src=0)
        dist.broadcast(b_global, src=0)
        dist.broadcast(gk_global, src=0)
        dist.broadcast(do_global, src=0)

        # Prepare cu_seqlens
        cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
        cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.long)

        # Step 2: Reference Run using dplr_recurrence (ground truth)
        # naive uses head-first format [B, H, T, D]
        ref_out = None
        ref_dq, ref_dk, ref_dv, ref_da, ref_db, ref_dgk = None, None, None, None, None, None

        if rank == 0:
            N = len(lengths)
            ref_outputs = []

            for i in range(N):
                seq_start = cu_seqlens_list[i]
                seq_end = cu_seqlens_list[i + 1]

                q_seq = q_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                k_seq = k_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                v_seq = v_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                a_seq = a_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                b_seq = b_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                gk_seq = gk_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                do_seq = do_global[:, seq_start:seq_end].clone()

                # transpose to head-first [B, H, T, D] for naive
                q_hf = q_seq.transpose(1, 2)
                k_hf = k_seq.transpose(1, 2)
                v_hf = v_seq.transpose(1, 2)
                a_hf = a_seq.transpose(1, 2)
                b_hf = b_seq.transpose(1, 2)
                gk_hf = gk_seq.transpose(1, 2)

                o_seq, _ = dplr_recurrence(
                    q=q_hf,
                    k=k_hf,
                    v=v_hf,
                    alpha=a_hf,
                    beta=b_hf,
                    gk=gk_hf,
                    initial_state=None,
                    output_final_state=False,
                )
                # transpose back to [B, T, H, D]
                o_seq = o_seq.transpose(1, 2)

                ref_outputs.append({
                    'o': o_seq,
                    'q': q_seq,
                    'k': k_seq,
                    'v': v_seq,
                    'a': a_seq,
                    'b': b_seq,
                    'gk': gk_seq,
                    'do': do_seq,
                })

            all_dq = []
            all_dk = []
            all_dv = []
            all_da = []
            all_db = []
            all_dgk = []

            for item in ref_outputs:
                (item['o'] * item['do']).sum().backward()
                all_dq.append(item['q'].grad.detach())
                all_dk.append(item['k'].grad.detach())
                all_dv.append(item['v'].grad.detach())
                all_da.append(item['a'].grad.detach())
                all_db.append(item['b'].grad.detach())
                all_dgk.append(item['gk'].grad.detach())

            ref_out = torch.cat([item['o'].detach() for item in ref_outputs], dim=1)
            ref_dq = torch.cat(all_dq, dim=1)
            ref_dk = torch.cat(all_dk, dim=1)
            ref_dv = torch.cat(all_dv, dim=1)
            ref_da = torch.cat(all_da, dim=1)
            ref_db = torch.cat(all_db, dim=1)
            ref_dgk = torch.cat(all_dgk, dim=1)

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
        a_local = a_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        b_local = b_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        gk_local = gk_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        do_local = do_global[:, start_idx:end_idx, :].clone()

        print(f"[Rank {rank}] chunk: [{start_idx}, {end_idx}), "
              f"cu_seqlens: {context.cu_seqlens.tolist()}, "
              f"pre_num_ranks: {context.pre_num_ranks}")
        dist.barrier()

        # CP Forward
        o_local, _ = chunk_dplr_delta_rule(
            q=q_local,
            k=k_local,
            v=v_local,
            a=a_local,
            b=b_local,
            gk=gk_local,
            cp_context=context,
            safe_gate=True,
            chunk_size=64,
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

        da_gathered = [torch.zeros_like(a_local.grad) for _ in range(world_size)]
        dist.all_gather(da_gathered, a_local.grad)
        da_cp_global = torch.cat(da_gathered, dim=1)

        db_gathered = [torch.zeros_like(b_local.grad) for _ in range(world_size)]
        dist.all_gather(db_gathered, b_local.grad)
        db_cp_global = torch.cat(db_gathered, dim=1)

        dgk_gathered = [torch.zeros_like(gk_local.grad) for _ in range(world_size)]
        dist.all_gather(dgk_gathered, gk_local.grad)
        dgk_cp_global = torch.cat(dgk_gathered, dim=1)

        test_passed = True
        if rank == 0:
            print(f"\n[{test_name}] Verifying results...")

            tensors_to_verify = [
                ("Output", ref_out, o_cp_global),
                ("dq", ref_dq, dq_cp_global),
                ("dk", ref_dk, dk_cp_global),
                ("dv", ref_dv, dv_cp_global),
                ("da", ref_da, da_cp_global),
                ("db", ref_db, db_cp_global),
                ("dgk", ref_dgk, dgk_cp_global),
            ]

            try:
                for name, ref, cp in tensors_to_verify:
                    assert_close(name, ref, cp, ratio=2.5e-2, warning=False)
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
    safe_gate: bool = False,
):
    """
    Run CP test using torch.multiprocessing.spawn.
    This allows running the test directly with pytest.
    """
    mp.start_processes(
        run_cp_dplr_test_worker,
        args=(world_size, test_name, T, H, D, lengths, dtype, safe_gate),
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
        T=1024, H=4, D=64,
        lengths=[400, 624],
        dtype=torch.bfloat16,
    )


def test_cp2_boundary_aligned():
    """CP2: sequence boundaries aligned with rank boundaries."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_BoundaryAligned",
        T=1024, H=4, D=64,
        lengths=[512, 512],
        dtype=torch.bfloat16,
    )


def test_cp4_complex():
    """CP4: complex sequence distribution, first sequence spans 3 ranks."""
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_Complex",
        T=1024, H=4, D=64,
        lengths=[700, 324],
        dtype=torch.bfloat16,
    )


def test_cp4_single_sequence():
    """CP4: single long sequence spanning all ranks."""
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_SingleSequence",
        T=1024, H=4, D=64,
        lengths=[1024],
        dtype=torch.bfloat16,
    )


def test_cp2_many_short_sequences():
    """CP2: many short sequences."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_ManyShortSequences",
        T=1024, H=4, D=64,
        lengths=[100, 200, 300, 424],
        dtype=torch.bfloat16,
    )


def test_cp2_safe_gate():
    """CP2: safe_gate=True with sequence cut."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_SafeGate",
        T=1024, H=4, D=64,
        lengths=[400, 624],
        dtype=torch.bfloat16,
        safe_gate=True,
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
