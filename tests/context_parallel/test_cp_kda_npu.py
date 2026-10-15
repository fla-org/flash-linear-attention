# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Eager context-parallel KDA on Ascend.

Compares ``chunk_kda`` under an HCCL process group with ``naive_recurrent_kda``
per sequence (``h0=0``). Device and rank selection use ``fla.utils`` helpers.
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
from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import IS_NPU, assert_close, device, device_platform, device_torch_lib

# mark, not importorskip: a fully skipped module reports "no tests collected" and exits
# with code 5, which CI per-file loops treat as a failure.
pytestmark = pytest.mark.skipif(not IS_NPU, reason="requires Ascend NPU")

if IS_NPU:
    # register the hccl backend before init_process_group
    import torch_npu  # noqa: F401

logging.basicConfig(level=logging.INFO, format='%(message)s')

DIST_BACKEND = 'hccl' if device_platform == 'npu' else 'nccl'
MASTER_PORT = '29617'


def init_distributed(rank, world_size):
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = MASTER_PORT
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(backend=DIST_BACKEND, rank=rank, world_size=world_size)
    device_torch_lib.set_device(rank)


def cleanup_distributed():
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
    layout: str = 'contiguous',
):
    try:
        init_distributed(rank, world_size)
        dev = torch.device(f'{device}:{rank}')

        assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
        if layout == 'zigzag':
            assert T % (2 * world_size) == 0, f"T={T} must be divisible by 2*world_size={2 * world_size}"
        assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Test: {test_name}")
            print(f"Config: T={T}, H={H}, D={D}, world_size={world_size}, "
                  f"disable_recompute={disable_recompute}, layout={layout}")
            print(f"use_gate_in_kernel={use_gate_in_kernel}, safe_gate={safe_gate}, lower_bound={lower_bound}")
            print(f"Sequence lengths: {lengths}")
            print(f"{'='*60}")

        # step 1: global data, generated on rank 0 and broadcast
        B = 1
        q_global = torch.empty(B, T, H, D, device=dev, dtype=dtype)
        k_global = torch.empty(B, T, H, D, device=dev, dtype=dtype)
        v_global = torch.empty(B, T, H, D, device=dev, dtype=dtype)
        g_global = torch.empty(B, T, H, D, device=dev, dtype=dtype if use_gate_in_kernel else torch.float)
        beta_global = torch.empty(B, T, H, device=dev, dtype=dtype)
        do_global = torch.empty(B, T, H, D, device=dev, dtype=dtype)

        if rank == 0:
            torch.manual_seed(42)
            q_global.copy_(torch.randn(B, T, H, D, device=dev, dtype=dtype) + 1.0)
            k_global.copy_(torch.randn(B, T, H, D, device=dev, dtype=dtype))
            v_global.copy_(torch.randn(B, T, H, D, device=dev, dtype=dtype))
            if use_gate_in_kernel:
                g_global.copy_(torch.randn(B, T, H, D, device=dev, dtype=dtype))
            else:
                g_global.copy_(F.logsigmoid(torch.randn(B, T, H, D, device=dev, dtype=torch.float)))
                g_global.clamp_(min=-5.0)
            beta_global.copy_(torch.randn(B, T, H, device=dev, dtype=dtype).sigmoid())
            do_global.copy_(torch.randn(B, T, H, D, device=dev, dtype=dtype))

        for t in (q_global, k_global, v_global, g_global, beta_global, do_global):
            dist.broadcast(t, src=0)

        import triton
        num_even_heads = triton.next_power_of_2(H)
        projection_size = H * D

        A_log_global = torch.empty(num_even_heads, dtype=torch.float32, device=dev)
        dt_bias_global = torch.empty(projection_size, dtype=torch.float32, device=dev)

        if rank == 0:
            import math
            if safe_gate and lower_bound is not None:
                A_log_global[:H] = torch.log(torch.linspace(1, 8, H, dtype=torch.float32, device=dev))
            else:
                A_log_global.copy_(torch.log(torch.empty(num_even_heads, dtype=torch.float32, device=dev).uniform_(1, 16)))
            dt = torch.exp(torch.rand(projection_size, device=dev) *
                           (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp_(min=1e-4)
            dt_bias_global.copy_(dt + torch.log(-torch.expm1(-dt)))

        dist.broadcast(A_log_global, src=0)
        dist.broadcast(dt_bias_global, src=0)

        cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
        cu_seqlens_global = torch.tensor(cu_seqlens_list, device=dev, dtype=torch.long)

        # step 2: reference (naive_recurrent_kda per sequence, h0=0)
        ref_out = None
        ref_dq, ref_dk, ref_dv, ref_dg, ref_db = None, None, None, None, None

        if rank == 0:
            ref_outputs = []
            for i in range(len(lengths)):
                seq_start, seq_end = cu_seqlens_list[i], cu_seqlens_list[i + 1]

                q_seq = q_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                k_seq = k_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                v_seq = v_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                g_seq = g_global[:, seq_start:seq_end].clone().detach()
                beta_seq = beta_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
                do_seq = do_global[:, seq_start:seq_end].clone()

                q_seq_norm = F.normalize(q_seq, p=2, dim=-1)
                k_seq_norm = F.normalize(k_seq, p=2, dim=-1)

                if use_gate_in_kernel:
                    g_seq_input = g_seq.requires_grad_(True)
                    if safe_gate and lower_bound is not None:
                        g_processed = naive_kda_lowerbound_gate(
                            g_seq_input.to(torch.float), A_log_global[:H].contiguous(),
                            dt_bias_global, lower_bound=lower_bound)
                    else:
                        g_processed = naive_kda_gate(
                            g_seq_input.to(torch.float), A_log_global[:H].contiguous(), dt_bias_global)
                    g_for_grad = g_seq_input
                else:
                    # pre-gated: g_global is already log-space decay (logsigmoid[, clamp]).
                    # match tests/ops/test_kda.py — naive takes per-token decay; chunk_kda
                    # only runs chunk_local_cumsum. Do not apply naive_kda_gate here.
                    g_processed = g_seq.to(torch.float).requires_grad_(True)
                    g_for_grad = g_processed

                o_seq, _ = naive_recurrent_kda(
                    q=q_seq_norm, k=k_seq_norm, v=v_seq, g=g_processed, beta=beta_seq,
                    initial_state=None, output_final_state=False)

                ref_outputs.append({'o': o_seq, 'q': q_seq, 'k': k_seq, 'v': v_seq,
                                    'g': g_for_grad, 'beta': beta_seq, 'do': do_seq})

            all_dq, all_dk, all_dv, all_dg, all_db = [], [], [], [], []
            for item in ref_outputs:
                (item['o'] * item['do']).sum().backward()
                all_dq.append(item['q'].grad.detach())
                all_dk.append(item['k'].grad.detach())
                all_dv.append(item['v'].grad.detach())
                all_dg.append(item['g'].grad.detach())
                all_db.append(item['beta'].grad.detach())

            ref_out = torch.cat([item['o'].detach() for item in ref_outputs], dim=1)
            ref_dq = torch.cat(all_dq, dim=1)
            ref_dk = torch.cat(all_dk, dim=1)
            ref_dv = torch.cat(all_dv, dim=1)
            ref_dg = torch.cat(all_dg, dim=1)
            ref_db = torch.cat(all_db, dim=1)

        # step 3: CP run
        dist.barrier()

        context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD, layout=layout)

        if layout == 'zigzag':
            part_len = T // (2 * world_size)
            front_start = rank * part_len
            back_start = (2 * world_size - 1 - rank) * part_len

            def slice_local(t):
                return torch.cat([t[:, front_start: front_start + part_len],
                                  t[:, back_start: back_start + part_len]], dim=1)
        else:
            chunk = T // world_size
            start_idx, end_idx = rank * chunk, (rank + 1) * chunk

            def slice_local(t):
                return t[:, start_idx:end_idx]

        q_local = slice_local(q_global).clone().detach().requires_grad_(True)
        k_local = slice_local(k_global).clone().detach().requires_grad_(True)
        v_local = slice_local(v_global).clone().detach().requires_grad_(True)
        g_local = slice_local(g_global).clone().detach().requires_grad_(True)
        beta_local = slice_local(beta_global).clone().detach().requires_grad_(True)
        do_local = slice_local(do_global).clone()

        print(f"[Rank {rank}] cu_seqlens: {context.cu_seqlens.tolist()}, "
              f"pre_num_ranks: {context.pre_num_ranks}")
        dist.barrier()

        o_local, _ = chunk_kda(
            q=q_local, k=k_local, v=v_local, g=g_local, beta=beta_local,
            cp_context=context,
            disable_recompute=disable_recompute,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            A_log=A_log_global[:H].contiguous() if use_gate_in_kernel else None,
            dt_bias=dt_bias_global if use_gate_in_kernel else None,
        )

        o_local.backward(do_local)

        # step 4: gather and verify on rank 0
        def gather_global(local):
            gathered = [torch.zeros_like(local) for _ in range(world_size)]
            dist.all_gather(gathered, local)
            if layout == 'zigzag':
                return torch.cat([g[:, :part_len] for g in gathered]
                                 + [g[:, part_len:] for g in reversed(gathered)], dim=1)
            return torch.cat(gathered, dim=1)

        o_cp_global = gather_global(o_local)
        dq_cp_global = gather_global(q_local.grad)
        dk_cp_global = gather_global(k_local.grad)
        dv_cp_global = gather_global(v_local.grad)
        dg_cp_global = gather_global(g_local.grad)
        db_cp_global = gather_global(beta_local.grad)

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
                    # dg/db warning-only, same rationale as the CUDA CP test
                    assert_close(name, ref, cp, ratio=8e-3, warning=name in ("dg", "db"))
                print(f"PASS [{test_name}]\n")
            except AssertionError as e:
                print(f"FAIL [{test_name}]: {e}\n")
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
    layout: str = 'contiguous',
):
    mp.start_processes(
        run_cp_kda_test_worker,
        args=(world_size, test_name, T, H, D, lengths, dtype, disable_recompute,
              use_gate_in_kernel, safe_gate, lower_bound, layout),
        nprocs=world_size,
        join=True,
        start_method='spawn',
    )


def skip_if_not_enough_devices(n):
    if device_torch_lib.device_count() < n:
        pytest.skip(f"At least {n} devices required")


GATE_KWARGS = dict(use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0)


def test_cp2_sequence_cut():
    """CP2: sequences cut across rank boundary, in-kernel safe gate."""
    skip_if_not_enough_devices(2)
    run_cp_test_with_spawn(2, "CP2_SequenceCut", T=10240, H=12, D=128,
                           lengths=[3000, 4000, 3240], **GATE_KWARGS)


def test_cp2_boundary_aligned():
    """CP2: sequence boundaries aligned with rank boundaries (no cross-rank state)."""
    skip_if_not_enough_devices(2)
    run_cp_test_with_spawn(2, "CP2_BoundaryAligned", T=10240, H=12, D=128,
                           lengths=[5120, 5120], **GATE_KWARGS)


def test_cp2_many_short_sequences():
    """CP2: many short sequences."""
    skip_if_not_enough_devices(2)
    run_cp_test_with_spawn(2, "CP2_ManyShortSequences", T=10240, H=12, D=128,
                           lengths=[1000, 1500, 2000, 2500, 1240, 1000, 1000], **GATE_KWARGS)


def test_cp4_complex():
    """CP4: first sequence spans 3 ranks."""
    skip_if_not_enough_devices(4)
    run_cp_test_with_spawn(4, "CP4_Complex", T=10240, H=12, D=128,
                           lengths=[7000, 3240], **GATE_KWARGS)


def test_cp4_single_sequence():
    """CP4: single long sequence spanning all ranks."""
    skip_if_not_enough_devices(4)
    run_cp_test_with_spawn(4, "CP4_SingleSequence", T=10240, H=12, D=128,
                           lengths=[10240], **GATE_KWARGS)


def test_cp2_pregated_nonsafe():
    """CP2: pre-gated mode (use_gate_in_kernel=False), non-safe gate."""
    skip_if_not_enough_devices(2)
    run_cp_test_with_spawn(2, "CP2_PreGated_NonSafe", T=10240, H=12, D=128,
                           lengths=[3000, 4000, 3240],
                           use_gate_in_kernel=False, safe_gate=False)


def test_cp2_pregated_safe():
    """CP2: pre-gated mode with bounded log-space gates (safe_gate=True)."""
    skip_if_not_enough_devices(2)
    run_cp_test_with_spawn(2, "CP2_PreGated_Safe", T=10240, H=12, D=128,
                           lengths=[3000, 4000, 3240],
                           use_gate_in_kernel=False, safe_gate=True, lower_bound=-5.0)


def test_cp2_disable_recompute():
    """CP2: disable_recompute=True."""
    skip_if_not_enough_devices(2)
    run_cp_test_with_spawn(2, "CP2_DisableRecompute", T=10240, H=12, D=128,
                           lengths=[3000, 4000, 3240], disable_recompute=True, **GATE_KWARGS)


def test_cp2_zigzag_single_sequence():
    """CP2 zigzag layout: single long sequence, state chain snakes through all parts."""
    skip_if_not_enough_devices(2)
    run_cp_test_with_spawn(2, "CP2_Zigzag_SingleSequence", T=10240, H=12, D=128,
                           lengths=[10240], layout='zigzag', **GATE_KWARGS)


def test_cp4_zigzag_complex():
    """CP4 zigzag layout: sequences cut at every part boundary."""
    skip_if_not_enough_devices(4)
    run_cp_test_with_spawn(4, "CP4_Zigzag_Complex", T=10240, H=12, D=128,
                           lengths=[1500, 2600, 3000, 3140], layout='zigzag', **GATE_KWARGS)
