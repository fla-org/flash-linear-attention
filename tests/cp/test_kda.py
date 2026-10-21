# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare distributed KDA outputs and gradients with the recurrent reference.

In fused-gate mode, the reference applies the same gate transform and Q/K normalization before its recurrent scan.
In pre-gated mode, both paths consume per-token log gates; chunk_kda computes its own chunk-local cumulative sums.
Each reference sequence starts with zero state, including sequences spanning rank boundaries.
Rank-local outputs and input gradients are gathered back into global token order before comparison.
"""

import contextlib
import logging
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

with contextlib.suppress(ImportError):
    import torch_npu  # noqa: F401

from fla.ops.cp import build_cp_context
from fla.ops.kda import chunk_kda
from fla.ops.kda.gate import naive_kda_lowerbound_gate
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import assert_close, device_platform, device_torch_lib
from fla.utils import device as fla_device

# configure logging to see assert_close messages
logging.basicConfig(level=logging.INFO, format='%(message)s')


def _init_distributed(rank, world_size):
    """Initialize distributed environment for a single process."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # the NPU backend requires HCCL; a separate CUDA port avoids collisions with convolution CP tests
    if device_platform == 'npu':
        backend, port = 'hccl', '29617'
    else:
        backend, port = 'nccl', '29501'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device_torch_lib.set_device(rank)


def _cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _run_chunk(
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
    state_v_first: bool = False,
    use_tf32x3_affine_chain: bool | None = None,
    layout: str = 'contiguous',
):
    """Run one rank of the distributed KDA reference comparison."""
    try:
        _init_distributed(rank, world_size)
        device = torch.device(f'{fla_device}:{rank}')

        assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
        if layout == 'zigzag':
            assert T % (2 * world_size) == 0, f"T={T} must be divisible by 2*world_size={2 * world_size} for zigzag"
        assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Test: {test_name}")
            print(f"Config: T={T}, H={H}, D={D}, world_size={world_size}, disable_recompute={disable_recompute}")
            print(f"use_gate_in_kernel={use_gate_in_kernel}, safe_gate={safe_gate}, lower_bound={lower_bound}")
            print(f"Sequence lengths: {lengths}")
            print(f"{'='*60}")

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
            # q's offset keeps L2 normalization observable; large k values exceed the bf16 reference error budget
            q_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype) + 1.0)
            k_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
            v_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
            if use_gate_in_kernel:
                g_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
            else:
                g_global.copy_(F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=torch.float)))
                g_global.clamp_(min=-5.0)
            beta_global.copy_(torch.randn(B, T, H, device=device, dtype=dtype).sigmoid())
            do_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))

        dist.broadcast(q_global, src=0)
        dist.broadcast(k_global, src=0)
        dist.broadcast(v_global, src=0)
        dist.broadcast(g_global, src=0)
        dist.broadcast(beta_global, src=0)
        dist.broadcast(do_global, src=0)

        # always generate these for reference, even when use_gate_in_kernel=False
        import triton
        num_even_heads = triton.next_power_of_2(H)
        projection_size = H * D

        A_log_global = torch.empty(num_even_heads, dtype=torch.float32, device=device)
        dt_bias_global = torch.empty(projection_size, dtype=torch.float32, device=device)

        if rank == 0:
            import math
            # distinct per-head gates expose missing head offsets in backward loads
            if safe_gate and lower_bound is not None:
                A_log_global[:H] = torch.log(torch.linspace(1, 8, H, dtype=torch.float32, device=device))
            else:
                A_log_global.copy_(torch.log(torch.empty(num_even_heads, dtype=torch.float32, device=device).uniform_(1, 16)))
            # dt initialization from real training
            dt = torch.exp(torch.rand(projection_size, device=device) *
                           (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp_(min=1e-4)
            dt_bias_global.copy_(dt + torch.log(-torch.expm1(-dt)))

        dist.broadcast(A_log_global, src=0)
        dist.broadcast(dt_bias_global, src=0)

        cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
        cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.long)

        ref_out = None
        ref_dq, ref_dk, ref_dv, ref_dg, ref_db = None, None, None, None, None

        if rank == 0:
            N = len(lengths)
            ref_outputs = []

            for i in range(N):
                seq_start = cu_seqlens_list[i]
                seq_end = cu_seqlens_list[i + 1]

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
                            g_seq_input.to(torch.float),
                            A_log_global[:H].contiguous(),
                            dt_bias_global,
                            lower_bound=lower_bound,
                        )
                    else:
                        from fla.ops.kda.gate import naive_kda_gate
                        g_processed = naive_kda_gate(
                            g_seq_input.to(torch.float),
                            A_log_global[:H].contiguous(),
                            dt_bias_global,
                        )
                    g_for_grad = g_seq_input
                else:
                    # both paths consume per-token log decay; only chunk_kda applies a chunk-local cumulative sum
                    g_processed = g_seq.to(torch.float).requires_grad_(True)
                    g_for_grad = g_processed

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

            ref_out = torch.cat([item['o'].detach() for item in ref_outputs], dim=1)
            ref_dq = torch.cat(all_dq, dim=1)
            ref_dk = torch.cat(all_dk, dim=1)
            ref_dv = torch.cat(all_dv, dim=1)
            ref_dg = torch.cat(all_dg, dim=1)
            ref_db = torch.cat(all_db, dim=1)

        dist.barrier()

        context = build_cp_context(
            cu_seqlens_global,
            group=dist.group.WORLD,
            **({'use_tf32x3_affine_chain': use_tf32x3_affine_chain} if use_tf32x3_affine_chain is not None else {}),
            layout=layout,
        )

        if layout == 'zigzag':
            # local input = [front part; back part]; part_len = T / (2W)
            part_len = T // (2 * world_size)
            front_start = rank * part_len
            back_start = (2 * world_size - 1 - rank) * part_len

            def slice_local(t):
                return torch.cat([t[:, front_start: front_start + part_len],
                                  t[:, back_start: back_start + part_len]], dim=1)
        else:
            chunk_size = T // world_size
            start_idx = rank * chunk_size
            end_idx = (rank + 1) * chunk_size

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
            state_v_first=state_v_first,
        )

        o_local.backward(do_local)

        def gather_global(local):
            gathered = [torch.zeros_like(local) for _ in range(world_size)]
            dist.all_gather(gathered, local)
            if layout == 'zigzag':
                # undo the zigzag permutation: fronts in rank order, then backs reversed
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

            # dg/db remain warning-only: reference errors reach ~0.0085 for dg and 1.5–2.4% for db
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
                    warn = name in ("dg", "db")
                    assert_close(name, ref, cp, ratio=8e-3, warning=warn)
                print(f"✅ [{test_name}] Test Passed!\n")
            except AssertionError as e:
                print(f"❌ [{test_name}] Test Failed: {e}\n")
                test_passed = False

        dist.barrier()
        _cleanup_distributed()

        if not test_passed:
            raise AssertionError(f"Test {test_name} failed on rank {rank}")

    except Exception as e:
        _cleanup_distributed()
        raise e


def _run_distributed(
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
    state_v_first: bool = False,
    use_tf32x3_affine_chain: bool | None = None,
    layout: str = 'contiguous',
):
    """
    Run CP test using torch.multiprocessing.spawn.
    This allows running the test directly with pytest.
    """
    mp.start_processes(
        _run_chunk,
        args=(world_size, test_name, T, H, D, lengths, dtype, disable_recompute,
              use_gate_in_kernel, safe_gate, lower_bound, state_v_first, use_tf32x3_affine_chain, layout),
        nprocs=world_size,
        join=True,
        start_method='spawn',
    )


@pytest.mark.parametrize(
    (
        'world_size',
        'T',
        'lengths',
        'disable_recompute',
        'use_gate_in_kernel',
        'safe_gate',
        'lower_bound',
        'state_v_first',
        'use_tf32x3_affine_chain',
        'layout',
    ),
    [
        pytest.param(2, 10240, [3000, 4000, 3240], False, True, True, -5.0, False, None, 'contiguous', id='cp2-sequence-cut'),
        pytest.param(2, 10240, [5120, 5120], False, True, True, -5.0, False, None, 'contiguous', id='cp2-boundary-aligned'),
        # the first sequence crosses three rank partitions
        pytest.param(4, 10240, [7000, 3240], False, True, True, -5.0, False, None, 'contiguous', id='cp4-sequence-cut'),
        pytest.param(4, 10240, [10240], False, True, True, -5.0, False, None, 'contiguous', id='cp4-single-sequence'),
        pytest.param(8, 65536, [65536], False, True, True, -5.0, False, None, 'contiguous', id='cp8-single-sequence'),
        pytest.param(
            2,
            10240,
            [1000, 1500, 2000, 2500, 1240, 1000, 1000],
            False,
            True,
            True,
            -5.0,
            False,
            None,
            'contiguous',
            id='cp2-short-sequences',
        ),
        pytest.param(2, 10240, [3000, 4000, 3240], True, True, True, -5.0, False, None, 'contiguous', id='cp2-no-recompute'),
        pytest.param(2, 10240, [3000, 4000, 3240], False, False, False, None, False, None, 'contiguous', id='cp2-pregated'),
        pytest.param(
            2,
            10240,
            [3000, 4000, 3240],
            False,
            False,
            True,
            -5.0,
            False,
            None,
            'contiguous',
            id='cp2-pregated-safe',
        ),
        pytest.param(2, 10240, [3000, 4000, 3240], False, True, True, -5.0, True, None, 'contiguous', id='cp2-v-first'),
        pytest.param(4, 10240, [10240], False, True, True, -5.0, True, None, 'contiguous', id='cp4-v-first'),
        pytest.param(
            2,
            10240,
            [3000, 4000, 3240],
            False,
            True,
            True,
            -5.0,
            False,
            True,
            'contiguous',
            id='cp2-sequence-cut-legacy-precision-true',
        ),
        pytest.param(
            4,
            10240,
            [10240],
            False,
            True,
            True,
            -5.0,
            False,
            True,
            'contiguous',
            id='cp4-single-sequence-legacy-precision-true',
        ),
        pytest.param(
            2,
            10240,
            [3000, 4000, 3240],
            False,
            True,
            True,
            -5.0,
            True,
            True,
            'contiguous',
            id='cp2-v-first-legacy-precision-true',
        ),
        # the state chain crosses all four zigzag parts
        pytest.param(2, 10240, [10240], False, True, True, -5.0, False, None, 'zigzag', id='cp2-zigzag-single-sequence'),
        pytest.param(
            2,
            10240,
            [3000, 4000, 3240],
            False,
            True,
            True,
            -5.0,
            False,
            None,
            'zigzag',
            id='cp2-zigzag-sequence-cut',
        ),
        pytest.param(
            4,
            10240,
            [1500, 2600, 3000, 3140],
            False,
            True,
            True,
            -5.0,
            False,
            None,
            'zigzag',
            id='cp4-zigzag-sequence-cut',
        ),
        # every sequence ends at a part boundary, so no state crosses ranks
        pytest.param(
            4,
            10240,
            [2560, 2560, 2560, 2560],
            False,
            True,
            True,
            -5.0,
            False,
            None,
            'zigzag',
            id='cp4-zigzag-boundary-aligned',
        ),
    ],
)
def test_chunk(
    request,
    world_size,
    T,
    lengths,
    disable_recompute,
    use_gate_in_kernel,
    safe_gate,
    lower_bound,
    state_v_first,
    use_tf32x3_affine_chain,
    layout,
):
    if device_torch_lib.device_count() < world_size:
        pytest.skip(f"At least {world_size} devices required")

    _run_distributed(
        world_size=world_size,
        test_name=request.node.callspec.id,
        T=T,
        H=12,
        D=128,
        lengths=lengths,
        dtype=torch.bfloat16,
        disable_recompute=disable_recompute,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        state_v_first=state_v_first,
        use_tf32x3_affine_chain=use_tf32x3_affine_chain,
        layout=layout,
    )


def _init_torchrun():
    """Initialize distributed environment for torchrun."""
    if 'RANK' not in os.environ:
        return False

    backend = 'hccl' if device_platform == 'npu' else 'nccl'
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    device_torch_lib.set_device(local_rank)
    return True
