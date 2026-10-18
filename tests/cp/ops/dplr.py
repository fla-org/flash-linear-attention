# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Shared DPLR and RWKV7 output and gradient comparisons against the recurrent reference."""


import torch
import torch.distributed as dist
import torch.nn.functional as F

from fla.ops.cp import build_cp_context
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule
from fla.ops.generalized_delta_rule.dplr.naive import dplr_recurrence
from fla.utils import assert_close
from tests.cp.utils import gather_global


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
    op=chunk_dplr_delta_rule,
    op_chunk_size: int = 64,
    assert_tilelang: bool = False,
):
    """
    Worker function for CP DPLR test.
    Runs in a spawned process with the given rank.
    """
    device = torch.device(f'cuda:{rank}')

    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
    assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"Config: T={T}, H={H}, D={D}, world_size={world_size}, safe_gate={safe_gate}")
        print(f"Sequence lengths: {lengths}")
        print(f"{'='*60}")

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

    route_spy = None
    if assert_tilelang:
        from fla.ops.generalized_delta_rule.dplr.backends.tilelang import DPLRTileLangBackend
        route_spy = []
        _orig = DPLRTileLangBackend.chunk_dplr_delta_rule

        def _spy(self, *args, **kwargs):
            route_spy.append(1)
            return _orig(self, *args, **kwargs)

        DPLRTileLangBackend.chunk_dplr_delta_rule = _spy

    try:
        # CP Forward
        o_local, _ = op(
            q=q_local,
            k=k_local,
            v=v_local,
            a=a_local,
            b=b_local,
            gk=gk_local,
            cp_context=context,
            safe_gate=True,
            chunk_size=op_chunk_size,
        )

        # CP Backward
        o_local.backward(do_local)

        if assert_tilelang:
            assert route_spy, "TileLang backend route was not taken"
    finally:
        if assert_tilelang:
            DPLRTileLangBackend.chunk_dplr_delta_rule = _orig

    o_cp_global = gather_global(o_local, world_size=world_size)

    dq_cp_global = gather_global(q_local.grad, world_size=world_size)

    dk_cp_global = gather_global(k_local.grad, world_size=world_size)

    dv_cp_global = gather_global(v_local.grad, world_size=world_size)

    da_cp_global = gather_global(a_local.grad, world_size=world_size)

    db_cp_global = gather_global(b_local.grad, world_size=world_size)

    dgk_cp_global = gather_global(gk_local.grad, world_size=world_size)

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

        ratio_threshold = 8e-3
        failures = []
        for name, ref, cp in tensors_to_verify:
            try:
                assert_close(name, ref, cp, ratio=ratio_threshold, warning=False)
            except AssertionError as e:
                failures.append((name, str(e).strip()))
        if failures:
            print(f"❌ [{test_name}] Test Failed on: {[f[0] for f in failures]}")
            for name, msg in failures:
                print(f"   - {msg}")
            print()
            test_passed = False
        else:
            print(f"✅ [{test_name}] Test Passed!\n")

    dist.barrier()

    if not test_passed:
        raise AssertionError(f"Test {test_name} failed on rank {rank}")
