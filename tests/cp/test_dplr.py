# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from cp.utils import gather_global
from fla.ops.cp import build_cp_context
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule
from fla.ops.generalized_delta_rule.dplr.backends import DPLRTileLangBackend
from fla.ops.generalized_delta_rule.dplr.naive import dplr_recurrence
from fla.ops.rwkv7 import chunk_rwkv7
from fla.utils import assert_close
from fla.utils import device as device_type


def _run_chunk(
    rank: int,
    world_size: int,
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
    device = torch.device(f'{device_type}:{rank}')

    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
    assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

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
        # match the normalized keys and bounded decay in fla/layers/rwkv7.py
        q_global.copy_(F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), dim=-1, p=2.0))
        k_global.copy_(F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), dim=-1, p=2.0))
        v_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        kk = F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), dim=-1, p=2.0)
        a_sigmoid = torch.randn(B, T, H, D, device=device, dtype=dtype).sigmoid()
        a_global.copy_(-kk)
        b_global.copy_(kk * a_sigmoid)
        gk_global.copy_(-0.6065306597126334 * torch.randn(B, T, H, D, device=device, dtype=torch.float).sigmoid())
        do_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))

    dist.broadcast(q_global, src=0)
    dist.broadcast(k_global, src=0)
    dist.broadcast(v_global, src=0)
    dist.broadcast(a_global, src=0)
    dist.broadcast(b_global, src=0)
    dist.broadcast(gk_global, src=0)
    dist.broadcast(do_global, src=0)

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

    context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)

    chunk_size = T // world_size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size

    q_local = q_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    k_local = k_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    v_local = v_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    a_local = a_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    b_local = b_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    gk_local = gk_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    do_local = do_global[:, start_idx:end_idx, :].clone()

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
            test_passed = False

    dist.barrier()

    if not test_passed:
        raise AssertionError(f"CP comparison failed on rank {rank}")


def _chunk_rwkv7(q, k, v, a, b, gk, scale=None, **kwargs):
    # use the DPLR reference scale for the RWKV7 wrapper as well
    return chunk_rwkv7(r=q, w=gk, k=k, v=v, a=a, b=b, scale=scale, **kwargs)


@pytest.mark.parametrize(
    ('world_size', 'lengths', 'safe_gate', 'op'),
    [
        pytest.param(2, [400, 624], False, chunk_dplr_delta_rule, id='cp2-sequence-cut'),
        pytest.param(2, [512, 512], False, chunk_dplr_delta_rule, id='cp2-boundary-aligned'),
        pytest.param(4, [700, 324], False, chunk_dplr_delta_rule, id='cp4-complex'),
        pytest.param(4, [1024], False, chunk_dplr_delta_rule, id='cp4-single-sequence'),
        pytest.param(2, [100, 200, 300, 424], False, chunk_dplr_delta_rule, id='cp2-many-short-sequences'),
        pytest.param(2, [400, 624], True, chunk_dplr_delta_rule, id='cp2-safe-gate'),
        pytest.param(2, [400, 624], False, _chunk_rwkv7, id='rwkv7-cp2-wrapper-sequence-cut'),
        pytest.param(4, [1024], False, _chunk_rwkv7, id='rwkv7-cp4-wrapper-single-sequence'),
    ],
)
def test_chunk(run_distributed, world_size, lengths, safe_gate, op):
    run_distributed(
        _run_chunk,
        world_size=world_size,
        T=1024,
        H=4,
        D=64,
        lengths=lengths,
        dtype=torch.bfloat16,
        safe_gate=safe_gate,
        op=op,
        op_chunk_size=64,
        assert_tilelang=False,
    )


@pytest.mark.parametrize(
    ('world_size',),
    [
        pytest.param(2, id='cp2-tilelang-route'),
    ],
)
def test_chunk_tilelang(run_distributed, world_size, monkeypatch):
    if not DPLRTileLangBackend.is_available():
        pytest.skip("TileLang backend not available")
    monkeypatch.setenv('FLA_TILELANG', '1')
    run_distributed(
        _run_chunk,
        world_size=world_size,
        T=1024,
        H=128,
        D=64,
        lengths=[400, 624],
        dtype=torch.bfloat16,
        safe_gate=False,
        op=chunk_dplr_delta_rule,
        op_chunk_size=32,
        assert_tilelang=True,
    )
