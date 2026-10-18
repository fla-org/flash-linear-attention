# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Shared DPLR and RWKV7 output and gradient comparisons against the recurrent reference."""


import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from fla.ops.cp import build_cp_context
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule
from fla.ops.generalized_delta_rule.dplr.backends import DPLRTileLangBackend
from fla.ops.generalized_delta_rule.dplr.naive import dplr_recurrence
from fla.ops.rwkv7 import chunk_rwkv7
from fla.utils import assert_close, device
from tests.cp.utils import gather_global, run_distributed

pytestmark = pytest.mark.skipif(device != "cuda", reason="requires CUDA or ROCm GPUs")


def _chunk_rwkv7(q, k, v, a, b, gk, scale=None, **kwargs):
    # chunk_rwkv7 defaults scale=1.0; align with chunk_dplr_delta_rule (scale=None
    # -> 1/sqrt(K)) so the wrapper matches the dplr_recurrence reference.
    return chunk_rwkv7(r=q, w=gk, k=k, v=v, a=a, b=b, scale=scale, **kwargs)


def _test_chunk(
    rank: int,
    world_size: int,
    T: int,
    H: int,
    D: int,
    lengths: list[int],
    dtype,
    op=chunk_dplr_delta_rule,
    op_chunk_size: int = 64,
    assert_tilelang: bool = False,
):
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

    if rank == 0:
        assert_close("Output", ref_out, o_cp_global, ratio=8e-3, warning=False)
        assert_close("dq", ref_dq, dq_cp_global, ratio=8e-3, warning=False)
        assert_close("dk", ref_dk, dk_cp_global, ratio=8e-3, warning=False)
        assert_close("dv", ref_dv, dv_cp_global, ratio=8e-3, warning=False)
        assert_close("da", ref_da, da_cp_global, ratio=8e-3, warning=False)
        assert_close("db", ref_db, db_cp_global, ratio=8e-3, warning=False)
        assert_close("dgk", ref_dgk, dgk_cp_global, ratio=8e-3, warning=False)


@pytest.mark.parametrize(
    ('world_size', 'lengths', 'op'),
    [
        pytest.param(2, [400, 624], chunk_dplr_delta_rule, id='CP2_SequenceCut'),
        pytest.param(2, [512, 512], chunk_dplr_delta_rule, id='CP2_BoundaryAligned'),
        pytest.param(4, [700, 324], chunk_dplr_delta_rule, id='CP4_Complex'),
        pytest.param(4, [1024], chunk_dplr_delta_rule, id='CP4_SingleSequence'),
        pytest.param(2, [100, 200, 300, 424], chunk_dplr_delta_rule, id='CP2_ManyShortSequences'),
        pytest.param(2, [400, 624], _chunk_rwkv7, id='CP2_RWKV7_SequenceCut'),
        pytest.param(4, [1024], _chunk_rwkv7, id='CP4_RWKV7_SingleSequence'),
    ],
)
def test_chunk(world_size, lengths, op):
    run_distributed(
        worker=_test_chunk,
        world_size=world_size,
        T=1024,
        H=4,
        D=64,
        lengths=lengths,
        dtype=torch.bfloat16,
        op=op,
    )


def test_chunk_tilelang(monkeypatch):
    """CP2 through the TileLang DPLR backend (FLA_TILELANG=1), with route assertion."""
    if not DPLRTileLangBackend.is_available():
        pytest.skip("TileLang backend not available")

    monkeypatch.setenv("FLA_TILELANG", "1")
    run_distributed(
        worker=_test_chunk,
        world_size=2,
        T=1024,
        H=128,
        D=64,
        lengths=[400, 624],
        dtype=torch.bfloat16,
        op_chunk_size=32,
        assert_tilelang=True,
    )
