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
from tests.cp.utils import gather_global, shard_tensor

pytestmark = pytest.mark.skipif(device != "cuda", reason="requires CUDA or ROCm GPUs")


def _chunk_rwkv7(q, k, v, a, b, gk, scale=None, **kwargs):
    # chunk_rwkv7 defaults scale=1.0; align with chunk_dplr_delta_rule (scale=None
    # -> 1/sqrt(K)) so the wrapper matches the dplr_recurrence reference.
    return chunk_rwkv7(r=q, w=gk, k=k, v=v, a=a, b=b, scale=scale, **kwargs)


def _make_inputs(rank, T, H, D, dtype):
    B = 1
    q = torch.empty(B, T, H, D, device=device, dtype=dtype)
    k = torch.empty(B, T, H, D, device=device, dtype=dtype)
    v = torch.empty(B, T, H, D, device=device, dtype=dtype)
    a = torch.empty(B, T, H, D, device=device, dtype=dtype)
    b = torch.empty(B, T, H, D, device=device, dtype=dtype)
    gk = torch.empty(B, T, H, D, device=device, dtype=torch.float)
    do = torch.empty(B, T, H, D, device=device, dtype=dtype)

    if rank == 0:
        torch.manual_seed(42)
        # Mirror fla/layers/rwkv7.py:
        #   q/k are l2-normalized over head_dim,
        #   a = -kk where kk = l2norm(...),
        #   b = kk * sigmoid(...),
        #   gk (w in rwkv7) = -0.6065306597126334 * sigmoid(...)  → range [-0.6065, 0]
        q.copy_(F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), dim=-1, p=2.0))
        k.copy_(F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), dim=-1, p=2.0))
        v.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        kk = F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), dim=-1, p=2.0)
        a_sigmoid = torch.randn(B, T, H, D, device=device, dtype=dtype).sigmoid()
        a.copy_(-kk)
        b.copy_(kk * a_sigmoid)
        gk.copy_(-0.6065306597126334 * torch.randn(B, T, H, D, device=device, dtype=torch.float).sigmoid())
        do.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))

    for tensor in (q, k, v, a, b, gk, do):
        dist.broadcast(tensor, src=0)
    return q, k, v, a, b, gk, do


def _reference(q, k, v, a, b, gk, do, offsets):
    results = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        q_seq = q[:, start:end].clone().detach().requires_grad_(True)
        k_seq = k[:, start:end].clone().detach().requires_grad_(True)
        v_seq = v[:, start:end].clone().detach().requires_grad_(True)
        a_seq = a[:, start:end].clone().detach().requires_grad_(True)
        b_seq = b[:, start:end].clone().detach().requires_grad_(True)
        gk_seq = gk[:, start:end].clone().detach().requires_grad_(True)
        do_seq = do[:, start:end].clone()

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
        (o_seq * do_seq).sum().backward()
        results.append((
            o_seq.detach(),
            q_seq.grad.detach(),
            k_seq.grad.detach(),
            v_seq.grad.detach(),
            a_seq.grad.detach(),
            b_seq.grad.detach(),
            gk_seq.grad.detach(),
        ))
    return tuple(torch.cat(parts, dim=1) for parts in zip(*results))


def _run_chunk(
    rank: int,
    world_size: int,
    H: int,
    D: int,
    lengths: list[int],
    dtype,
    op=chunk_dplr_delta_rule,
    op_chunk_size: int = 64,
    assert_tilelang: bool = False,
):
    T = sum(lengths)
    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"

    q_global, k_global, v_global, a_global, b_global, gk_global, do_global = _make_inputs(
        rank=rank,
        T=T,
        H=H,
        D=D,
        dtype=dtype,
    )

    cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
    cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.long)

    if rank == 0:
        ref_out, ref_dq, ref_dk, ref_dv, ref_da, ref_db, ref_dgk = _reference(
            q=q_global,
            k=k_global,
            v=v_global,
            a=a_global,
            b=b_global,
            gk=gk_global,
            do=do_global,
            offsets=cu_seqlens_list,
        )

    dist.barrier()

    context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)

    q_local = shard_tensor(q_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    k_local = shard_tensor(k_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    v_local = shard_tensor(v_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    a_local = shard_tensor(a_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    b_local = shard_tensor(b_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    gk_local = shard_tensor(gk_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    do_local = shard_tensor(do_global, rank=rank, world_size=world_size).clone()

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
        pytest.param(2, [400, 624], chunk_dplr_delta_rule, id='cp2-sequence-cut'),
        pytest.param(2, [512, 512], chunk_dplr_delta_rule, id='cp2-boundary-aligned'),
        pytest.param(4, [700, 324], chunk_dplr_delta_rule, id='cp4-sequence-cut'),
        pytest.param(4, [1024], chunk_dplr_delta_rule, id='cp4-single-sequence'),
        pytest.param(2, [100, 200, 300, 424], chunk_dplr_delta_rule, id='cp2-short-sequences'),
        pytest.param(2, [400, 624], _chunk_rwkv7, id='cp2-rwkv7-sequence-cut'),
        pytest.param(4, [1024], _chunk_rwkv7, id='cp4-rwkv7-single-sequence'),
    ],
)
def test_chunk(run_distributed, world_size, lengths, op):
    run_distributed(worker=_run_chunk, world_size=world_size, H=4, D=64, lengths=lengths, dtype=torch.bfloat16, op=op)


def test_chunk_tilelang(run_distributed, monkeypatch):
    """CP2 through the TileLang DPLR backend (FLA_TILELANG=1), with route assertion."""
    if not DPLRTileLangBackend.is_available():
        pytest.skip("TileLang backend not available")

    monkeypatch.setenv("FLA_TILELANG", "1")
    run_distributed(
        worker=_run_chunk,
        world_size=2,
        H=128,
        D=64,
        lengths=[400, 624],
        dtype=torch.bfloat16,
        op_chunk_size=32,
        assert_tilelang=True,
    )
