# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare CP GDN outputs and gradients with a full-sequence chunk reference."""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.utils import IS_TF32_SUPPORTED, assert_close, device
from tests.cp.utils import gather_global, shard_tensor

pytestmark = pytest.mark.skipif(not IS_TF32_SUPPORTED, reason="requires NVIDIA Ampere or newer")

pytestmark = pytest.mark.skipif(device != "cuda", reason="requires CUDA or ROCm GPUs")


def _run_chunk(
    rank: int,
    world_size: int,
    H: int,
    D: int,
    lengths: list[int],
    dtype,
    state_v_first: bool = False,
    Hq: int | None = None,
):
    T = sum(lengths)
    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"

    B = 1
    Hq_actual = Hq if Hq is not None else H
    q_global = torch.empty(B, T, Hq_actual, D, device=device, dtype=dtype)
    k_global = torch.empty(B, T, Hq_actual, D, device=device, dtype=dtype)
    v_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    g_global = torch.empty(B, T, H, device=device, dtype=dtype)
    beta_global = torch.empty(B, T, H, device=device, dtype=torch.float32)
    do_global = torch.empty(B, T, H, D, device=device, dtype=dtype)

    if rank == 0:
        torch.manual_seed(42)
        q_global.copy_(F.normalize(
            input=torch.randn(B, T, Hq_actual, D, device=device, dtype=torch.float32),
            p=2,
            dim=-1,
        ).to(dtype))
        k_global.copy_(F.normalize(
            input=torch.randn(B, T, Hq_actual, D, device=device, dtype=torch.float32),
            p=2,
            dim=-1,
        ).to(dtype))
        v_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        g_global.copy_(F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype)))
        beta_global.copy_(torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid())
        do_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))

    dist.broadcast(q_global, src=0)
    dist.broadcast(k_global, src=0)
    dist.broadcast(v_global, src=0)
    dist.broadcast(g_global, src=0)
    dist.broadcast(beta_global, src=0)
    dist.broadcast(do_global, src=0)

    cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
    cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.long)

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
            state_v_first=state_v_first,
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

    context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)

    q_local = shard_tensor(q_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    k_local = shard_tensor(k_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    v_local = shard_tensor(v_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    g_local = shard_tensor(g_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    beta_local = shard_tensor(beta_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    do_local = shard_tensor(do_global, rank=rank, world_size=world_size).clone()

    dist.barrier()

    o_local, _ = chunk_gated_delta_rule(
        q=q_local,
        k=k_local,
        v=v_local,
        g=g_local,
        beta=beta_local,
        state_v_first=state_v_first,
        cp_context=context,
    )

    o_local.backward(do_local)

    o_cp_global = gather_global(o_local, world_size=world_size)
    dq_cp_global = gather_global(q_local.grad, world_size=world_size)
    dk_cp_global = gather_global(k_local.grad, world_size=world_size)
    dv_cp_global = gather_global(v_local.grad, world_size=world_size)
    dg_cp_global = gather_global(g_local.grad, world_size=world_size)
    db_cp_global = gather_global(beta_local.grad, world_size=world_size)

    if rank == 0:
        assert_close("Output", ref_out, o_cp_global, ratio=3e-3, warning=False)
        assert_close("dq", ref_dq, dq_cp_global, ratio=3e-3, warning=False)
        assert_close("dk", ref_dk, dk_cp_global, ratio=3e-3, warning=False)
        assert_close("dv", ref_dv, dv_cp_global, ratio=3e-3, warning=False)
        assert_close("dg", ref_dg, dg_cp_global, ratio=3e-3, warning=False)
        assert_close("db", ref_db, db_cp_global, ratio=3e-3, warning=False)


@pytest.mark.parametrize(
    ('world_size', 'lengths', 'H', 'D', 'Hq', 'state_v_first'),
    [
        pytest.param(2, [3000, 4000, 3240], 4, 64, 4, False, id='cp2-sequence-cut'),
        pytest.param(2, [5120, 5120], 4, 128, 4, False, id='cp2-boundary-aligned'),
        pytest.param(4, [7000, 3240], 4, 128, 4, False, id='cp4-sequence-cut'),
        pytest.param(4, [10240], 4, 128, 4, False, id='cp4-single-sequence'),
        pytest.param(8, [65536], 4, 128, 4, False, id='cp8-single-sequence'),
        pytest.param(2, [1000, 1500, 2000, 2500, 1240, 1000, 1000], 4, 128, 4, False, id='cp2-short-sequences'),
        pytest.param(2, [3000, 4000, 3240], 4, 64, 2, False, id='cp2-gva-sequence-cut'),
        pytest.param(2, [10240], 8, 64, 2, False, id='cp2-gva-single-sequence'),
        pytest.param(2, [3000, 4000, 3240], 4, 128, 4, True, id='cp2-v-first'),
        pytest.param(4, [10240], 4, 128, 4, True, id='cp4-v-first'),
    ],
)
def test_chunk(run_distributed, world_size, lengths, H, D, Hq, state_v_first):
    run_distributed(
        worker=_run_chunk,
        world_size=world_size,
        H=H,
        D=D,
        lengths=lengths,
        dtype=torch.bfloat16,
        state_v_first=state_v_first,
        Hq=Hq,
    )
