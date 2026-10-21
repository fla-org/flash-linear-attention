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

from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.utils import assert_close
from fla.utils import device as device_type
from tests.cp.utils import gather_global


def _run_chunk(
    rank: int,
    world_size: int,
    T: int,
    H: int,
    D: int,
    lengths: list[int],
    dtype,
    state_v_first: bool = False,
    Hq: int | None = None,
):
    device = torch.device(f'{device_type}:{rank}')

    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
    assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

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
        q_global.copy_(F.normalize(torch.randn(B, T, Hq_actual, D, device=device,
                       dtype=torch.float32), p=2, dim=-1).to(dtype))
        k_global.copy_(F.normalize(torch.randn(B, T, Hq_actual, D, device=device,
                       dtype=torch.float32), p=2, dim=-1).to(dtype))
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
            cu_seqlens=cu_seqlens_global,
            state_v_first=state_v_first,
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

    dist.barrier()

    o_local, _ = chunk_gated_delta_rule(
        q=q_local,
        k=k_local,
        v=v_local,
        g=g_local,
        beta=beta_local,
        cp_context=context,
        state_v_first=state_v_first,
    )

    o_local.backward(do_local)

    o_cp_global = gather_global(o_local, world_size=world_size)

    dq_cp_global = gather_global(q_local.grad, world_size=world_size)

    dk_cp_global = gather_global(k_local.grad, world_size=world_size)

    dv_cp_global = gather_global(v_local.grad, world_size=world_size)

    dg_cp_global = gather_global(g_local.grad, world_size=world_size)

    db_cp_global = gather_global(beta_local.grad, world_size=world_size)

    test_passed = True
    if rank == 0:

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
                assert_close(name, ref, cp, ratio=3e-3, warning=False)
        except AssertionError:
            test_passed = False

    dist.barrier()

    if not test_passed:
        raise AssertionError(f"CP comparison failed on rank {rank}")


@pytest.mark.parametrize(
    ('world_size', 'T', 'H', 'D', 'lengths', 'state_v_first', 'Hq'),
    [
        pytest.param(2, 10240, 4, 64, [3000, 4000, 3240], False, None, id='cp2-sequence-cut'),
        pytest.param(2, 10240, 4, 128, [5120, 5120], False, None, id='cp2-boundary-aligned'),
        pytest.param(4, 10240, 4, 128, [7000, 3240], False, None, id='cp4-complex'),
        pytest.param(4, 10240, 4, 128, [10240], False, None, id='cp4-single-sequence'),
        pytest.param(8, 65536, 4, 128, [65536], False, None, id='cp8-single-sequence'),
        pytest.param(2, 10240, 4, 128, [1000, 1500, 2000, 2500, 1240, 1000, 1000], False, None, id='cp2-many-short-sequences'),
        pytest.param(2, 10240, 4, 64, [3000, 4000, 3240], False, 2, id='cp2-gqa-sequence-cut'),
        pytest.param(2, 10240, 8, 64, [10240], False, 2, id='cp2-gqa-single-sequence'),
        pytest.param(2, 10240, 4, 128, [3000, 4000, 3240], True, None, id='cp2-state-v-first'),
        pytest.param(4, 10240, 4, 128, [10240], True, None, id='cp4-state-v-first'),
    ],
)
def test_chunk(run_distributed, world_size, T, H, D, lengths, state_v_first, Hq):
    run_distributed(
        _run_chunk,
        world_size=world_size,
        T=T,
        H=H,
        D=D,
        lengths=lengths,
        dtype=torch.bfloat16,
        state_v_first=state_v_first,
        Hq=Hq,
    )
