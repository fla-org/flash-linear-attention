# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import pytest
import torch
import torch.distributed as dist

from cp.utils import gather_global
from fla.modules.token_shift import token_shift_ref
from fla.modules.token_shift_cp import token_shift_cp
from fla.ops.cp import build_cp_context
from fla.utils import assert_close
from fla.utils import device as device_type


def _run_chunk(
    rank: int,
    world_size: int,
    T: int,
    D: int,
    lengths: list[int],
    dtype,
):
    device = torch.device(f'{device_type}:{rank}')

    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
    assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

    torch.manual_seed(42)
    B = 1

    x_global = torch.randn(B, T, D, device=device, dtype=dtype) * 10
    dy_global = torch.randn(B, T, D, device=device, dtype=dtype)

    cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
    cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.int32)

    ref_out, ref_dx = None, None

    if rank == 0:
        x_ref = x_global.clone().detach().requires_grad_(True)
        y_ref = token_shift_ref(x_ref, cu_seqlens=cu_seqlens_global)
        y_ref.backward(dy_global)
        ref_out = y_ref.detach()
        ref_dx = x_ref.grad.detach()

    dist.barrier()

    context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)

    chunk_size = T // world_size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size

    x_local = x_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    dy_local = dy_global[:, start_idx:end_idx, :].clone()

    dist.barrier()

    y_local = token_shift_cp(
        x=x_local,
        cp_context=context,
        cu_seqlens=context.cu_seqlens,
    )

    y_local.backward(dy_local)

    y_cp_global = gather_global(y_local, world_size=world_size)

    dx_cp_global = gather_global(x_local.grad, world_size=world_size)

    test_passed = True
    if rank == 0:
        try:
            assert_close("Output", ref_out, y_cp_global, ratio=0.001)
            assert_close("dx", ref_dx, dx_cp_global, ratio=0.001)
        except AssertionError:
            test_passed = False

    dist.barrier()

    if not test_passed:
        raise AssertionError(f"CP comparison failed on rank {rank}")


@pytest.mark.parametrize(
    ('world_size', 'lengths'),
    [
        pytest.param(2, [300, 400, 324], id='cp2-sequence-cut'),
        pytest.param(2, [512, 512], id='cp2-boundary-aligned'),
        pytest.param(4, [700, 324], id='cp4-complex'),
        pytest.param(4, [1024], id='cp4-single-sequence'),
        pytest.param(2, [100, 150, 200, 250, 124, 100, 100], id='cp2-many-short-sequences'),
        pytest.param(2, [513, 511], id='cp2-short-tail-len1'),
        pytest.param(2, [514, 510], id='cp2-short-tail-len2'),
    ],
)
def test_token_shift(run_distributed, world_size, lengths):
    run_distributed(
        _run_chunk,
        world_size=world_size,
        T=1024,
        D=128,
        lengths=lengths,
        dtype=torch.float32,
    )
