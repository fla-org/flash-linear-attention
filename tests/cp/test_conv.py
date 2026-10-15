# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import pytest
import torch
import torch.distributed as dist

from cp.utils import gather_global, slice_local
from fla.modules.convolution import causal_conv1d
from fla.ops.cp import build_cp_context
from fla.utils import assert_close
from fla.utils import device as device_type


def _run_chunk(
    rank: int,
    world_size: int,
    T: int,
    D: int,
    W: int,
    lengths: list[int],
    dtype,
    layout: str = 'contiguous',
):
    device = torch.device(f'{device_type}:{rank}')

    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
    if layout == 'zigzag':
        assert T % (2 * world_size) == 0, f"T={T} must be divisible by 2*world_size={2 * world_size} for zigzag"
    assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

    torch.manual_seed(42)
    B = 1

    x_global = torch.randn(B, T, D, device=device, dtype=dtype) * 100
    dy_global = torch.randn(B, T, D, device=device, dtype=dtype)

    weight = torch.randn(D, W, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype)

    dist.broadcast(weight, src=0)
    dist.broadcast(bias, src=0)

    cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
    cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.int32)

    activation = 'swish'

    ref_out, ref_dx, ref_dw, ref_db = None, None, None, None

    if rank == 0:
        x_ref = x_global.clone().detach().requires_grad_(True)
        weight_ref = weight.clone().detach().requires_grad_(True)
        bias_ref = bias.clone().detach().requires_grad_(True)

        y_ref, _ = causal_conv1d(
            x=x_ref,
            weight=weight_ref,
            bias=bias_ref,
            activation=activation,
            backend='triton',
            cu_seqlens=cu_seqlens_global,
        )

        y_ref.backward(dy_global)

        ref_out = y_ref.detach()
        ref_dx = x_ref.grad.detach()
        ref_dw = weight_ref.grad.detach()
        ref_db = bias_ref.grad.detach()

    dist.barrier()

    context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD, conv1d_kernel_size=W, layout=layout)

    x_local = slice_local(x_global, rank=rank, world_size=world_size, layout=layout).clone().detach().requires_grad_(True)
    dy_local = slice_local(dy_global, rank=rank, world_size=world_size, layout=layout).clone()
    weight_local = weight.clone().detach().requires_grad_(True)
    bias_local = bias.clone().detach().requires_grad_(True)

    dist.barrier()

    y_local, _ = causal_conv1d(
        x=x_local,
        weight=weight_local,
        bias=bias_local,
        activation=activation,
        cp_context=context,
    )

    y_local.backward(dy_local)

    y_cp_global = gather_global(y_local, world_size=world_size, layout=layout)
    dx_cp_global = gather_global(x_local.grad, world_size=world_size, layout=layout)

    dw_cp = weight_local.grad.clone()
    db_cp = bias_local.grad.clone()
    dist.all_reduce(dw_cp, op=dist.ReduceOp.SUM)
    dist.all_reduce(db_cp, op=dist.ReduceOp.SUM)

    test_passed = True
    if rank == 0:
        try:
            assert_close("Output", ref_out, y_cp_global, ratio=0.001)
            assert_close("dx", ref_dx, dx_cp_global, ratio=0.001)
            assert_close("dw", ref_dw, dw_cp, ratio=0.001)
            assert_close("db", ref_db, db_cp, ratio=0.001)
        except AssertionError:
            test_passed = False

    dist.barrier()

    if not test_passed:
        raise AssertionError(f"CP comparison failed on rank {rank}")


@pytest.mark.parametrize(
    ('world_size', 'lengths', 'layout'),
    [
        pytest.param(2, [1024], 'zigzag', id='cp2-zigzag-single-sequence'),
        pytest.param(2, [300, 400, 324], 'zigzag', id='cp2-zigzag-sequence-cut'),
        pytest.param(4, [150, 260, 300, 314], 'zigzag', id='cp4-zigzag-complex'),
        pytest.param(4, [256, 256, 256, 256], 'zigzag', id='cp4-zigzag-boundary-aligned'),
        pytest.param(2, [300, 400, 324], 'contiguous', id='cp2-sequence-cut'),
        pytest.param(2, [512, 512], 'contiguous', id='cp2-boundary-aligned'),
        pytest.param(4, [700, 324], 'contiguous', id='cp4-complex'),
        pytest.param(4, [1024], 'contiguous', id='cp4-single-sequence'),
        pytest.param(2, [100, 150, 200, 250, 124, 100, 100], 'contiguous', id='cp2-many-short-sequences'),
    ],
)
def test_causal_conv1d(run_distributed, world_size, lengths, layout):
    run_distributed(
        _run_chunk,
        world_size=world_size,
        T=1024,
        D=128,
        W=4,
        lengths=lengths,
        dtype=torch.float32,
        layout=layout,
    )


@pytest.mark.parametrize(
    ('world_size', 'lengths'),
    [
        pytest.param(2, [513, 511], id='cp2-short-tail-len1'),
        pytest.param(2, [514, 510], id='cp2-short-tail-len2'),
        pytest.param(4, [257, 255, 257, 255], id='cp4-every-rank-gets-short-tail'),
        pytest.param(2, [200, 313, 511], id='cp2-multiple-short-tails'),
        pytest.param(2, [512, 1, 511], id='cp2-global-len1-sequence'),
        pytest.param(4, [257, 253, 257, 257], id='cp4-multiple-short-tails'),
        pytest.param(4, [257, 1, 253, 257, 1, 253, 2], id='cp4-worst-case-many-len1'),
    ],
)
def test_causal_conv1d_short_sequence(run_distributed, world_size, lengths):
    run_distributed(
        _run_chunk,
        world_size=world_size,
        T=1024,
        D=128,
        W=4,
        lengths=lengths,
        dtype=torch.float32,
        layout='contiguous',
    )
