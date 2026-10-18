# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare CP token shift outputs and gradients across packed sequence boundaries."""


import pytest
import torch
import torch.distributed as dist

from fla.modules.token_shift import token_shift_ref
from fla.modules.token_shift_cp import token_shift_cp
from fla.ops.cp import build_cp_context
from fla.utils import assert_close, device
from tests.cp.utils import gather_global, run_distributed

pytestmark = pytest.mark.skipif(device != "cuda", reason="requires CUDA or ROCm GPUs")


def run_cp_token_shift_test_worker(
    rank: int,
    world_size: int,
    test_name: str,
    T: int,
    D: int,
    lengths: list[int],
    dtype,
):
    """Worker function for CP token shift test."""
    device = torch.device(f'cuda:{rank}')

    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
    assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"Config: T={T}, D={D}, world_size={world_size}")
        print(f"Sequence lengths: {lengths}")
        print(f"{'='*60}")

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

    print(f"[Rank {rank}] chunk: [{start_idx}, {end_idx}), "
          f"cu_seqlens: {context.cu_seqlens.tolist()}, "
          f"pre_num_ranks: {context.pre_num_ranks}")
    dist.barrier()

    # CP Forward
    y_local = token_shift_cp(
        x=x_local,
        cp_context=context,
        cu_seqlens=context.cu_seqlens,
    )

    # CP Backward
    y_local.backward(dy_local)

    y_cp_global = gather_global(y_local, world_size=world_size)

    dx_cp_global = gather_global(x_local.grad, world_size=world_size)

    test_passed = True
    if rank == 0:
        print(f"\n[{test_name}] Verification Results:")
        try:
            assert_close("Output", ref_out, y_cp_global, ratio=0.001)
            assert_close("dx", ref_dx, dx_cp_global, ratio=0.001)
            print(f"✅ [{test_name}] Test Passed!\n")
        except AssertionError as e:
            print(f"❌ [{test_name}] Test Failed: {e}\n")
            test_passed = False

    dist.barrier()

    if not test_passed:
        raise AssertionError(f"Test {test_name} failed on rank {rank}")


@pytest.mark.parametrize(
    ('world_size', 'lengths'),
    [
        pytest.param(2, [300, 400, 324], id='CP2_SequenceCut'),
        pytest.param(2, [512, 512], id='CP2_BoundaryAligned'),
        pytest.param(4, [700, 324], id='CP4_Complex'),
        pytest.param(4, [1024], id='CP4_SingleSequence'),
        pytest.param(2, [100, 150, 200, 250, 124, 100, 100], id='CP2_ManyShortSequences'),
        pytest.param(2, [513, 511], id='CP2_ShortTail_Len1'),
        pytest.param(2, [514, 510], id='CP2_ShortTail_Len2'),
    ],
)
def test_token_shift(request, world_size, lengths):
    run_distributed(
        worker=run_cp_token_shift_test_worker,
        world_size=world_size,
        T=1024,
        D=128,
        lengths=lengths,
        dtype=torch.float32,
        test_name=request.node.callspec.id,
    )
