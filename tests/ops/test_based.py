# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.based import fused_chunk_based, parallel_based
from fla.ops.based.naive import naive_parallel_based
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 60, torch.float16),
            (3, 111, 2, 64, torch.float16),
            (3, 1024, 4, 100, torch.float16),
            (3, 1024, 8, 128, torch.float16),
            (4, 2048, 8, 256, torch.float16),
        ]
    ],
)
@pytest.mark.smoke
def test_based(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    # the ops take [B, T, H, D]; the naive reference attends over dim -2, so it gets [B, H, T, D]
    q = torch.randn((B, T, H, 16), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, 16), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn_like(v)
    # the reference runs in fp32 so the tolerances below measure the kernels' error
    ref = naive_parallel_based(*(x.float().transpose(1, 2) for x in (q, k, v)), use_norm=True).transpose(1, 2)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    for fn in (parallel_based, fused_chunk_based):
        tri = fn(q, k, v, use_norm=True)
        tri.backward(do)
        tri_dq, q.grad = q.grad.clone(), None
        tri_dk, k.grad = k.grad.clone(), None
        tri_dv, v.grad = v.grad.clone(), None

        assert_close(f'{fn.__name__} o', ref, tri, 0.005)
        assert_close(f'{fn.__name__} dq', ref_dq, tri_dq, 0.005)
        assert_close(f'{fn.__name__} dk', ref_dk, tri_dk, 0.005)
        assert_close(f'{fn.__name__} dv', ref_dv, tri_dv, 0.005)
