# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.abc import chunk_abc
from fla.ops.abc.naive import naive_recurrent_abc
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'M', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-M{}-{}".format(*test))
        for test in [
            (1, 1, 1, 32, 16, torch.float32),
            (2, 64, 2, 32, 16, torch.float32),
            # T not a multiple of the chunk size, so the last chunk is partial
            (2, 100, 2, 32, 32, torch.float32),
            (2, 130, 2, 64, 64, torch.float16),
        ]
    ],
)
def test_chunk_abc(B: int, T: int, H: int, D: int, M: int, dtype: torch.dtype):
    torch.manual_seed(42)
    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    s = torch.randn((B, T, H, M), dtype=dtype, device=device).requires_grad_()
    do = torch.randn_like(v)

    ref, (ref_hk, ref_hv) = naive_recurrent_abc(
        *(x.float().transpose(1, 2) for x in (q, k, v, s)),
        output_final_state=True,
    )
    ref = ref.transpose(1, 2)
    ref.backward(do.float())
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None

    tri, (tri_hk, tri_hv) = chunk_abc(q, k, v, s, output_final_state=True)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_ds, s.grad = s.grad.clone(), None

    assert_close('  o', ref, tri, 0.005)
    assert_close(' hk', ref_hk, tri_hk, 0.005)
    assert_close(' hv', ref_hv, tri_hv, 0.005)
    assert_close(' dq', ref_dq, tri_dq, 0.005)
    assert_close(' dk', ref_dk, tri_dk, 0.005)
    assert_close(' dv', ref_dv, tri_dv, 0.005)
    assert_close(' ds', ref_ds, tri_ds, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'M'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-M{}".format(*test))
        for test in [
            (1, 1, 1, 32, 16),
            (2, 77, 2, 32, 64),
            (2, 128, 2, 32, 32),
        ]
    ],
)
def test_chunk_abc_initial_state(B: int, T: int, H: int, D: int, M: int):
    torch.manual_seed(42)
    q, k, v = (torch.randn((B, T, H, D), device=device) for _ in range(3))
    s = torch.randn((B, T, H, M), device=device)
    h0 = (torch.randn((B, H, D, M), device=device), torch.randn((B, H, M, D), device=device))

    ref, (ref_hk, ref_hv) = naive_recurrent_abc(
        *(x.transpose(1, 2) for x in (q, k, v, s)),
        initial_state=h0,
        output_final_state=True,
    )
    tri, (tri_hk, tri_hv) = chunk_abc(q, k, v, s, initial_state=h0, output_final_state=True)

    assert_close('  o', ref.transpose(1, 2), tri, 0.005)
    assert_close(' hk', ref_hk, tri_hk, 0.005)
    assert_close(' hv', ref_hv, tri_hv, 0.005)
