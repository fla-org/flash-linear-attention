# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.decay_max_scan import chunk_decay_max_scan, fused_recurrent_decay_max_scan
from fla.ops.decay_max_scan.naive import naive_decay_max_scan
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'H', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-D{}-H{}-{}".format(*test))
        for test in [
            (1, 63, 32, 2, torch.float32),
            (2, 200, 64, 4, torch.float32),
            (2, 1000, 64, 4, torch.float32),
            (2, 500, 48, 8, torch.float16),
        ]
    ],
)
def test_chunk(B: int, T: int, D: int, H: int, dtype: torch.dtype):
    torch.manual_seed(42)
    x = torch.randn(B, T, D, H, dtype=dtype).to(device).requires_grad_(True)
    v = (torch.randn(D, dtype=dtype) * 0.05).to(device).requires_grad_(True)
    b = (torch.randn(D, dtype=dtype) * 0.05).to(device).requires_grad_(True)
    do = torch.rand_like(x, dtype=torch.float32)

    tri, tri_s = chunk_decay_max_scan(x=x.clone(), decay_v=v.clone(), decay_b=b.clone())
    (tri * do).sum().backward()
    tri_dx, tri_dv, tri_db = x.grad, v.grad, b.grad
    x.grad = v.grad = b.grad = None

    ref, ref_s = naive_decay_max_scan(x=x.clone(), decay_v=v.clone(), decay_b=b.clone())
    (ref * do).sum().backward()
    ref_dx, ref_dv, ref_db = x.grad, v.grad, b.grad

    assert_close('o', ref, tri, 0.006)
    assert_close('state', ref_s, tri_s, 0.006)
    assert_close('dx', ref_dx, tri_dx, 0.006)
    assert_close('dv', ref_dv, tri_dv, 0.006)
    assert_close('db', ref_db, tri_db, 0.006)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'H'),
    [
        (2, 63, 32, 2),
        (2, 500, 64, 4),
    ],
)
def test_chunk_varlen(B: int, T: int, D: int, H: int):
    torch.manual_seed(42)
    x = torch.randn(B, T, D, H).to(device).requires_grad_(True)
    v = (torch.randn(D) * 0.05).to(device).requires_grad_(True)
    b = (torch.randn(D) * 0.05).to(device).requires_grad_(True)
    cu = torch.tensor([0, 40, 100, T], device=device)

    tri, _ = chunk_decay_max_scan(x=x.clone(), decay_v=v.clone(), decay_b=b.clone(), cu_seqlens=cu)
    do = torch.rand_like(tri)
    (tri * do).sum().backward()
    tri_dx = x.grad
    x.grad = None

    ref = torch.empty_like(tri)
    for i in range(len(cu) - 1):
        ref[:, cu[i]:cu[i + 1]], _ = naive_decay_max_scan(
            x=x[:, cu[i]:cu[i + 1]].clone(), decay_v=v.clone(), decay_b=b.clone())
    (ref * do).sum().backward()

    assert_close('o', ref, tri, 0.006)
    assert_close('dx', x.grad, tri_dx, 0.006)


@pytest.mark.parametrize(('B', 'D', 'H'), [(2, 32, 2), (2, 48, 8)])
def test_fused_recurrent(B: int, D: int, H: int):
    torch.manual_seed(42)
    T = 50
    x = torch.randn(B, T, D, H).to(device)
    v = (torch.randn(D) * 0.05).to(device)
    b = (torch.randn(D) * 0.05).to(device)

    ref, ref_s = naive_decay_max_scan(x=x, decay_v=v, decay_b=b)
    s = None
    for t in range(T):
        e_t, s = fused_recurrent_decay_max_scan(x=x[:, t:t + 1], decay_v=v, decay_b=b, initial_state=s)
        assert_close(f'o_t{t}', ref[:, t], e_t[:, 0], 0.006)
    assert_close('state', ref_s, s, 0.006)
