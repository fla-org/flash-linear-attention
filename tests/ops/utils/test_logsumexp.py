# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.utils import logsumexp_fwd
from fla.utils import assert_close, device


def test_logsumexp_noncontiguous_transpose():
    x = torch.tensor([[0.0, 2.0], [1.0, 3.0]], device=device).t()
    got = logsumexp_fwd(x)
    expected = x.logsumexp(-1)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize('scale', [None, 0.5, -0.5])
@pytest.mark.parametrize(
    'layout',
    ['transpose_2d', 'transpose_batch', 'transpose_last', 'strided_slice', 'expanded'],
)
def test_logsumexp_noncontiguous_layouts(layout: str, scale: float | None, dtype: torch.dtype):
    torch.manual_seed(42)
    if layout == 'transpose_2d':
        x = torch.randn(65, 33, device=device, dtype=dtype).t()
    elif layout == 'transpose_batch':
        x = torch.randn(5, 3, 1025, device=device, dtype=dtype).transpose(0, 1)
    elif layout == 'transpose_last':
        x = torch.randn(2, 129, 17, device=device, dtype=dtype).transpose(-1, -2)
    elif layout == 'strided_slice':
        x = torch.randn(4, 8, 2050, device=device, dtype=dtype)[::2, 1::2, ::2]
    elif layout == 'expanded':
        x = torch.randn(1, 3, 97, device=device, dtype=dtype).expand(4, 3, 97)
    else:
        raise ValueError(f'Unknown layout: {layout}')

    scaled = x.float() * scale if scale is not None else x.float()
    ref = scaled.logsumexp(-1)
    tri = logsumexp_fwd(x, scale=scale)
    assert tri.shape == x.shape[:-1]
    assert tri.dtype == torch.float32
    assert_close('lse', ref, tri, 1e-3)

    tri_cast = logsumexp_fwd(x, scale=scale, dtype=dtype)
    assert tri_cast.dtype == dtype
    assert_close('lse_cast', ref.to(dtype), tri_cast, 1e-2)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=str)
def test_logsumexp_infinite_tile_maxima(dtype: torch.dtype):
    neg_inf = float('-inf')
    pos_inf = float('inf')
    cases = [
        torch.tensor([[neg_inf]], device=device, dtype=dtype),
        torch.tensor([[pos_inf]], device=device, dtype=dtype),
        torch.full((3, 129), neg_inf, device=device, dtype=dtype),
        torch.tensor([[pos_inf, 0.0, -1.0], [neg_inf, neg_inf, 2.0], [pos_inf, pos_inf, neg_inf]], device=device, dtype=dtype),
    ]
    for x in cases:
        got = logsumexp_fwd(x)
        expected = x.float().logsumexp(-1)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    # Negative scale flips -inf <-> +inf
    x_scaled = torch.tensor([[neg_inf, -2.0], [pos_inf, pos_inf]], device=device, dtype=dtype)
    got_scaled = logsumexp_fwd(x_scaled, scale=-0.5)
    expected_scaled = (x_scaled.float() * -0.5).logsumexp(-1)
    torch.testing.assert_close(got_scaled, expected_scaled, atol=1e-5, rtol=1e-5)


def test_logsumexp_multi_tile_infinite_and_nan():
    neg_inf = float('-inf')
    pos_inf = float('inf')
    nan = float('nan')
    # Pack all multi-tile (D = 65537 -> 2 tiles of 65536 + 1) edge cases into one launch shape
    row0 = torch.cat([torch.full((1, 65536), neg_inf, device=device), torch.zeros((1, 1), device=device)], dim=-1)
    row1 = torch.cat([torch.zeros((1, 65536), device=device), torch.full((1, 1), neg_inf, device=device)], dim=-1)
    row2 = torch.full((1, 65537), neg_inf, device=device)
    row3 = torch.full((1, 65537), pos_inf, device=device)
    x_multi = torch.cat([row0, row1, row2, row3], dim=0)
    got_multi = logsumexp_fwd(x_multi)
    expected_multi = x_multi.logsumexp(-1)
    torch.testing.assert_close(got_multi, expected_multi, atol=1e-5, rtol=1e-5)

    x_nan = torch.tensor(
        [
            [pos_inf, nan, 0.0],
            [nan, pos_inf, 0.0],
            [neg_inf, nan, neg_inf],
            [0.0, nan, -1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    assert logsumexp_fwd(x_nan).isnan().all()
    assert logsumexp_fwd(torch.tensor([[pos_inf]], device=device), scale=0.0).isnan().all()
