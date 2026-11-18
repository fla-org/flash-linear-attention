# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.utils import softmax_bwd, softmax_fwd
from fla.utils import assert_close, device


def test_softmax_noncontiguous_transpose():
    x = torch.tensor([[0.0, 2.0], [1.0, 3.0]], device=device).t()
    p = x.softmax(-1)
    dp = torch.tensor([[1.0, 3.0], [2.0, 4.0]], device=device).t()

    got_p = softmax_fwd(x)
    ref_p = x.softmax(-1)
    assert got_p.is_contiguous()
    torch.testing.assert_close(got_p, ref_p, atol=1e-6, rtol=1e-5)

    got_ds = softmax_bwd(p, dp)
    ref_ds = p * (dp - (p * dp).sum(-1, keepdim=True))
    assert got_ds.is_contiguous()
    torch.testing.assert_close(got_ds, ref_ds, atol=1e-6, rtol=1e-5)

    got_ds_both_nc = softmax_bwd(x.softmax(-1).t().contiguous().t(), dp)
    torch.testing.assert_close(got_ds_both_nc, ref_ds, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    'layout',
    ['contiguous', 'transpose_2d', 'transpose_batch', 'transpose_last', 'strided_slice', 'expanded_dp'],
)
def test_softmax_fwd_bwd_layouts(layout: str, dtype: torch.dtype):
    torch.manual_seed(42)
    if layout == 'contiguous':
        x = torch.randn(2, 7, 129, device=device, dtype=dtype)
        dp = torch.randn_like(x)
    elif layout == 'transpose_2d':
        x = torch.randn(65, 33, device=device, dtype=dtype).t()
        dp = torch.randn(65, 33, device=device, dtype=dtype).t()
    elif layout == 'transpose_batch':
        x = torch.randn(5, 3, 97, device=device, dtype=dtype).transpose(0, 1)
        dp = torch.randn(5, 3, 97, device=device, dtype=dtype).transpose(0, 1)
    elif layout == 'transpose_last':
        x = torch.randn(2, 65, 19, device=device, dtype=dtype).transpose(-1, -2)
        dp = torch.randn(2, 65, 19, device=device, dtype=dtype).transpose(-1, -2)
    elif layout == 'strided_slice':
        base_x = torch.randn(6, 10, 260, device=device, dtype=dtype)
        base_dp = torch.randn(6, 10, 260, device=device, dtype=dtype)
        x = base_x[::2, 1::2, ::2]
        dp = base_dp[1::2, ::2, 1::2]
    elif layout == 'expanded_dp':
        x = torch.randn(3, 4, 71, device=device, dtype=dtype).transpose(0, 1)
        dp = torch.randn(1, 3, 71, device=device, dtype=dtype).expand_as(x)
    else:
        raise ValueError(f'Unknown layout: {layout}')

    ref_p = x.float().softmax(-1)
    tri_p = softmax_fwd(x, dtype=torch.float32)
    assert tri_p.shape == x.shape
    assert tri_p.dtype == torch.float32
    assert tri_p.is_contiguous()
    assert_close('p_fp32', ref_p, tri_p, 1e-3)

    tri_p_same = softmax_fwd(x, dtype=None)
    assert tri_p_same.dtype == dtype
    assert tri_p_same.is_contiguous()
    assert_close('p_same', ref_p.to(dtype), tri_p_same, 1e-2)

    p_in = ref_p.to(dtype)
    if layout in ('transpose_2d', 'transpose_last'):
        p_in = p_in.t().contiguous().t() if p_in.ndim == 2 else p_in.transpose(-1, -2).contiguous().transpose(-1, -2)
    ref_ds = (p_in.float() * (dp.float() - (p_in.float() * dp.float()).sum(-1, keepdim=True))).to(dtype)
    tri_ds = softmax_bwd(p_in, dp, dtype=None)
    assert tri_ds.shape == p_in.shape
    assert tri_ds.dtype == dtype
    assert tri_ds.is_contiguous()
    assert_close('ds', ref_ds, tri_ds, 1e-2)
