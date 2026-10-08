# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.modules.convolution import causal_conv1d
from fla.modules.l2norm import l2_norm
from fla.ops.convolution.fused_short_conv import fused_short_conv
from fla.ops.utils import prepare_chunk_indices
from fla.utils import assert_close, device


@pytest.mark.parametrize('W', [4, 17])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_fused_short_conv_chunk_indices(W, dtype):
    torch.manual_seed(42)
    lengths = [0, 1, 17, 0, 63, 65, 0]
    H, head_dim, T = 2, 80, sum(lengths)
    D = H * head_dim
    cu_seqlens = torch.tensor([0, *lengths], device=device, dtype=torch.long).cumsum(0)
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    x = torch.randn(1, T, D, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(D, W, device=device, dtype=dtype, requires_grad=True)
    initial_state = torch.randn(len(lengths), D, W, device=device, dtype=dtype, requires_grad=True)
    kwargs = dict(
        x=x,
        weight=weight,
        initial_state=initial_state,
        output_final_state=True,
        activation='silu',
        cu_seqlens=cu_seqlens,
    )
    ref, ref_state = causal_conv1d(backend='triton', **kwargs)
    ref = l2_norm(x=ref.view(1, T, H, head_dim)).view_as(x)
    actual, actual_state = fused_short_conv(
        chunk_indices=chunk_indices,
        use_norm=True,
        head_dim=head_dim,
        **kwargs,
    )
    assert torch.isfinite(actual).all()
    assert torch.isfinite(actual_state).all()
    assert_close('y', ref, actual, 1e-3)
    assert_close('state', ref_state, actual_state, 1e-3)

    dy = torch.randn_like(ref)
    dht = torch.randn_like(ref_state)
    inputs = (x, weight, initial_state)
    ref_grads = torch.autograd.grad((ref, ref_state), inputs, grad_outputs=(dy, dht))
    actual_grads = torch.autograd.grad((actual, actual_state), inputs, grad_outputs=(dy, dht))
    for name, ref_grad, actual_grad in zip(('dx', 'dw', 'dh0'), ref_grads, actual_grads, strict=True):
        assert torch.isfinite(actual_grad).all(), name
        assert_close(name, ref_grad, actual_grad, 1e-3)
