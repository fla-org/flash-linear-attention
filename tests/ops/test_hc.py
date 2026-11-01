# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.hc import naive_hc_post, naive_hc_pre
from fla.utils import assert_close


@pytest.mark.parametrize(
    ('shape', 'dtype', 'ratio'),
    [
        pytest.param((1, 1), torch.float32, 5e-5, id='single-stream'),
        pytest.param((2, 3, 3, 65), torch.float16, 2e-3, id='fp16-dense'),
        pytest.param((7, 5, 17), torch.bfloat16, 6e-3, id='bf16-packed'),
        pytest.param((3, 7), torch.float32, 5e-5, id='fp32-unbatched'),
    ],
)
def test_naive_hc_pre(shape: tuple[int, ...], dtype: torch.dtype, ratio: float):
    torch.manual_seed(42)
    x = torch.randn(*shape[:-2], shape[-1], shape[-2], dtype=dtype).transpose(-1, -2).requires_grad_()
    pre = torch.randn(shape[:-1], dtype=torch.float32, requires_grad=True)
    ref = (x.float() * pre.float().unsqueeze(-1)).sum(-2).to(dtype)
    out = naive_hc_pre(x=x, pre=pre)
    do = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=(x, pre), grad_outputs=do)
    grads = torch.autograd.grad(outputs=out, inputs=(x, pre), grad_outputs=do)
    assert out.shape == ref.shape and out.dtype == dtype
    assert_close('o', ref, out, ratio)
    for name, tensor, ref_grad, grad in zip(('dx', 'dpre'), (x, pre), ref_grads, grads):
        assert grad.dtype == tensor.dtype
        assert_close(name, ref_grad, grad, ratio)


@pytest.mark.parametrize(
    ('shape', 'dtype', 'ratio', 'use_mixing'),
    [
        pytest.param((1, 1), torch.float32, 5e-5, False, id='ihc-single-stream'),
        pytest.param((2, 3, 3, 65), torch.float16, 2e-3, True, id='mhc-fp16-dense'),
        pytest.param((7, 5, 17), torch.bfloat16, 6e-3, False, id='ihc-bf16-packed'),
        pytest.param((3, 7), torch.float32, 5e-5, True, id='mhc-fp32-unbatched'),
        pytest.param((7, 5, 17), torch.bfloat16, 6e-3, True, id='mhc-bf16-packed'),
    ],
)
def test_naive_hc_post(shape: tuple[int, ...], dtype: torch.dtype, ratio: float, use_mixing: bool):
    torch.manual_seed(42)
    x = torch.randn(*shape[:-2], shape[-1], shape[-2], dtype=dtype).transpose(-1, -2).requires_grad_()
    branch = torch.randn((*shape[:-2], shape[-1]), dtype=dtype, requires_grad=True)
    post = torch.randn(shape[:-1], dtype=torch.float32, requires_grad=True)
    mixing = torch.randn((*shape[:-1], shape[-2]), dtype=torch.float32, requires_grad=True) if use_mixing else None
    inputs = (x, branch, post, mixing) if use_mixing else (x, branch, post)
    ref = x.float()
    if use_mixing:
        ref = (mixing.float().unsqueeze(-1) * ref.unsqueeze(-3)).sum(-2)
    ref = (ref + post.float().unsqueeze(-1) * branch.float().unsqueeze(-2)).to(dtype)
    out = naive_hc_post(x=x, branch=branch, post=post, mixing=mixing)
    do = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=inputs, grad_outputs=do)
    grads = torch.autograd.grad(outputs=out, inputs=inputs, grad_outputs=do)
    assert out.shape == ref.shape and out.dtype == dtype
    assert_close('o', ref, out, ratio)
    for name, tensor, ref_grad, grad in zip(('dx', 'dbranch', 'dpost', 'dmixing'), inputs, ref_grads, grads):
        assert grad.dtype == tensor.dtype
        assert_close(name, ref_grad, grad, ratio)
