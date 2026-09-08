# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from itertools import product

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.utils import IS_INTEL, IS_NPU, assert_close, device, device_platform


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [512, 1024])
@pytest.mark.parametrize("V", [32000, 100000])
@pytest.mark.parametrize("softcap", [None, 30.0], ids=["plain", "softcap"])
@pytest.mark.parametrize("reduction", ['mean'])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure",
)
def test_fused_cross_entropy(
    B: int,
    T: int,
    V: int,
    softcap: float | None,
    reduction: str,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    logits = torch.randn(B * T, V).to(device).to(dtype=dtype).requires_grad_()
    target = torch.randint(0, V, (B, T)).to(device)
    target = torch.cat((target[..., 1:], torch.full_like(target[..., :1], -100)), -1)
    target = target.flatten()

    transformed = logits if softcap is None else softcap * torch.tanh(logits.float() / softcap)
    ref = nn.CrossEntropyLoss(reduction=reduction)(transformed, target).to(dtype=dtype)
    do = torch.randn_like(ref).to(device).to(dtype=dtype)

    ref.backward(do)
    ref_d, logits.grad = logits.grad.clone(), None

    tri = FusedCrossEntropyLoss(reduction=reduction, logit_softcapping=softcap)(logits, target).to(dtype=dtype)
    tri.backward(do)
    tri_d, logits.grad = logits.grad.clone(), None

    assert_close(" o", ref, tri, ratio=1e-2)
    assert_close("dl", ref_d, tri_d, ratio=1e-2)


@pytest.mark.parametrize(
    ('V', 'scale', 'softcap', 'z_scale', 'strided', 'inplace_backward', 'reduction', 'dtype'),
    [
        pytest.param(
            V, 0.3, 3.0, 0.01, True, inplace, reduction, dtype,
            id=f'V{V}-transformed-{inplace=}-{reduction}-{dtype}',
        )
        for V, inplace, reduction, dtype in product(
            (4103, 65539), (False, True), ('mean', 'sum', 'none'), (torch.bfloat16, torch.float16, torch.float32),
        )
    ] + [
        pytest.param(V, scale, softcap, 0.0, False, False, 'mean', dtype, id=f'V{V}-{scale=}-{softcap=}-{dtype}')
        for V, scale, softcap, dtype in product((4103, 65537), (0.0, -0.5), (None, 3.0), (torch.bfloat16, torch.float32))
    ],
)
@pytest.mark.skipif(IS_INTEL or IS_NPU, reason="Covers the default Triton GPU kernels")
def test_fused_cross_entropy_options(V, scale, softcap, z_scale, strided, inplace_backward, reduction, dtype):
    torch.manual_seed(42)
    if strided:
        logits = torch.randn(2, 7, V, device=device, dtype=dtype).transpose(0, 1).reshape(7, 2 * V)[:, ::2]
    else:
        logits = torch.randn(7, V, device=device, dtype=dtype)
    logits = logits.detach().requires_grad_()
    target = torch.randint(V, (7,), device=device)
    target[::3] = -100
    target[1] = V - 1
    transformed = logits.float() * scale
    if softcap is not None:
        transformed = softcap * torch.tanh(transformed / softcap)
    z_loss = z_scale * transformed.logsumexp(-1).square()
    z_loss = z_loss.masked_fill(target == -100, 0)
    ref = F.cross_entropy(transformed, target, reduction='none', label_smoothing=0.1) + z_loss
    if reduction == 'mean':
        ref, z_loss = ref.sum() / (target != -100).sum(), z_loss.sum() / (target != -100).sum()
    elif reduction == 'sum':
        ref, z_loss = ref.sum(), z_loss.sum()
    do = torch.randn_like(ref) if strided else torch.ones_like(ref)
    ref_grad, = torch.autograd.grad(ref, logits, grad_outputs=do)
    tri, tri_z_loss = FusedCrossEntropyLoss(
        reduction=reduction,
        label_smoothing=0.1,
        logit_scale=scale,
        lse_square_scale=z_scale,
        logit_softcapping=softcap,
        inplace_backward=inplace_backward,
        return_z_loss=True,
    )(logits, target)
    tri_grad, = torch.autograd.grad(tri, logits, grad_outputs=do)

    assert not tri_z_loss.requires_grad
    assert_close("loss", ref, tri, ratio=1e-2)
    assert_close("z_loss", z_loss, tri_z_loss, ratio=1e-2)
    assert_close("dlogits", ref_grad, tri_grad, ratio=1e-2)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'V', 'smoothing', 'scale', 'softcap', 'reduction',
     'with_bias', 'strided', 'ignore_all', 'accumulate_grad_in_fp32', 'dtype'),
    [
        pytest.param(
            2, T, D, V, 0.0, scale, softcap, 'mean', True, False, False, fp32_grad, torch.bfloat16,
            id=f'T{T}-D{D}-V{V}-{scale=}-{softcap=}-{fp32_grad=}',
        )
        for T, D, V, (scale, softcap, fp32_grad) in product(
            (512, 1024), (1024, 2048), (32000, 100000),
            ((1.0, None, False), (0.5, None, False), (1.0, None, True), (0.5, None, True), (1.0, 30.0, True)),
        )
    ] + [
        pytest.param(
            3, 7, 32, V, smoothing, scale, softcap, reduction, with_bias, True, ignore_all, True, dtype,
            id=f'V{V}-{smoothing=}-{scale=}-{softcap=}-{with_bias=}-{reduction}-{dtype}',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        )
        for (V, smoothing, scale, softcap, ignore_all), with_bias, reduction, dtype in product(
            ((4103, 0.1, 0.3, None, False), (65539, 0.1, 0.5, 3.0, False), (129, 0.0, 1.0, None, True)),
            (False, True), ('mean', 'sum'), (torch.bfloat16, torch.float16, torch.float32),
        )
    ],
)
@pytest.mark.skipif(IS_INTEL, reason="Intel Triton Failure")
def test_fused_linear_cross_entropy(
    B: int,
    T: int,
    D: int,
    V: int,
    smoothing: float,
    scale: float,
    softcap: float | None,
    reduction: str,
    with_bias: bool,
    strided: bool,
    ignore_all: bool,
    accumulate_grad_in_fp32: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    if strided:
        x = torch.randn(B, T, D, device=device, dtype=dtype).transpose(0, 1).requires_grad_()
        weight = (torch.randn(V, D, device=device) / D ** 0.5).to(dtype).requires_grad_()
        bias = torch.randn(V, device=device, dtype=dtype).requires_grad_() if with_bias else None
        target = torch.randint(V, (T, B), device=device)
        target[::3] = -100
        target[1, 0] = V - 1
    else:
        x = torch.randn(B * T, D).to(device).to(dtype=dtype).requires_grad_()
        target = torch.randint(0, V, (B, T)).to(device)
        target = torch.cat((target[..., 1:], torch.full_like(target[..., :1], -100)), -1).flatten()
        weight = torch.randn(V, D).to(device).to(dtype=dtype).requires_grad_()
        bias = torch.randn(V).to(device).to(dtype=dtype).requires_grad_() if with_bias else None
    if ignore_all:
        target.fill_(-100)
    inputs = (x, weight, bias) if with_bias else (x, weight)
    logits = F.linear(x, weight, bias)
    if strided:
        logits = logits.float() * scale
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        if ignore_all:
            ref = logits.sum() * 0
        else:
            ref = F.cross_entropy(logits.reshape(-1, V), target.flatten(), label_smoothing=smoothing, reduction=reduction)
        do = 2
    else:
        ref = FusedCrossEntropyLoss(reduction=reduction, logit_scale=scale, logit_softcapping=softcap)(logits, target)
        do = torch.randn_like(ref).to(device).to(dtype=dtype)
    ref_grads = torch.autograd.grad(ref * do, inputs)
    tri = FusedLinearCrossEntropyLoss(
        label_smoothing=smoothing,
        logit_scale=scale,
        logit_softcapping=softcap,
        reduction=reduction,
        accumulate_grad_in_fp32=accumulate_grad_in_fp32,
    )(x, target, weight, bias)
    tri_grads = torch.autograd.grad(tri * do, inputs)

    assert_close("loss", ref, tri, ratio=1e-2)
    for name, expected, actual in zip(('dx', 'dw', 'db'), ref_grads, tri_grads):
        assert_close(name, expected, actual, ratio=1e-2)
    if ignore_all:
        assert tri.item() == 0
        for grad in tri_grads:
            assert torch.count_nonzero(grad).item() == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=['bf16', 'fp16', 'fp32'])
@pytest.mark.parametrize("reduction", ['mean', 'sum'])
@pytest.mark.parametrize("num_chunks", [1, 8])
@pytest.mark.parametrize(
    ('N', 'V', 'target_id', 'gap'),
    [(128, 128, 0, 12.0), (63, 4103, 4102, 15.0), (63, 65539, 65538, 18.0)],
    ids=['single_tile', 'backward_tail', 'forward_tail'],
)
@pytest.mark.skipif(IS_INTEL or IS_NPU, reason="Covers the default Triton GPU kernels")
def test_fused_linear_cross_entropy_target_gradient(N, V, target_id, gap, num_chunks, reduction, dtype):
    """Preserve small target gradients when the correct class has probability close to one."""
    torch.manual_seed(42)
    x = torch.zeros(N, 64, device=device, dtype=dtype)
    x[:, 0] = 1
    x.requires_grad_()
    weight = torch.zeros(V, 64, device=device, dtype=dtype)
    weight[target_id, 0] = gap
    weight.requires_grad_()
    bias = torch.zeros(V, device=device, dtype=dtype, requires_grad=True)
    target = torch.full((N,), target_id, device=device, dtype=torch.long)

    ref = F.cross_entropy(F.linear(x, weight, bias).float(), target, reduction=reduction)
    ref_grads = torch.autograd.grad(ref, (x, weight, bias))
    tri = FusedLinearCrossEntropyLoss(num_chunks=num_chunks, reduction=reduction)(x, target, weight, bias)
    tri_grads = torch.autograd.grad(tri, (x, weight, bias))

    assert_close("loss", ref, tri, ratio=1e-2, err_atol=0)
    for name, expected, actual in zip(('dx', 'dw', 'db'), ref_grads, tri_grads):
        assert_close(name, expected, actual, ratio=1e-2, err_atol=0)


@pytest.mark.parametrize(('name', 'legacy_name'), [('fwd', 'forward'), ('bwd', 'backward')])
def test_fused_linear_cross_entropy_backend_dispatch(monkeypatch, name, legacy_name):
    import fla.modules.fused_linear_cross_entropy as linear_ce
    from fla.modules.backends.triton_ascend import TritonAscendBackend
    from fla.ops.backends import _DISPATCH_DISABLED

    if _DISPATCH_DISABLED:
        pytest.skip("Backend dispatch was disabled before import")
    method = f'fused_linear_cross_entropy_{name}'
    assert hasattr(TritonAscendBackend, method)
    result = object()
    monkeypatch.setattr(TritonAscendBackend, 'is_available', classmethod(lambda cls: True))
    monkeypatch.setattr(TritonAscendBackend, 'is_enabled', classmethod(lambda cls: True))
    monkeypatch.setattr(TritonAscendBackend, method, lambda self, **kwargs: result)
    args = dict(x=None, target=None, weight=None) if name == 'fwd' else dict(do=None, dx=None, dw=None, db=None)

    assert getattr(linear_ce, method)(**args) is result
    assert getattr(linear_ce, f'fused_linear_cross_entropy_{legacy_name}')(**args) is result
