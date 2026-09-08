# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.utils import IS_INTEL, IS_NPU, assert_close, device, device_platform


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [512, 1024])
@pytest.mark.parametrize("D", [1024, 2048])
@pytest.mark.parametrize("V", [32000, 100000])
@pytest.mark.parametrize("reduction", ['mean'])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure",
)
def test_fused_cross_entropy(
    B: int,
    T: int,
    D: int,
    V: int,
    reduction: str,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    logits = torch.randn(B * T, V).to(device).to(dtype=dtype).requires_grad_()
    target = torch.randint(0, V, (B, T)).to(device)
    target = torch.cat((target[..., 1:], torch.full_like(target[..., :1], -100)), -1)
    target = target.flatten()

    ref = nn.CrossEntropyLoss(reduction=reduction)(logits, target).to(dtype=dtype)
    do = torch.randn_like(ref).to(device).to(dtype=dtype)

    ref.backward(do)
    ref_d, logits.grad = logits.grad.clone(), None

    tri = FusedCrossEntropyLoss(reduction=reduction)(logits, target).to(dtype=dtype)
    tri.backward(do)
    tri_d, logits.grad = logits.grad.clone(), None

    assert_close(" o", ref, tri, ratio=1e-2)
    assert_close("dl", ref_d, tri_d, ratio=1e-2)


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [512, 1024])
@pytest.mark.parametrize("D", [1024, 2048])
@pytest.mark.parametrize("V", [32000, 100000])
@pytest.mark.parametrize("softcap", [30.0])
@pytest.mark.parametrize("reduction", ['mean'])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure",
)
def test_fused_cross_entropy_softcap(
    B: int,
    T: int,
    D: int,
    V: int,
    softcap: float,
    reduction: str,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    logits = torch.randn(B * T, V).to(device).to(dtype=dtype).requires_grad_()
    target = torch.randint(0, V, (B, T)).to(device)
    target = torch.cat((target[..., 1:], torch.full_like(target[..., :1], -100)), -1)
    target = target.flatten()

    # reference: manually apply softcap, then standard CE
    capped = softcap * torch.tanh(logits.float() / softcap)
    ref = nn.CrossEntropyLoss(reduction=reduction)(capped, target).to(dtype=dtype)
    do = torch.randn_like(ref).to(device).to(dtype=dtype)

    ref.backward(do)
    ref_d, logits.grad = logits.grad.clone(), None

    tri = FusedCrossEntropyLoss(logit_softcapping=softcap, reduction=reduction)(logits, target).to(dtype=dtype)
    tri.backward(do)
    tri_d, logits.grad = logits.grad.clone(), None

    assert_close(" o", ref, tri, ratio=1e-2)
    assert_close("dl", ref_d, tri_d, ratio=1e-2)


@pytest.mark.parametrize('V', [4103, 65537])
@pytest.mark.parametrize('scale', [0.0, -0.5])
@pytest.mark.parametrize('softcap', [None, 3.0])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.skipif(IS_INTEL or IS_NPU, reason="Covers the default Triton GPU kernels")
def test_fused_cross_entropy_nonpositive_scale(V, scale, softcap, dtype):
    torch.manual_seed(42)
    logits = torch.randn(7, V, device=device, dtype=dtype).requires_grad_()
    target = torch.randint(V, (7,), device=device)
    target[::3] = -100
    target[1] = V - 1
    transformed = logits.float() * scale
    if softcap is not None:
        transformed = softcap * torch.tanh(transformed / softcap)
    ref = F.cross_entropy(transformed, target, label_smoothing=0.1)
    ref_grad, = torch.autograd.grad(ref, logits)
    tri = FusedCrossEntropyLoss(label_smoothing=0.1, logit_scale=scale, logit_softcapping=softcap)(logits, target)
    tri_grad, = torch.autograd.grad(tri, logits)

    assert_close('loss', ref, tri, ratio=1e-2)
    assert_close('dlogits', ref_grad, tri_grad, ratio=1e-2)


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [512, 1024])
@pytest.mark.parametrize("D", [1024, 2048])
@pytest.mark.parametrize("V", [32000, 100000])
@pytest.mark.parametrize("scale", [1., 0.5])
@pytest.mark.parametrize("reduction", ['mean'])
@pytest.mark.parametrize("accumulate_grad_in_fp32", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure",
)
def test_fused_linear_cross_entropy(
    B: int,
    T: int,
    D: int,
    V: int,
    scale: float,
    reduction: str,
    accumulate_grad_in_fp32: bool,
    dtype: torch.dtype
):
    torch.manual_seed(42)

    x = torch.randn(B * T, D).to(device).to(dtype=dtype).requires_grad_()
    target = torch.randint(0, V, (B, T)).to(device)
    target = torch.cat((target[..., 1:], torch.full_like(target[..., :1], -100)), -1)
    target = target.flatten()
    weight = torch.randn(V, D).to(device).to(dtype=dtype).requires_grad_()
    bias = torch.randn(V).to(device).to(dtype=dtype).requires_grad_()

    logits = F.linear(x, weight, bias)
    ref = FusedCrossEntropyLoss(logit_scale=scale, reduction=reduction)(logits, target)
    do = torch.randn_like(ref).to(device).to(dtype=dtype)

    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dw, weight.grad = weight.grad.clone(), None
    ref_db, bias.grad = bias.grad.clone(), None

    tri = FusedLinearCrossEntropyLoss(
        logit_scale=scale,
        reduction=reduction,
        accumulate_grad_in_fp32=accumulate_grad_in_fp32,
    )(x, target, weight, bias)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dw, weight.grad = weight.grad.clone(), None
    tri_db, bias.grad = bias.grad.clone(), None

    assert_close(" o", ref, tri, ratio=1e-2)
    assert_close("dx", ref_dx, tri_dx, ratio=1e-2)
    assert_close("dw", ref_dw, tri_dw, ratio=1e-2)
    assert_close("db", ref_db, tri_db, ratio=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=['bf16', 'fp16', 'fp32'])
@pytest.mark.parametrize("reduction", ['mean', 'sum'])
@pytest.mark.parametrize("num_chunks", [1, 8])
@pytest.mark.parametrize(
    ('N', 'V', 'target_id', 'gap'),
    [(128, 128, 0, 12.0), (63, 4103, 4102, 15.0), (63, 65539, 65538, 18.0)],
    ids=['single_tile', 'backward_tail', 'forward_tail'],
)
@pytest.mark.skipif(IS_INTEL or IS_NPU, reason="Covers the default Triton GPU kernels")
def test_fused_linear_cross_entropy_confident(N, V, target_id, gap, num_chunks, reduction, dtype):
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
    tri = FusedLinearCrossEntropyLoss(reduction=reduction, num_chunks=num_chunks)(x, target, weight, bias)
    tri_grads = torch.autograd.grad(tri, (x, weight, bias))

    assert_close("loss", ref, tri, ratio=1e-2, err_atol=0)
    for name, expected, actual in zip(('dx', 'dw', 'db'), ref_grads, tri_grads):
        assert_close(name, expected, actual, ratio=1e-2, err_atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=['bf16', 'fp16', 'fp32'])
@pytest.mark.parametrize("reduction", ['mean', 'sum'])
@pytest.mark.parametrize("with_bias", [False, True], ids=['no_bias', 'bias'])
@pytest.mark.parametrize(
    ('V', 'smoothing', 'scale', 'softcap'),
    [(4103, 0.1, 0.3, None), (65539, 0.1, 0.5, 3.0)],
    ids=['scaled_smoothing', 'softcap_tail'],
)
@pytest.mark.skipif(IS_INTEL or IS_NPU, reason="Covers the default Triton GPU kernels")
def test_fused_linear_cross_entropy_options(V, smoothing, scale, softcap, with_bias, reduction, dtype):
    torch.manual_seed(42)
    x = torch.randn(3, 7, 32, device=device, dtype=dtype).transpose(0, 1).requires_grad_()
    weight = (torch.randn(V, 32, device=device) / 32 ** 0.5).to(dtype).requires_grad_()
    bias = torch.randn(V, device=device, dtype=dtype).requires_grad_() if with_bias else None
    target = torch.randint(V, (7, 3), device=device)
    target[::3] = -100
    target[1, 0] = V - 1
    inputs = (x, weight, bias) if with_bias else (x, weight)
    logits = F.linear(x, weight, bias).float() * scale
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    ref = F.cross_entropy(logits.reshape(-1, V), target.flatten(), label_smoothing=smoothing, reduction=reduction)
    ref_grads = torch.autograd.grad(ref * 2, inputs)
    tri = FusedLinearCrossEntropyLoss(
        label_smoothing=smoothing,
        logit_scale=scale,
        logit_softcapping=softcap,
        reduction=reduction,
    )(x, target, weight, bias)
    tri_grads = torch.autograd.grad(tri * 2, inputs)

    assert_close("loss", ref, tri, ratio=1e-2)
    for name, expected, actual in zip(('dx', 'dw', 'db'), ref_grads, tri_grads):
        assert_close(name, expected, actual, ratio=1e-2)


@pytest.mark.parametrize("reduction", ['mean', 'sum'])
@pytest.mark.skipif(IS_INTEL or IS_NPU, reason="Covers the default Triton GPU kernels")
def test_fused_linear_cross_entropy_all_ignored(reduction):
    torch.manual_seed(42)
    x = torch.randn(7, 32, device=device, requires_grad=True)
    weight = torch.randn(129, 32, device=device, requires_grad=True)
    bias = torch.randn(129, device=device, requires_grad=True)
    target = torch.full((7,), -100, device=device, dtype=torch.long)
    loss = FusedLinearCrossEntropyLoss(reduction=reduction)(x, target, weight, bias)
    loss.backward()

    assert loss.item() == 0
    for grad in (x.grad, weight.grad, bias.grad):
        assert torch.count_nonzero(grad).item() == 0


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=['bf16', 'fp16', 'fp32'])
@pytest.mark.parametrize("reduction", ['mean', 'sum', 'none'])
@pytest.mark.parametrize("inplace_backward", [False, True], ids=['out_of_place', 'inplace'])
@pytest.mark.parametrize("V", [4103, 65539])
@pytest.mark.skipif(IS_INTEL or IS_NPU, reason="Covers the default Triton GPU kernels")
def test_fused_cross_entropy_options(V, inplace_backward, reduction, dtype):
    torch.manual_seed(42)
    logits = torch.randn(2, 7, V, device=device, dtype=dtype).transpose(0, 1).reshape(7, 2 * V)[:, ::2]
    logits = logits.detach().requires_grad_()
    target = torch.randint(V, (7,), device=device)
    target[::3] = -100
    target[1] = V - 1
    transformed = 3 * torch.tanh(logits.float() * 0.3 / 3)
    z_loss = 0.01 * transformed.logsumexp(-1).square()
    z_loss = z_loss.masked_fill(target == -100, 0)
    ref = F.cross_entropy(transformed, target, reduction='none', label_smoothing=0.1) + z_loss
    if reduction == 'mean':
        ref, z_loss = ref.sum() / (target != -100).sum(), z_loss.sum() / (target != -100).sum()
    elif reduction == 'sum':
        ref, z_loss = ref.sum(), z_loss.sum()
    do = torch.randn_like(ref)
    ref_grad, = torch.autograd.grad(ref, logits, grad_outputs=do)
    tri, tri_z_loss = FusedCrossEntropyLoss(
        reduction=reduction,
        label_smoothing=0.1,
        logit_scale=0.3,
        lse_square_scale=0.01,
        logit_softcapping=3.0,
        inplace_backward=inplace_backward,
        return_z_loss=True,
    )(logits, target)
    tri_grad, = torch.autograd.grad(tri, logits, grad_outputs=do)

    assert not tri_z_loss.requires_grad
    assert_close("loss", ref, tri, ratio=1e-2)
    assert_close("z_loss", z_loss, tri_z_loss, ratio=1e-2)
    assert_close("dlogits", ref_grad, tri_grad, ratio=1e-2)


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [512, 1024])
@pytest.mark.parametrize("D", [1024, 2048])
@pytest.mark.parametrize("V", [32000, 100000])
@pytest.mark.parametrize("softcap", [30.0])
@pytest.mark.parametrize("reduction", ['mean'])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure",
)
def test_fused_linear_cross_entropy_softcap(
    B: int,
    T: int,
    D: int,
    V: int,
    softcap: float,
    reduction: str,
    dtype: torch.dtype
):
    torch.manual_seed(42)

    x = torch.randn(B * T, D).to(device).to(dtype=dtype).requires_grad_()
    target = torch.randint(0, V, (B, T)).to(device)
    target = torch.cat((target[..., 1:], torch.full_like(target[..., :1], -100)), -1)
    target = target.flatten()
    weight = torch.randn(V, D).to(device).to(dtype=dtype).requires_grad_()
    bias = torch.randn(V).to(device).to(dtype=dtype).requires_grad_()

    logits = F.linear(x, weight, bias)
    ref = FusedCrossEntropyLoss(logit_softcapping=softcap, reduction=reduction)(logits, target)
    do = torch.randn_like(ref).to(device).to(dtype=dtype)

    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dw, weight.grad = weight.grad.clone(), None
    ref_db, bias.grad = bias.grad.clone(), None

    tri = FusedLinearCrossEntropyLoss(logit_softcapping=softcap, reduction=reduction)(x, target, weight, bias)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dw, weight.grad = weight.grad.clone(), None
    tri_db, bias.grad = bias.grad.clone(), None

    assert_close(" o", ref, tri, ratio=1e-2)
    assert_close("dx", ref_dx, tri_dx, ratio=1e-2)
    assert_close("dw", ref_dw, tri_dw, ratio=1e-2)
    assert_close("db", ref_db, tri_db, ratio=1e-2)
