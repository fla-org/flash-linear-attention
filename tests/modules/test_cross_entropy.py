# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math
from datetime import timedelta
from itertools import product

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules.l2warp import l2_warp
from fla.utils import IS_INTEL, IS_NPU, IS_NVIDIA, assert_close, device, device_torch_lib


@pytest.mark.parametrize(
    ('N', 'V', 'smoothing', 'scale', 'softcap', 'z_scale', 'strided', 'inplace_backward', 'reduction', 'dtype'),
    [
        pytest.param(1024, 32000, 0.0, 1.0, None, 0.0, False, False, 'mean', torch.bfloat16, id='basic'),
        pytest.param(63, 131071, 0.0, 1.0, None, 0.0, False, True, 'mean', torch.bfloat16, id='large_vocab'),
        pytest.param(
            127, 4103, 0.1, 0.3, None, 0.0, True, True, 'none', torch.float16,
            id='strided_tail',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        ),
        pytest.param(
            2047, 65539, 0.1, 0.5, 3.0, 0.0, False, True, 'mean', torch.bfloat16,
            id='softcap_tail',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        ),
        pytest.param(
            1025, 32000, 0.1, 0.3, None, 0.01, False, False, 'sum', torch.bfloat16,
            id='smoothing',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        ),
        pytest.param(
            511, 32003, 0.1, 0.3, 3.0, 0.01, True, False, 'mean', torch.float32,
            id='z_loss',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        ),
        pytest.param(
            255, 65537, 0.1, 0.0, None, 0.0, False, False, 'mean', torch.bfloat16,
            id='zero_scale',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        ),
        pytest.param(
            65, 65537, 0.0, -0.5, 3.0, 0.0, True, True, 'none', torch.float32,
            id='negative_scale',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        ),
    ],
)
@pytest.mark.skipif(IS_INTEL, reason="Intel Triton Failure")
def test_fused_cross_entropy(
    N: int,
    V: int,
    smoothing: float,
    scale: float,
    softcap: float | None,
    z_scale: float,
    strided: bool,
    inplace_backward: bool,
    reduction: str,
    dtype: torch.dtype,
):
    """Match CE loss, optional z-loss, and logits gradients against PyTorch."""
    torch.manual_seed(42)
    logits = torch.randn(N, 2 * V if strided else V, device=device, dtype=dtype)
    if strided:
        logits = logits[:, ::2]
    logits = logits.detach().requires_grad_()
    target = torch.randint(V, (N,), device=device)
    target[::3] = -100
    target[1] = V - 1

    transformed = logits.float() * scale
    if softcap is not None:
        transformed = softcap * torch.tanh(transformed / softcap)
    ref = F.cross_entropy(transformed, target, reduction=reduction, label_smoothing=smoothing)
    if z_scale > 0:
        z_loss = (z_scale * transformed.logsumexp(-1).square()).masked_fill(target == -100, 0)
        if reduction == 'mean':
            z_loss = z_loss.sum() / (target != -100).sum()
        elif reduction == 'sum':
            z_loss = z_loss.sum()
        ref = ref + z_loss
    do = torch.randn_like(ref)
    ref_grad, = torch.autograd.grad(ref, logits, grad_outputs=do)
    tri = FusedCrossEntropyLoss(
        reduction=reduction,
        label_smoothing=smoothing,
        logit_scale=scale,
        lse_square_scale=z_scale,
        logit_softcapping=softcap,
        inplace_backward=inplace_backward,
        return_z_loss=z_scale > 0,
    )(logits, target)
    if z_scale > 0:
        tri, tri_z_loss = tri
    tri_grad, = torch.autograd.grad(tri, logits, grad_outputs=do)

    assert_close("loss", ref, tri, ratio=1e-2)
    assert_close("dlogits", ref_grad, tri_grad, ratio=1e-2)
    if z_scale > 0:
        assert not tri_z_loss.requires_grad
        assert_close("z_loss", z_loss, tri_z_loss, ratio=1e-2)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'V', 'smoothing', 'scale', 'softcap', 'num_chunks', 'reduction',
     'with_bias', 'strided', 'ignore_all', 'confident_target', 'accumulate_grad_in_fp32', 'dtype'),
    [
        pytest.param(
            2, T, D, V, 0.0, scale, softcap, 8, 'mean', True, False, False, False, fp32_grad, torch.bfloat16,
            id=f'T{T}-D{D}-V{V}-{scale=}-{softcap=}-{fp32_grad=}',
        )
        for T, D, V, (scale, softcap, fp32_grad) in product(
            (512, 1024), (1024, 2048), (32000, 100000),
            ((1.0, None, False), (0.5, None, False), (1.0, None, True), (0.5, None, True), (1.0, 30.0, True)),
        )
    ] + [
        pytest.param(
            3, 7, 32, V, smoothing, scale, softcap, 8, reduction, with_bias, True, ignore_all, False, True, dtype,
            id=f'V{V}-{smoothing=}-{scale=}-{softcap=}-{with_bias=}-{reduction}-{dtype}',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        )
        for (V, smoothing, scale, softcap, ignore_all), with_bias, reduction, dtype in product(
            ((4103, 0.1, 0.3, None, False), (65539, 0.1, 0.5, 3.0, False), (129, 0.0, 1.0, None, True)),
            (False, True), ('mean', 'sum'), (torch.bfloat16, torch.float16, torch.float32),
        )
    ] + [
        pytest.param(
            1, T, 64, V, 0.0, 1.0, None, num_chunks, reduction, True, False, False, True, True, dtype,
            id=f'T{T}-V{V}-confident_target-{num_chunks=}-{reduction}-{dtype}',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        )
        for (T, V), num_chunks, reduction, dtype in product(
            ((128, 128), (63, 4103), (63, 65539)), (1, 8), ('mean', 'sum'),
            (torch.bfloat16, torch.float16, torch.float32),
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
    num_chunks: int,
    reduction: str,
    with_bias: bool,
    strided: bool,
    ignore_all: bool,
    confident_target: bool,
    accumulate_grad_in_fp32: bool,
    dtype: torch.dtype,
):
    """Match linear CE loss and input/parameter gradients against the reference implementations."""
    torch.manual_seed(42)
    if confident_target:
        x = torch.zeros(B * T, D, device=device, dtype=dtype)
        x[:, 0] = 1
        x.requires_grad_()
        weight = torch.zeros(V, D, device=device, dtype=dtype)
        # keep the target probability near one across vocabulary sizes
        weight[-1, 0] = math.log(V - 1) + 7
        weight.requires_grad_()
        bias = torch.zeros(V, device=device, dtype=dtype, requires_grad=True) if with_bias else None
        target = torch.full((B * T,), V - 1, device=device, dtype=torch.long)
    elif strided:
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
    if strided or confident_target:
        logits = logits.float() * scale
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        if ignore_all:
            ref = logits.sum() * 0
        else:
            ref = F.cross_entropy(logits.reshape(-1, V), target.flatten(), label_smoothing=smoothing, reduction=reduction)
        do = 1 if confident_target else 2
    else:
        ref = FusedCrossEntropyLoss(reduction=reduction, logit_scale=scale, logit_softcapping=softcap)(logits, target)
        do = torch.randn_like(ref).to(device).to(dtype=dtype)
    ref_grads = torch.autograd.grad(ref * do, inputs)
    tri = FusedLinearCrossEntropyLoss(
        label_smoothing=smoothing,
        logit_scale=scale,
        logit_softcapping=softcap,
        num_chunks=num_chunks,
        reduction=reduction,
        accumulate_grad_in_fp32=accumulate_grad_in_fp32,
    )(x, target, weight, bias)
    tri_grads = torch.autograd.grad(tri * do, inputs)

    err_atol = 0 if confident_target else 1e-6
    assert_close("loss", ref, tri, ratio=1e-2, err_atol=err_atol)
    for name, expected, actual in zip(('dx', 'dw', 'db'), ref_grads, tri_grads):
        assert_close(name, expected, actual, ratio=1e-2, err_atol=err_atol)
    if ignore_all:
        assert tri.item() == 0
        for grad in tri_grads:
            assert torch.count_nonzero(grad).item() == 0


def _check_parallel_linear_cross_entropy(rank, world_size, local_vocab, dtype, option, reduction):
    torch.manual_seed(42)
    N, H, V = 63, 64, world_size * local_vocab
    x = torch.randn(1, N, H, device=device, dtype=dtype).requires_grad_()
    weight = (torch.randn(V, H, device=device) / H ** 0.5).to(dtype).requires_grad_()
    with_bias = option in ('combined', 'l2', 'l2_tie')
    bias = torch.randn(V, device=device, dtype=dtype).requires_grad_() if with_bias else None
    target = torch.randint(V, (1, N), device=device)
    target[:, ::7] = -100
    target[0, :4] = torch.tensor([0, local_vocab - 1, min(local_vocab, V - 1), V - 1], device=device)
    if option == 'ignored':
        target.fill_(-100)
    if option == 'confident':
        with torch.no_grad():
            x.zero_()
            x[..., 0] = 1
            weight.zero_()
            weight[-1, 0] = 12
            target.fill_(V - 1)
    if option == 'l2_tie':
        with torch.no_grad():
            weight.zero_()
            bias.fill_(1)

    kwargs = dict(reduction=reduction)
    if option == 'combined':
        kwargs.update(label_smoothing=0.1, logit_scale=0.3, logit_softcapping=3.0)
    if option in ('l2', 'l2_tie'):
        kwargs.update(use_l2warp=True, l2_penalty_factor=0.7)
    raw = F.linear(x, weight, bias)
    logits = raw.float() * kwargs.get('logit_scale', 1)
    if 'logit_softcapping' in kwargs:
        logits = 3 * torch.tanh(logits / 3)
    if option == 'ignored':
        ref = logits.sum() * 0
    else:
        ref = F.cross_entropy(
            logits.flatten(0, 1),
            target.flatten(),
            label_smoothing=kwargs.get('label_smoothing', 0),
            reduction=reduction,
        )
    if kwargs.get('use_l2warp'):
        ref = l2_warp(ref, raw.float(), kwargs['l2_penalty_factor'])
    inputs = (x, weight, bias) if with_bias else (x, weight)
    ref_grads = torch.autograd.grad(ref * 2, inputs)

    start, end = rank * local_vocab, (rank + 1) * local_vocab
    local_x = x.detach().clone().requires_grad_()
    local_weight = weight[start:end].detach().clone().requires_grad_()
    local_bias = bias[start:end].detach().clone().requires_grad_() if with_bias else None
    local_inputs = (local_x, local_weight, local_bias) if with_bias else (local_x, local_weight)
    tri = FusedLinearCrossEntropyLoss(process_group=dist.group.WORLD, **kwargs)(
        local_x, target, local_weight, local_bias,
    )
    tri_grads = torch.autograd.grad(tri * 2, local_inputs)
    expected_grads = (ref_grads[0],) + tuple(grad[start:end] for grad in ref_grads[1:])
    for name, expected, actual in zip(('loss', 'dx', 'dw', 'db'), (ref, *expected_grads), (tri, *tri_grads)):
        assert torch.isfinite(actual).all(), name
        assert_close(name, expected.float(), actual.float(), ratio=1e-2, err_atol=0 if option == 'confident' else 1e-6)


def _run_parallel_linear_cross_entropy_worker(rank, world_size, init_file):
    device_torch_lib.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        backend='nccl',
        init_method=f'file://{init_file}',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5),
    )
    try:
        for local_vocab, dtype, option, reduction in product(
            (67, 4103, 65539),
            (torch.bfloat16, torch.float32),
            ('plain', 'combined', 'ignored', 'l2', 'l2_tie'),
            ('mean', 'sum'),
        ):
            try:
                _check_parallel_linear_cross_entropy(rank, world_size, local_vocab, dtype, option, reduction)
            except AssertionError as error:
                raise AssertionError(f'{local_vocab=}, {dtype=}, {option=}, {reduction=}: {error}') from error
        _check_parallel_linear_cross_entropy(rank, world_size, 67, torch.bfloat16, 'confident', 'mean')
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('world_size', [1, 2, 4])
@pytest.mark.skipif(not IS_NVIDIA, reason="Vocabulary-parallel fused linear CE requires the default GPU backend")
def test_fused_linear_cross_entropy_parallel(world_size, tmp_path):
    if device_torch_lib.device_count() < world_size:
        pytest.skip(f"Requires {world_size} devices")
    mp.spawn(
        fn=_run_parallel_linear_cross_entropy_worker,
        args=(world_size, str(tmp_path / 'init')),
        nprocs=world_size,
        join=True,
    )


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


def test_fused_linear_cross_entropy_parallel_ascend_rejection(monkeypatch):
    from fla.modules.backends.triton_ascend import TritonAscendBackend

    monkeypatch.setattr(dist, 'get_world_size', lambda group: 2)
    with pytest.raises(NotImplementedError, match="Vocabulary-parallel"):
        TritonAscendBackend().fused_linear_cross_entropy_fwd(x=None, target=None, weight=None, process_group=object())
