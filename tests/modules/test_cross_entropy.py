# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import inspect
import math
from datetime import timedelta
from itertools import product
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from fla import backends
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules.l2warp import l2_warp
from fla.utils import IS_INTEL, IS_NPU, IS_NVIDIA, assert_close, device, device_torch_lib


@pytest.mark.parametrize(
    ('N', 'V', 'smoothing', 'scale', 'z_scale', 'softcap', 'strided', 'ignore_all', 'inplace_backward', 'reduction', 'dtype'),
    [
        pytest.param(1024, 32000, 0.0, 1.0, 0.0, None, False, False, False, 'mean', torch.bfloat16, id='basic'),
        pytest.param(63, 131071, 0.0, 1.0, 0.0, None, False, False, True, 'mean', torch.bfloat16, id='large_vocab'),
        pytest.param(
            127,
            4103,
            0.1,
            0.3,
            0.0,
            None,
            True,
            False,
            True,
            'none',
            torch.float16,
            id='strided_tail',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        ),
        pytest.param(
            2047,
            65539,
            0.1,
            0.5,
            0.0,
            3.0,
            False,
            False,
            True,
            'mean',
            torch.bfloat16,
            id='softcap_tail',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        ),
        pytest.param(
            1025,
            32000,
            0.1,
            0.3,
            0.01,
            None,
            False,
            False,
            False,
            'sum',
            torch.bfloat16,
            id='smoothing',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        ),
        pytest.param(
            511,
            32003,
            0.1,
            0.3,
            0.01,
            3.0,
            True,
            False,
            False,
            'mean',
            torch.float32,
            id='z_loss',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        ),
        pytest.param(
            255,
            65537,
            0.1,
            0.0,
            0.0,
            None,
            False,
            False,
            False,
            'mean',
            torch.bfloat16,
            id='zero_scale',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        ),
        pytest.param(
            65,
            65537,
            0.0,
            -0.5,
            0.0,
            3.0,
            True,
            False,
            True,
            'none',
            torch.float32,
            id='negative_scale',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        ),
        pytest.param(2, 1, 0.0, 1.0, 0.0, None, False, False, False, 'sum', torch.float32, id='single_class'),
        pytest.param(
            7,
            129,
            0.1,
            0.5,
            0.01,
            3.0,
            False,
            True,
            True,
            'none',
            torch.float16,
            id='ignored',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        ),
        pytest.param(7, 16384, 0.1, 0.5, 0.01, None, True, False, True, 'none', torch.bfloat16, id='gluon_full_row'),
        pytest.param(7, 16385, 0.1, 0.5, 0.01, None, True, False, True, 'none', torch.float32, id='gluon_split_row'),
        pytest.param(7, 65536, 0.0, 1.0, 0.0, None, False, False, False, 'sum', torch.bfloat16, id='split_boundary'),
        pytest.param(3, 1025, 0.1, 0.5, 0.01, 3.0, False, False, True, 'none', torch.float16, id='row_tail_softcap'),
        pytest.param(5, 8193, 0.1, 1.0, 0.01, None, False, False, False, 'sum', torch.float32, id='row_tail_split'),
    ],
)
@pytest.mark.parametrize(
    ('backend', 'backward_backend', 'autocast'),
    [
        pytest.param('triton', 'triton', False, id='triton'),
        pytest.param('gluon', 'gluon', False, id='gluon'),
        pytest.param('triton', 'gluon', True, id='autocast-triton-gluon'),
        pytest.param('gluon', 'triton', True, id='autocast-gluon-triton'),
    ],
)
@pytest.mark.skipif(IS_INTEL, reason="Intel Triton Failure")
def test_fused_cross_entropy(
    N: int,
    V: int,
    smoothing: float,
    scale: float,
    z_scale: float,
    softcap: float | None,
    strided: bool,
    ignore_all: bool,
    inplace_backward: bool,
    reduction: str,
    dtype: torch.dtype,
    backend: str,
    backward_backend: str,
    autocast: bool,
    monkeypatch,
):
    """Match CE loss, optional z-loss, and logits gradients against PyTorch."""
    from fla.backends import _DISPATCH_DISABLED, GluonBackend

    monkeypatch.setenv('FLA_GLUON', '0')
    monkeypatch.setenv('FLA_CE_GLUON', '1' if backend == 'gluon' else '0')
    calls = []
    if 'gluon' in (backend, backward_backend):
        if not GluonBackend.is_available() or _DISPATCH_DISABLED:
            pytest.skip("Gluon dispatch is unavailable")
        from fla.modules.fused_cross_entropy import ops

        for entry in (ops.cross_entropy_fwd, ops.cross_entropy_bwd):
            candidate = backends._function_registries[inspect.unwrap(entry)]._backends['gluon']
            call = Mock(wraps=candidate.implementation)
            monkeypatch.setattr(candidate, 'implementation', call)
            calls.append(call)
    torch.manual_seed(42)
    logits = torch.randn(N, 2 * V if strided else V, device=device, dtype=dtype)
    if strided:
        logits = logits[:, ::2]
    logits = logits.detach().requires_grad_()
    target = torch.randint(V, (N,), device=device)
    target[::3] = -100
    target[1] = V - 1
    if ignore_all:
        target.fill_(-100)

    transformed = logits.float() * scale
    if softcap is not None:
        transformed = softcap * torch.tanh(transformed / softcap)
    ref = F.cross_entropy(input=transformed, target=target, reduction=reduction, label_smoothing=smoothing)
    if z_scale > 0:
        z_loss = (z_scale * transformed.logsumexp(-1).square()).masked_fill(target == -100, 0)
        if reduction == 'mean':
            z_loss = z_loss.sum() / (target != -100).sum()
        elif reduction == 'sum':
            z_loss = z_loss.sum()
        ref = ref + z_loss
    do = torch.randn_like(ref)
    ref_grad, = torch.autograd.grad(outputs=ref, inputs=logits, grad_outputs=do)
    logits_before = logits.detach().clone() if not inplace_backward else None
    criterion = FusedCrossEntropyLoss(
        reduction=reduction,
        label_smoothing=smoothing,
        logit_scale=scale,
        lse_square_scale=z_scale,
        logit_softcapping=softcap,
        inplace_backward=inplace_backward,
        return_z_loss=z_scale > 0,
    )
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
        tri = criterion(input=logits, target=target)
    if z_scale > 0:
        tri, tri_z_loss = tri
    monkeypatch.setenv('FLA_CE_GLUON', '1' if backward_backend == 'gluon' else '0')
    tri_grad, = torch.autograd.grad(outputs=tri, inputs=logits, grad_outputs=do)

    for call, expected in zip(calls, (backend, backward_backend)):
        assert call.call_count == int(expected == 'gluon')
    if inplace_backward and not strided:
        assert tri_grad.data_ptr() == logits.data_ptr()
    if not inplace_backward:
        torch.testing.assert_close(logits, logits_before, rtol=0, atol=0)
    if ignore_all:
        assert torch.count_nonzero(tri) == torch.count_nonzero(tri_grad) == 0
    assert tri.dtype == torch.float32
    assert tri_grad.dtype == dtype
    assert_close("loss", ref, tri, ratio=1e-2)
    assert_close("dlogits", ref_grad, tri_grad, ratio=1e-2)
    if z_scale > 0:
        assert not tri_z_loss.requires_grad
        if ignore_all:
            assert torch.count_nonzero(tri_z_loss) == 0
        assert_close("z_loss", z_loss, tri_z_loss, ratio=1e-2)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'V', 'smoothing', 'scale', 'softcap', 'num_chunks', 'reduction',
     'with_bias', 'strided', 'ignore_all', 'confident_target', 'accumulate_grad_in_fp32', 'dtype', 'gradient_mode', 'use_l2warp'),
    [
        pytest.param(
            2,
            T,
            D,
            V,
            0.0,
            scale,
            softcap,
            8,
            'mean',
            True,
            False,
            False,
            False,
            fp32_grad,
            torch.bfloat16,
            'all',
            False,
            id=f'T{T}-D{D}-V{V}-{scale=}-{softcap=}-{fp32_grad=}',
        )
        for T, D, V, (scale, softcap, fp32_grad) in product(
            (512, 1024),
            (1024, 2048),
            (32000, 100000),
            ((1.0, None, False), (0.5, None, False), (1.0, None, True), (0.5, None, True), (1.0, 30.0, True)),
        )
    ] + [
        pytest.param(
            3,
            7,
            32,
            V,
            smoothing,
            scale,
            softcap,
            8,
            reduction,
            with_bias,
            True,
            ignore_all,
            False,
            True,
            dtype,
            'all',
            False,
            id=f'V{V}-{smoothing=}-{scale=}-{softcap=}-{with_bias=}-{reduction}-{dtype}',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        )
        for (V, smoothing, scale, softcap, ignore_all), with_bias, reduction, dtype in product(
            ((4103, 0.1, 0.3, None, False), (65539, 0.1, 0.5, 3.0, False), (129, 0.0, 1.0, None, True)),
            (False, True),
            ('mean', 'sum'),
            (torch.bfloat16, torch.float16, torch.float32),
        )
    ] + [
        pytest.param(
            1,
            T,
            64,
            V,
            0.0,
            1.0,
            None,
            num_chunks,
            reduction,
            True,
            False,
            False,
            True,
            True,
            dtype,
            'all',
            False,
            id=f'T{T}-V{V}-confident_target-{num_chunks=}-{reduction}-{dtype}',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers the default Triton GPU kernels"),
        )
        for (T, V), num_chunks, reduction, dtype in product(
            ((128, 128), (63, 4103), (63, 65539)),
            (1, 8),
            ('mean', 'sum'),
            (torch.bfloat16, torch.float16, torch.float32),
        )
    ] + [
        pytest.param(
            2, 63, 64, 4103, 0.1, 0.3, 3.0, 8, reduction, True, True, False, False, True, dtype,
            gradient_mode, use_l2warp, id=f'demand-{gradient_mode}-{use_l2warp=}-{reduction}-{dtype}',
            marks=pytest.mark.skipif(IS_NPU, reason="Covers transformed logits in the default GPU kernels"),
        )
        for gradient_mode, use_l2warp, reduction, dtype in product(
            ('input', 'weight', 'bias', 'frozen', 'no_grad'), (False, True), ('mean', 'sum'),
            (torch.bfloat16, torch.float32),
        )
    ] + [
        pytest.param(
            2, 63, 64, 4103, 0.0, 1.0, None, 8, 'mean', True, False, False, False, True, torch.bfloat16,
            gradient_mode, use_l2warp, id=f'demand-backend-{gradient_mode}-{use_l2warp=}',
        )
        for gradient_mode, use_l2warp in product(('input', 'weight', 'bias', 'frozen', 'no_grad'), (False, True))
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
    gradient_mode: str,
    use_l2warp: bool,
    monkeypatch,
):
    """Match linear CE loss and input/parameter gradients against the reference implementations."""
    if IS_NVIDIA:
        # reduced-precision GEMM accumulation depends on the row chunk size
        monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction', False)
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
    for name, value in zip(('input', 'weight', 'bias'), inputs):
        value.requires_grad_(gradient_mode in ('all', 'no_grad', name))
    gradient_names = tuple(name for name, value in zip(('dx', 'dw', 'db'), inputs)
                           if value.requires_grad and gradient_mode != 'no_grad')
    inputs = tuple(value for value in inputs if value.requires_grad and gradient_mode != 'no_grad')
    raw_logits = F.linear(x, weight, bias)
    logits = raw_logits
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
    if use_l2warp:
        ref = l2_warp(ref, raw_logits.float().reshape(1, -1, V))
    ref_grads = torch.autograd.grad(ref * do, inputs) if inputs else ()
    saved = []

    def pack(tensor):
        saved.append(tensor.numel() * tensor.element_size())
        return tensor

    with torch.set_grad_enabled(gradient_mode != 'no_grad'), torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        tri = FusedLinearCrossEntropyLoss(
            label_smoothing=smoothing,
            logit_scale=scale,
            logit_softcapping=softcap,
            num_chunks=num_chunks,
            reduction=reduction,
            accumulate_grad_in_fp32=accumulate_grad_in_fp32,
            use_l2warp=use_l2warp,
        )(x, target, weight, bias)
    expected_saved = [value.numel() * value.element_size() for value in inputs]
    if saved != expected_saved:
        pytest.fail(f'Gradient storage bytes: expected {expected_saved}, got {saved}')
    if tri.requires_grad != bool(inputs):
        pytest.fail(f'Loss gradient requirement: expected {bool(inputs)}, got {tri.requires_grad}')
    tri_grads = torch.autograd.grad(tri * do, inputs) if inputs else ()

    err_atol = 0 if confident_target else 1e-6
    assert_close("loss", ref, tri, ratio=1e-2, err_atol=err_atol)
    for name, expected, actual in zip(gradient_names, ref_grads, tri_grads):
        assert_close(name, expected, actual, ratio=1e-2, err_atol=err_atol)
    if ignore_all:
        assert tri.item() == 0
        for grad in tri_grads:
            assert torch.count_nonzero(grad).item() == 0


def _check_parallel_linear_cross_entropy(rank, world_size, local_vocab, dtype, option, reduction, gradient_mode="all"):
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
    for name, value in zip(('input', 'weight', 'bias'), local_inputs):
        need_grad = gradient_mode in ('all', 'no_grad', name)
        if gradient_mode == 'mixed':
            need_grad = rank == 0 if name == 'input' else rank != 0
        value.requires_grad_(need_grad)
    expected_grads = (ref_grads[0],) + tuple(grad[start:end] for grad in ref_grads[1:])
    selected = [(name, value, grad) for name, value, grad in zip(('dx', 'dw', 'db'), local_inputs, expected_grads)
                if value.requires_grad and gradient_mode != 'no_grad']
    with torch.set_grad_enabled(gradient_mode != 'no_grad'):
        tri = FusedLinearCrossEntropyLoss(process_group=dist.group.WORLD, **kwargs)(
            local_x, target, local_weight, local_bias,
        )
    tri_grads = torch.autograd.grad(tri * 2, tuple(value for _, value, _ in selected)) if selected else ()
    names = ('loss',) + tuple(name for name, _, _ in selected)
    expected_grads = tuple(grad for _, _, grad in selected)
    for name, expected, actual in zip(names, (ref, *expected_grads), (tri, *tri_grads)):
        assert torch.isfinite(actual).all(), name
        assert_close(name, expected.float(), actual.float(), ratio=1e-2, err_atol=0 if option == 'confident' else 1e-6)


def _run_parallel_linear_cross_entropy_worker(rank, world_size, init_file):
    device_torch_lib.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group(
        backend='nccl',
        init_method=f'file://{init_file}',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5),
    )
    try:
        for gradient_mode, option in product(('input', 'weight', 'bias', 'frozen', 'no_grad', 'mixed'), ('combined', 'l2')):
            _check_parallel_linear_cross_entropy(rank, world_size, 4103, torch.float32, option, 'mean', gradient_mode)
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


def test_fused_linear_cross_entropy_parallel_ascend_rejection(monkeypatch):
    from triton.language import math as triton_math

    # rejection happens before any Ascend kernel is executed.
    if not hasattr(triton_math, 'tanh'):
        monkeypatch.setattr(triton_math, 'tanh', None, raising=False)
    from fla.modules.fused_linear_cross_entropy.triton_ascend import fused_linear_cross_entropy_fwd_npu

    monkeypatch.setattr(dist, 'get_world_size', lambda group: 2)
    with pytest.raises(NotImplementedError, match="Vocabulary-parallel"):
        fused_linear_cross_entropy_fwd_npu(x=None, target=None, weight=None, process_group=object())


@pytest.mark.parametrize(
    ('route', 'defaults'),
    [('gluon', True), ('gluon', False), ('device', True), ('dtype', True), ('target_device', True)],
)
@pytest.mark.parametrize('direction', ['fwd', 'bwd'])
@pytest.mark.skipif(not IS_NVIDIA, reason="Gluon requires NVIDIA GPUs")
def test_cross_entropy_gluon_dispatch(route: str, defaults: bool, direction: str, monkeypatch):
    from fla.backends import _DISPATCH_DISABLED, GluonBackend
    from fla.modules.fused_cross_entropy import ops

    if not GluonBackend.is_available() or _DISPATCH_DISABLED:
        pytest.skip("Gluon dispatch is unavailable")
    monkeypatch.setenv('FLA_GLUON', '0')
    monkeypatch.setenv('FLA_CE_GLUON', '1')
    entry = getattr(ops, f'cross_entropy_{direction}')
    registry = backends._function_registries[inspect.unwrap(entry)]
    implementation = Mock()
    monkeypatch.setattr(registry._backends['gluon'], 'implementation', implementation)
    kernel = MagicMock()
    kernel.__getitem__.return_value.side_effect = RuntimeError('Triton fallback')
    monkeypatch.setattr(ops, f'cross_entropy_{direction}_kernel', kernel)
    logits = torch.empty(
        7,
        129,
        device='cpu' if route == 'device' else device,
        dtype=torch.float64 if route == 'dtype' else torch.float32,
    )
    target = torch.empty(7, device='cpu' if route == 'target_device' else device, dtype=torch.long)
    kwargs = dict(logits=logits, target=target)
    if direction == 'bwd':
        kwargs.update(lse=torch.empty(7, device=device), dloss=torch.empty(7, device=device))
    if not defaults:
        kwargs.update(label_smoothing=0.1, logit_scale=0.5, lse_square_scale=0.01, logit_softcapping=3.0, ignore_index=-7)
        if direction == 'bwd':
            kwargs.update(total_classes=258, class_start_idx=129, inplace_backward=True)
    if route == 'gluon':
        assert entry(**kwargs) is implementation.return_value
        implementation.assert_called_once_with(**kwargs)
        kernel.__getitem__.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match='Triton fallback'):
            entry(**kwargs)
        kernel.__getitem__.return_value.assert_called_once()
        implementation.assert_not_called()


def run_parallel_cross_entropy_worker(rank: int, world_size: int, init_file: str, backend: str):
    from fla.modules.fused_cross_entropy import ops

    calls = []
    if backend == 'gluon':
        for entry in (ops.cross_entropy_fwd, ops.cross_entropy_bwd):
            registry = backends._function_registries[inspect.unwrap(entry)]
            candidate = registry._backends['gluon']
            candidate.implementation = Mock(wraps=candidate.implementation)
            calls.append(candidate.implementation)
    device_torch_lib.set_device(rank)
    dist.init_process_group(
        backend='nccl',
        init_method=f'file://{init_file}',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5),
    )
    try:
        for V, dtype in product((128, 129, 32000, 65539), (torch.float16, torch.bfloat16, torch.float32)):
            for call in calls:
                call.reset_mock()
            torch.manual_seed(42)
            logits = torch.randn(7, world_size * V, device=device, dtype=dtype).requires_grad_()
            target = torch.randint(world_size * V, (7,), device=device)
            target[0], target[1], target[2] = -100, V - 1, V
            transformed = 3 * torch.tanh(logits.float() * 0.5 / 3)
            z_loss = (0.01 * transformed.logsumexp(-1).square()).masked_fill(target == -100, 0)
            ref = F.cross_entropy(input=transformed, target=target, reduction='none', label_smoothing=0.1) + z_loss
            dloss = torch.randn_like(ref)
            ref_grad, = torch.autograd.grad(outputs=ref, inputs=logits, grad_outputs=dloss)
            local = logits[:, rank * V:(rank + 1) * V].detach().clone().requires_grad_()
            loss, actual_z = ops.cross_entropy_loss(
                logits=local,
                target=target,
                label_smoothing=0.1,
                logit_scale=0.5,
                lse_square_scale=0.01,
                logit_softcapping=3.,
                inplace_backward=True,
                process_group=dist.group.WORLD,
            )
            assert loss.dtype == actual_z.dtype == torch.float32
            grad, = torch.autograd.grad(outputs=loss, inputs=local, grad_outputs=dloss)
            for call in calls:
                call.assert_called_once()
            assert_close('loss', ref, loss, ratio=1e-2)
            assert_close('z_loss', z_loss, actual_z, ratio=1e-2)
            assert_close('grad', ref_grad[:, rank * V:(rank + 1) * V], grad, ratio=1e-2)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('backend', ['triton', 'gluon'])
@pytest.mark.skipif(not IS_NVIDIA, reason="Vocabulary-parallel CE uses NCCL")
def test_cross_entropy_loss_parallel(backend: str, tmp_path, monkeypatch):
    from fla.backends import _DISPATCH_DISABLED, GluonBackend

    if backend == 'gluon' and (not GluonBackend.is_available() or _DISPATCH_DISABLED):
        pytest.skip("Gluon dispatch is unavailable")
    if device_torch_lib.device_count() < 2:
        pytest.skip("Requires 2 devices")
    monkeypatch.setenv('FLA_GLUON', '0')
    monkeypatch.setenv('FLA_CE_GLUON', '1' if backend == 'gluon' else '0')
    mp.spawn(fn=run_parallel_cross_entropy_worker, args=(2, str(tmp_path / 'init'), backend), nprocs=2, join=True)
