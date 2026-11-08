# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.layers.phba import PHBALinear
from fla.ops.phba.backends.cublaslt import exact_backend
from fla.utils import IS_NVIDIA, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='PHBA optional NVIDIA projection path')


def _offset_copy(tensor, offset):
    result = torch.empty(tensor.numel() + offset, device=tensor.device, dtype=tensor.dtype)[offset:].view_as(tensor)
    result.copy_(tensor.detach())
    assert result.is_contiguous() and result.data_ptr() % 16 == offset * tensor.element_size() % 16
    return result


def _compare_linear(out_features, input_offset=None, grad_offset=0):
    torch.manual_seed(42)
    module = PHBALinear(2048, out_features, bias=False).to(device).train()
    dtype = torch.float32 if input_offset is None else torch.bfloat16
    x = torch.randn(1, 8192, 2048, device=device, dtype=dtype)
    x = _offset_copy(x, input_offset or 0).requires_grad_()
    baseline_x = _offset_copy(x, input_offset or 0).requires_grad_()
    baseline_w = module.weight.detach().clone().requires_grad_()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = module(x)
        expected = F.linear(baseline_x, baseline_w)
    grad = _offset_copy(torch.randn_like(output), grad_offset)
    actual_grads = torch.autograd.grad(output, (x, module.weight), grad)
    expected_grads = torch.autograd.grad(expected, (baseline_x, baseline_w), grad)
    assert torch.equal(output, expected)
    for actual, reference in zip(actual_grads, expected_grads):
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


def test_linear_unscreened_fallback():
    torch.manual_seed(42)
    module = PHBALinear(32, 17, bias=True).to(device)
    x = torch.randn(3, 33, 32, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    weight = module.weight.detach().clone().requires_grad_()
    bias = module.bias.detach().clone().requires_grad_()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        actual = module(x)
        expected = F.linear(y, weight, bias)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, module.weight, module.bias), grad)
    expected_grads = torch.autograd.grad(expected, (y, weight, bias), grad)
    assert torch.equal(actual, expected)
    for dx, dy in zip(actual_grads, expected_grads):
        assert torch.equal(dx, dy)


def _compare_swiglu_down(input_offset=None, grad_offset=0):
    from fla.modules.activations import swiglu_linear
    from fla.ops.phba.backends.cublaslt import _SwiGLULinear

    torch.manual_seed(42)
    x = torch.randn(1, 8192, 4272, device=device, dtype=torch.bfloat16)
    y = _offset_copy(torch.randn_like(x), input_offset or 0).requires_grad_()
    x = _offset_copy(x, input_offset or 0).requires_grad_()
    weight = torch.randn(2048, 4272, device=device, requires_grad=True)
    reference_inputs = [_offset_copy(tensor, input_offset or 0).requires_grad_() for tensor in (x, y)]
    reference_inputs.append(weight.detach().clone().requires_grad_())
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        actual = _SwiGLULinear.apply(x, y, weight, None)
        expected = swiglu_linear(*reference_inputs, None)
    grad = _offset_copy(torch.randn_like(actual), grad_offset)
    actual_gradients = torch.autograd.grad(actual, (x, y, weight), grad)
    expected_gradients = torch.autograd.grad(expected, reference_inputs, grad)
    assert torch.equal(actual, expected)
    for result, reference in zip(actual_gradients, expected_gradients):
        assert torch.equal(result.view(torch.uint8), reference.view(torch.uint8))


def _compare_projection(geometry, input_offset=None, grad_offset=0):
    if geometry == 'ffn-down':
        _compare_swiglu_down(input_offset, grad_offset)
    else:
        _compare_linear(2048 if geometry == 'square' else 4272, input_offset, grad_offset)


@pytest.mark.parametrize('geometry', ['square', 'ffn-up', 'ffn-down'])
def test_projection_native_bytes(geometry):
    _compare_projection(geometry)


def _exact_backend_or_skip():
    import os

    backend = exact_backend(torch.device('cuda', torch.cuda.current_device()))
    if backend is None:
        if os.environ.get('PHBA_REQUIRE_EXACT_GEMM') == '1':
            pytest.fail('The pinned H20 gate requires a working exact backend')
        pytest.skip('Exact backend runtime, library fingerprint or build prerequisites do not match')
    return backend


@pytest.mark.parametrize(
    ('geometry', 'input_offset', 'grad_offset', 'expected_calls'),
    [
        pytest.param('square', None, 0, 1, id='square-fp32-autocast'),
        pytest.param('ffn-up', None, 0, 1, id='ffn-up-fp32-autocast'),
        pytest.param('ffn-down', None, 0, 1, id='ffn-down-default'),
        pytest.param('square', 8, 8, 1, id='square-aligned-bf16-views'),
        pytest.param('ffn-up', 8, 8, 1, id='ffn-up-aligned-bf16-views'),
        pytest.param('ffn-down', 8, 8, 1, id='ffn-down-aligned-bf16-views'),
        pytest.param('square', 1, 0, 0, id='square-unaligned-input'),
        pytest.param('ffn-up', 1, 0, 0, id='ffn-up-unaligned-input'),
        # SwiGLU materializes an aligned GEMM operand even when its inputs are unaligned
        pytest.param('ffn-down', 1, 0, 1, id='ffn-down-unaligned-activation'),
        pytest.param('square', 0, 1, 0, id='square-unaligned-gradient'),
        pytest.param('ffn-up', 0, 1, 0, id='ffn-up-unaligned-gradient'),
        pytest.param('ffn-down', 0, 1, 0, id='ffn-down-unaligned-gradient'),
    ],
)
def test_exact_backend_executes_geometry(geometry, input_offset, grad_offset, expected_calls, monkeypatch):
    backend = _exact_backend_or_skip()
    calls = []
    native = backend.library.run_algo

    def observe(*args):
        calls.append(args[0])
        return native(*args)

    monkeypatch.setattr(backend.library, 'run_algo', observe)
    _compare_projection(geometry, input_offset, grad_offset)
    assert len(calls) == expected_calls, f'{geometry} selected the wrong alignment path'


@pytest.mark.parametrize(
    ('m', 'n', 'operand'),
    [(2048, 2048, 'a'), (2048, 2048, 'b'), (4272, 2048, 'a'),
     (4272, 2048, 'b'), (2048, 4272, 'a'), (2048, 4272, 'b')],
    ids=['square-a', 'square-b', 'ffn-up-a', 'ffn-up-b', 'ffn-down-a', 'ffn-down-b'],
)
def test_exact_backend_unaligned_operands(m, n, operand, monkeypatch):
    torch.manual_seed(42)
    backend = _exact_backend_or_skip()
    a = _offset_copy(torch.randn(8192, m, device=device, dtype=torch.bfloat16), int(operand == 'a')).t()
    b = _offset_copy(torch.randn(8192, n, device=device, dtype=torch.bfloat16), int(operand == 'b'))

    def unexpected(*args):
        pytest.fail('Unaligned GEMM operands must use the native weight gradient')

    monkeypatch.setattr(backend.library, 'run_algo', unexpected)
    actual = backend.weight_gradient(a=a, b=b)
    expected = torch.mm(a, b)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize('condition', ['cuda-version', 'library-fingerprint', 'build-failure'])
def test_runtime_fallback_gradients(condition, monkeypatch):
    from fla.ops.phba.backends import cublaslt

    if condition != 'cuda-version':
        _exact_backend_or_skip()

    monkeypatch.setattr(cublaslt, '_INITIALIZED', False)
    monkeypatch.setattr(cublaslt, '_BACKEND', None)
    if condition == 'cuda-version':
        monkeypatch.setattr(torch.version, 'cuda', 'unsupported')
    elif condition == 'library-fingerprint':
        monkeypatch.setattr(cublaslt, 'EXACT_GEMM_ALGORITHMS', {
            **cublaslt.EXACT_GEMM_ALGORITHMS, 'library_sha256': '0' * 64,
        })
    else:
        def unavailable(*args, **kwargs):
            raise RuntimeError('test compiler unavailable')

        monkeypatch.setattr(cublaslt, '_ExactBackend', unavailable)
        with pytest.warns(UserWarning, match='test compiler unavailable'):
            assert exact_backend(torch.device('cuda', torch.cuda.current_device())) is None
    assert exact_backend(torch.device('cuda', torch.cuda.current_device())) is None
    _compare_linear(2048)
