# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy

import pytest
import torch
import torch.nn.functional as F

from fla.modules import DynamicShortConvolution
from fla.modules.dynamic_conv import dynamic_conv1d
from fla.modules.dynamic_conv.naive import naive_dynamic_conv1d
from fla.ops.utils import prepare_chunk_indices
from fla.utils import assert_close, device, device_platform

pytestmark = pytest.mark.skipif(device_platform not in ("cuda", "hip"), reason="dynamic_conv1d requires CUDA/HIP")


@pytest.fixture(autouse=True)
def disable_tf32(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'R', 'W', 'lengths', 'state_dtype', 'bias', 'residual', 'strided', 'dtype', 'activation'),
    [
        (2, 63, 60, 7, 4, None, None, False, False, False, torch.float32, None),
        (2, 1, 33, 3, 8, None, torch.bfloat16, True, True, False, torch.bfloat16, 'swish'),
        (1, 5, 64, 1, 1, None, torch.float16, False, False, False, torch.float16, None),
        (2, 257, 128, 32, 4, None, torch.bfloat16, True, False, False, torch.bfloat16, 'swish'),
        (1, 80, 96, 17, 3, [0, 1, 0, 2, 77], torch.float32, True, True, False, torch.float32, 'swish'),
        (1, 129, 65, 52, 4, [63, 1, 65], None, False, True, True, torch.bfloat16, None),
        (2, 32, 32, 128, 2, None, None, True, True, False, torch.float16, 'swish'),
        (1, 0, 32, 4, 4, None, torch.float32, True, True, False, torch.float32, None),
        (1, 2051, 128, 16, 4, None, None, True, False, False, torch.bfloat16, None),
        (1, 97, 192, 16, 5, [2, 31, 64], torch.float16, True, False, True, torch.float16, 'swish'),
        (1, 0, 33, 7, 4, [0, 0], torch.bfloat16, True, True, False, torch.bfloat16, 'swish'),
        (1, 1048583, 1, 1, 1, None, None, False, False, False, torch.float32, None),
        (1, 35, 33, 129, 9, [0, 1, 2, 32], torch.float32, True, True, True, torch.float32, 'swish'),
        (2, 33, 65, 257, 17, None, torch.float32, True, True, False, torch.bfloat16, None),
        (2, 1, 16, 512, 32, None, torch.float16, False, False, False, torch.float16, 'swish'),
        (1, 18, 17, 16, 65, [0, 1, 17], torch.float32, True, False, False, torch.float32, None),
    ],
)
def test_dynamic_conv(B, T, D, R, W, lengths, state_dtype, bias, residual, strided, dtype, activation):
    torch.manual_seed(42)
    stride = 2 if strided else 1
    x = torch.randn(B, T, D * stride, device=device, dtype=dtype)[..., ::stride].requires_grad_()
    z = (torch.randn(B, T, R, device=device, dtype=dtype) / R**0.5).requires_grad_()
    out_proj_weight = torch.randn(R, W, D, device=device, dtype=dtype).requires_grad_()
    out_proj_bias = torch.randn(W, D, device=device, dtype=dtype).requires_grad_() if bias else None
    skip = torch.randn_like(x).requires_grad_() if residual else None
    cu = F.pad(torch.tensor(lengths, device=device, dtype=torch.int32).cumsum(0), (1, 0)) if lengths is not None else None
    N = len(lengths) if lengths is not None else B
    h0 = torch.randn(N, D, W, device=device, dtype=state_dtype).requires_grad_() if state_dtype is not None else None
    args = dict(
        x=x,
        z=z,
        out_proj_weight=out_proj_weight,
        out_proj_bias=out_proj_bias,
        residual=skip,
        initial_state=h0,
        output_final_state=True,
        activation=activation,
        cu_seqlens=cu,
    )
    ref, ref_ht = naive_dynamic_conv1d(**args)
    indices = prepare_chunk_indices(cu_seqlens=cu, chunk_size=16) if cu is not None and strided else None
    out, ht = dynamic_conv1d(**args, chunk_indices=indices)
    dy, dht = torch.randn_like(ref), torch.randn_like(ref_ht)
    tensors = [tensor for tensor in (x, z, out_proj_weight, out_proj_bias, skip, h0) if tensor is not None]
    ref_grads = torch.autograd.grad((ref, ref_ht), tensors, (dy, dht))
    grads = torch.autograd.grad((out, ht), tensors, (dy, dht))
    for name, expected, actual in [('output', ref, out), ('state', ref_ht, ht),
                                   *[(f'grad_{i}', a, b) for i, (a, b) in enumerate(zip(ref_grads, grads))]]:
        assert torch.isfinite(actual).all()
        if actual.numel():
            assert_close(name, expected, actual, 1e-3)


@pytest.mark.parametrize(
    ('kernel_size', 'bias', 'activation', 'dtype', 'rank', 'generator_activation', 'autocast'),
    [
        (4, True, None, torch.float32, 7, None, False),
        (4, True, 'swish', torch.float32, 7, None, False),
        (4, False, None, torch.bfloat16, 7, 'silu', False),
        (17, True, 'swish', torch.bfloat16, 129, 'silu', False),
        (4, True, None, torch.bfloat16, 7, None, True),
        (4, True, 'swish', torch.bfloat16, 7, None, True),
    ],
)
def test_dynamic_conv_module(kernel_size, bias, activation, dtype, rank, generator_activation, autocast):
    torch.manual_seed(42)
    self_conditioned = generator_activation is None and not autocast
    module = DynamicShortConvolution(
        hidden_size=48,
        kernel_size=kernel_size,
        bias=bias,
        activation=activation,
        device=device,
        dtype=torch.float32 if autocast else dtype,
        rank=rank,
        generator_input_size=None if self_conditioned else 32,
        generator_activation=generator_activation,
    )
    assert module.state_size == 48 * kernel_size
    x = torch.randn(2, 19, 48, device=device, dtype=dtype).requires_grad_()
    generator = x if self_conditioned else torch.randn(2, 19, 32, device=device, dtype=dtype).requires_grad_()
    residual = torch.randn_like(x).requires_grad_()
    mask = torch.ones(2, 19, device=device) if generator_activation is not None else None
    if mask is not None:
        mask[:, -3:] = 0
    values = x if mask is None else x * mask.unsqueeze(-1).to(x.dtype)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast):
        initial, _ = module(x=x, residual=residual, mask=mask, generator_input=None if self_conditioned else generator)
        with torch.autocast(device_type="cuda", enabled=False):
            weight = module.out_proj_bias.float() if bias else values.new_zeros(kernel_size, 48, dtype=torch.float32)
            static = F.conv1d(
                values.float().transpose(1, 2),
                weight.T.flip(-1).unsqueeze(1),
                padding=kernel_size - 1,
                groups=48,
            )[..., :19].transpose(1, 2)
            if activation is not None:
                static = F.silu(static)
            static = static + residual.float()
        assert_close('static initialization', static.to(dtype), initial, 1e-3)
        with torch.no_grad():
            module.out_proj_weight.normal_(std=0.1)
        reference = copy.deepcopy(module)
        ref_x = x.detach().clone().requires_grad_()
        ref_generator = ref_x if self_conditioned else generator.detach().clone().requires_grad_()
        ref_residual = residual.detach().clone().requires_grad_()
        ref_values = ref_x if mask is None else ref_x * mask.unsqueeze(-1).to(ref_x.dtype)
        gen_values = ref_generator if mask is None else ref_generator * mask.unsqueeze(-1).to(ref_generator.dtype)
        z = reference.in_proj(gen_values)
        if generator_activation is not None:
            z = F.silu(z)
        ref, _ = naive_dynamic_conv1d(
            x=ref_values,
            z=z,
            out_proj_weight=reference.out_proj_weight,
            out_proj_bias=reference.out_proj_bias,
            residual=ref_residual,
            activation=activation,
        )
        out, _ = module(x=x, residual=residual, mask=mask, generator_input=None if self_conditioned else generator)
        dy = torch.randn_like(out)
    out.backward(dy)
    ref.backward(dy)
    assert_close('module', ref, out, 1e-3)
    expected_inputs = [ref_x, ref_generator, ref_residual, *reference.parameters()]
    actual_inputs = [x, generator, residual, *module.parameters()]
    for expected, actual in zip(expected_inputs, actual_inputs):
        assert_close('module gradient', expected.grad, actual.grad, 1e-3)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast):
        cache, parts = None, []
        for start, end in [(0, 2), (2, 7), (7, 18), (18, 19)]:
            old_cache = cache.clone() if cache is not None else None
            part, new_cache = module(
                x=x[:, start:end],
                residual=residual[:, start:end],
                mask=mask[:, start:end] if mask is not None else None,
                cache=cache,
                output_final_state=True,
                generator_input=None if self_conditioned else generator[:, start:end],
            )
            if old_cache is not None:
                if end - start == 1:
                    assert new_cache is cache
                else:
                    torch.testing.assert_close(cache, old_cache, rtol=0, atol=0)
            cache = new_cache
            parts.append(part)
        assert_close('cached continuation', out, torch.cat(parts, dim=1), 1e-3)
        torch.testing.assert_close(cache, values[:, -kernel_size:].transpose(1, 2), rtol=0, atol=0)


def test_dynamic_conv_packed_step():
    torch.manual_seed(42)
    module = DynamicShortConvolution(hidden_size=16, kernel_size=4, device=device, dtype=torch.bfloat16)
    x = torch.randn(1, 3, 16, device=device, dtype=torch.bfloat16)
    cache = torch.randn(3, 16, 4, device=device, dtype=torch.bfloat16)
    cu = torch.arange(4, device=device, dtype=torch.int32)
    with torch.no_grad():
        expected, state = module(x=x, cache=cache.clone(), output_final_state=True, cu_seqlens=cu)
        output, returned = module(x=x, cache=cache, cu_seqlens=cu)
    assert returned is None
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(cache, state, rtol=0, atol=0)


@pytest.mark.parametrize('invalid', ['rank', 'width', 'out_proj_bias', 'residual', 'state', 'packed_batch', 'activation'])
def test_dynamic_conv_invalid(invalid):
    torch.manual_seed(42)
    x = torch.randn(2, 5, 16, device=device)
    z = torch.randn(2, 5, 4, device=device)
    out_proj_weight = torch.randn(4, 3, 16, device=device)
    kwargs = {}
    if invalid == 'rank':
        z = torch.randn(2, 5, 0, device=device)
        out_proj_weight = torch.randn(0, 3, 16, device=device)
    elif invalid == 'width':
        out_proj_weight = torch.randn(4, 0, 16, device=device)
    elif invalid == 'out_proj_bias':
        kwargs['out_proj_bias'] = torch.randn(16, device=device)
    elif invalid == 'residual':
        kwargs['residual'] = torch.randn(2, 4, 16, device=device)
    elif invalid == 'state':
        kwargs['initial_state'] = torch.randn(2, 16, 2, device=device)
    elif invalid == 'packed_batch':
        kwargs['cu_seqlens'] = torch.tensor([0, 5], device=device, dtype=torch.int32)
    elif invalid == 'activation':
        kwargs['activation'] = 'relu'
    with pytest.raises(ValueError):
        dynamic_conv1d(x=x, z=z, out_proj_weight=out_proj_weight, **kwargs)
