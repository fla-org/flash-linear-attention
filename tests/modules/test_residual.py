# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fla.modules import AttentionResidual, StandardResidual
from fla.ops.attnres import naive_attnres
from fla.utils import assert_close, device, device_platform


@pytest.mark.parametrize('block_size', [0, -1, True, 1.5])
def test_attnres_block_size(block_size):
    with pytest.raises(ValueError, match='positive integer'):
        AttentionResidual(hidden_size=8, block_size=block_size)


@pytest.mark.parametrize('module_class', [StandardResidual, AttentionResidual])
def test_initialization(module_class):
    torch.manual_seed(42)
    kwargs = dict(hidden_size=8, fuse_norm=False)
    module = module_class(**kwargs)
    defaults = {name: value.clone() for name, value in module.state_dict().items()}
    for name, value in defaults.items():
        expected = torch.zeros_like(value) if name == 'query.weight' else torch.ones_like(value)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_()
    restored = module_class(**kwargs)
    restored.load_state_dict(module.state_dict(), strict=True)
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, module.state_dict()[name], rtol=0, atol=0)
    module.reset_parameters()
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, defaults[name], rtol=0, atol=0)


@pytest.mark.parametrize('output_norm', [False, True])
def test_standard_empty_state(output_norm):
    torch.manual_seed(42)
    module = StandardResidual(hidden_size=8, fuse_norm=False)
    x = torch.randn(2, 7, 8, requires_grad=True)
    state = module.initialize(x)
    ref = F.rms_norm(x, (8,), module.norm.weight, module.norm.eps) if output_norm else x
    actual = module(state) if output_norm else module.readout(state)
    torch.testing.assert_close(actual, ref)
    do = torch.randn_like(x)
    torch.testing.assert_close(torch.autograd.grad(actual, x, do)[0], torch.autograd.grad(ref, x, do)[0])


@pytest.mark.skipif(device_platform == 'cpu', reason='Residual numerical tests require an accelerator')
@pytest.mark.parametrize('output_norm', [False, True])
@pytest.mark.parametrize(('dtype', 'input_dtype', 'fuse_norm', 'fuse_residual'), [
    pytest.param(torch.float32, torch.float32, False, False, id='fp32'),
    pytest.param(torch.float16, torch.float16, True, False, id='fp16-fused-norm'),
    pytest.param(torch.bfloat16, torch.bfloat16, False, True, id='bf16-unfused-norm'),
    pytest.param(torch.bfloat16, torch.bfloat16, True, True, id='bf16-fused-add-norm'),
    pytest.param(torch.float16, torch.float16, True, True, id='fp16-fused-add-norm'),
    pytest.param(torch.bfloat16, torch.float32, True, False, id='mixed-bf16'),
    pytest.param(torch.bfloat16, torch.float32, True, True, id='mixed-bf16-fused-add-norm'),
    pytest.param(torch.float16, torch.float32, True, True, id='mixed-fp16-fused-add-norm'),
])
def test_standard(dtype, input_dtype, fuse_norm, fuse_residual, output_norm):
    torch.manual_seed(42)
    modules = nn.ModuleList([
        StandardResidual(hidden_size=128, fuse_norm=fuse_norm, fuse_residual=fuse_residual) for _ in range(7)
    ]).to(device=device, dtype=dtype)
    x = torch.randn(2, 7, 128, device=device, dtype=input_dtype, requires_grad=True)
    state = modules[0].initialize(x)
    ref_h = x.clone()
    for i, module in enumerate(modules[:-1]):
        ref_dtype = dtype if fuse_residual and fuse_norm and i > 0 else input_dtype
        ref_input = F.rms_norm(ref_h.float(), (128,), module.norm.weight.float(), module.norm.eps).to(ref_dtype)
        actual_input, context = module.pre(state)
        assert actual_input.dtype == ref_dtype
        assert_close('input', ref_input, actual_input, 0.005)
        assert_close('context', ref_h.to(input_dtype), context, 0.005)
        previous = state
        previous_h = module.readout(previous).detach().clone()
        state = module.post(F.silu(actual_input).to(dtype), context)
        torch.testing.assert_close(module.readout(previous), previous_h, rtol=0, atol=0)
        ref_branch = F.silu(ref_input).to(dtype)
        # fused prenorm reads the fp32 sum but stores its residual in input_dtype.
        if fuse_residual and fuse_norm:
            ref_h = ref_h.to(input_dtype).float() + ref_branch.float()
        else:
            ref_h = ref_h + ref_branch
        actual_h = module.readout(state)
        assert actual_h.dtype == input_dtype
        assert_close('state', ref_h.to(input_dtype), actual_h, 0.005)

    ref = ref_h.to(input_dtype)
    if output_norm:
        ref = F.rms_norm(ref.float(), (128,), modules[-1].norm.weight.float(), modules[-1].norm.eps).to(input_dtype)
    actual = modules[-1](state) if output_norm else modules[-1].readout(state)
    assert actual.dtype == ref.dtype
    assert_close('output', ref, actual, 0.005)
    do = torch.randn_like(actual)
    parameters = {'x': x, **dict(modules.named_parameters())}
    ref_grads = torch.autograd.grad(ref, tuple(parameters.values()), do, allow_unused=True)
    actual_grads = torch.autograd.grad(actual, tuple(parameters.values()), do, allow_unused=True)
    for name, ref_grad, actual_grad in zip(parameters, ref_grads, actual_grads):
        if ref_grad is None:
            assert actual_grad is None, name
        else:
            assert actual_grad is not None, name
            assert_close(name, ref_grad, actual_grad, 0.005)


@pytest.mark.skipif(device_platform == 'cpu', reason='AttnRes requires an accelerator')
@pytest.mark.parametrize('output_norm', [False, True])
@pytest.mark.parametrize(('block_size', 'dtype', 'input_dtype'), [
    pytest.param(1, torch.float32, torch.float32, id='full-fp32'),
    pytest.param(4, torch.float16, torch.float16, id='block-fp16'),
    pytest.param(4, torch.bfloat16, torch.bfloat16, id='block-bf16'),
    pytest.param(4, torch.bfloat16, torch.float32, id='block-mixed-dtype'),
    pytest.param(1, torch.bfloat16, torch.bfloat16, id='full-bf16'),
])
def test_attnres(block_size, dtype, input_dtype, output_norm, request):
    torch.manual_seed(42)
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    request.addfinalizer(lambda: setattr(torch.backends.cuda.matmul, 'allow_tf32', allow_tf32))
    torch.backends.cuda.matmul.allow_tf32 = False
    modules = nn.ModuleList([
        AttentionResidual(hidden_size=128, block_size=block_size) for _ in range(7)
    ]).to(device=device, dtype=dtype)
    with torch.no_grad():
        for module in modules:
            module.query.weight.normal_(std=0.03)
    x = torch.randn(2, 7, 128, device=device, dtype=input_dtype, requires_grad=True)
    state = modules[0].initialize(x)
    ref_blocks, ref_branches = [x.clone()], []
    for i, module in enumerate(modules[:-1]):
        if i == 0:
            ref_input = F.rms_norm(x.float(), (128,), module.norm.weight.float(), module.norm.eps).to(input_dtype)
        else:
            ref_input = naive_attnres(
                query=module.query.weight,
                residuals=ref_blocks,
                rms_weight=module.attnres_norm.weight,
                output_rms_weight=module.norm.weight,
                rms_eps=module.norm.eps,
            )
        actual_input, context = module.pre(state)
        assert actual_input.dtype == ref_input.dtype
        assert_close('input', ref_input, actual_input, 0.005)
        previous = state
        previous_blocks = [block.detach().clone() for block in previous.blocks]
        state = module.post(F.silu(actual_input).to(dtype), context)
        for before, after in zip(previous_blocks, previous.blocks):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        ref_branches.append(F.silu(ref_input).to(dtype))
        ref_blocks = [x.to(dtype)] + [
            sum(ref_branches[start:start + block_size]) for start in range(0, len(ref_branches), block_size)
        ]
        assert state.num_branches == i + 1
        assert len(state.blocks) == len(ref_blocks)
        for ref, actual in zip(ref_blocks, state.blocks):
            assert actual.dtype == ref.dtype
            assert_close('block', ref, actual, 0.005)

    module = modules[-1]
    ref = naive_attnres(
        query=module.query.weight,
        residuals=ref_blocks,
        rms_weight=module.attnres_norm.weight,
        output_rms_weight=module.norm.weight if output_norm else None,
        rms_eps=module.norm.eps,
    )
    actual = module(state) if output_norm else module.readout(state)
    assert actual.dtype == ref.dtype
    assert_close('output', ref, actual, 0.005)
    do = torch.randn_like(actual)
    parameters = {'x': x, **dict(modules.named_parameters())}
    ref_grads = torch.autograd.grad(ref, tuple(parameters.values()), do, allow_unused=True)
    actual_grads = torch.autograd.grad(actual, tuple(parameters.values()), do, allow_unused=True)
    for name, ref_grad, actual_grad in zip(parameters, ref_grads, actual_grads):
        if ref_grad is None:
            assert actual_grad is None, name
        else:
            assert actual_grad is not None, name
            assert_close(name, ref_grad, actual_grad, 0.005)
