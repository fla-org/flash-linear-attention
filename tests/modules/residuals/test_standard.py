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

from fla.modules import StandardResidual
from fla.utils import assert_close, device, device_platform


def test_initialization():
    torch.manual_seed(42)
    module = StandardResidual(hidden_size=8, fuse_norm=False)
    expected = torch.ones_like(module.norm.weight)
    torch.testing.assert_close(module.norm.weight, expected, rtol=0, atol=0)
    with torch.no_grad():
        module.norm.weight.normal_()
    restored = StandardResidual(hidden_size=8, fuse_norm=False)
    restored.load_state_dict(module.state_dict(), strict=True)
    torch.testing.assert_close(restored.norm.weight, module.norm.weight, rtol=0, atol=0)
    module.reset_parameters()
    torch.testing.assert_close(module.norm.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize('output_norm', [False, True], ids=['readout', 'output-norm'])
def test_empty_state(output_norm):
    torch.manual_seed(42)
    module = StandardResidual(hidden_size=8, fuse_norm=False)
    x = torch.randn(2, 7, 8, requires_grad=True)
    state = module.initialize(x)
    ref = F.rms_norm(x, (8,), module.norm.weight, module.norm.eps) if output_norm else x
    actual = module(state) if output_norm else module.readout(state)
    torch.testing.assert_close(actual, ref)
    do = torch.randn_like(x)
    torch.testing.assert_close(torch.autograd.grad(actual, x, do)[0], torch.autograd.grad(ref, x, do)[0])


@pytest.mark.parametrize('fuse_residual', [False, True], ids=['eager-add', 'pending-add'])
def test_post(fuse_residual):
    torch.manual_seed(42)
    module = StandardResidual(hidden_size=8, fuse_norm=False, fuse_residual=fuse_residual)
    x = torch.randn(2, 7, 8)
    branch = torch.randn_like(x, dtype=torch.bfloat16)
    previous = module.initialize(x)
    _, context = module.pre(previous)
    state = module.post(branch, context)
    if fuse_residual:
        torch.testing.assert_close(state.residual, x, rtol=0, atol=0)
        assert state.branch_output is branch
    else:
        torch.testing.assert_close(state.residual, x + branch, rtol=0, atol=0)
        assert state.branch_output is None
    torch.testing.assert_close(module.readout(state), x + branch, rtol=0, atol=0)
    torch.testing.assert_close(module.readout(previous), x, rtol=0, atol=0)
    assert previous.branch_output is None


@pytest.mark.skipif(device_platform == 'cpu', reason='Residual numerical tests require an accelerator')
@pytest.mark.parametrize(('dtype', 'input_dtype', 'fuse_norm', 'fuse_residual', 'output_norm'), [
    pytest.param(torch.float32, torch.float32, False, False, True, id='fp32'),
    pytest.param(torch.float16, torch.float16, True, False, True, id='fp16-fused-norm'),
    pytest.param(torch.bfloat16, torch.bfloat16, False, True, True, id='bf16-unfused-norm'),
    pytest.param(torch.bfloat16, torch.bfloat16, True, True, True, id='bf16-fused-add-norm'),
    pytest.param(torch.float16, torch.float16, True, True, False, id='fp16-readout'),
    pytest.param(torch.bfloat16, torch.float32, True, False, False, id='mixed-bf16-readout'),
    pytest.param(torch.bfloat16, torch.float32, True, True, True, id='mixed-bf16-fused-add-norm'),
    pytest.param(torch.float16, torch.float32, True, True, True, id='mixed-fp16-fused-add-norm'),
])
def test_forward_backward(dtype, input_dtype, fuse_norm, fuse_residual, output_norm):
    torch.manual_seed(42)
    modules = nn.ModuleList([
        StandardResidual(hidden_size=128, fuse_norm=fuse_norm, fuse_residual=fuse_residual) for _ in range(7)
    ]).to(device=device, dtype=dtype)
    x = torch.randn(2, 7, 128, device=device, dtype=input_dtype, requires_grad=True)

    ref_h = x.clone()
    ref_inputs, ref_states = [], [ref_h]
    for i, module in enumerate(modules[:-1]):
        ref_dtype = dtype if fuse_residual and fuse_norm and i > 0 else input_dtype
        ref_input = F.rms_norm(ref_h.float(), (128,), module.norm.weight.float(), module.norm.eps).to(ref_dtype)
        ref_inputs.append(ref_input)
        ref_branch = F.silu(ref_input).to(dtype)
        # fused prenorm reads the fp32 sum but stores its residual in input_dtype.
        if fuse_residual and fuse_norm:
            ref_h = ref_h.to(input_dtype).float() + ref_branch.float()
        else:
            ref_h = ref_h + ref_branch
        ref_states.append(ref_h.to(input_dtype))
    ref = ref_states[-1]
    if output_norm:
        ref = F.rms_norm(ref.float(), (128,), modules[-1].norm.weight.float(), modules[-1].norm.eps).to(input_dtype)

    state = modules[0].initialize(x)
    inputs, contexts, states = [], [], [modules[0].readout(state)]
    for module in modules[:-1]:
        branch_input, context = module.pre(state)
        inputs.append(branch_input)
        contexts.append(context)
        state = module.post(F.silu(branch_input).to(dtype), context)
        states.append(module.readout(state))
    actual = modules[-1](state) if output_norm else modules[-1].readout(state)

    for i, (expected, branch_input, context) in enumerate(zip(ref_inputs, inputs, contexts)):
        assert branch_input.dtype == expected.dtype
        assert_close(f'input_{i}', expected, branch_input, 0.005)
        assert_close(f'context_{i}', ref_states[i], context, 0.005)
    for i, (expected, hidden) in enumerate(zip(ref_states, states)):
        assert hidden.dtype == expected.dtype
        assert_close(f'state_{i}', expected, hidden, 0.005)
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
