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


@pytest.mark.parametrize('fuse_residual', [False, True], ids=['eager-add', 'pending-add'])
def test_standard_post(fuse_residual: bool):
    """A branch contributes once, whether addition happens in post or during readout."""
    torch.manual_seed(42)
    module = StandardResidual(hidden_size=8, fuse_norm=False, fuse_residual=fuse_residual)
    x = torch.randn(2, 7, 8)
    branch = torch.randn_like(x, dtype=torch.bfloat16)
    previous = module.initialize(x=x)
    _, context = module.pre(state=previous)
    state = module.post(branch_output=branch, context=context)
    assert state.residual.dtype == x.dtype
    if fuse_residual:
        assert torch.equal(state.residual, x)
        assert state.branch_output is branch
    else:
        assert torch.equal(state.residual, x + branch)
        assert state.branch_output is None
    assert torch.equal(module.readout(state=state), x + branch)
    assert torch.equal(module.readout(state=previous), x)
    assert previous.branch_output is None


@pytest.mark.skipif(device_platform == 'cpu', reason='Residual numerical tests require an accelerator')
@pytest.mark.parametrize(
    ('dtype', 'input_dtype', 'fuse_norm', 'fuse_residual', 'output_norm'),
    [
        pytest.param(torch.float32, torch.float32, False, False, True, id='fp32'),
        pytest.param(torch.float16, torch.float16, True, False, True, id='fp16-fused-norm'),
        pytest.param(torch.bfloat16, torch.bfloat16, False, True, True, id='bf16-unfused-norm'),
        pytest.param(torch.bfloat16, torch.bfloat16, True, True, True, id='bf16-fused-add-norm'),
        pytest.param(torch.float16, torch.float16, True, True, False, id='fp16-readout'),
        pytest.param(torch.bfloat16, torch.float32, True, False, False, id='mixed-bf16-readout'),
        pytest.param(torch.bfloat16, torch.float32, True, True, True, id='mixed-bf16-fused-add-norm'),
        pytest.param(torch.float16, torch.float32, True, True, True, id='mixed-fp16-fused-add-norm'),
    ],
)
def test_standard(dtype: torch.dtype, input_dtype: torch.dtype, fuse_norm: bool, fuse_residual: bool, output_norm: bool):
    """Check a connected residual chain, including fused rounding and gradients across six branches."""
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

    state = modules[0].initialize(x=x)
    inputs, contexts, states = [], [], [modules[0].readout(state=state)]
    for module in modules[:-1]:
        branch_input, context = module.pre(state=state)
        inputs.append(branch_input)
        contexts.append(context)
        state = module.post(branch_output=F.silu(branch_input).to(dtype), context=context)
        states.append(module.readout(state=state))
    out = modules[-1](state=state) if output_norm else modules[-1].readout(state=state)

    for i, (expected, branch_input, context) in enumerate(zip(ref_inputs, inputs, contexts)):
        assert branch_input.dtype == expected.dtype
        assert_close(f'input_{i}', expected, branch_input, 0.005)
        assert_close(f'context_{i}', ref_states[i], context, 0.005)
    for i, (expected, hidden) in enumerate(zip(ref_states, states)):
        assert hidden.dtype == expected.dtype
        assert_close(f'state_{i}', expected, hidden, 0.005)
    assert out.dtype == ref.dtype
    assert_close('o', ref, out, 0.005)

    do = torch.randn_like(out)
    parameters = {'x': x, **dict(modules.named_parameters())}
    ref_grads = torch.autograd.grad(ref, tuple(parameters.values()), do, allow_unused=True)
    out_grads = torch.autograd.grad(out, tuple(parameters.values()), do, allow_unused=True)
    for name, ref_grad, out_grad in zip(parameters, ref_grads, out_grads):
        if ref_grad is None:
            assert out_grad is None, name
        else:
            assert out_grad is not None, name
            assert_close(name, ref_grad, out_grad, 0.005)
