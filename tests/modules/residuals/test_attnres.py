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

from fla.modules import AttentionResidual
from fla.ops.attnres import naive_attnres
from fla.utils import assert_close, device, device_platform


@pytest.mark.parametrize('block_size', [0, -1, True, 1.5], ids=['zero', 'negative', 'bool', 'float'])
def test_attnres_invalid_block_size(block_size: int | float):
    with pytest.raises(ValueError, match='positive integer'):
        AttentionResidual(hidden_size=8, block_size=block_size)


@pytest.mark.skipif(device_platform == 'cpu', reason='AttnRes requires an accelerator')
@pytest.mark.parametrize(('block_size', 'reset_parameters'), [(1, False), (4, True)], ids=['full-init', 'block-reset'])
def test_attnres_uniform_readout(block_size: int, reset_parameters: bool):
    """A zero query must average the available residual sources after construction or reset."""
    torch.manual_seed(42)
    module = AttentionResidual(hidden_size=128, block_size=block_size).to(device=device)
    if reset_parameters:
        with torch.no_grad():
            module.query.weight.normal_()
        module.reset_parameters()
    x = torch.randn(2, 7, 128, device=device)
    branches = [torch.randn_like(x) for _ in range(6)]
    sources = [x] + [sum(branches[start:start + block_size]) for start in range(0, len(branches), block_size)]
    ref = torch.stack(sources).mean(0)

    state = module.initialize(x=x)
    for branch in branches:
        state = module.post(branch_output=branch, context=state)
    out = module.readout(state=state)

    assert_close('o', ref, out, 0.005)


@pytest.mark.parametrize('block_size', [1, 4], ids=['full', 'block'])
def test_attnres_post(block_size: int):
    """Starting or extending a block must leave every previously returned state unchanged."""
    torch.manual_seed(42)
    module = AttentionResidual(hidden_size=8, block_size=block_size, fuse_norm=False)
    x = torch.randn(2, 7, 8)
    branches = [torch.randn_like(x, dtype=torch.bfloat16) for _ in range(6)]
    state = module.initialize(x=x)
    assert torch.equal(module.readout(state=state), x)
    for i, branch in enumerate(branches):
        previous = state
        snapshots = [block.clone() for block in previous.blocks]
        state = module.post(branch_output=branch, context=previous)
        assert previous.num_branches == i
        assert len(previous.blocks) == len(snapshots)
        for before, after in zip(snapshots, previous.blocks):
            assert torch.equal(after, before)
        expected = [x.to(branch.dtype)] + [
            sum(branches[start:min(start + block_size, i + 1)]) for start in range(0, i + 1, block_size)
        ]
        assert state.num_branches == i + 1
        assert len(state.blocks) == len(expected)
        for ref, out in zip(expected, state.blocks):
            assert out.dtype == ref.dtype
            assert torch.equal(out, ref)


@pytest.mark.skipif(device_platform == 'cpu', reason='AttnRes requires an accelerator')
@pytest.mark.parametrize(
    ('block_size', 'dtype', 'input_dtype', 'output_norm'),
    [
        pytest.param(1, torch.float32, torch.float32, False, id='full-fp32-readout'),
        pytest.param(4, torch.float16, torch.float16, True, id='block-fp16'),
        pytest.param(4, torch.bfloat16, torch.bfloat16, False, id='block-bf16-readout'),
        pytest.param(4, torch.bfloat16, torch.float32, True, id='block-mixed-dtype'),
        pytest.param(1, torch.bfloat16, torch.bfloat16, True, id='full-bf16'),
    ],
)
def test_attnres(
    block_size: int,
    dtype: torch.dtype,
    input_dtype: torch.dtype,
    output_norm: bool,
    request: pytest.FixtureRequest,
):
    """Check cross-layer attention and block sums against a differentiable six-branch reference."""
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

    ref_inputs, ref_branches, ref_states = [], [], [[x.clone()]]
    for i, module in enumerate(modules[:-1]):
        if i == 0:
            ref_input = F.rms_norm(x.float(), (128,), module.norm.weight.float(), module.norm.eps).to(input_dtype)
        else:
            ref_input = naive_attnres(
                query=module.query.weight,
                residuals=ref_states[-1],
                rms_weight=module.attnres_norm.weight,
                output_rms_weight=module.norm.weight,
                rms_eps=module.norm.eps,
            )
        ref_inputs.append(ref_input)
        ref_branches.append(F.silu(ref_input).to(dtype))
        ref_states.append([x.to(dtype)] + [
            sum(ref_branches[start:start + block_size]) for start in range(0, len(ref_branches), block_size)
        ])
    module = modules[-1]
    ref = naive_attnres(
        query=module.query.weight,
        residuals=ref_states[-1],
        rms_weight=module.attnres_norm.weight,
        output_rms_weight=module.norm.weight if output_norm else None,
        rms_eps=module.norm.eps,
    )

    state = modules[0].initialize(x=x)
    inputs, states = [], [state]
    for module in modules[:-1]:
        branch_input, context = module.pre(state=state)
        inputs.append(branch_input)
        state = module.post(branch_output=F.silu(branch_input).to(dtype), context=context)
        states.append(state)
    out = modules[-1](state=state) if output_norm else modules[-1].readout(state=state)

    for i, (expected, branch_input) in enumerate(zip(ref_inputs, inputs)):
        assert branch_input.dtype == expected.dtype
        assert_close(f'input_{i}', expected, branch_input, 0.005)
    for i, (expected, state) in enumerate(zip(ref_states, states)):
        assert state.num_branches == i
        assert len(state.blocks) == len(expected)
        for j, (ref_block, block) in enumerate(zip(expected, state.blocks)):
            assert block.dtype == ref_block.dtype
            assert_close(f'state_{i}_block_{j}', ref_block, block, 0.005)
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
