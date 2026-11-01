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
from torch import nn

from fla import modules as fla_modules
from fla.modules import residuals
from fla.modules.residuals import (
    AttentionResidual,
    BaseResidual,
    StandardDeferredResidual,
    StandardResidual,
    get_residual_class,
    register_residual,
)
from fla.ops.attnres import naive_attnres
from fla.utils import assert_close, device, device_platform


def test_registry(monkeypatch):
    class CustomResidual(StandardResidual):
        pass

    monkeypatch.setattr(residuals, '_RESIDUAL_CLASSES', residuals._RESIDUAL_CLASSES.copy())
    for name, module_class in [
        ('standard', StandardResidual),
        ('standard_defer', StandardDeferredResidual),
        ('attnres', AttentionResidual),
    ]:
        assert get_residual_class(name) is module_class
        assert getattr(fla_modules, module_class.__name__) is module_class
    register_residual('test.custom', CustomResidual)
    module = get_residual_class('test.custom')(hidden_size=8, sub_layer_idx=0, num_sublayers=2, fuse_norm=False)
    assert isinstance(module, CustomResidual)
    with pytest.raises(ValueError, match='already registered'):
        register_residual('test.custom', StandardResidual)
    assert get_residual_class('test.custom') is CustomResidual
    with pytest.raises(ValueError, match='register_residual'):
        get_residual_class('test.missing')


@pytest.mark.parametrize(('name', 'module_class', 'error'), [
    pytest.param('', StandardResidual, ValueError, id='empty-name'),
    pytest.param('two words', StandardResidual, ValueError, id='whitespace-name'),
    pytest.param(None, StandardResidual, ValueError, id='non-string-name'),
    pytest.param('test.abstract', BaseResidual, TypeError, id='abstract-class'),
    pytest.param('test.unrelated', nn.Linear, TypeError, id='unrelated-class'),
])
def test_registry_validation(name, module_class, error):
    with pytest.raises(error):
        register_residual(name, module_class)


@pytest.mark.parametrize('block_size', [0, -1, True, 1.5])
def test_attnres_block_size(block_size):
    with pytest.raises(ValueError, match='positive integer'):
        AttentionResidual(hidden_size=8, sub_layer_idx=0, num_sublayers=2, block_size=block_size)


@pytest.mark.parametrize('module_class', [StandardResidual, StandardDeferredResidual, AttentionResidual])
def test_initialization(module_class):
    torch.manual_seed(42)
    kwargs = dict(hidden_size=8, sub_layer_idx=0, num_sublayers=2, fuse_norm=False)
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


@pytest.mark.skipif(device_platform == 'cpu', reason='Residual numerical tests require an accelerator')
@pytest.mark.parametrize(('residual_cls', 'batch_size', 'dtype', 'fuse_norm', 'input_dtype'), [
    pytest.param(StandardResidual, 1, torch.float32, False, torch.float32, id='standard-fp32'),
    pytest.param(StandardResidual, 2, torch.bfloat16, True, torch.bfloat16, id='standard-bf16-fused'),
    pytest.param(StandardResidual, 1, torch.float16, True, torch.float16, id='standard-fp16-fused'),
    pytest.param(StandardDeferredResidual, 1, torch.float32, False, torch.float32, id='deferred-fp32'),
    pytest.param(StandardDeferredResidual, 2, torch.float16, True, torch.float16, id='deferred-fp16-fused'),
    pytest.param(StandardDeferredResidual, 2, torch.bfloat16, False, torch.bfloat16, id='deferred-bf16'),
    pytest.param(StandardResidual, 1, torch.bfloat16, True, torch.float32, id='standard-mixed-dtype'),
    pytest.param(StandardDeferredResidual, 1, torch.bfloat16, True, torch.float32, id='deferred-mixed-dtype'),
])
def test_standard(residual_cls, batch_size, dtype, fuse_norm, input_dtype):
    deferred = residual_cls is StandardDeferredResidual
    torch.manual_seed(42)
    modules = nn.ModuleList([
        residual_cls(hidden_size=128, sub_layer_idx=i, num_sublayers=2, is_output=i == 2, fuse_norm=fuse_norm)
        for i in range(3)
    ]).to(device=device, dtype=dtype)
    reference = copy.deepcopy(modules)
    x = torch.randn(batch_size, 7, 128, device=device, dtype=input_dtype, requires_grad=True)
    branches = [torch.randn_like(x, dtype=dtype, requires_grad=True) for _ in range(2)]
    ref_x = x.detach().clone().requires_grad_()
    ref_branches = [branch.detach().clone().requires_grad_() for branch in branches]
    state = modules[-1].initialize(x)
    if deferred:
        assert state.x is x and state.branch_output is None
    else:
        assert state is x
    assert modules[-1].get_hidden_state(state) is x
    residual, pending = ref_x, None
    loss, ref_loss = 0, 0
    output_norm_inputs = []
    output_norm_hook = modules[-1].norm.register_forward_pre_hook(lambda module, inputs: output_norm_inputs.append(inputs[0]))
    for i, (module, ref_module) in enumerate(zip(modules, reference)):
        output, context = module.pre(state)
        value = residual if pending is None else residual + pending
        output_dtype = value.dtype
        ref_context = value
        if deferred and i == 1 and fuse_norm:
            value = residual.float() + pending.float()
            ref_context = value.to(residual.dtype)
            output_dtype = pending.dtype
        expected = F.rms_norm(value.float(), (128,), ref_module.norm.weight.float(), ref_module.norm.eps).to(output_dtype)
        assert output.shape == x.shape and output.dtype == expected.dtype
        assert_close('output', expected, output, 0.005)
        grad = torch.randn_like(output)
        loss = loss + (output * grad).sum()
        ref_loss = ref_loss + (expected * grad).sum()
        if deferred:
            assert isinstance(context, torch.Tensor)
            assert context.dtype == ref_context.dtype
            assert_close('context', ref_context, context, 0.005)
        else:
            assert context is None
        if i == 2:
            output_norm_hook.remove()
            if deferred:
                assert context is output_norm_inputs[0]
                grad = torch.randn_like(context)
                loss = loss + (context * grad).sum()
                ref_loss = ref_loss + (ref_context * grad).sum()
            break
        previous = state
        state = module.post(state, branches[i], context)
        assert state is not previous
        if deferred:
            assert state.x is context and state.branch_output is branches[i]
            residual, pending = ref_context, ref_branches[i]
            expected_state = residual + pending
        else:
            residual, pending = ref_context + ref_branches[i], None
            expected_state = residual
            assert module.get_hidden_state(state) is state
        hidden = module.get_hidden_state(state)
        assert hidden.dtype == expected_state.dtype
        assert_close('state', expected_state, hidden, 0.005)
        grad = torch.randn_like(hidden)
        loss = loss + (hidden * grad).sum()
        ref_loss = ref_loss + (expected_state * grad).sum()
    loss.backward()
    ref_loss.backward()
    assert_close('dx', ref_x.grad, x.grad, 0.005)
    for ref, actual in zip(ref_branches, branches):
        assert_close('dbranch', ref.grad, actual.grad, 0.005)
    for ref_module, module in zip(reference, modules):
        assert_close('dweight', ref_module.norm.weight.grad, module.norm.weight.grad, 0.005)


@pytest.mark.skipif(device_platform == 'cpu', reason='AttnRes requires an accelerator')
@pytest.mark.parametrize(('block_size', 'dtype', 'input_dtype'), [
    pytest.param(1, torch.float32, torch.float32, id='full-fp32'),
    pytest.param(4, torch.float16, torch.float16, id='block-fp16'),
    pytest.param(4, torch.bfloat16, torch.bfloat16, id='block-bf16'),
    pytest.param(4, torch.bfloat16, torch.float32, id='block-mixed-dtype'),
])
def test_attnres(block_size, dtype, input_dtype, monkeypatch):
    torch.manual_seed(42)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    modules = torch.nn.ModuleList([
        AttentionResidual(hidden_size=128, block_size=block_size, sub_layer_idx=i, num_sublayers=6, is_output=i == 6)
        for i in range(7)
    ]).to(device=device, dtype=dtype)
    with torch.no_grad():
        for module in modules:
            module.query.weight.normal_(std=0.03)
    reference = copy.deepcopy(modules)
    x = torch.randn(2, 7, 128, device=device, dtype=input_dtype, requires_grad=True)
    branches = [torch.randn_like(x, dtype=dtype, requires_grad=True) for _ in range(6)]
    ref_x = x.detach().clone().requires_grad_()
    ref_branches = [branch.detach().clone().requires_grad_() for branch in branches]
    state, ref_state = modules[-1].initialize(x), [ref_x]
    assert modules[-1].get_hidden_state(state) is x
    loss, ref_loss = 0, 0
    for i, (module, ref_module) in enumerate(zip(modules, reference)):
        output, context = module.pre(state)
        if i == 0:
            expected = F.rms_norm(ref_x.float(), (128,), ref_module.norm.weight.float(), ref_module.norm.eps).to(input_dtype)
        else:
            expected = naive_attnres(
                query=ref_module.query.weight,
                residuals=ref_state,
                rms_weight=ref_module.key_norm.weight,
                output_rms_weight=ref_module.norm.weight,
                rms_eps=ref_module.norm.eps,
            )
        assert output.dtype == ref_state[0].dtype
        assert_close('output', expected, output, 0.005)
        grad = torch.randn_like(output)
        loss = loss + (output * grad).sum()
        ref_loss = ref_loss + (expected * grad).sum()
        if i == 6:
            break
        previous = tuple(state)
        state = module.post(state, branches[i], context)
        assert len(previous) == len(ref_state)
        for ref, actual in zip(ref_state, previous):
            assert_close('previous_state', ref, actual, 0.005)
        if i == 0:
            ref_state = [ref_x.to(dtype)]
        if i % block_size == 0:
            ref_state = [*ref_state, ref_branches[i]]
        else:
            ref_state = [*ref_state[:-1], ref_state[-1] + ref_branches[i]]
        assert len(state) == len(ref_state)
        for ref, tri in zip(ref_state, state):
            assert_close('state', ref, tri, 0.005)
        assert_close('hidden_state', ref_state[-1], module.get_hidden_state(state), 0.005)
    loss.backward()
    ref_loss.backward()
    assert_close('dx', ref_x.grad, x.grad, 0.005)
    for ref, tri in zip(ref_branches, branches):
        assert_close('dbranch', ref.grad, tri.grad, 0.005)
    for (name, ref), (_, tri) in zip(reference.named_parameters(), modules.named_parameters()):
        if ref.grad is None:
            assert tri.grad is None, name
        else:
            assert_close(name, ref.grad, tri.grad, 0.005)
