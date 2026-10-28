# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fla.modules.residuals import AttentionResidual, ManifoldHyperConnection, StandardDeferredResidual, StandardResidual
from fla.modules.residuals.mhc import MHCState
from fla.ops.attnres import naive_attnres
from fla.ops.hc import naive_hc_post, naive_hc_pre
from fla.ops.utils.sinkhorn import naive_sinkhorn
from fla.utils import IS_NVIDIA, assert_close, device_name, device_platform


@pytest.fixture(params=['native', 'reference'])
def residual_backend(request):
    """Keep native numerical coverage separate from CPU state and routing coverage."""
    if request.param == 'reference':
        request.getfixturevalue('reference_residual_ops')
    return request.param


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize(('dtype', 'autocast'), [
    (torch.float32, False), (torch.float16, False), (torch.bfloat16, False),
    (torch.float16, True), (torch.bfloat16, True),
])
@pytest.mark.parametrize('fuse_norm', [False, True])
@pytest.mark.parametrize('residual_cls', [StandardResidual, StandardDeferredResidual], ids=['standard', 'standard_defer'])
def test_standard(batch_size, dtype, autocast, fuse_norm, residual_cls, residual_backend):
    if residual_backend == 'native' and device_platform == 'cpu' and (fuse_norm or autocast):
        pytest.skip('Native fused norm and autocast coverage require an accelerator')
    test_device = 'cpu' if residual_backend == 'reference' else device_name
    param_dtype = torch.float32 if autocast else dtype
    deferred = residual_cls is StandardDeferredResidual
    torch.manual_seed(42)
    modules = nn.ModuleList([
        residual_cls(hidden_size=128, sub_layer_idx=i, num_sublayers=2, is_output=i == 2, fuse_norm=fuse_norm)
        for i in range(3)
    ]).to(device=test_device, dtype=param_dtype)
    reference = copy.deepcopy(modules)
    x = torch.randn(batch_size, 7, 128, device=test_device, dtype=param_dtype, requires_grad=True)
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
        with torch.autocast(device_type=torch.device(test_device).type, dtype=dtype, enabled=autocast):
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


@pytest.mark.parametrize('block_size', [1, 4])
@pytest.mark.parametrize(('dtype', 'autocast'), [
    (torch.float32, False), (torch.float16, False), (torch.bfloat16, False),
    (torch.float16, True), (torch.bfloat16, True),
])
def test_attnres(block_size, dtype, autocast, residual_backend, monkeypatch):
    if residual_backend == 'native' and device_platform == 'cpu':
        pytest.skip('Native AttnRes requires an accelerator')
    if residual_backend == 'native' and autocast and device_platform != 'cuda':
        pytest.skip('Autocast requires a CUDA reduced-precision dtype')
    test_device = 'cpu' if residual_backend == 'reference' else device_name
    torch.manual_seed(42)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    param_dtype = torch.float32 if autocast else dtype
    modules = torch.nn.ModuleList([
        AttentionResidual(hidden_size=128, block_size=block_size, sub_layer_idx=i, num_sublayers=6, is_output=i == 6)
        for i in range(7)
    ]).to(device=test_device, dtype=param_dtype)
    with torch.no_grad():
        for module in modules:
            module.query.weight.normal_(std=0.03)
    reference = copy.deepcopy(modules)
    x = torch.randn(2, 7, 128, device=test_device, dtype=param_dtype, requires_grad=True)
    branches = [torch.randn_like(x, dtype=dtype, requires_grad=True) for _ in range(6)]
    ref_x = x.detach().clone().requires_grad_()
    ref_branches = [branch.detach().clone().requires_grad_() for branch in branches]
    state, ref_state = modules[-1].initialize(x), [ref_x]
    assert modules[-1].get_hidden_state(state) is x
    loss, ref_loss = 0, 0
    for i, (module, ref_module) in enumerate(zip(modules, reference)):
        with torch.autocast(device_type=torch.device(test_device).type, dtype=dtype, enabled=autocast):
            output, context = module.pre(state)
        if i == 0:
            expected = F.rms_norm(ref_x.float(), (128,), ref_module.norm.weight.float(), ref_module.norm.eps).to(param_dtype)
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


def naive_mhc_coefficients(module, state):
    streams = module.num_streams
    flat = state.float().flatten(-2)
    flat = flat * (flat.square().mean(-1, keepdim=True) + module.norm.eps).rsqrt()
    logits = flat @ module.weight.float().T
    pre = torch.sigmoid(logits[..., :streams] * module.scale[0].float() + module.pre_bias.float())
    post = 2 * torch.sigmoid(logits[..., streams:2 * streams] * module.scale[1].float() + module.post_bias.float())
    mixing = logits[..., 2 * streams:].reshape(*state.shape[:-2], streams, streams)
    mixing = naive_sinkhorn(logits=mixing * module.scale[2].float() + module.res_bias.float(), num_iters=module.num_iters)
    return pre, post, mixing


@pytest.mark.parametrize('num_streams', [2, 4, 8])
@pytest.mark.parametrize(('dtype', 'ratio', 'autocast'), [
    (torch.float32, 5e-5, False), (torch.float16, 2e-3, False), (torch.bfloat16, 6e-3, False),
    (torch.float16, 2e-3, True), (torch.bfloat16, 6e-3, True),
])
def test_mhc(num_streams, dtype, ratio, autocast, residual_backend, monkeypatch):
    if residual_backend == 'native' and not IS_NVIDIA:
        pytest.skip('HC and Sinkhorn require NVIDIA CUDA')
    test_device = 'cpu' if residual_backend == 'reference' else device_name
    torch.manual_seed(42)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    param_dtype = torch.float32 if autocast else dtype
    modules = torch.nn.ModuleList([
        ManifoldHyperConnection(hidden_size=128, num_streams=num_streams, sub_layer_idx=i, num_sublayers=4, is_output=i == 4)
        for i in range(5)
    ]).to(device=test_device, dtype=param_dtype)
    with torch.no_grad():
        for i, module in enumerate(modules):
            module.norm.weight.uniform_(0.7, 1.3)
            assert module.is_output == (i == 4)
            assert module.is_last_sublayer == (i == 3)
            for name in ['weight', 'scale', 'pre_bias', 'post_bias', 'res_bias']:
                assert hasattr(module, name) == (i != 3)
            if i != 3:
                module.scale.fill_(0.2)
                module.pre_bias.add_(0.1 * i)
                module.res_bias.normal_(std=0.3)
    reference = copy.deepcopy(modules)
    for module in modules:
        monkeypatch.setattr(module, 'prepare_coefficients', Mock(wraps=module.prepare_coefficients))
    x = torch.randn(2, 7, num_streams, 128, device=test_device, dtype=param_dtype, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    branches = [torch.randn(2, 7, 128, device=test_device, dtype=dtype, requires_grad=True) for _ in range(4)]
    ref_branches = [branch.detach().clone().requires_grad_() for branch in branches]
    with torch.autocast(device_type=torch.device(test_device).type, dtype=dtype, enabled=autocast):
        state = MHCState(x, *modules[-1].prepare_coefficients(x))
    ref_state = ref_x
    ref_coefficients = naive_mhc_coefficients(reference[-1], ref_state)
    assert state.streams.dtype == param_dtype
    loss, ref_loss = 0, 0
    for i, (module, ref_module) in enumerate(zip(modules[:-1], reference[:-1])):
        pre, post, mixing = ref_coefficients
        for name, expected in [('pre', pre), ('post', post), ('mixing', mixing)]:
            actual = getattr(state, name)
            assert actual.dtype == torch.float32
            assert_close(name, expected, actual, ratio)
        call_counts = [m.prepare_coefficients.call_count for m in modules]
        raw_hidden = naive_hc_pre(x=ref_state, pre=pre)
        assert_close('hidden_state', raw_hidden, module.get_hidden_state(state), ratio)
        with torch.autocast(device_type=torch.device(test_device).type, dtype=dtype, enabled=autocast):
            output, context = module.pre(state)
            assert context is None
            assert [m.prepare_coefficients.call_count for m in modules] == call_counts
            previous = state
            state = module.post(state, branches[i], context)
        assert state is not previous
        assert_close('previous_state', ref_state, previous.streams, ratio)
        expected = F.rms_norm(raw_hidden.float(), (128,), ref_module.norm.weight.float(), ref_module.norm.eps).to(param_dtype)
        ref_state = naive_hc_post(x=ref_state, branch=ref_branches[i], post=post, mixing=mixing)
        if i < 3:
            # prepare before state-loss reads to preserve reduced-precision gradient accumulation order.
            ref_coefficients = naive_mhc_coefficients(ref_module, ref_state)
        assert output.dtype == param_dtype and state.streams.dtype == param_dtype
        assert_close('output', expected, output, ratio)
        assert_close('state', ref_state, state.streams, ratio)
        do, ds = torch.randn_like(output), torch.randn_like(state.streams)
        loss = loss + (output * do).sum() + (state.streams * ds).sum()
        ref_loss = ref_loss + (expected * do).sum() + (ref_state * ds).sum()
    assert state.pre is None and state.post is None and state.mixing is None
    assert [m.prepare_coefficients.call_count for m in modules] == [1, 1, 1, 0, 1]
    output, _ = modules[-1].pre(state)
    expected = ref_state.float().mean(-2).to(ref_state.dtype)
    assert_close('final_hidden_state', expected, modules[-1].get_hidden_state(state), ratio)
    expected = F.rms_norm(expected.float(), (128,), reference[-1].norm.weight.float(), modules[-1].norm.eps).to(param_dtype)
    assert_close('readout', expected, output, ratio)
    do = torch.randn_like(output)
    (loss + (output * do).sum()).backward()
    (ref_loss + (expected * do).sum()).backward()
    assert_close('dx', ref_x.grad, x.grad, ratio)
    for ref, tri in zip(ref_branches, branches):
        assert_close('dbranch', ref.grad, tri.grad, ratio)
    for (name, ref), (_, tri) in zip(reference.named_parameters(), modules.named_parameters()):
        if ref.grad is None:
            assert tri.grad is None, name
        else:
            assert tri.grad is not None and torch.isfinite(tri.grad).all(), name
            assert_close(name, ref.grad, tri.grad, ratio)


@pytest.mark.parametrize('num_streams', [2, 4, 8])
def test_mhc_initialize(num_streams, residual_backend):
    if residual_backend == 'native' and not IS_NVIDIA:
        pytest.skip('HC and Sinkhorn require NVIDIA CUDA')
    test_device = 'cpu' if residual_backend == 'reference' else device_name
    torch.manual_seed(42)
    module = ManifoldHyperConnection(
        hidden_size=128,
        num_streams=num_streams,
        sub_layer_idx=2,
        num_sublayers=2,
        is_output=True,
    ).to(test_device)
    assert module.is_output and not module.is_last_sublayer
    x = torch.randn(2, 7, 128, device=test_device, requires_grad=True)
    state = module.initialize(x)
    torch.testing.assert_close(state.streams, x.unsqueeze(-2).repeat(1, 1, num_streams, 1), rtol=0, atol=0)
    expected = naive_mhc_coefficients(module, state.streams)
    coefficients = state.pre, state.post, state.mixing
    for actual, reference in zip(coefficients, expected):
        assert actual.dtype == torch.float32
        assert_close('initial_coefficients', reference, actual, 5e-5)
    state.streams.sum().backward(retain_graph=True)
    torch.testing.assert_close(x.grad, torch.full_like(x, num_streams), rtol=0, atol=0)
    x.grad = None
    sum((coefficient * torch.randn_like(coefficient)).sum() for coefficient in coefficients).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name in ['weight', 'scale', 'pre_bias', 'post_bias', 'res_bias']:
        grad = getattr(module, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name
    assert module.norm.weight.grad is None


@pytest.mark.parametrize(('module_class', 'kwargs', 'match'), [
    (AttentionResidual, {'block_size': 0}, 'block_size'),
    (ManifoldHyperConnection, {'num_streams': 1}, 'num_streams'),
    (ManifoldHyperConnection, {'num_streams': 9}, 'num_streams'),
    (ManifoldHyperConnection, {'num_iters': 0}, 'num_iters'),
    (ManifoldHyperConnection, {'init_scale': 0}, 'init_scale'),
])
def test_validation(module_class, kwargs, match):
    with pytest.raises(ValueError, match=match):
        module_class(hidden_size=128, sub_layer_idx=0, num_sublayers=2, **kwargs)
