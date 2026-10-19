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

from fla.modules.residuals import AttentionResidual, ManifoldHyperConnection, StandardResidual
from fla.ops.attnres import naive_attnres
from fla.ops.hc import naive_hc_post, naive_hc_pre
from fla.ops.utils.sinkhorn import naive_sinkhorn
from fla.utils import IS_NVIDIA, assert_close, device, device_platform


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('fuse_norm', [False, True])
def test_standard(dtype, fuse_norm):
    torch.manual_seed(42)
    module = StandardResidual(hidden_size=128, fuse_norm=fuse_norm).to(device=device, dtype=dtype)
    x = torch.randn(2, 7, 128, device=device, dtype=dtype, requires_grad=True)
    branch = torch.randn_like(x, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_branch = branch.detach().clone().requires_grad_()
    weight = module.norm.weight.detach().clone().requires_grad_()
    output, context = module.pre(module.initialize(x))
    state = module.post(x, branch, context)
    expected = F.rms_norm(ref_x.float(), (128,), weight.float(), module.norm.eps).to(dtype)
    ref_state = ref_x + ref_branch
    do, ds = torch.randn_like(output), torch.randn_like(state)
    ((output * do).sum() + (state * ds).sum()).backward()
    ((expected * do).sum() + (ref_state * ds).sum()).backward()
    assert_close('output', expected, output, 0.005)
    assert_close('state', ref_state, state, 0.005)
    for name, ref, tri in [('dx', ref_x.grad, x.grad), ('dbranch', ref_branch.grad, branch.grad),
                           ('dweight', weight.grad, module.norm.weight.grad)]:
        assert_close(name, ref, tri, 0.005)


@pytest.mark.parametrize('block_size', [1, 4])
@pytest.mark.parametrize(('dtype', 'autocast'), [
    (torch.float32, False), (torch.float16, False), (torch.bfloat16, False),
    (torch.float16, True), (torch.bfloat16, True),
])
def test_attnres(block_size, dtype, autocast, monkeypatch):
    if autocast and device_platform != 'cuda':
        pytest.skip('Autocast requires a CUDA reduced-precision dtype')
    torch.manual_seed(42)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    param_dtype = torch.float32 if autocast else dtype
    modules = torch.nn.ModuleList([
        AttentionResidual(hidden_size=128, block_size=block_size, sub_layer_idx=i, is_output=i == 6)
        for i in range(7)
    ]).to(device=device, dtype=param_dtype)
    with torch.no_grad():
        for module in modules:
            module.query.weight.normal_(std=0.03)
    reference = copy.deepcopy(modules)
    x = torch.randn(2, 7, 128, device=device, dtype=param_dtype, requires_grad=True)
    branches = [torch.randn_like(x, dtype=dtype, requires_grad=True) for _ in range(6)]
    ref_x = x.detach().clone().requires_grad_()
    ref_branches = [branch.detach().clone().requires_grad_() for branch in branches]
    state, ref_state = modules[0].initialize(x), [ref_x]
    loss, ref_loss = 0, 0
    for i, (module, ref_module) in enumerate(zip(modules, reference)):
        with torch.autocast(device_type=device_platform, dtype=dtype, enabled=autocast):
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
        state = module.post(state, branches[i], context)
        if i == 0:
            ref_state = [ref_x.to(dtype)]
        if i % block_size == 0:
            ref_state = [*ref_state, ref_branches[i]]
        else:
            ref_state = [*ref_state[:-1], ref_state[-1] + ref_branches[i]]
        assert len(state) == len(ref_state)
        for ref, tri in zip(ref_state, state):
            assert_close('state', ref, tri, 0.005)
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


def naive_mhc_pre(module, state):
    streams = module.num_streams
    flat = state.float().flatten(-2)
    flat = flat * (flat.square().mean(-1, keepdim=True) + module.norm.eps).rsqrt()
    logits = flat @ module.weight.float().T
    pre = torch.sigmoid(logits[..., :streams] * module.scale[0].float() + module.pre_bias.float())
    post = 2 * torch.sigmoid(logits[..., streams:2 * streams] * module.scale[1].float() + module.post_bias.float())
    mixing = logits[..., 2 * streams:].reshape(*state.shape[:-2], streams, streams)
    mixing = naive_sinkhorn(logits=mixing * module.scale[2].float() + module.res_bias.float(), num_iters=module.num_iters)
    output = naive_hc_pre(x=state, pre=pre)
    output = F.rms_norm(output.float(), (module.hidden_size,), module.norm.weight.float(), module.norm.eps).to(state.dtype)
    return output, (post, mixing)


@pytest.mark.skipif(not IS_NVIDIA, reason='HC and Sinkhorn require NVIDIA CUDA')
@pytest.mark.parametrize('num_streams', [2, 4, 8])
@pytest.mark.parametrize(('dtype', 'ratio', 'autocast'), [
    (torch.float32, 5e-5, False), (torch.float16, 2e-3, False), (torch.bfloat16, 6e-3, False),
    (torch.float16, 2e-3, True), (torch.bfloat16, 6e-3, True),
])
def test_mhc(num_streams, dtype, ratio, autocast, monkeypatch):
    torch.manual_seed(42)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    param_dtype = torch.float32 if autocast else dtype
    modules = torch.nn.ModuleList([
        ManifoldHyperConnection(hidden_size=128, num_streams=num_streams, sub_layer_idx=i, is_output=i == 2)
        for i in range(3)
    ]).to(device=device, dtype=param_dtype)
    with torch.no_grad():
        for module in modules[:-1]:
            module.scale.fill_(0.2)
            module.res_bias.normal_(std=0.3)
    reference = copy.deepcopy(modules)
    x = torch.randn(2, 7, num_streams, 128, device=device, dtype=param_dtype, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    branches = [torch.randn(2, 7, 128, device=device, dtype=dtype, requires_grad=True) for _ in range(2)]
    ref_branches = [branch.detach().clone().requires_grad_() for branch in branches]
    state, ref_state = x, ref_x
    loss, ref_loss = 0, 0
    for i, (module, ref_module) in enumerate(zip(modules[:-1], reference[:-1])):
        with torch.autocast(device_type=device_platform, dtype=dtype, enabled=autocast):
            output, context = module.pre(state)
            state = module.post(state, branches[i], context)
        expected, (post, mixing) = naive_mhc_pre(ref_module, ref_state)
        ref_state = naive_hc_post(x=ref_state, branch=ref_branches[i], post=post, mixing=mixing)
        assert_close('output', expected, output, ratio)
        assert_close('state', ref_state, state, ratio)
        do, ds = torch.randn_like(output), torch.randn_like(state)
        loss = loss + (output * do).sum() + (state * ds).sum()
        ref_loss = ref_loss + (expected * do).sum() + (ref_state * ds).sum()
    output, _ = modules[-1].pre(state)
    expected = ref_state.float().mean(-2).to(ref_state.dtype)
    expected = F.rms_norm(expected.float(), (128,), reference[-1].norm.weight.float(), modules[-1].norm.eps).to(state.dtype)
    assert_close('readout', expected, output, ratio)
    do = torch.randn_like(output)
    (loss + (output * do).sum()).backward()
    (ref_loss + (expected * do).sum()).backward()
    assert_close('dx', ref_x.grad, x.grad, ratio)
    for ref, tri in zip(ref_branches, branches):
        assert_close('dbranch', ref.grad, tri.grad, ratio)
    for (name, ref), (_, tri) in zip(reference.named_parameters(), modules.named_parameters()):
        assert_close(name, ref.grad, tri.grad, ratio)


@pytest.mark.parametrize('num_streams', [2, 4, 8])
def test_mhc_initialize(num_streams):
    torch.manual_seed(42)
    module = ManifoldHyperConnection(hidden_size=128, num_streams=num_streams).to(device)
    x = torch.randn(2, 7, 128, device=device, requires_grad=True)
    state = module.initialize(x)
    torch.testing.assert_close(state, x.unsqueeze(-2).repeat(1, 1, num_streams, 1), rtol=0, atol=0)
    state.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, num_streams), rtol=0, atol=0)


@pytest.mark.parametrize(('module_class', 'kwargs', 'match'), [
    (AttentionResidual, {'block_size': 0}, 'block_size'),
    (ManifoldHyperConnection, {'num_streams': 1}, 'num_streams'),
    (ManifoldHyperConnection, {'num_streams': 9}, 'num_streams'),
    (ManifoldHyperConnection, {'num_iters': 0}, 'num_iters'),
    (ManifoldHyperConnection, {'init_scale': 0}, 'init_scale'),
])
def test_validation(module_class, kwargs, match):
    with pytest.raises(ValueError, match=match):
        module_class(hidden_size=128, **kwargs)
