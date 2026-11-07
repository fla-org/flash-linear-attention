# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fla.modules import ManifoldHyperConnection
from fla.modules.residuals.mhc import MHCState
from fla.ops.hc import naive_hc_post, naive_hc_pre
from fla.ops.utils.sinkhorn import naive_sinkhorn
from fla.utils import IS_NVIDIA, assert_close, device


def naive_mhc_coefficients(module: ManifoldHyperConnection, streams: torch.Tensor):
    num_streams = module.num_streams
    flat = streams.float().flatten(-2)
    flat = flat * (flat.square().mean(-1, keepdim=True) + module.norm.eps).rsqrt()
    logits = flat @ module.weight.float().T
    pre = torch.sigmoid(logits[..., :num_streams] * module.scale[0].float() + module.pre_bias.float())
    post = 2 * torch.sigmoid(logits[..., num_streams:2 * num_streams] * module.scale[1].float() + module.post_bias.float())
    mixing = logits[..., 2 * num_streams:].reshape(*streams.shape[:-2], num_streams, num_streams)
    mixing = naive_sinkhorn(logits=mixing * module.scale[2].float() + module.res_bias.float(), num_iters=module.num_iters)
    return pre, post, mixing


@pytest.mark.skipif(not IS_NVIDIA, reason='HC and Sinkhorn require NVIDIA CUDA')
@pytest.mark.parametrize(
    ('num_streams', 'dtype', 'fuse_norm', 'output_norm', 'from_embeddings', 'ratio'),
    [
        pytest.param(2, torch.float32, False, True, False, 5e-5, id='2-streams-fp32'),
        pytest.param(4, torch.float16, True, False, False, 2e-3, id='4-streams-fp16-readout'),
        pytest.param(8, torch.bfloat16, True, True, False, 6e-3, id='8-streams-bf16'),
        pytest.param(4, torch.bfloat16, True, True, True, 6e-3, id='4-streams-bf16-embeddings'),
    ],
)
def test_mhc(
    num_streams: int,
    dtype: torch.dtype,
    fuse_norm: bool,
    output_norm: bool,
    from_embeddings: bool,
    ratio: float,
    request: pytest.FixtureRequest,
):
    """Check learned routing and stream updates through four connected branches, including final readout gradients."""
    torch.manual_seed(42)
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    request.addfinalizer(lambda: setattr(torch.backends.cuda.matmul, 'allow_tf32', allow_tf32))
    torch.backends.cuda.matmul.allow_tf32 = False
    modules = nn.ModuleList([
        ManifoldHyperConnection(
            hidden_size=128,
            num_streams=num_streams,
            fuse_norm=fuse_norm,
            sub_layer_idx=i,
            num_sublayers=4,
        )
        for i in range(5)
    ]).to(device=device, dtype=dtype)
    with torch.no_grad():
        for i, module in enumerate(modules):
            module.norm.weight.uniform_(0.7, 1.3)
            if module.prepare_next:
                module.scale.fill_(0.2)
                module.pre_bias.add_(0.1 * i)
                module.res_bias.normal_(std=0.3)
                # preserve stream diversity across the chain for routing-gradient comparisons.
                module.res_bias.add_(math.log(num_streams) * torch.eye(num_streams, device=device))
    shape = (2, 7, 128) if from_embeddings else (2, 7, num_streams, 128)
    x = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    # independent branch features prevent repeated mixing from collapsing the stream directions.
    branches = [torch.randn(2, 7, 128, device=device, dtype=dtype, requires_grad=True) for _ in range(4)]
    ref_streams = x.clone()
    if from_embeddings:
        ref_streams = ref_streams.unsqueeze(-2).expand(*x.shape[:-1], num_streams, 128)
    ref_states = [MHCState(ref_streams, *naive_mhc_coefficients(modules[-1], ref_streams))]
    ref_inputs, ref_readouts = [], []
    for i, module in enumerate(modules[:-1]):
        ref_state = ref_states[-1]
        ref_hidden = naive_hc_pre(x=ref_state.streams, pre=ref_state.pre)
        ref_readouts.append(ref_hidden)
        ref_input = F.rms_norm(ref_hidden.float(), (128,), module.norm.weight.float(), module.norm.eps).to(dtype)
        ref_inputs.append(ref_input)
        ref_streams = naive_hc_post(
            x=ref_state.streams,
            branch=F.silu(ref_input) + branches[i],
            post=ref_state.post,
            mixing=ref_state.mixing,
        )
        coefficients = naive_mhc_coefficients(module, ref_streams) if module.prepare_next else (None, None, None)
        ref_states.append(MHCState(ref_streams, *coefficients))
    ref = ref_streams.float().mean(-2).to(dtype)
    if output_norm:
        ref = F.rms_norm(ref.float(), (128,), modules[-1].norm.weight.float(), modules[-1].norm.eps).to(dtype)

    if from_embeddings:
        state = modules[-1].initialize(x=x)
    else:
        state = MHCState(x, *modules[-1].prepare_coefficients(x))
    inputs, readouts, states = [], [], [state]
    for i, module in enumerate(modules[:-1]):
        readouts.append(module.readout(state=state))
        branch_input, context = module.pre(state=state)
        inputs.append(branch_input)
        state = module.post(branch_output=F.silu(branch_input) + branches[i], context=context)
        states.append(state)
    out = modules[-1](state=state) if output_norm else modules[-1].readout(state=state)

    for i, (expected, branch_input) in enumerate(zip(ref_inputs, inputs)):
        assert branch_input.dtype == dtype
        assert_close(f'input_{i}', expected, branch_input, ratio)
        assert_close(f'readout_{i}', ref_readouts[i], readouts[i], ratio)
    for i, (expected, state) in enumerate(zip(ref_states, states)):
        assert state.streams.dtype == dtype
        assert_close(f'streams_{i}', expected.streams, state.streams, ratio)
        for name in ('pre', 'post', 'mixing'):
            ref_coefficient, coefficient = getattr(expected, name), getattr(state, name)
            if ref_coefficient is None:
                assert coefficient is None
            else:
                assert coefficient.dtype == torch.float32
                assert_close(f'{name}_{i}', ref_coefficient, coefficient, ratio)
    assert out.dtype == dtype
    assert_close('o', ref, out, ratio)

    do = torch.randn_like(out)
    ref_loss, loss = (ref.float() * do).sum(), (out.float() * do).sum()
    for expected, branch_input in zip(ref_inputs, inputs):
        dh = torch.randn_like(branch_input)
        ref_loss = ref_loss + (expected.float() * dh).sum()
        loss = loss + (branch_input.float() * dh).sum()
    # final stream averaging alone can hide errors in the mixing gradients.
    for expected, state in zip(ref_states[1:], states[1:]):
        ds = torch.randn_like(state.streams)
        ref_loss = ref_loss + (expected.streams.float() * ds).sum()
        loss = loss + (state.streams.float() * ds).sum()
    for expected, state in zip(ref_states[:-1], states[:-1]):
        for ref_coefficient, coefficient in zip(expected[1:], state[1:]):
            dc = torch.randn_like(coefficient)
            ref_loss = ref_loss + (ref_coefficient * dc).sum()
            loss = loss + (coefficient * dc).sum()
    parameters = {'x': x, **{f'branch_{i}': branch for i, branch in enumerate(branches)}, **dict(modules.named_parameters())}
    ref_grads = torch.autograd.grad(ref_loss, tuple(parameters.values()), allow_unused=True)
    out_grads = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
    for name, ref_grad, out_grad in zip(parameters, ref_grads, out_grads):
        if ref_grad is None:
            assert out_grad is None, name
        else:
            assert out_grad is not None and torch.isfinite(out_grad).all(), name
            assert_close(name, ref_grad, out_grad, ratio)
