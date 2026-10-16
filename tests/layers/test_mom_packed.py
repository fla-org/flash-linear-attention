# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers import mom
from fla.models.mom import MomConfig, MomForCausalLM
from fla.models.utils import Cache
from fla.utils import assert_close, device


@pytest.mark.parametrize('T', [1, 7, 65], ids=lambda T: f'T{T}')
@pytest.mark.parametrize('topk', [1, 3], ids=lambda topk: f'topk{topk}')
def test_pack_slots(T: int, topk: int):
    torch.manual_seed(42)
    batch_size, num_memories = 3, 7
    x = torch.randn(batch_size, T, 8, device=device, requires_grad=True)
    scores = torch.randn(batch_size, T, num_memories, device=device)
    selected_memories = scores.topk(topk, dim=-1).indices
    routing_mask = torch.zeros_like(scores, dtype=torch.int32).scatter(-1, selected_memories, 1)
    padded, indices, sorted_indices, _, mask, memory_mask = mom.transform(
        x=x,
        routing_mask=routing_mask,
        num_memories=num_memories,
        selected_memories=selected_memories,
        attention_mask=None,
    )
    order, inverse, cu_seqlens = mom._pack_slots(selected_memories=selected_memories, num_memories=num_memories)
    packed = x.flatten(0, 1)[order // topk]
    assert torch.equal(packed, padded[memory_mask])
    assert torch.equal(cu_seqlens[1:] - cu_seqlens[:-1], memory_mask.sum(dim=-1).flatten())
    assert torch.equal(order[inverse], torch.arange(order.numel(), device=device))

    routing_weights = torch.randn(batch_size, T, topk, device=device, requires_grad=True)
    actual = (packed[inverse].reshape(batch_size, T, topk, -1) * routing_weights[..., None]).sum(dim=2)
    expected = mom.reconstruct(
        transformed_x=padded,
        indices=indices,
        sorted_indices=sorted_indices,
        batch_size=batch_size,
        seq_len=T,
        topk=topk,
        routing_weights=routing_weights,
        mask=mask,
    )
    assert_close('o', expected, actual, 0.005)
    do = torch.randn_like(actual)
    grads = torch.autograd.grad(outputs=actual, inputs=(x, routing_weights), grad_outputs=do)
    refs = torch.autograd.grad(outputs=expected, inputs=(x, routing_weights), grad_outputs=do)
    for name, ref, grad in zip(('dx', 'drouting_weights'), refs, grads):
        assert_close(name, ref, grad, 0.008)


@pytest.mark.parametrize('T', [1, 7, 65], ids=lambda T: f'T{T}')
@pytest.mark.parametrize('single_kv_proj', [False, True], ids=['independent_kv', 'shared_kv'])
@pytest.mark.parametrize('shared_mem', [False, True], ids=['no_shared_mem', 'shared_mem'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16], ids=['fp32', 'bf16'])
def test_forward_backward(T: int, single_kv_proj: bool, shared_mem: bool, dtype: torch.dtype):
    torch.manual_seed(42)
    layer = mom.MomAttention(
        hidden_size=64,
        head_dim=16,
        num_heads=4,
        expand_v=1.5,
        num_memories=7,
        topk=3,
        shared_mem=shared_mem,
        single_kv_proj=single_kv_proj,
    ).to(device=device).train()
    x = torch.randn(3, T, 64, device=device, requires_grad=True)
    mask = torch.ones(3, T, device=device, dtype=torch.long)
    with torch.autocast(device_type=torch.device(device).type, dtype=dtype, enabled=dtype != torch.float32):
        actual = layer(hidden_states=x)[0]
        expected = layer(hidden_states=x, attention_mask=mask)[0]
    assert_close('o', expected, actual, 0.005)
    do = torch.randn_like(actual)
    inputs = {'x': x, **dict(layer.named_parameters())}
    grads = torch.autograd.grad(outputs=actual, inputs=tuple(inputs.values()), grad_outputs=do, allow_unused=True)
    refs = torch.autograd.grad(outputs=expected, inputs=tuple(inputs.values()), grad_outputs=do, allow_unused=True)
    for name, grad, ref in zip(inputs, grads, refs):
        assert (grad is None) == (ref is None), name
        if grad is not None:
            assert_close(f'd{name}', ref, grad, 0.008)


@pytest.mark.parametrize('T', [1, 7, 65], ids=lambda T: f'T{T}')
@pytest.mark.parametrize('training', [False, True], ids=['eval', 'train'])
def test_forward_empty_memories(T: int, training: bool, monkeypatch: pytest.MonkeyPatch):
    torch.manual_seed(42)
    layer = mom.MomAttention(
        hidden_size=64,
        head_dim=16,
        num_heads=4,
        expand_v=1,
        num_memories=7,
        topk=2,
    ).to(device=device).train(training)
    x = torch.randn(2, T, 64, device=device)
    gate_forward = layer.gate.forward
    bias = x.new_tensor([100, 99, -100, -100, -100, -100, -100])
    monkeypatch.setattr(layer.gate, 'forward', lambda x: gate_forward(x) + bias)
    projected_lengths = []

    def record_length(module, inputs):
        projected_lengths.append(inputs[0].shape[1])

    handles = [proj.register_forward_pre_hook(record_length) for proj in layer.k_proj]
    try:
        with torch.no_grad():
            actual = layer(hidden_states=x)[0]
    finally:
        for handle in handles:
            handle.remove()
    assert projected_lengths == [2 * T, 2 * T, 0, 0, 0, 0, 0]
    mask = torch.ones(2, T, device=device, dtype=torch.long)
    with torch.no_grad():
        expected = layer(hidden_states=x, attention_mask=mask)[0]
    assert_close('o', expected, actual, 0.005)


@pytest.mark.parametrize('layout', ['mask', 'varlen', 'cache'])
def test_forward_fallback(layout: str, monkeypatch: pytest.MonkeyPatch):
    torch.manual_seed(42)
    layer = mom.MomAttention(
        hidden_size=64,
        head_dim=16,
        num_heads=4,
        expand_v=1,
        num_memories=3,
        topk=2,
        layer_idx=0,
    ).to(device=device).eval()
    x = torch.randn(2, 65, 64, device=device)

    def forbidden(*args, **kwargs):
        pytest.fail('fallback called the packed path')

    monkeypatch.setattr(mom, '_pack_slots', forbidden)
    kwargs = {}
    if layout == 'mask':
        kwargs['attention_mask'] = torch.ones(2, 65, device=device, dtype=torch.long)
        kwargs['attention_mask'][0, :7] = 0
    elif layout == 'varlen':
        x = x.reshape(1, 130, 64)
        kwargs['cu_seqlens'] = torch.tensor([0, 61, 130], device=device, dtype=torch.int32)
    else:
        kwargs.update(past_key_values=Cache(), use_cache=True)
    with torch.no_grad():
        output = layer(hidden_states=x, **kwargs)[0]
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_model_forward_backward():
    torch.manual_seed(42)
    config = MomConfig(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        expand_v=1,
        num_hidden_layers=2,
        vocab_size=128,
        num_memories=7,
        topk=3,
        shared_mem=True,
        use_cache=False,
        fuse_cross_entropy=False,
    )
    model = MomForCausalLM(config=config).to(device=device).train()
    ids = torch.randint(0, 128, (2, 65), device=device)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_forward_zero_routing_weights(monkeypatch: pytest.MonkeyPatch):
    torch.manual_seed(42)
    layer = mom.MomAttention(
        hidden_size=64,
        head_dim=16,
        num_heads=4,
        expand_v=1,
        num_memories=7,
        topk=3,
    ).to(device=device).train()
    x = torch.randn(2, 7, 64, device=device)
    gate_forward = layer.gate.forward
    bias = x.new_tensor([100, 99, -100, -100, -100, -100, -100])
    monkeypatch.setattr(layer.gate, 'forward', lambda x: gate_forward(x) + bias)

    def forbidden(*args, **kwargs):
        pytest.fail('zero routing weights called the packed path')

    monkeypatch.setattr(mom, '_pack_slots', forbidden)
    mask = torch.ones(2, 7, device=device, dtype=torch.long)
    actual = layer(hidden_states=x)[0]
    expected = layer(hidden_states=x, attention_mask=mask)[0]
    assert_close('o', expected, actual, 0.005)
