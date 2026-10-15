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


@pytest.mark.parametrize('T', [1, 7, 65])
@pytest.mark.parametrize('single_kv_proj', [False, True])
@pytest.mark.parametrize('shared_mem', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_forward_backward(T, single_kv_proj, shared_mem, dtype):
    """Packed projection preserves the padded path's output, input gradients, and parameter gradients."""
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
        actual = layer(x)[0]
        expected = layer(x, attention_mask=mask)[0]
    assert_close('o', expected, actual, 0.005)
    do = torch.randn_like(actual)
    inputs = [x, *layer.parameters()]
    grads = torch.autograd.grad(actual, inputs, grad_outputs=do, allow_unused=True)
    refs = torch.autograd.grad(expected, inputs, grad_outputs=do, allow_unused=True)
    for i, (grad, ref) in enumerate(zip(grads, refs)):
        assert (grad is None) == (ref is None)
        if grad is not None:
            assert_close(f'grad_{i}', ref, grad, 0.008)


@pytest.mark.parametrize('T', [1, 7, 65])
@pytest.mark.parametrize('training', [False, True])
def test_forward_empty_memories(T, training, monkeypatch):
    """Unused memories receive no projected tokens even when all tokens choose the same memories."""
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
    counts = []
    handles = [proj.register_forward_pre_hook(lambda module, inputs: counts.append(inputs[0].shape[1]))
               for proj in layer.k_proj]
    with torch.no_grad():
        actual = layer(x)[0]
    for handle in handles:
        handle.remove()
    assert counts == [2 * T, 2 * T, 0, 0, 0, 0, 0]
    with torch.no_grad():
        expected = layer(x, attention_mask=torch.ones(2, T, device=device, dtype=torch.long))[0]
    assert_close('o', expected, actual, 0.005)


@pytest.mark.parametrize('layout', ['mask', 'varlen', 'cache'])
def test_forward_fallback(layout, monkeypatch):
    """Masked, variable-length, and cached calls retain the padded implementation."""
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
        output = layer(x, **kwargs)[0]
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_model_forward_backward():
    """The packed path supports a causal LM loss and all model parameter gradients."""
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
    model = MomForCausalLM(config).to(device=device).train()
    ids = torch.randint(0, 128, (2, 65), device=device)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_forward_zero_routing_weights(monkeypatch):
    """Underflowed routing probabilities retain the padded path's slot masking."""
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
    actual = layer(x)[0]
    expected = layer(x, attention_mask=torch.ones(2, 7, device=device, dtype=torch.long))[0]
    assert_close('o', expected, actual, 0.005)
