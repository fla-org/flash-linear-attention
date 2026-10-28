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

from fla.layers import MemoryAttention
from fla.models import MemoryAttnConfig, MemoryAttnForCausalLM
from fla.utils import assert_close, device, find_spec_cached

pytestmark = pytest.mark.skipif(find_spec_cached('flash_attn') is None, reason='MemoryAttention requires flash-attn.')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16], ids=['fp16', 'bf16'])
@pytest.mark.parametrize('num_kv_heads', [2, 4], ids=['gqa', 'mha'])
@pytest.mark.parametrize('qk_norm', [False, True], ids=['plain', 'qk-norm'])
def test_attention(dtype, num_kv_heads, qk_norm):
    torch.manual_seed(42)
    layer = MemoryAttention(hidden_size=256, num_heads=4, num_kv_heads=num_kv_heads,
                            qk_norm=qk_norm, vocab_size=128).to(device=device, dtype=dtype)
    reference = copy.deepcopy(layer)
    hidden = torch.randn(2, 63, 256, device=device, dtype=dtype, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ids = torch.randint(0, 128, (2, 63), device=device)
    output = layer(hidden_states=hidden, input_ids=ids)[0]
    q = reference.q_proj(ref_hidden).reshape(2, 63, 4, 64)
    k = reference.k_proj(ref_hidden).reshape(2, 63, num_kv_heads, 64)
    memory = reference.m_proj(ids).reshape(2, 63, num_kv_heads, 64)
    v = k + reference.m_norm(memory)
    if qk_norm:
        q, k = reference.q_norm(q), reference.k_norm(k)
    q, k = reference.rotary(q, k, max_seqlen=63)
    k = k.repeat_interleave(4 // num_kv_heads, dim=2)
    v = v.repeat_interleave(4 // num_kv_heads, dim=2)
    scores = torch.einsum('bthd,bshd->bhts', q.float(), k.float()) / 8
    mask = torch.ones(63, 63, device=device, dtype=torch.bool).triu(1)
    probabilities = scores.masked_fill(mask, -torch.inf).softmax(dim=-1)
    expected = torch.einsum('bhts,bshd->bthd', probabilities, v.float()).to(dtype).reshape(2, 63, 256)
    expected = reference.o_proj(expected)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    expected.backward(gradient)
    assert_close('output', expected, output, 0.006)
    assert_close('hidden gradient', ref_hidden.grad, hidden.grad, 0.006)
    for (name, parameter), (_, ref_parameter) in zip(layer.named_parameters(), reference.named_parameters()):
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert_close(name, ref_parameter.grad, parameter.grad, 0.006)


def _create_model():
    config = MemoryAttnConfig(hidden_size=128, num_hidden_layers=2, num_heads=2, num_kv_heads=1,
                              vocab_size=128, intermediate_size=256, fuse_cross_entropy=False)
    return MemoryAttnForCausalLM(config).to(device=device, dtype=torch.bfloat16)


def test_modeling():
    torch.manual_seed(42)
    model = _create_model()
    ids = torch.randint(0, 128, (2, 63), device=device)
    dense = model(input_ids=ids, labels=ids, use_cache=False)
    expected_loss = F.cross_entropy(dense.logits[:, :-1].float().reshape(-1, 128), ids[:, 1:].reshape(-1))
    assert_close('loss', expected_loss, dense.loss.float(), 0.006)
    cu_seqlens = torch.tensor([0, 63, 126], device=device, dtype=torch.int32)
    attention_mask = torch.ones(1, 126, device=device, dtype=torch.long)
    attention_mask[:, [0, 1, 2, 63, 64, 65]] = 0
    packed = model(
        input_ids=ids.reshape(1, -1),
        attention_mask=attention_mask,
        cu_seqlens=cu_seqlens,
        use_cache=True,
    )
    assert_close('packed logits', dense.logits.reshape(1, 126, 128), packed.logits, 0.006)
    dense.loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for layer in model.model.layers:
        assert layer.attn.m_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16], ids=['fp16', 'bf16'])
def test_attention_autocast(dtype):
    torch.manual_seed(42)
    layer = MemoryAttention(hidden_size=256, num_heads=4, num_kv_heads=2, vocab_size=128).to(device=device)
    hidden = torch.randn(2, 63, 256, device=device)
    ids = torch.randint(0, 128, (2, 63), device=device)
    with torch.autocast(device_type=device, dtype=dtype):
        output = layer(hidden_states=hidden, input_ids=ids)[0]
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_inputs_embeds_unsupported():
    model = _create_model()
    ids = torch.randint(0, 128, (2, 63), device=device)
    with pytest.raises(ValueError, match='requires `input_ids`'):
        model(inputs_embeds=model.model.embeddings(ids))


@pytest.mark.parametrize('window_size', [None, 16], ids=['full', 'window'])
@torch.no_grad()
def test_generation(window_size):
    torch.manual_seed(42)
    model = _create_model().eval()
    for layer in model.model.layers:
        layer.attn.window_size = window_size
    ids = torch.randint(3, 128, (2, 31), device=device)
    mask = torch.ones_like(ids)
    mask[0, :3] = 0
    mask[1, :5] = 0
    full = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    prefill = model(input_ids=ids[:, :8], attention_mask=mask[:, :8], use_cache=True)
    outputs, cache = [prefill.logits], prefill.past_key_values
    for end in range(9, 32):
        result = model(input_ids=ids[:, end-1:end], attention_mask=mask[:, :end],
                       past_key_values=cache, use_cache=True)
        outputs.append(result.logits)
        cache = result.past_key_values
    cached = torch.cat(outputs, dim=1)
    assert torch.isfinite(cached).all()
    assert_close('cached logits', full[mask.bool()], cached[mask.bool()], 0.006)


@pytest.mark.parametrize('continuation_length', [3, 20], ids=['chunk', 'chunk-larger-than-window'])
@torch.no_grad()
def test_cached_continuation(continuation_length):
    torch.manual_seed(42)
    model = _create_model().eval()
    for layer in model.model.layers:
        layer.attn.window_size = 16
    ids = torch.randint(3, 128, (1, 64), device=device)
    full = model(input_ids=ids, use_cache=False).logits
    prefill = model(input_ids=ids[:, :24], use_cache=True)
    cached = model(
        input_ids=ids[:, 24:24 + continuation_length],
        past_key_values=prefill.past_key_values,
        use_cache=True,
    ).logits
    assert_close('cached continuation', full[:, 24:24 + continuation_length], cached, 0.006)
