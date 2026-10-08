# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy

import pytest
import torch

from fla.layers import StickBreakingAttention, stickbreaking_attn
from fla.ops.stickbreaking_attn import naive_stickbreaking_attn
from fla.utils import IS_AMD, IS_INTEL, IS_NVIDIA, assert_close, device

TOL = {torch.float16: 0.005, torch.bfloat16: 0.02}


def _reference(q, k, v, attend_current=False, cu_seqlens=None):
    dtype = q.dtype
    q, k, v = q.double(), k.double(), v.double()
    if cu_seqlens is None:
        o, rem = naive_stickbreaking_attn(q=q, k=k, v=v, attend_current=attend_current)
    else:
        outputs, remainders = [], []
        for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True):
            o, rem = naive_stickbreaking_attn(q=q[:, bos:eos], k=k[:, bos:eos], v=v[:, bos:eos], attend_current=attend_current)
            outputs.append(o)
            remainders.append(rem)
        o, rem = torch.cat(outputs, dim=1), torch.cat(remainders, dim=1)
    return o.to(dtype), rem.to(dtype)


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_INTEL), reason="Requires a supported GPU")
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16], ids=['fp16', 'bf16'])
@pytest.mark.parametrize('attend_current', [False, True])
@pytest.mark.parametrize('layout', ['dense', 'packed', 'padded', 'empty-row'])
@pytest.mark.parametrize('qk_norm', [False, True])
def test_forward_backward(dtype, attend_current, layout, qk_norm, monkeypatch):
    torch.manual_seed(42)
    layer = StickBreakingAttention(
        hidden_size=128,
        num_heads=2,
        num_kv_heads=1,
        qkv_bias=True,
        qk_norm=qk_norm,
        attend_current=attend_current,
    ).to(device=device, dtype=dtype)
    reference = copy.deepcopy(layer)
    B, T = (1, 101) if layout == 'packed' else (2, 67)
    x = torch.randn(B, T, 128, dtype=dtype, device=device)
    do = torch.randn_like(x)
    kwargs = {}
    if layout == 'packed':
        kwargs['cu_seqlens'] = torch.tensor([0, 29, 101], dtype=torch.int32, device=device)
    elif layout in ('padded', 'empty-row'):
        mask = torch.ones(B, T, dtype=torch.long, device=device)
        mask[0, :13], mask[1, 55:] = 0, 0
        if layout == 'empty-row':
            mask[0] = 0
        kwargs['attention_mask'] = mask

    actual_x = x.clone().requires_grad_()
    actual, _, cache = layer(hidden_states=actual_x, **kwargs)
    assert cache is None
    actual.backward(do)
    ref_x = x.clone().requires_grad_()
    with monkeypatch.context() as patch:
        patch.setattr(stickbreaking_attn, 'parallel_stickbreaking_attn', _reference)
        expected, _, _ = reference(hidden_states=ref_x, **kwargs)
        expected.backward(do)
    assert_close(prefix='output', ref=expected, tri=actual, ratio=TOL[dtype])
    assert_close(prefix='dx', ref=ref_x.grad, tri=actual_x.grad, ratio=TOL[dtype])
    for (name, param), (ref_name, ref_param) in zip(layer.named_parameters(), reference.named_parameters(), strict=True):
        assert name == ref_name
        assert_close(prefix=name, ref=ref_param.grad, tri=param.grad, ratio=TOL[dtype])
    if layout in ('padded', 'empty-row'):
        assert torch.count_nonzero(actual[mask == 0]) == 0
        assert torch.count_nonzero(actual_x.grad[mask == 0]) == 0


@pytest.mark.parametrize(
    ('hidden_size', 'num_heads', 'num_kv_heads'),
    [(33, 2, 1), (32, 0, 1), (32, 2, 0), (32, 2, 3), (514, 2, 1)],
    ids=['hidden-size', 'zero-query-heads', 'zero-kv-heads', 'gqa', 'head-dim'],
)
def test_rejects_invalid_heads(hidden_size, num_heads, num_kv_heads):
    with pytest.raises(ValueError):
        StickBreakingAttention(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads)


@pytest.mark.parametrize(
    'kwargs',
    [{'use_cache': True}, {'past_key_values': ()}, {'output_attentions': True}],
    ids=['cache', 'past', 'attention-matrix'],
)
def test_rejects_unsupported_outputs(kwargs):
    layer = StickBreakingAttention(hidden_size=32, num_heads=2)
    with pytest.raises(NotImplementedError):
        layer(hidden_states=torch.zeros(1, 3, 32), **kwargs)


def test_rejects_cpu_forward():
    layer = StickBreakingAttention(hidden_size=32, num_heads=2)
    with pytest.raises(NotImplementedError, match='requires a CUDA/HIP or Intel GPU'):
        layer(hidden_states=torch.zeros(1, 3, 32))


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_INTEL), reason="Requires a supported GPU")
def test_rejects_invalid_layout():
    layer = StickBreakingAttention(hidden_size=32, num_heads=2).to(device=device, dtype=torch.bfloat16)
    x = torch.zeros(2, 3, 32, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='attention_mask must have shape'):
        layer(hidden_states=x, attention_mask=torch.ones(2, 4, device=device))
    with pytest.raises(ValueError, match='batch size 1'):
        layer(hidden_states=x, cu_seqlens=torch.tensor([0, 6], device=device))


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_INTEL), reason="Requires a supported GPU")
def test_rejects_fp32_projection():
    layer = StickBreakingAttention(hidden_size=32, num_heads=2).to(device=device)
    with pytest.raises(TypeError, match='matching fp16/bf16'):
        layer(hidden_states=torch.zeros(1, 3, 32, device=device))
