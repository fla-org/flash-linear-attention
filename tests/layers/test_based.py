# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy

import pytest
import torch
import triton
from einops import rearrange

from fla.layers.based import BasedLinearAttention
from fla.utils import assert_close, device


def _quadratic_attention(layer: BasedLinearAttention, hidden_states: torch.Tensor) -> torch.Tensor:
    q = rearrange(layer.q_proj(hidden_states), 'b t (h d) -> b h t d', h=layer.num_heads)
    k = rearrange(layer.k_proj(hidden_states), 'b t (h d) -> b h t d', h=layer.num_heads)
    v = rearrange(layer.v_proj(hidden_states), 'b t (h d) -> b h t d', h=layer.num_heads)
    scores = (q @ k.transpose(-1, -2)) * layer.feature_dim ** -0.5
    scores = 1 + scores + 0.5 * scores.square()
    if layer.causal:
        scores = scores.tril()
    o = (scores @ v) / (scores.sum(-1, keepdim=True) + layer.eps)
    return layer.o_proj(rearrange(o, 'b h t d -> b t (h d)'))


@pytest.mark.parametrize('mode', ['parallel', 'chunk', 'fused_chunk'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V'),
    [
        pytest.param(*test, id='B{}-T{}-H{}-K{}-V{}'.format(*test))
        for test in [(1, 1, 2, 16, 16), (2, 63, 3, 16, 32), (1, 129, 2, 8, 24), (1, 17, 1, 1, 16)]
    ],
)
def test_forward(B: int, T: int, H: int, K: int, V: int, dtype: torch.dtype, mode: str, monkeypatch: pytest.MonkeyPatch):
    if mode == 'chunk' and dtype != torch.float32:
        # 16-bit chunk kernels reject odd K/V (see the assert in chunk_bwd_dv);
        # the Taylor feature map expands K to 1 + K + K*(K+1)/2.
        if (1 + K + K * (K + 1) // 2) % 2 == 1 or V % 2 == 1:
            pytest.skip('odd expanded K/V is not supported for 16-bit chunk kernels')
    monkeypatch.setenv('TRITON_F32_DEFAULT', 'ieee')
    if hasattr(triton, 'knobs'):
        monkeypatch.setattr(triton.knobs.language, 'fp32_default', 'ieee')
    torch.manual_seed(42)
    layer = BasedLinearAttention(
        hidden_size=H * V,
        feature_dim=K,
        num_key_value_heads=H,
        num_heads=H,
        mode=mode,
    ).to(device=device, dtype=dtype)
    reference = copy.deepcopy(layer).float()
    reference.eps = 1e-6 if mode == 'parallel' else 1e-10
    x = torch.randn(B, T, H * V, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().float().requires_grad_(True)
    do = torch.randn_like(x)

    ref = _quadratic_attention(reference, x_ref)
    ref.backward(do.float())
    actual = layer(hidden_states=x)
    actual.backward(do)

    assert actual.shape == x.shape
    assert actual.dtype == dtype
    tol = {torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 2e-2}[dtype]
    for name, expected, result in [
        ('o', ref, actual),
        ('dx', x_ref.grad, x.grad),
        *[(name, parameter.grad, layer.get_parameter(name).grad) for name, parameter in reference.named_parameters()],
    ]:
        assert torch.isfinite(expected).all(), name
        assert torch.isfinite(result).all(), name
        # single-token attention has near-zero query/key gradients after normalization
        atol = tol if T == 1 and dtype != torch.float32 and name in ['q_proj.weight', 'k_proj.weight'] else 1e-6
        assert_close(name, expected, result, tol, err_atol=atol)


@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('eps', [1e-12, 0.1])
@pytest.mark.parametrize(('hidden_size', 'H', 'K'), [(48, 2, 8), (10, 4, 2)])
def test_forward_reference(causal: bool, eps: float, hidden_size: int, H: int, K: int):
    torch.manual_seed(42)
    layer = BasedLinearAttention(
        hidden_size=hidden_size,
        feature_dim=K,
        num_key_value_heads=H,
        num_heads=H,
        eps=eps,
        causal=causal,
    )
    x = torch.randn(2, 17, hidden_size, requires_grad=True)
    do = torch.randn_like(x)
    ref = _quadratic_attention(layer, x)
    ref.backward(do)
    ref_grads = [tensor.grad.clone() for tensor in (x, *layer.parameters())]
    x.grad = None
    layer.zero_grad(set_to_none=True)
    actual = layer.forward_reference(hidden_states=x)
    actual.backward(do)
    actual_grads = [tensor.grad for tensor in (x, *layer.parameters())]

    assert_close('o', ref, actual, 1e-5)
    for i, (expected, result) in enumerate(zip(ref_grads, actual_grads)):
        assert torch.isfinite(expected).all()
        assert torch.isfinite(result).all()
        assert_close(f'grad_{i}', expected, result, 1e-5)
    if not causal:
        assert_close('noncausal', actual, layer(hidden_states=x), 1e-5)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        pytest.param({'mode': 'unknown'}, 'mode', id='mode'),
        pytest.param({'feature_name': 'relu'}, 'feature', id='feature-map'),
        pytest.param({'num_heads': 6}, 'equal', id='unequal-heads'),
        pytest.param({'hidden_size': 2, 'num_heads': 4, 'num_key_value_heads': 4}, 'at least', id='hidden-size'),
        pytest.param({'feature_dim': 0}, 'positive', id='feature-dim'),
        pytest.param({'num_key_value_heads': 0}, 'positive', id='zero-heads'),
        pytest.param({'feature_dim': 129}, '128', id='parallel-feature-dim'),
    ],
)
def test_invalid_config(kwargs: dict, message: str):
    config = {'hidden_size': 192, **kwargs}
    with pytest.raises(AssertionError, match=message):
        BasedLinearAttention(**config)
