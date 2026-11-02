# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest import mock

import pytest
import torch

from fla.layers.gsa2 import GatedSlotAttention2
from fla.models.cache_utils import Cache
from fla.utils import IS_AMD, IS_NPU, IS_NVIDIA, assert_close, device

# the layer runs the GDN-2 kernels, which tests/ops/test_gdn2.py guards the same way
pytestmark = pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")


@pytest.mark.parametrize(
    ('num_heads', 'num_v_heads', 'use_short_conv', 'num_slots', 'level2_axis'),
    [
        pytest.param(*case, id="H{}-HV{}-conv{}-M{}-{}".format(*case))
        for case in [
            (2, 2, False, 64, 'slot'),
            (2, 4, False, 64, 'head'),
            (2, 2, True, 128, 'head'),
        ]
    ],
)
def test_gsa2_layer(num_heads: int, num_v_heads: int, use_short_conv: bool, num_slots: int, level2_axis: str):
    """Both memories and their projections receive finite training gradients."""
    torch.manual_seed(42)
    hidden_size, head_dim, B, T = 128, 32, 2, 128
    layer = GatedSlotAttention2(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        num_slots=num_slots,
        level2_axis=level2_axis,
        use_short_conv=use_short_conv,
        use_w_conv=use_short_conv,
        layer_idx=0,
    ).to(device=device, dtype=torch.float32).train()

    x = torch.randn(B, T, hidden_size, device=device, dtype=torch.float32, requires_grad=True)
    o, _, _ = layer(x)
    assert o.shape == (B, T, hidden_size)
    assert torch.isfinite(o).all()

    o.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in layer.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name}.grad is None"
            assert torch.isfinite(p.grad).all(), f"{name}.grad has non-finite values"


@pytest.mark.parametrize(
    ('B', 'T', 'num_slots', 'level2_axis'),
    [
        pytest.param(1, 65, 64, 'head', id='single-partial-chunk'),
        pytest.param(1, 129, 256, 'slot', id='single-slot-axis'),
        pytest.param(2, 65, 128, 'head', id='multiple-sequences'),
    ],
)
def test_gsa2_masked_packed_gradients(B: int, T: int, num_slots: int, level2_axis: str):
    torch.manual_seed(42)
    layer = GatedSlotAttention2(
        hidden_size=128,
        head_dim=64,
        num_heads=2,
        num_slots=num_slots,
        level2_axis=level2_axis,
        use_short_conv=True,
        use_w_conv=True,
        layer_idx=0,
    ).to(device=device, dtype=torch.bfloat16).train()
    inputs = torch.randn(B, T, 128, device=device, dtype=torch.bfloat16)
    mask = torch.arange(T, device=device)[None, :].expand(B, -1) >= 3
    cu_seqlens_cpu = torch.arange(0, (B + 1) * (T - 3), T - 3, dtype=torch.int32)
    cu_seqlens = cu_seqlens_cpu.to(device)
    cotangent = torch.randn_like(inputs)[mask]
    dense = inputs.detach().clone().requires_grad_(True)
    packed = inputs.detach().clone().requires_grad_(True)
    expected = layer(dense, attention_mask=mask, use_cache=False)[0][mask]
    actual = layer(
        packed[mask].unsqueeze(0),
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        use_cache=False,
    )[0].squeeze(0)
    assert_close('masked packed output', expected, actual, 0.001)
    expected_grads = torch.autograd.grad(expected, (dense, *layer.parameters()), cotangent, allow_unused=True)
    actual_grads = torch.autograd.grad(actual, (packed, *layer.parameters()), cotangent, allow_unused=True)
    for index, (expected_grad, actual_grad) in enumerate(zip(expected_grads, actual_grads, strict=True)):
        assert (expected_grad is None) == (actual_grad is None), index
        if expected_grad is not None:
            assert torch.isfinite(expected_grad).all() and torch.isfinite(actual_grad).all()
            assert_close(f'masked packed gradient {index}', expected_grad, actual_grad, 1e-4)
    assert torch.count_nonzero(actual_grads[0][~mask]) == 0


@pytest.mark.parametrize('mode', ['chunk', 'fused_recurrent'])
@pytest.mark.parametrize(('num_slots', 'requires_grad'), [(64, True), (256, True), (256, False)])
def test_chunk_fallback(mode: str, num_slots: int, requires_grad: bool):
    torch.manual_seed(42)
    layer = GatedSlotAttention2(
        hidden_size=128,
        head_dim=32,
        num_heads=2,
        num_slots=num_slots,
        mode=mode,
        layer_idx=0,
    ).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(1, 32, 128, device=device, dtype=torch.bfloat16, requires_grad=True)

    message = "this input requires the chunk kernel"
    with (
        torch.set_grad_enabled(requires_grad),
        mock.patch('fla.layers.gsa2.fused_recurrent_gated_oja_rule2', side_effect=AssertionError(message)) as oja2,
        mock.patch('fla.layers.gsa2.fused_recurrent_gdn2', side_effect=AssertionError(message)) as gdn2,
    ):
        y = layer(x)[0]
        if requires_grad:
            y.sum().backward()

    oja2.assert_not_called()
    gdn2.assert_not_called()
    assert torch.isfinite(y).all()
    if requires_grad:
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


@torch.no_grad()
@pytest.mark.parametrize('T', [1, 3, 65])
def test_gsa2_read_only_call_keeps_cache(T: int):
    """A call with an existing cache and `use_cache=False` leaves every cached state and the token count unchanged."""
    torch.manual_seed(42)
    layer = GatedSlotAttention2(
        hidden_size=128,
        head_dim=32,
        num_heads=2,
        num_slots=64,
        use_short_conv=True,
        use_w_conv=True,
        layer_idx=0,
    ).to(device=device, dtype=torch.float32).eval()
    cache = Cache()
    layer(torch.randn(1, 8, 128, device=device), past_key_values=cache, use_cache=True)

    state = cache[0]
    states = [s.clone() for s in (*state['recurrent_state'], *state['conv_state'])]
    seen = cache.get_seq_length()

    x = torch.randn(1, T, 128, device=device)
    actual = layer(x, past_key_values=cache, use_cache=False)[0]

    state = cache[0]
    assert cache.get_seq_length() == seen
    for saved, current in zip(states, (*state['recurrent_state'], *state['conv_state'])):
        assert torch.equal(saved, current)
    expected = layer(x, past_key_values=cache, use_cache=True)[0]
    assert_close('o', expected, actual, 1e-6)
