# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy

import pytest
import torch

from fla.layers.lightnet import LightNetAttention
from fla.models.utils import Cache
from fla.utils import assert_close, device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("T_prefill,T_continue", [(8, 4), (32, 80)], ids=["T8-T4", "T32-T80"])
@pytest.mark.parametrize("use_short_conv", [False, True], ids=["no-conv", "conv"])
def test_warm_cache_multi_token_continuation(
    dtype: torch.dtype,
    T_prefill: int,
    T_continue: int,
    use_short_conv: bool,
):
    B, D = 2, 64
    torch.manual_seed(42)
    layer = LightNetAttention(
        hidden_size=D,
        num_heads=2,
        expand_ratio=8,
        use_short_conv=use_short_conv,
        layer_idx=0,
    ).to(device=device, dtype=dtype).eval()
    prefill = torch.randn(B, T_prefill, D, device=device, dtype=dtype)
    continuation = torch.randn(B, T_continue, D, device=device, dtype=dtype)
    tol = 0.005 if dtype == torch.float16 else 0.02

    with torch.no_grad():
        expected, _, _ = layer(hidden_states=torch.cat([prefill, continuation], dim=1))
        cache = Cache()
        actual_prefill, _, returned_cache = layer(hidden_states=prefill, past_key_values=cache, use_cache=True)
        actual, _, _ = layer(hidden_states=continuation, past_key_values=cache, use_cache=True)

    assert returned_cache is cache
    assert_close("prefill", expected[:, :T_prefill], actual_prefill, tol)
    assert_close("continuation", expected[:, T_prefill:], actual, tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "use_short_conv,singleton",
    [(False, False), (True, False), (True, True)],
    ids=["no-conv", "conv", "singleton-conv"],
)
@pytest.mark.parametrize("layout", ["padded", "packed"])
@pytest.mark.parametrize("warm_cache", [False, True], ids=["cold", "warm"])
def test_varlen_forward_backward(dtype: torch.dtype, use_short_conv: bool, singleton: bool, layout: str, warm_cache: bool):
    torch.manual_seed(42)
    B, T, D = (2, 9, 64) if singleton else (5, 79, 64)
    layer = LightNetAttention(
        hidden_size=D,
        num_heads=2,
        expand_ratio=8,
        use_short_conv=use_short_conv,
        layer_idx=0,
    ).to(device=device, dtype=dtype)
    reference = copy.deepcopy(layer)
    hidden_states = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    reference_inputs = hidden_states.detach().clone().requires_grad_()
    attention_mask = torch.ones(B, T, device=device, dtype=torch.bool)
    if singleton:
        attention_mask[:, :-1] = False
    else:
        attention_mask[0] = False
        attention_mask[1, :6] = False
        attention_mask[2, 67:] = False
        attention_mask[3, [1, 17, 64]] = False
        attention_mask[4] = False
        attention_mask[4, 31] = True
    do = torch.randn_like(hidden_states)
    tol = 0.005 if dtype == torch.float16 else 0.02
    cache = Cache() if warm_cache else None
    reference_caches = [Cache() if warm_cache else None for _ in range(B)]
    if warm_cache:
        prefill = torch.randn(B, 11, D, device=device, dtype=dtype)
        with torch.no_grad():
            layer(hidden_states=prefill, past_key_values=cache, use_cache=True)
            for i, reference_cache in enumerate(reference_caches):
                reference(hidden_states=prefill[i:i+1], past_key_values=reference_cache, use_cache=True)

    if layout == "padded":
        actual, _, _ = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=warm_cache,
        )
        assert torch.count_nonzero(actual[~attention_mask]) == 0
        actual = actual[attention_mask]
    else:
        lengths = attention_mask.sum(-1, dtype=torch.int32)
        cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)])
        packed = hidden_states[attention_mask].unsqueeze(0)
        actual, _, _ = layer(
            hidden_states=packed,
            attention_mask=torch.zeros_like(packed[..., 0], dtype=torch.bool),
            past_key_values=cache,
            use_cache=warm_cache,
            cu_seqlens=cu_seqlens,
        )
        actual = actual.squeeze(0)

    expected = torch.cat([
        reference(
            hidden_states=reference_inputs[i, mask].unsqueeze(0),
            past_key_values=reference_caches[i],
            use_cache=warm_cache,
        )[0].squeeze(0)
        for i, mask in enumerate(attention_mask)
        if mask.any()
    ])
    assert_close("output", expected, actual, tol)
    (actual.float() * do[attention_mask].float()).sum().backward()
    (expected.float() * do[attention_mask].float()).sum().backward()
    assert_close("input gradient", reference_inputs.grad, hidden_states.grad, tol)
    assert torch.count_nonzero(hidden_states.grad[~attention_mask]) == 0
    for (name, parameter), (_, reference_parameter) in zip(layer.named_parameters(), reference.named_parameters()):
        assert_close(name, reference_parameter.grad, parameter.grad, tol)


@pytest.mark.parametrize("use_short_conv", [False, True], ids=["no-conv", "conv"])
@pytest.mark.parametrize("T_continue", [1, 7, 79], ids=["T1", "T7", "T79"])
@pytest.mark.parametrize("layout", ["padded", "packed"])
@torch.no_grad()
def test_varlen_warm_cache(use_short_conv: bool, T_continue: int, layout: str):
    torch.manual_seed(42)
    B, T_prefill, D = 3, 9, 64
    dtype, tol = torch.bfloat16, 0.02
    layer = LightNetAttention(
        hidden_size=D,
        num_heads=2,
        expand_ratio=8,
        use_short_conv=use_short_conv,
        layer_idx=0,
    ).to(device=device, dtype=dtype).eval()
    prefill = torch.randn(B, T_prefill, D, device=device, dtype=dtype)
    continuation = torch.randn(B, T_continue, D, device=device, dtype=dtype)
    prefill_mask = torch.ones(B, T_prefill, device=device, dtype=torch.bool)
    prefill_mask[0, :2] = False
    prefill_mask[1] = False
    prefill_mask[2, 5:] = False
    continuation_mask = torch.ones(B, T_continue, device=device, dtype=torch.bool)
    continuation_mask[0] = False
    continuation_mask[2, 1::7] = False
    cache = Cache()

    for hidden_states, mask, history_mask in [
        (prefill, prefill_mask, prefill_mask),
        (continuation, continuation_mask, torch.cat([prefill_mask, continuation_mask], dim=1)),
    ]:
        if layout == "padded":
            actual, _, returned_cache = layer(
                hidden_states=hidden_states,
                attention_mask=history_mask,
                past_key_values=cache,
                use_cache=True,
            )
            assert torch.count_nonzero(actual[~mask]) == 0
            actual = actual[mask]
        else:
            lengths = mask.sum(-1, dtype=torch.int32)
            cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)])
            actual, _, returned_cache = layer(
                hidden_states=hidden_states[mask].unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
                cu_seqlens=cu_seqlens,
            )
            actual = actual.squeeze(0)
        assert returned_cache is cache

    expected = []
    for i in range(B):
        reference_cache = Cache()
        sequence = torch.cat([prefill[i, prefill_mask[i]], continuation[i, continuation_mask[i]]]).unsqueeze(0)
        output, _, _ = layer(hidden_states=sequence, past_key_values=reference_cache, use_cache=True)
        expected.append(output[:, prefill_mask[i].sum():].squeeze(0))
        for name in ("recurrent_state", "ffn_state"):
            assert_close(name, reference_cache[0][name], cache[0][name][i:i+1], tol)
        if use_short_conv:
            for reference_state, state in zip(reference_cache[0]["conv_state"], cache[0]["conv_state"]):
                assert_close("conv_state", reference_state, state[i:i+1], tol)
    assert_close("continuation", torch.cat(expected), actual, tol)

    previous_state = copy.deepcopy(cache[0])
    empty_output, _, _ = layer(
        hidden_states=continuation,
        attention_mask=torch.zeros_like(continuation_mask),
        past_key_values=cache,
        use_cache=True,
    )
    assert torch.count_nonzero(empty_output) == 0
    for name in ("recurrent_state", "ffn_state"):
        assert torch.equal(previous_state[name], cache[0][name])
    if use_short_conv:
        for previous, state in zip(previous_state["conv_state"], cache[0]["conv_state"]):
            assert torch.equal(previous, state)


@pytest.mark.parametrize("use_short_conv", [False, True], ids=["no-conv", "conv"])
@pytest.mark.parametrize("layout", ["padded", "packed"])
def test_all_padding_forward_backward(use_short_conv: bool, layout: str):
    torch.manual_seed(42)
    layer = LightNetAttention(
        hidden_size=64,
        num_heads=2,
        expand_ratio=8,
        use_short_conv=use_short_conv,
        layer_idx=0,
    ).to(device=device, dtype=torch.bfloat16)
    shape = (3, 9, 64) if layout == "padded" else (1, 0, 64)
    hidden_states = torch.randn(*shape, device=device, dtype=torch.bfloat16, requires_grad=True)
    if layout == "padded":
        kwargs = {"attention_mask": torch.zeros(3, 9, device=device, dtype=torch.bool)}
    else:
        kwargs = {"cu_seqlens": torch.zeros(4, device=device, dtype=torch.int32)}
    cache = Cache()
    output, _, _ = layer(
        hidden_states=hidden_states,
        past_key_values=cache,
        use_cache=True,
        **kwargs,
    )
    assert torch.count_nonzero(output) == 0
    output.float().sum().backward()
    assert torch.count_nonzero(hidden_states.grad) == 0
    for parameter in layer.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0

    with torch.no_grad():
        next_input = torch.randn(3, 1, 64, device=device, dtype=torch.bfloat16)
        actual, _, _ = layer(hidden_states=next_input, past_key_values=cache, use_cache=True)
        expected, _, _ = layer(hidden_states=next_input)
    assert_close("first valid token", expected, actual, 0.02)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('T_continue', [4, 80])
@pytest.mark.parametrize('use_short_conv', [False, True])
def test_warm_cache_ignores_left_padding_content(dtype: torch.dtype, T_continue: int, use_short_conv: bool):
    torch.manual_seed(42)
    layer = LightNetAttention(
        hidden_size=64, num_heads=2, expand_ratio=8, use_short_conv=use_short_conv, layer_idx=0,
    ).to(device=device, dtype=dtype).eval()
    prefill = torch.randn(2, 8, 64, device=device, dtype=dtype)
    continuation = torch.randn(2, T_continue, 64, device=device, dtype=dtype)
    mask = torch.ones(2, T_continue, device=device, dtype=torch.bool)
    mask[:, :3] = False
    changed = continuation.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 5
    outputs, states = [], []
    with torch.no_grad():
        for current in (continuation, changed):
            cache = Cache()
            layer(hidden_states=prefill, past_key_values=cache, use_cache=True)
            output, _, _ = layer(hidden_states=current, attention_mask=mask, past_key_values=cache, use_cache=True)
            outputs.append(output[mask])
            states.append(cache[0]['ffn_state'].clone())
    assert torch.isfinite(outputs[0]).all()
    assert torch.isfinite(states[0]).all()
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(states[0], states[1])


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_warm_cache_all_padding_keeps_state(dtype: torch.dtype):
    torch.manual_seed(42)
    layer = LightNetAttention(hidden_size=64, num_heads=2, expand_ratio=8, layer_idx=0).to(device=device, dtype=dtype).eval()
    prefill = torch.randn(2, 8, 64, device=device, dtype=dtype)
    padding = torch.randn(2, 3, 64, device=device, dtype=dtype)
    continuation = torch.randn(2, 5, 64, device=device, dtype=dtype)
    mask = torch.zeros(2, 3, device=device, dtype=torch.bool)
    outputs = []
    with torch.no_grad():
        for skip_padding in (True, False):
            cache = Cache()
            layer(hidden_states=prefill, past_key_values=cache, use_cache=True)
            if not skip_padding:
                old_normalizer = cache[0]['ffn_state'].clone()
                old_recurrent_state = cache[0]['recurrent_state'].clone()
                layer(hidden_states=padding, attention_mask=mask, past_key_values=cache, use_cache=True)
                assert torch.equal(cache[0]['ffn_state'], old_normalizer)
                assert torch.equal(cache[0]['recurrent_state'], old_recurrent_state)
            output, _, _ = layer(hidden_states=continuation, past_key_values=cache, use_cache=True)
            outputs.append(output)
    assert torch.isfinite(outputs[0]).all()
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('T_continue', [4, 80])
def test_warm_cache_padding_has_zero_gradient(dtype: torch.dtype, T_continue: int):
    torch.manual_seed(42)
    layer = LightNetAttention(hidden_size=64, num_heads=2, expand_ratio=8, layer_idx=0).to(device=device, dtype=dtype).train()
    prefill = torch.randn(2, 8, 64, device=device, dtype=dtype)
    continuation = torch.randn(2, T_continue, 64, device=device, dtype=dtype, requires_grad=True)
    mask = torch.ones(2, T_continue, device=device, dtype=torch.bool)
    mask[:, :3] = False
    cache = Cache()
    with torch.no_grad():
        layer(hidden_states=prefill, past_key_values=cache, use_cache=True)
    output, _, _ = layer(hidden_states=continuation, attention_mask=mask, past_key_values=cache, use_cache=True)
    valid_output = output[mask]
    (valid_output.float() * torch.randn_like(valid_output)).sum().backward()
    assert torch.isfinite(continuation.grad).all()
    assert torch.count_nonzero(continuation.grad[~mask]) == 0
    assert torch.count_nonzero(continuation.grad[mask]) > 0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('T_continue', [4, 80])
def test_warm_cache_padding_before_first_valid_token(dtype: torch.dtype, T_continue: int):
    torch.manual_seed(42)
    layer = LightNetAttention(hidden_size=64, num_heads=2, expand_ratio=8, layer_idx=0).to(device=device, dtype=dtype).train()
    prefill = torch.randn(2, 8, 64, device=device, dtype=dtype)
    continuation = torch.randn(2, T_continue, 64, device=device, dtype=dtype, requires_grad=True)
    mask = torch.ones(2, T_continue, device=device, dtype=torch.bool)
    mask[:, :3] = False
    cache = Cache()
    with torch.no_grad():
        layer(hidden_states=prefill, attention_mask=torch.zeros(2, 8, device=device, dtype=torch.bool),
              past_key_values=cache, use_cache=True)
        expected, _, _ = layer(hidden_states=continuation[:, 3:].detach(), use_cache=False)
    output, _, _ = layer(hidden_states=continuation, attention_mask=mask, past_key_values=cache, use_cache=True)
    valid_output = output[:, 3:]
    assert torch.isfinite(valid_output).all()
    assert torch.isfinite(cache[0]['recurrent_state']).all()
    assert_close('empty-prefix continuation', expected, valid_output, 2e-2 if dtype == torch.bfloat16 else 2e-3)
    (valid_output.float() * torch.randn_like(valid_output)).sum().backward()
    assert torch.isfinite(continuation.grad).all()
    assert torch.count_nonzero(continuation.grad[~mask]) == 0
    assert torch.count_nonzero(continuation.grad[mask]) > 0
