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
@pytest.mark.parametrize("use_short_conv", [False, True], ids=["no-conv", "conv"])
@pytest.mark.parametrize("warm_cache", [False, True], ids=["cold", "warm"])
def test_masked_single_token_decode(dtype: torch.dtype, use_short_conv: bool, warm_cache: bool):
    torch.manual_seed(42)
    layer = LightNetAttention(
        hidden_size=64,
        num_heads=2,
        expand_ratio=8,
        use_short_conv=use_short_conv,
        layer_idx=0,
    ).to(device=device, dtype=dtype)
    reference = copy.deepcopy(layer)
    cache, reference_cache = Cache(), Cache()
    history_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1]], device=device, dtype=torch.bool)
    if warm_cache:
        prefill = torch.randn(2, 6, 64, device=device, dtype=dtype)
        with torch.no_grad():
            for module, state in ((layer, cache), (reference, reference_cache)):
                module(hidden_states=prefill, attention_mask=history_mask[:, :-1], past_key_values=state, use_cache=True)
    inference_cache, reference_inference_cache = copy.deepcopy(cache), copy.deepcopy(reference_cache)
    x = torch.randn(2, 1, 64, device=device, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual, _, _ = layer(hidden_states=x, attention_mask=history_mask, past_key_values=cache, use_cache=True)
    expected, _, _ = reference(hidden_states=reference_x, past_key_values=reference_cache, use_cache=True)
    do = torch.randn_like(actual)
    actual.backward(do)
    expected.backward(do)
    tol = 0.005 if dtype == torch.float16 else 0.02
    assert_close("input gradient", reference_x.grad, x.grad, tol)
    for (name, parameter), (_, reference_parameter) in zip(layer.named_parameters(), reference.named_parameters()):
        assert_close(name, reference_parameter.grad, parameter.grad, tol)

    with torch.no_grad():
        inference, _, _ = layer(
            hidden_states=x,
            attention_mask=history_mask,
            past_key_values=inference_cache,
            use_cache=True,
        )
        reference_inference, _, _ = reference(hidden_states=x, past_key_values=reference_inference_cache, use_cache=True)
    for output, state in ((actual, cache), (inference, inference_cache), (reference_inference, reference_inference_cache)):
        assert_close("output", expected, output, tol)
        for name in ("recurrent_state", "ffn_state", "conv_state"):
            expected_state, actual_state = reference_cache[0][name], state[0][name]
            if expected_state is None:
                assert actual_state is None
                continue
            if isinstance(expected_state, torch.Tensor):
                expected_state, actual_state = (expected_state,), (actual_state,)
            for ref, result in zip(expected_state, actual_state):
                assert_close(name, ref, result, tol)
