# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from copy import deepcopy
from functools import partial

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from fla.models.modeling_layers import GradientCheckpointingLayer


class CheckpointedLayer(GradientCheckpointingLayer):

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, hidden_states, **kwargs):
        self.cache_kwargs = kwargs
        return self.proj(hidden_states).sin()


@pytest.mark.parametrize('use_reentrant', [False, True])
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('cache_kwargs', [
    {},
    {'use_cache': False, 'past_key_values': None},
    {'use_cache': True, 'past_key_value': (), 'past_key_values': (), 'layer_past': ()},
])
def test_gradient_checkpointing(use_reentrant, training, enabled, cache_kwargs):
    torch.manual_seed(42)
    layer = CheckpointedLayer().train(training)
    reference = deepcopy(layer)
    layer.gradient_checkpointing = enabled
    layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=use_reentrant)
    x = torch.randn(2, 4, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)

    output = layer(x, **cache_kwargs)
    ref_output = reference(ref_x, **cache_kwargs)
    torch.testing.assert_close(output, ref_output)
    output.sum().backward()
    ref_output.sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for parameter, ref_parameter in zip(layer.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad)

    if enabled and training:
        assert layer.cache_kwargs == {
            name: False if name == 'use_cache' else None for name in cache_kwargs
        }
    else:
        assert layer.cache_kwargs == cache_kwargs
    assert reference.cache_kwargs == cache_kwargs
