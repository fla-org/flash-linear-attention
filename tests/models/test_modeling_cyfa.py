# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
from transformers import AutoModelForCausalLM

from fla.models import CyclicFlowAttentionConfig
from fla.utils import IS_NVIDIA, assert_close, device

from .test_modeling_base import run_test_generation, run_test_model_forward_backward
from .test_modeling_utils import create_model_and_config

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='CyFA kernels currently require NVIDIA GPUs')


def make_model(dtype=torch.bfloat16, **kwargs):
    config = CyclicFlowAttentionConfig(
        hidden_size=128, num_heads=2, head_dim=64, num_slots=32, num_hidden_layers=2,
        vocab_size=256, intermediate_size=256, **kwargs,
    )
    return AutoModelForCausalLM.from_config(config).to(device=device, dtype=dtype)


@pytest.mark.parametrize('checkpoint_level', [0, 1])
@pytest.mark.parametrize('use_short_conv', [False, True])
def test_modeling(checkpoint_level, use_short_conv):
    torch.manual_seed(42)
    run_test_model_forward_backward(
        4, 4, 1024, 4, 64, CyclicFlowAttentionConfig, use_l2warp=False, dtype=torch.bfloat16,
        head_dim=64, num_slots=128,
        checkpoint_level=checkpoint_level, use_short_conv=use_short_conv,
    )


@pytest.mark.parametrize('use_short_conv', [False, True])
def test_generation(use_short_conv):
    torch.manual_seed(42)
    model, config = create_model_and_config(
        CyclicFlowAttentionConfig, 2, 8, 64, dtype=torch.float16,
        head_dim=64, num_slots=128, use_short_conv=use_short_conv,
    )
    run_test_generation(2, 4, 2000, 8, 64, CyclicFlowAttentionConfig, torch.float16, model=model, config=config)


@torch.no_grad()
def test_logits_to_keep_with_labels():
    torch.manual_seed(42)
    model = make_model(fuse_cross_entropy=False).eval()
    tokens = torch.randint(3, 256, (2, 65), device=device)
    expected = model(tokens, labels=tokens, use_cache=False)
    actual = model(tokens, labels=tokens, logits_to_keep=1, use_cache=False)
    assert actual.logits.shape == expected.logits.shape
    assert_close('loss', expected.loss, actual.loss, 1e-6)
    assert model(tokens, logits_to_keep=1, use_cache=False).logits.shape == (2, 1, 256)
