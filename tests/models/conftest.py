# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
from transformers import AutoModelForCausalLM

from fla.models import RWKV7Config


@pytest.fixture
def make_model():
    """Construct a small zero-layer model for API tests."""
    def create(config_class, **kwargs):
        if config_class is RWKV7Config and not torch._C.has_lapack:
            pytest.skip('RWKV7 initialization requires LAPACK.')
        config = config_class(**{
            'hidden_size': 64,
            'num_heads': 2,
            'num_hidden_layers': 0,
            'num_self_decoder_layers': 0,
            'intermediate_size': 128,
            'vocab_size': 64,
            'fuse_norm': False,
            'fuse_cross_entropy': False,
            **kwargs,
        })
        return AutoModelForCausalLM.from_config(config)

    return create
