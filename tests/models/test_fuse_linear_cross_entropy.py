# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import (
    BitNetConfig,
    DeltaFormerConfig,
    ForgettingTransformerConfig,
    PaTHAttentionConfig,
)
from fla.utils import assert_close, device, find_spec_cached

from .test_modeling_utils import create_model_and_config

# Mark (not importorskip): a fully skipped module reports "no tests collected" and exits
# with code 5, which CI per-file loops treat as a failure.
pytestmark = pytest.mark.skipif(find_spec_cached("flash_attn") is None,
                                reason="BitNet/DeltaFormer attention requires flash-attn (`pip install flash-attn --no-build-isolation`).")


# ===================================================================================
# Test for Fused Linear Cross Entropy (training loss + no-labels logits)
# ===================================================================================
@pytest.mark.parametrize(
    "config_class",
    [
        pytest.param(config_class, id=config_class.__name__)
        for config_class in [
            BitNetConfig,
            DeltaFormerConfig,
            ForgettingTransformerConfig,
            PaTHAttentionConfig,
        ]
    ],
)
def test_fuse_linear_cross_entropy(config_class):
    L, B, T, H, D = 2, 2, 64, 4, 64
    model, config = create_model_and_config(config_class, L, H, D, dtype=torch.bfloat16)
    model.eval()

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T), device=device)
    labels = input_ids.clone()

    with torch.no_grad():
        expected = model(input_ids, labels=labels).loss

        model.config.fuse_linear_cross_entropy = True
        outputs = model(input_ids, labels=labels)
        assert outputs.logits is None
        assert_close("loss", expected, outputs.loss, 0.02)

        # the generate path passes no labels; logits must still be materialized
        outputs = model(input_ids)
        assert outputs.logits is not None
        output_ids = model.generate(input_ids[:, :T // 2], max_new_tokens=2)
    assert output_ids.shape[1] == T // 2 + 2
