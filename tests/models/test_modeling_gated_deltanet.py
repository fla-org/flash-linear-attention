# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetBlock
from fla.utils import assert_close, device
from tests.models.test_modeling_base import (
    run_test_generate_matches_forward,
    run_test_generation,
    run_test_model_forward_backward,
)


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'attnres_block_size', 'output_gate_activation', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-bs{}-gate{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, True,  None, 'swish',   torch.bfloat16),
            (4, 4, 1024, 4, 64, False, None, 'swish',   torch.bfloat16),
            (4, 4, 1024, 4, 64, False, 1,    'swish',   torch.bfloat16),
            (4, 4, 1024, 4, 64, False, 4,    'swish',   torch.bfloat16),
            (4, 4, 1024, 4, 64, False, None, 'sigmoid', torch.bfloat16),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    use_l2warp: bool,
    attnres_block_size: int | None,
    output_gate_activation: str,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    run_test_model_forward_backward(
        L=L,
        B=B,
        T=T,
        H=H,
        D=D,
        config_class=GatedDeltaNetConfig,
        use_l2warp=use_l2warp,
        dtype=dtype,
        attnres_block_size=attnres_block_size,
        output_gate_activation=output_gate_activation,
    )


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 4, 2000, 8, 64, torch.float16),
        ]
    ],
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    run_test_generation(L, B, T, H, D, GatedDeltaNetConfig, dtype)


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 2, 64, 8, 64, torch.float32),
        ]
    ],
)
def test_generate_prefill(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    run_test_generate_matches_forward(L, B, T, H, D, GatedDeltaNetConfig, dtype)


@torch.no_grad()
def test_logits_to_keep_with_labels():
    B, T, H, D, V = 2, 8, 2, 8, 32
    config = GatedDeltaNetConfig(
        hidden_size=H * D,
        num_hidden_layers=1,
        num_heads=H,
        head_dim=D,
        expand_v=1,
        vocab_size=V,
        fuse_cross_entropy=False,
    )
    model = GatedDeltaNetForCausalLM(config).eval().to(device)
    input_ids = torch.randint(V, (B, T), device=device)

    expected = model(input_ids, labels=input_ids)
    actual = model(input_ids, labels=input_ids, logits_to_keep=1)
    inference = model(input_ids, logits_to_keep=1)

    assert actual.logits.shape == expected.logits.shape == (B, T, V)
    assert inference.logits.shape == (B, 1, V)
    assert torch.isfinite(actual.loss)
    assert_close('loss', expected.loss, actual.loss, 1e-6)


@pytest.mark.parametrize('activation', [None, 'silu', 'sigmoid'], ids=['default', 'silu', 'sigmoid'])
@pytest.mark.parametrize('use_gate', [False, True], ids=['ungated', 'gated'])
def test_output_gate_config(activation: str | None, use_gate: bool, tmp_path):
    torch.manual_seed(42)
    kwargs = {} if activation is None else {'output_gate_activation': activation}
    config = GatedDeltaNetConfig(
        hidden_size=128,
        expand_v=1,
        use_gate=use_gate,
        head_dim=32,
        num_heads=2,
        intermediate_size=256,
        num_hidden_layers=1,
        vocab_size=128,
        **kwargs,
    )
    config.save_pretrained(tmp_path)
    restored = GatedDeltaNetConfig.from_pretrained(tmp_path)
    expected = activation or 'swish'
    assert restored.output_gate_activation == expected
    block = GatedDeltaNetBlock(config=restored, layer_idx=0)
    if use_gate:
        assert block.attn.o_norm.activation == expected
    else:
        assert not hasattr(block.attn, 'g_proj')


@pytest.mark.parametrize('use_gate', [False, True], ids=['ungated', 'gated'])
def test_output_gate_config_validation(use_gate: bool):
    with pytest.raises(ValueError, match='output_gate_activation'):
        GatedDeltaNetConfig(use_gate=use_gate, output_gate_activation='relu')
