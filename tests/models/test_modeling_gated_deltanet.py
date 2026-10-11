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

from .test_modeling_base import (
    run_test_generate_matches_forward,
    run_test_generation,
    run_test_model_forward_backward,
)
from .test_modeling_utils import create_model_and_config


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize('output_gate_activation', ['swish', 'sigmoid'])
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'attnres_block_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-bs{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, True,  None, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, None, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, 1,    torch.bfloat16),
            (4, 4, 1024, 4, 64, False, 4,    torch.bfloat16),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    output_gate_activation: str,
    use_l2warp: bool,
    attnres_block_size: int | None,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    run_test_model_forward_backward(
        L,
        B,
        T,
        H,
        D,
        GatedDeltaNetConfig,
        use_l2warp=use_l2warp,
        attnres_block_size=attnres_block_size,
        output_gate_activation=output_gate_activation,
        dtype=dtype,
    )


# ===================================================================================
# Test for Generation
# ===================================================================================
@pytest.mark.parametrize('output_gate_activation', ['swish', 'sigmoid'])
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
    output_gate_activation: str,
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    model, config = create_model_and_config(
        config_class=GatedDeltaNetConfig,
        L=L,
        H=H,
        D=D,
        dtype=dtype,
        output_gate_activation=output_gate_activation,
    )
    run_test_generation(L, B, T, H, D, GatedDeltaNetConfig, dtype, model=model, config=config)


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


@pytest.mark.parametrize('activation', [None, 'swish', 'silu', 'sigmoid'], ids=['legacy', 'swish', 'silu', 'sigmoid'])
@pytest.mark.parametrize('use_gate', [False, True], ids=['ungated', 'gated'])
def test_output_gate_config(activation: str | None, use_gate: bool, tmp_path):
    torch.manual_seed(42)
    config = GatedDeltaNetConfig(
        hidden_size=128,
        expand_v=1,
        use_gate=use_gate,
        head_dim=32,
        num_heads=2,
        intermediate_size=256,
        num_hidden_layers=1,
        vocab_size=128,
    )
    legacy = GatedDeltaNetBlock(config=config, layer_idx=0)
    config_dict = config.to_dict()
    if activation is None:
        config_dict.pop('output_gate_activation')
    else:
        config_dict['output_gate_activation'] = activation
    config = GatedDeltaNetConfig.from_dict(config_dict)
    config.save_pretrained(tmp_path)
    restored = GatedDeltaNetConfig.from_pretrained(tmp_path)
    expected = activation or 'swish'
    assert restored.output_gate_activation == expected
    assert restored.to_dict()['output_gate_activation'] == expected
    block = GatedDeltaNetBlock(config=restored, layer_idx=0)
    assert block.attn.output_gate_activation == expected
    if use_gate:
        assert block.attn.o_norm.activation == expected
    else:
        assert not hasattr(block.attn, 'g_proj')
    block.load_state_dict(legacy.state_dict(), strict=True)
    legacy.load_state_dict(block.state_dict(), strict=True)


@pytest.mark.parametrize('use_gate', [False, True], ids=['ungated', 'gated'])
def test_output_gate_config_validation(use_gate: bool):
    with pytest.raises(ValueError, match='output_gate_activation'):
        GatedDeltaNetConfig(use_gate=use_gate, output_gate_activation='relu')


def test_config_positional_compatibility():
    attn_spec = {"layers": [0], "num_heads": 2}
    cfg_pos = GatedDeltaNetConfig(
        "chunk",
        128,
        1.0,
        True,
        True,
        False,
        4,
        32,
        2,
        2,
        512,
        2,
        256,
        "swish",
        2,
        1e-5,
        attn_spec,
        False,
    )
    cfg_kw = GatedDeltaNetConfig(
        attn_mode="chunk",
        hidden_size=128,
        expand_v=1.0,
        use_gate=True,
        use_short_conv=True,
        allow_neg_eigval=False,
        conv_size=4,
        head_dim=32,
        num_heads=2,
        num_v_heads=2,
        max_position_embeddings=512,
        hidden_ratio=2,
        intermediate_size=256,
        hidden_act="swish",
        num_hidden_layers=2,
        norm_eps=1e-5,
        attn=attn_spec,
        use_cache=False,
    )
    assert cfg_pos.norm_eps == cfg_kw.norm_eps == 1e-5
    assert cfg_pos.attn == cfg_kw.attn
    assert cfg_pos.use_cache == cfg_kw.use_cache is False
    assert cfg_pos.output_gate_activation == cfg_kw.output_gate_activation == "swish"
