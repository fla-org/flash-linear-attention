# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetBlock
from fla.utils import device


def _tiny_config(**overrides):
    base = dict(
        hidden_size=128,
        head_dim=32,
        num_heads=2,
        num_v_heads=2,
        expand_v=1,
        intermediate_size=256,
        hidden_ratio=2,
        max_position_embeddings=512,
        num_hidden_layers=2,
        vocab_size=512,
    )
    base.update(overrides)
    return GatedDeltaNetConfig(**base)


def test_gated_deltanet_default_output_gate_activation_is_swish():
    layer = GatedDeltaNet(hidden_size=128, head_dim=32, num_heads=2, expand_v=1)
    assert layer.output_gate_activation == "swish"
    assert layer.o_norm.activation == "swish"
    cfg = _tiny_config()
    assert cfg.output_gate_activation == "swish"
    block = GatedDeltaNetBlock(cfg, layer_idx=0)
    assert block.attn.output_gate_activation == "swish"
    assert block.attn.o_norm.activation == "swish"


@pytest.mark.parametrize("gate", ["sigmoid", "silu"])
def test_gated_deltanet_output_gate_activation_wiring(gate):
    layer = GatedDeltaNet(hidden_size=128, head_dim=32, num_heads=2, expand_v=1, output_gate_activation=gate)
    assert layer.output_gate_activation == gate
    assert layer.o_norm.activation == gate
    cfg = _tiny_config(output_gate_activation=gate)
    block = GatedDeltaNetBlock(cfg, layer_idx=0)
    assert block.attn.output_gate_activation == gate
    assert block.attn.o_norm.activation == gate
    assert cfg.to_dict()["output_gate_activation"] == gate


def test_gated_deltanet_config_serialization():
    cfg = _tiny_config(output_gate_activation="sigmoid")
    d = cfg.to_dict()
    assert d["output_gate_activation"] == "sigmoid"
    cfg2 = GatedDeltaNetConfig.from_dict(d)
    assert cfg2.output_gate_activation == "sigmoid"
    block = GatedDeltaNetBlock(cfg2, layer_idx=0)
    assert block.attn.o_norm.activation == "sigmoid"
    # old checkpoint without field defaults to swish
    cfg3 = _tiny_config(output_gate_activation="sigmoid")
    d3 = cfg3.to_dict()
    d3.pop("output_gate_activation")
    restored = GatedDeltaNetConfig.from_dict(d3)
    assert restored.output_gate_activation == "swish"
    block = GatedDeltaNetBlock(restored, layer_idx=0)
    assert block.attn.output_gate_activation == "swish"
    assert block.attn.o_norm.activation == "swish"


def test_gated_deltanet_state_dict_compatibility():
    tiny_kwargs = dict(hidden_size=128, head_dim=32, num_heads=2, expand_v=1)
    base = GatedDeltaNet(**tiny_kwargs)
    swish = GatedDeltaNet(**tiny_kwargs, output_gate_activation="swish")
    sigmoid = GatedDeltaNet(**tiny_kwargs, output_gate_activation="sigmoid")
    # activation choice adds no parameter/state-dict key
    assert "output_gate_activation" not in base.state_dict()
    assert set(base.state_dict().keys()) == set(swish.state_dict().keys()) == set(sigmoid.state_dict().keys())
    assert sum(p.numel() for p in base.parameters()) == sum(
        p.numel() for p in swish.parameters()
    ) == sum(p.numel() for p in sigmoid.parameters())
    # strict load between variants
    sd = base.state_dict()
    swish.load_state_dict(sd, strict=True)
    sigmoid.load_state_dict(sd, strict=True)


def test_gated_deltanet_sigmoid_forward_backward():
    torch.manual_seed(42)

    layer = GatedDeltaNet(
        hidden_size=128,
        head_dim=32,
        num_heads=2,
        expand_v=1,
        output_gate_activation="sigmoid",
    ).to(device).train()

    x = torch.randn(2, 16, 128, device=device, requires_grad=True)
    y, _, _ = layer(x)

    assert torch.isfinite(y).all()

    y.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_gated_deltanet_validation_semantics():
    with pytest.raises(ValueError):
        GatedDeltaNetConfig(output_gate_activation="relu")
    with pytest.raises(ValueError):
        GatedDeltaNet(hidden_size=128, head_dim=32, num_heads=2, expand_v=1, output_gate_activation="relu")


def test_gated_deltanet_config_positional_compatibility():
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
