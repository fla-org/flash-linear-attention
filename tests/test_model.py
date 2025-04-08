# -*- coding: utf-8 -*-

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from fla.models import (
    ABCConfig,
    BitNetConfig,
    DeltaNetConfig,
    ForgettingTransformerConfig,
    GatedDeltaNetConfig,
    GatedDeltaProductConfig,
    GLAConfig,
    GSAConfig,
    HGRN2Config,
    HGRNConfig,
    LightNetConfig,
    LinearAttentionConfig,
    Mamba2Config,
    MambaConfig,
    NSAConfig,
    RetNetConfig,
    RWKV6Config,
    RWKV7Config,
    SambaConfig,
    TransformerConfig
)
from fla.ops.utils.testing import assert_close
from fla.utils import device, device_platform


@pytest.mark.parametrize("L", [4])
@pytest.mark.parametrize("B", [8])
@pytest.mark.parametrize("T", [2048])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("config_class", [
    ABCConfig,
    BitNetConfig,
    DeltaNetConfig,
    ForgettingTransformerConfig,
    GatedDeltaNetConfig,
    GatedDeltaProductConfig,
    GLAConfig,
    GSAConfig,
    HGRN2Config,
    HGRNConfig,
    LightNetConfig,
    LinearAttentionConfig,
    Mamba2Config,
    MambaConfig,
    NSAConfig,
    RetNetConfig,
    RWKV6Config,
    RWKV7Config,
    SambaConfig,
    TransformerConfig
])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure"
)
def test_model(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    config_class: AutoConfig,
    dtype: torch.dtype
):
    if config_class in [
        ABCConfig, LinearAttentionConfig, LightNetConfig,
        Mamba2Config, MambaConfig, SambaConfig, GatedDeltaProductConfig
    ]:
        pytest.skip("Variable length not supported yet")
    config = config_class(**{
        'hidden_size': int(H * D),
        'num_hidden_layers': L,
        **({'num_heads': H} if config_class != NSAConfig else {})
    })
    model = AutoModelForCausalLM.from_config(config)
    model.to(dtype).to(device)

    cu_seqlens = torch.cat([
        torch.arange(0, B * T, T),
        torch.tensor([B * T], dtype=torch.long)
    ], 0).to(device).to(torch.int32)

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(1, B * T)).to(device)
    output = model(input_ids.view(B, T), output_hidden_states=True).hidden_states[-1]
    assert output.shape == (B, T, config.hidden_size)

    output_var = model(input_ids, output_hidden_states=True, cu_seqlens=cu_seqlens).hidden_states[-1]
    assert output_var.shape == (1, B * T, config.hidden_size)
    assert_close('output', output.view(1, B * T, -1), output_var, ratio=1e-3)
