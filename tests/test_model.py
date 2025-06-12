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
    MesaNetConfig,
    NSAConfig,
    PaTHAttentionConfig,
    RetNetConfig,
    RodimusConfig,
    RWKV6Config,
    RWKV7Config,
    SambaConfig,
    TransformerConfig
)
from fla.utils import assert_close, device, is_intel_alchemist, is_nvidia_hopper


@pytest.mark.parametrize("L", [4])
@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [2048])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("D", [64, 128])
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
    MesaNetConfig,
    NSAConfig,
    PaTHAttentionConfig,
    RetNetConfig,
    RodimusConfig,
    RWKV6Config,
    RWKV7Config,
    SambaConfig,
    TransformerConfig
])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_l2warp", [True, False])
@pytest.mark.skipif(
    is_intel_alchemist is True,
    reason="Skipping test on Intel Alchemist due to known issues with SRAM."
)
def test_model(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    config_class: AutoConfig,
    dtype: torch.dtype,
    use_l2warp: bool
):
    if not is_nvidia_hopper and D == 128 or config_class in [GatedDeltaNetConfig]:
        pytest.skip("D=128 is only Tested on Hopper GPUs")
    if config_class in [
        ABCConfig, ForgettingTransformerConfig, LinearAttentionConfig, LightNetConfig,
        Mamba2Config, MambaConfig, MesaNetConfig, SambaConfig, GatedDeltaProductConfig,
        RodimusConfig,
    ]:
        pytest.skip("Variable length not supported yet")
    if config_class in [PaTHAttentionConfig]:
        pytest.skip("PaTHAttentionConfig does not adopted for testing yet")
    config = config_class(**{
        'hidden_size': int(H * D),
        'num_hidden_layers': L,
        **({'num_heads': H} if config_class != NSAConfig else {})
    })
    config.use_l2warp = use_l2warp
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
    # Test backward pass
    # Do nothing, just to ensure no errors occur during backward pass
    output_var.sum().backward()
