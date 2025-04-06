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
@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("H", [8])
@pytest.mark.parametrize("D", [64])
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
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure"
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    config_class: AutoConfig,
    dtype: torch.dtype
):
    if config_class in [
        ABCConfig, BitNetConfig, LinearAttentionConfig, LightNetConfig,
        Mamba2Config, MambaConfig, SambaConfig, GatedDeltaProductConfig
    ]:
        pytest.skip()
    config = config_class(**{
        'hidden_size': int(H * D),
        'num_hidden_layers': L,
        **({'num_heads': H} if config_class != NSAConfig else {})
    })
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    model.to(dtype).to(device)

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T)).to(device)
    attention_mask = torch.ones((B, T), dtype=torch.bool).to(device)
    seq_start = torch.randint(low=0, high=T, size=(B,))
    attention_mask[torch.arange(T) < seq_start[:, None]] = False
    ref = model(input_ids=input_ids, attention_mask=attention_mask).logits

    num_chunks = 4
    chunk_size = T // num_chunks
    logits = []

    out = model(
        input_ids=input_ids[:, :chunk_size],
        attention_mask=attention_mask[:, :chunk_size],
        use_cache=True,
        past_key_values=None,
    )
    logits, past_key_values = [out.logits], out.past_key_values
    for i in range(1, num_chunks):
        start, end = i * chunk_size, (i + 1) * chunk_size
        for j in range(start, end):
            out = model(
                input_ids=input_ids[:, j:j+1],
                attention_mask=attention_mask[:, :j+1],
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits.append(out.logits)
            past_key_values = out.past_key_values
    gen = torch.cat(logits, dim=1)

    assert_close('logits', ref, gen, 1e-3)
