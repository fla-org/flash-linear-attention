# -*- coding: utf-8 -*-

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from fla.models import TransformerConfig
from fla.ops.utils.testing import assert_close
from fla.utils import device, device_platform


@pytest.mark.parametrize("L", [4])
@pytest.mark.parametrize("N", [8])
@pytest.mark.parametrize("B", [8])
@pytest.mark.parametrize("T", [2048])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("config", [
    TransformerConfig
])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure"
)
def test_model(
    L: int,
    N: int,
    B: int,
    T: int,
    H: int,
    D: int,
    config: AutoConfig,
    dtype: torch.dtype
):
    config = config(
        hidden_size=int(H * D),
        num_hidden_layers=L,
        num_heads=H,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.to(dtype).to(device)

    N = min(1, N) if T < 64 else N
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(0, B * T, T),
        torch.tensor([B * T], dtype=torch.long)
    ], 0).to(device).to(torch.int32)

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(1, B * T)).to(device)
    output = model(input_ids.view(B, T), output_hidden_states=True).hidden_states[-1]
    assert output.shape == (B, T, config.hidden_size)

    output_var = model(input_ids, output_hidden_states=True, cu_seqlens=cu_seqlens).hidden_states[-1]
    assert output_var.shape == (1, B * T, config.hidden_size)
    assert_close('output', output.view(1, B * T, -1), output_var, ratio=1e-3)
