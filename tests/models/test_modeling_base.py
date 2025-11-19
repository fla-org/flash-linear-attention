# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

from fla.utils import IS_INTEL_ALCHEMIST, IS_NVIDIA_HOPPER, assert_close, device

from .test_modeling_utils import (
    GENERATION_UNSUPPORTED,
    HOPPER_EXCLUSIVE,
    MODELING_UNSUPPORTED_VARLEN,
    NOT_READY_FOR_TESTING,
    create_model_and_config,
)


# ===================================================================================
# BASE TEST FOR MODELING (FORWARD/BACKWARD PASS)
# ===================================================================================
@pytest.mark.skipif(
    IS_INTEL_ALCHEMIST,
    reason="Skipping test on Intel Alchemist due to known issues with SRAM.",
)
def run_test_model_forward_backward(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    config_class: type,
    use_l2warp: bool,
    dtype: torch.dtype,
):
    """
    A foundational test for the forward and backward passes of a model.
    """
    if not IS_NVIDIA_HOPPER and D == 128:
        pytest.skip("D=128 is only tested on Hopper GPUs to save CI time.")
    if not IS_NVIDIA_HOPPER and config_class.__name__ in HOPPER_EXCLUSIVE:
        pytest.skip(f"{config_class.__name__} requires Hopper-specific features.")
    if config_class.__name__ in NOT_READY_FOR_TESTING:
        pytest.skip(f"{config_class.__name__} is not yet ready for testing.")

    model, config = create_model_and_config(config_class, L, H, D, use_l2warp=use_l2warp, dtype=dtype)
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T), device=device)
    output_fixed = model(input_ids, output_hidden_states=True).hidden_states[-1]
    assert output_fixed.shape == (B, T, config.hidden_size)

    if config_class.__name__ in MODELING_UNSUPPORTED_VARLEN:
        pytest.skip(f"Variable length not supported for {config_class.__name__}.")

    cu_seqlens = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=device)
    output_var = model(
        input_ids.view(1, B * T), output_hidden_states=True, cu_seqlens=cu_seqlens,
    ).hidden_states[-1]
    assert output_var.shape == (1, B * T, config.hidden_size)
    assert_close("output", output_fixed.view(1, B * T, -1), output_var, 1e-3)
    output_var.backward(torch.randn_like(output_var))


# ===================================================================================
# BASE TEST FOR GENERATION (K/V CACHE)
# ===================================================================================
def run_test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    config_class: type,
    dtype: torch.dtype,
    use_l2warp: bool = False,
    model: torch.nn.Module | None = None,
    config: PretrainedConfig | None = None,
    tol: float = 2e-3,
):
    """
    A foundational test for K/V cache-based generation.
    """
    torch.manual_seed(42)
    if config_class.__name__ in GENERATION_UNSUPPORTED:
        pytest.skip(f"Generation test not supported for {config_class.__name__}.")
    if config_class.__name__ in NOT_READY_FOR_TESTING:
        pytest.skip(f"{config_class.__name__} is not yet ready for testing.")

    if model is None:
        model, config = create_model_and_config(config_class, L, H, D, use_l2warp=use_l2warp, dtype=dtype)
    model.eval()
    model = model.to(dtype).to(device)

    num_chunks = 4
    chunk_size = T // num_chunks
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T)).to(device)
    attention_mask = torch.ones((B, T), dtype=torch.bool).to(device)
    seq_start = torch.randint(low=1, high=chunk_size - 1, size=(B,))
    attention_mask[torch.arange(T) < seq_start[:, None]] = False
    ref = torch.cat([
        model(input_ids=input_ids[i:i+1, start:], use_cache=False).logits
        for i, start in enumerate(seq_start)
    ], dim=1)

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
    gen = torch.cat(logits, 1)
    gen = torch.cat([gen[i:i+1, start:] for i, start in enumerate(seq_start)], 1)
    assert_close('logits', ref, gen, tol)
