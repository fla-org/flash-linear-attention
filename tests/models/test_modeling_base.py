# -*- coding: utf-8 -*-

import pytest
import torch

from fla.utils import assert_close, device, is_intel_alchemist, is_nvidia_hopper

from .testing_utils import (
    GENERATION_UNSUPPORTED,
    HOPPER_EXCLUSIVE,
    MODELING_UNSUPPORTED_VAR_LEN,
    NOT_READY_FOR_TESTING,
    create_model_and_config
)


# ===================================================================================
# BASE TEST FOR MODELING (FORWARD/BACKWARD PASS)
# ===================================================================================
@pytest.mark.skipif(
    is_intel_alchemist,
    reason="Skipping test on Intel Alchemist due to known issues with SRAM."
)
def run_test_model_forward_backward(
    L: int, B: int, T: int, H: int, D: int,
    config_class: type, dtype: torch.dtype, use_l2warp: bool
):
    """
    A foundational test for the forward and backward passes of a model.
    """
    if not is_nvidia_hopper and D == 128:
        pytest.skip("D=128 is only tested on Hopper GPUs to save CI time.")
    if not is_nvidia_hopper and config_class.__name__ in HOPPER_EXCLUSIVE:
        pytest.skip(f"{config_class.__name__} requires Hopper-specific features.")
    if config_class.__name__ in NOT_READY_FOR_TESTING:
        pytest.skip(f"{config_class.__name__} is not yet ready for testing.")

    model, config = create_model_and_config(config_class, L, H, D, dtype, use_l2warp=use_l2warp)
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T), device=device)
    output_fixed = model(input_ids, output_hidden_states=True).hidden_states[-1]
    assert output_fixed.shape == (B, T, config.hidden_size)

    if config_class.__name__ in MODELING_UNSUPPORTED_VAR_LEN:
        pytest.skip(f"Variable length not supported for {config_class.__name__}.")

    cu_seqlens = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=device)
    output_var = model(
        input_ids.view(1, B * T), output_hidden_states=True, cu_seqlens=cu_seqlens
    ).hidden_states[-1]
    assert output_var.shape == (1, B * T, config.hidden_size)
    assert_close("output", output_fixed.view(1, B * T, -1), output_var, 1e-3)
    output_var.backward(torch.randn_like(output_var))


# ===================================================================================
# BASE TEST FOR GENERATION (K/V CACHE)
# ===================================================================================
@pytest.mark.skipif(is_nvidia_hopper is False, reason="Only run on Hopper GPUs")
def run_test_generation(
    L: int, B: int, T: int, H: int, D: int,
    config_class: type, dtype: torch.dtype
):
    """
    A foundational test for K/V cache-based generation.
    """
    torch.manual_seed(42)
    if config_class.__name__ in GENERATION_UNSUPPORTED:
        pytest.skip(f"Generation test not supported for {config_class.__name__}.")
    if config_class.__name__ in NOT_READY_FOR_TESTING:
        pytest.skip(f"{config_class.__name__} is not yet ready for testing.")

    model, config = create_model_and_config(config_class, L, H, D, dtype, use_l2warp=None)
    model.eval()

    num_chunks = 4
    chunk_size = T // num_chunks
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T), device=device)

    # Generate reference logits by running the model on the full sequence
    ref_logits = model(input_ids=input_ids, use_cache=False).logits

    # Generate logits autoregressively using K/V cache
    out = model(input_ids=input_ids[:, :chunk_size], use_cache=True)
    gen_logits_list = [out.logits]
    past_key_values = out.past_key_values
    for i in range(chunk_size, T):
        out = model(
            input_ids=input_ids[:, i:i+1],
            use_cache=True,
            past_key_values=past_key_values,
        )
        gen_logits_list.append(out.logits)
        past_key_values = out.past_key_values
    gen_logits = torch.cat(gen_logits_list, 1)

    # Correctly call assert_close with the positional argument 'ratio'
    assert_close('logits', ref_logits, gen_logits, 2e-3)
