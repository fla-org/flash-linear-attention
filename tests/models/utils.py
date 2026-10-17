# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math
from contextlib import contextmanager

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM, PretrainedConfig

from fla.utils import IS_INTEL_ALCHEMIST, IS_NVIDIA_HOPPER, assert_close, device

MODELING_UNSUPPORTED_VARLEN = [
    "ABCConfig", "CATConfig", "ForgettingTransformerConfig", "LinearAttentionConfig",
    "Mamba2Config", "MambaConfig", "MesaNetConfig", "SambaConfig",
    "RodimusConfig", "WallTransformerConfig",
]

NOT_READY_FOR_TESTING = ['RodimusConfig']

HOPPER_EXCLUSIVE = []

GENERATION_UNSUPPORTED = [
    "ABCConfig",
    "CATConfig",
    "DeltaFormerConfig",
    "MoBAConfig",
    "WallTransformerConfig",
]


def create_model_and_config(config_class, L, H, D, dtype, **kwargs):
    config_params = {
        'hidden_size': H * D,
        'num_hidden_layers': L,
        **({'num_heads': H} if config_class.__name__ != 'NSAConfig' else {}),
        **kwargs,
    }
    config = config_class(**config_params)
    model = AutoModelForCausalLM.from_config(config)
    model.apply(_init_weights)
    model.to(dtype).to(device)
    return model, config


def _init_weights(module):
    """Use asymmetric weights to expose numerical-path differences."""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(module.weight, a=math.sqrt(5))
        with torch.no_grad():
            shape = module.weight.shape
            if len(shape) > 1:
                quarter_size = shape[0] // 4
                module.weight[:quarter_size] *= 1.2
                module.weight[-quarter_size:] *= 0.8
                if shape[0] == shape[1]:
                    idx = torch.arange(min(shape[0], shape[1]))
                    module.weight[idx, idx] += 0.05
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)
            with torch.no_grad():
                module.bias[::3] *= 1.1
                module.bias[1::3] *= 0.9
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            vocab_size, dim = module.weight.shape
            pattern = 0.01 * torch.sin(torch.arange(dim) * (6.28 / dim))
            for i in range(min(100, vocab_size)):
                module.weight[i] += pattern * (1 + i % 5) * 0.2


def check_forward(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    config_class: type,
    use_l2warp: bool,
    dtype: torch.dtype,
    varlen: bool = True,
    **kwargs,
):
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST:
        pytest.skip('Model tests require more SRAM than Intel Alchemist provides.')
    if varlen and config_class.__name__ in MODELING_UNSUPPORTED_VARLEN:
        pytest.skip(f'Variable length not supported for {config_class.__name__}.')
    if not IS_NVIDIA_HOPPER and D == 128:
        pytest.skip("D=128 is only tested on Hopper GPUs to save CI time.")
    if not IS_NVIDIA_HOPPER and config_class.__name__ in HOPPER_EXCLUSIVE:
        pytest.skip(f"{config_class.__name__} requires Hopper-specific features.")
    if config_class.__name__ in NOT_READY_FOR_TESTING:
        pytest.skip(f"{config_class.__name__} is not yet ready for testing.")

    model, config = create_model_and_config(config_class, L, H, D, use_l2warp=use_l2warp, dtype=dtype, **kwargs)
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T), device=device)
    outputs = model(input_ids, labels=input_ids if not varlen else None, output_hidden_states=True)
    output_fixed = outputs.hidden_states[-1]
    assert output_fixed.shape == (B, T, config.hidden_size)
    if not varlen:
        assert torch.isfinite(outputs.loss)
        outputs.loss.backward()
        return

    cu_seqlens = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=device)
    output_var = model(
        input_ids.view(1, B * T), output_hidden_states=True, cu_seqlens=cu_seqlens,
    ).hidden_states[-1]
    assert output_var.shape == (1, B * T, config.hidden_size)
    assert_close("output", output_fixed.view(1, B * T, -1), output_var, 1e-3)
    output_var.backward(torch.randn_like(output_var))


@contextmanager
def _triton_conv():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv('FLA_CONV_BACKEND', 'triton')
        yield


@torch.no_grad()
@_triton_conv()
def check_generation(
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


@_triton_conv()
def check_generate(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    config_class: type,
    dtype: torch.dtype,
    num_new_tokens: int = 8,
):
    """Check that generation prefills the full prompt."""
    torch.manual_seed(42)
    if config_class.__name__ in GENERATION_UNSUPPORTED:
        pytest.skip(f"Generation test not supported for {config_class.__name__}.")
    if config_class.__name__ in NOT_READY_FOR_TESTING:
        pytest.skip(f"{config_class.__name__} is not yet ready for testing.")

    model, config = create_model_and_config(config_class, L, H, D, dtype=dtype)
    model.eval()
    model = model.to(dtype).to(device)
    # avoid the pad/eos id so attention_mask is all ones and tokens are unambiguous
    input_ids = torch.randint(low=1, high=config.vocab_size, size=(B, T)).to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)
        manual = [next_token]
        for _ in range(num_new_tokens - 1):
            out = model(input_ids=next_token, use_cache=True, past_key_values=past_key_values)
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1:].argmax(dim=-1)
            manual.append(next_token)
        manual = torch.cat(manual, dim=1)

        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=num_new_tokens,
            do_sample=False,
        )[:, T:]

    assert torch.equal(generated, manual), (
        "generate() output diverges from a manual greedy decode that prefills the full prompt; "
        "prepare_inputs_for_generation likely dropped the prompt on the first step.\n"
        f"generate={generated.tolist()}\nmanual={manual.tolist()}"
    )
