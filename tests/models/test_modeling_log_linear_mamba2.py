# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch

import fla.layers.log_linear_mamba2 as log_linear_mamba2
from fla.models import LogLinearMamba2Config, LogLinearMamba2ForCausalLM
from fla.utils import device


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'attnres_block_size', 'dtype', 'conv_backend'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-bs{}-{}-conv-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64,  None, torch.bfloat16, 'cuda'),
            (4, 4, 1024, 4, 64,  None, torch.bfloat16, 'triton'),
            (4, 4, 1024, 4, 128, None, torch.bfloat16, 'cuda'),
            (4, 4, 1024, 4, 64,  1,    torch.bfloat16, 'cuda'),
            (4, 4, 1024, 4, 64,  4,    torch.bfloat16, 'cuda'),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    attnres_block_size: int | None,
    dtype: torch.dtype,
    conv_backend: str,
):
    """
    Test the forward and backward pass of the Mamba2 model by manually
    instantiating the configuration and the model.
    """
    os.environ['FLA_CONV_BACKEND'] = conv_backend

    # Manually create a consistent configuration
    # The key relationship is: num_heads = expand * hidden_size / head_dim
    # To ensure consistency, we derive hidden_size from other parameters.
    expand = 2
    hidden_size = H * D // expand

    config = LogLinearMamba2Config(
        num_hidden_layers=L,
        hidden_size=hidden_size,
        expand=expand,
        num_heads=H,
        head_dim=D,
        attnres_block_size=attnres_block_size,
        vocab_size=1000,  # dummy vocab size
    )

    model = LogLinearMamba2ForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()

    # Create random input tensor
    x = torch.randint(0, config.vocab_size, (B, T), device=device)

    # Forward pass
    y = model(x)

    # Assert output shape is correct
    assert y.logits.shape == (B, T, config.vocab_size)

    # Backward pass
    y.logits.sum().backward()
    print(f"Test test_modeling passed with H={H}, D={D}, backend={conv_backend}.")


def test_fused_conv_padding_mask(monkeypatch):
    batch_size, seq_len, num_heads, head_dim, state_size, num_levels = 1, 3, 1, 4, 2, 3
    dim = num_heads * head_dim
    conv_dim = dim + 2 * state_size
    projected_size = 2 * dim + 2 * state_size + num_heads * (num_levels + 1)
    observed = {}

    def conv(*, x, **kwargs):
        observed['conv_input'] = x
        return x + 1, None

    def scan(**kwargs):
        observed['scan_input'] = kwargs['x']
        return kwargs['x'], None

    monkeypatch.setattr(log_linear_mamba2, 'hmamba_chunk_scan_combined', scan)
    output = log_linear_mamba2.hmamba_split_conv1d_scan_combined(
        zxbcdtdl=torch.ones(batch_size, seq_len, projected_size),
        conv1d_weight=torch.ones(conv_dim, 4),
        conv1d_bias=torch.ones(conv_dim),
        dt_bias=torch.ones(num_heads),
        A=-torch.ones(num_heads),
        L=torch.ones(num_heads, num_levels),
        D=torch.ones(num_heads),
        chunk_size=64,
        outproj_weight=torch.eye(dim),
        headdim=head_dim,
        conv1d_fn=conv,
        conv_backend='triton',
        attention_mask=torch.tensor([[0, 1, 1]]),
    )

    assert torch.count_nonzero(observed['conv_input'][:, 0]) == 0
    assert torch.count_nonzero(observed['scan_input'][:, 0]) == 0
    assert output.shape == (batch_size, seq_len, dim)


def test_conv_backend_padding_and_cached_decode_parity(monkeypatch):
    torch.manual_seed(42)
    config = LogLinearMamba2Config(
        num_hidden_layers=1,
        hidden_size=32,
        expand=2,
        num_heads=1,
        head_dim=64,
        state_size=64,
        use_bias=True,
        vocab_size=128,
    )

    cuda_available = (
        log_linear_mamba2.causal_conv1d_fn is not None and log_linear_mamba2.causal_conv1d_update is not None
    )
    backends = ['cuda', 'triton'] if cuda_available else ['triton']
    models = []
    for backend in backends:
        monkeypatch.setenv('FLA_CONV_BACKEND', backend)
        model = LogLinearMamba2ForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()
        if models:
            model.load_state_dict(models[0].state_dict())
        models.append(model)
    assert [model.backbone.layers[0].mixer.backend for model in models] == backends

    input_ids = torch.randint(0, config.vocab_size, (2, 74), device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[0, :2] = False
    attention_mask[1, :1] = False
    prefill_len = 70

    def cached_logits(model):
        output = model(
            input_ids=input_ids[:, :prefill_len],
            attention_mask=attention_mask[:, :prefill_len],
            use_cache=True,
        )
        logits = [output.logits]
        for index in range(prefill_len, input_ids.shape[1]):
            output = model(
                input_ids=input_ids[:, index:index + 1],
                attention_mask=attention_mask[:, :index + 1],
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            logits.append(output.logits)
        return torch.cat(logits, dim=1)

    with torch.no_grad():
        full = [model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits for model in models]
        cached = [cached_logits(model) for model in models]

    if cuda_available:
        torch.testing.assert_close(full[0], full[1], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(cached[0], cached[1], rtol=2e-2, atol=2e-2)
    for expected, actual in zip(full, cached):
        torch.testing.assert_close(expected[attention_mask], actual[attention_mask], rtol=2e-2, atol=2e-2)
