# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import LoLCATsConfig, LoLCATsForCausalLM
from fla.utils import assert_close, device

from .utils import check_forward, check_generation, create_model_and_config


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'attnres_block_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-bs{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, None, torch.bfloat16),
            (4, 4, 1024, 4, 64, 4, torch.bfloat16),
        ]
    ],
)
def test_modeling(L: int, B: int, T: int, H: int, D: int, attnres_block_size: int | None, dtype: torch.dtype):
    check_forward(
        L, B, T, H, D, LoLCATsConfig, use_l2warp=False, attnres_block_size=attnres_block_size, dtype=dtype, varlen=False,
        feature_dim=16,
    )


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 4, 400, 4, 64, torch.float32),
        ]
    ],
)
def test_generation(L: int, B: int, T: int, H: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    model, config = create_model_and_config(LoLCATsConfig, L, H, D, dtype, num_kv_heads=2, feature_dim=16, window_size=32)
    check_generation(L, B, T, H, D, LoLCATsConfig, dtype, model=model, config=config)


def test_load_llama():
    """With a window covering the whole input, a model holding the weights of Llama reproduces it exactly."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(42)
    llama = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, rms_norm_eps=1e-6,
    )).to(device).eval()
    model = LoLCATsForCausalLM(LoLCATsConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_heads=4, num_kv_heads=2,
        window_size=64, feature_dim=16, rope_theta=llama.config.rope_parameters['rope_theta'], fuse_cross_entropy=False,
    )).to(device).eval()
    renames = [('embed_tokens', 'embeddings'), ('self_attn', 'attn'), ('input_layernorm', 'attn_norm'),
               ('post_attention_layernorm', 'mlp_norm')]
    state_dict = {}
    for name, value in llama.state_dict().items():
        for old, new in renames:
            name = name.replace(old, new)
        state_dict[name] = value
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not unexpected
    assert all('feature_map' in name or 'window_factors' in name for name in missing)

    input_ids = torch.randint(0, 128, (2, 50), device=device)
    with torch.no_grad():
        assert_close('logits', llama(input_ids).logits, model(input_ids).logits, 1e-4)
