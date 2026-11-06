# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import MLSTMConfig, MLSTMForCausalLM
from fla.utils import device

from .utils import check_forward, check_generate, check_generation


@pytest.mark.parametrize(
    ["L", "B", "T", "H", "D", "use_l2warp", "dtype", "varlen"],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-use_l2warp{}-{}-varlen{}".format(*test))
        for test in [
            (2, 2, 127, 2, 32, True, torch.bfloat16, False),
            (2, 2, 257, 2, 32, False, torch.bfloat16, True),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    use_l2warp: bool,
    dtype: torch.dtype,
    varlen: bool,
):
    check_forward(
        L=L,
        B=B,
        T=T,
        H=H,
        D=D,
        config_class=MLSTMConfig,
        use_l2warp=use_l2warp,
        dtype=dtype,
        varlen=varlen,
        vocab_size=256,
    )


@pytest.mark.parametrize(
    ["L", "B", "T", "H", "D", "dtype"],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 2, 132, 2, 32, torch.float16),
        ]
    ],
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    check_generation(L=L, B=B, T=T, H=H, D=D, config_class=MLSTMConfig, dtype=dtype)
    check_generate(L=L, B=B, T=T, H=H, D=D, config_class=MLSTMConfig, dtype=dtype)


@pytest.mark.parametrize('num_hidden_layers', [1, 2], ids=['single-layer', 'multi-layer'])
def test_initialization(num_hidden_layers: int):
    torch.manual_seed(42)
    config = MLSTMConfig(
        hidden_size=64,
        num_hidden_layers=num_hidden_layers,
        num_heads=2,
        proj_factor=1,
        hidden_ratio=1,
        vocab_size=32,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = MLSTMForCausalLM(config).to(device)
    for block in model.model.layers:
        layer = block.attn
        assert torch.count_nonzero(layer.fgate.weight) == 0
        assert torch.count_nonzero(layer.igate.weight) == 0
        assert torch.equal(layer.learnable_skip, torch.ones_like(layer.learnable_skip))
        assert layer.fgate.bias.requires_grad
        expected_bias = torch.linspace(3, 6, config.num_heads, device=device)
        assert torch.equal(layer.fgate.bias, expected_bias)
    input_ids = torch.randint(config.vocab_size, (1, 7), device=device)
    model(input_ids=input_ids, labels=input_ids).loss.backward()
    for block in model.model.layers:
        assert block.attn.fgate.bias.grad is not None
        assert torch.isfinite(block.attn.fgate.bias.grad).all()


@pytest.mark.parametrize('tie_word_embeddings', [False, True], ids=['untied', 'tied'])
def test_save_load(tmp_path, tie_word_embeddings: bool):
    torch.manual_seed(42)
    config = MLSTMConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_heads=2,
        proj_factor=1,
        vocab_size=32,
        tie_word_embeddings=tie_word_embeddings,
    )
    model = MLSTMForCausalLM(config)
    model.save_pretrained(tmp_path)
    restored = MLSTMForCausalLM.from_pretrained(tmp_path)
    assert restored.state_dict().keys() == model.state_dict().keys()
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(weight, restored.state_dict()[name], rtol=0, atol=0)
    assert (restored.lm_head.weight is restored.model.embeddings.weight) == tie_word_embeddings
