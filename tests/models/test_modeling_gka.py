# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import GKAConfig, GKAForCausalLM

from .test_modeling_base import run_test_generation, run_test_model_forward_backward


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'attnres_block_size', 'num_v_heads', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-bs{}-hv{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64,  True,  None, None, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, None, None, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, None, None, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 1,    None, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 4,    None, torch.bfloat16),
            (4, 4, 1024, 2, 64,  False, None, 4,    torch.bfloat16),
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
    attnres_block_size: int | None,
    num_v_heads: int | None,
    dtype: torch.dtype,
):
    run_test_model_forward_backward(
        L,
        B,
        T,
        H,
        D,
        GKAConfig,
        use_l2warp=use_l2warp,
        attnres_block_size=attnres_block_size,
        head_dim=D,
        num_v_heads=num_v_heads,
        dtype=dtype,
    )


# ===================================================================================
# Test for Generation
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 4, 2000, 8, 64, torch.float16),
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
    # chunked prefill and token-by-token decoding round differently, and the ridge solve amplifies it at random init,
    # so the default 2e-3 is too tight
    run_test_generation(L, B, T, H, D, GKAConfig, dtype, tol=0.03)


def test_tied_embeddings(tmp_path):
    config = GKAConfig(hidden_size=64, num_hidden_layers=2, num_heads=2, head_dim=32, vocab_size=128,
                       tie_word_embeddings=True)
    model = GKAForCausalLM(config)
    assert model.lm_head.weight.data_ptr() == model.model.embeddings.weight.data_ptr()

    model.save_pretrained(tmp_path)
    reloaded = GKAForCausalLM.from_pretrained(tmp_path)
    assert reloaded.lm_head.weight.data_ptr() == reloaded.model.embeddings.weight.data_ptr()
    assert torch.equal(reloaded.model.embeddings.weight, model.model.embeddings.weight)


@pytest.mark.parametrize('use_forgetting_gate', [True, False])
def test_init_from_meta_device(use_forgetting_gate: bool):
    config = GKAConfig(hidden_size=64, num_hidden_layers=2, num_heads=2, head_dim=32, vocab_size=128,
                       use_forgetting_gate=use_forgetting_gate)
    with torch.device('meta'):
        model = GKAForCausalLM(config)
    model.to_empty(device='cpu')
    model.apply(model._init_weights)

    for layer in model.model.layers:
        attn = layer.attn
        if not use_forgetting_gate:
            assert not hasattr(attn, 'A_log') and not hasattr(attn, 'dt_bias')
            continue
        A, dt = attn.A_log.exp(), torch.nn.functional.softplus(attn.dt_bias)
        assert torch.isfinite(attn.A_log).all() and torch.isfinite(attn.dt_bias).all()
        assert ((A > 0) & (A <= 16)).all()
        assert ((dt >= 1e-4 * (1 - 1e-3)) & (dt <= 0.1 * (1 + 1e-3))).all()
        assert attn.A_log._no_weight_decay and attn.dt_bias._no_weight_decay
