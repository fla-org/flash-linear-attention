# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import YOCOConfig, YOCOForCausalLM

from .test_modeling_base import run_test_generation, run_test_model_forward_backward


def _create_yoco_config_kwargs(
    L: int,
    H: int,
    D: int,
    vocab_size: int = 1000,
    self_decoder_attn_type: str = 'gated_deltanet',
    self_window_size: int | None = None,
    attnres_block_size: int | None = None,
):
    self_decoder_attn = {
        'type': self_decoder_attn_type,
        'mode': 'chunk',
        'num_heads': H,
    }
    if self_decoder_attn_type == 'gated_deltanet':
        self_decoder_attn.update({
            'num_v_heads': H,
            'head_dim': D,
        })
    elif self_decoder_attn_type == 'swa':
        self_decoder_attn['window_size'] = self_window_size if self_window_size is not None else 64

    return {
        'num_hidden_layers': L,
        'num_self_decoder_layers': L // 2,
        'hidden_size': H * D,
        'self_decoder_attn': self_decoder_attn,
        'cross_decoder_attn': {
            'num_heads': H,
            'num_kv_heads': H,
        },
        'intermediate_size': 4 * H * D,
        'vocab_size': vocab_size,
        'fuse_norm': False,
        'fuse_swiglu': False,
        'fuse_cross_entropy': False,
        'attnres_block_size': attnres_block_size,
    }


def _create_yoco_config(
    L: int,
    H: int,
    D: int,
    use_l2warp: bool = False,
    vocab_size: int = 1000,
    self_decoder_attn_type: str = 'gated_deltanet',
    self_window_size: int | None = None,
    attnres_block_size: int | None = None,
):
    return YOCOConfig(
        **_create_yoco_config_kwargs(
            L,
            H,
            D,
            vocab_size=vocab_size,
            self_decoder_attn_type=self_decoder_attn_type,
            self_window_size=self_window_size,
            attnres_block_size=attnres_block_size,
        ),
        use_l2warp=use_l2warp,
    )


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'self_decoder_attn_type', 'use_l2warp', 'attnres_block_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-attn{}-use_l2warp{}-bs{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, 'gated_deltanet', True,  None, torch.bfloat16),
            (4, 4, 1024, 4, 64, 'gated_deltanet', False, None, torch.bfloat16),
            (4, 4, 1024, 4, 64, 'gated_deltanet', False, 1,    torch.bfloat16),
            (4, 4, 1024, 4, 64, 'gated_deltanet', False, 4,    torch.bfloat16),
            (4, 2, 32, 4, 32, 'gated_retention', False, None, torch.bfloat16),
            (4, 2, 96, 4, 32, 'gated_retention', False, None, torch.bfloat16),
            (4, 2, 128, 4, 32, 'swa', False, None, torch.bfloat16),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    self_decoder_attn_type: str,
    use_l2warp: bool,
    attnres_block_size: int | None,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    run_test_model_forward_backward(
        L,
        B,
        T,
        H,
        D,
        YOCOConfig,
        use_l2warp=use_l2warp,
        dtype=dtype,
        **_create_yoco_config_kwargs(
            L,
            H,
            D,
            self_decoder_attn_type=self_decoder_attn_type,
            attnres_block_size=attnres_block_size,
        ),
    )


# ===================================================================================
# Test for Generation
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'self_decoder_attn_type', 'dtype', 'tol'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-attn{}-{}".format(*test))
        for test in [
            (4, 4, 2000, 8, 64, 'gated_deltanet', torch.float16, 3e-3),
            (4, 2, 256, 4, 32, 'gated_retention', torch.float16, 3e-3),
            (4, 2, 256, 4, 32, 'swa', torch.float16, 3e-3),
        ]
    ],
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    self_decoder_attn_type: str,
    dtype: torch.dtype,
    tol: float,
):
    config = _create_yoco_config(
        L,
        H,
        D,
        vocab_size=128,
        self_decoder_attn_type=self_decoder_attn_type,
    )
    model = YOCOForCausalLM(config)
    run_test_generation(L, B, T, H, D, YOCOConfig, dtype, model=model, config=config, tol=tol)
