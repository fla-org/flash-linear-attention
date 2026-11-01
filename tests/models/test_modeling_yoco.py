# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import YOCOConfig, YOCOForCausalLM
from fla.models.cache_utils import Cache
from fla.utils import assert_close, device

from .utils import check_forward, check_generation

try:
    import flash_attn  # noqa: F401
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


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


@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
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
    varlen: bool,
):
    if self_decoder_attn_type == 'swa' and not HAS_FLASH_ATTN:
        pytest.skip(reason="YOCO swa attention requires flash-attn (`pip install flash-attn --no-build-isolation`).")
    torch.manual_seed(42)
    check_forward(
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
        varlen=varlen,
    )


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
    if self_decoder_attn_type == 'swa' and not HAS_FLASH_ATTN:
        pytest.skip(reason="YOCO swa attention requires flash-attn (`pip install flash-attn --no-build-isolation`).")
    config = _create_yoco_config(
        L,
        H,
        D,
        vocab_size=128,
        self_decoder_attn_type=self_decoder_attn_type,
    )
    model = YOCOForCausalLM(config)
    check_generation(L, B, T, H, D, YOCOConfig, dtype, model=model, config=config, tol=tol)


@torch.no_grad()
def test_packed_with_mask_matches_per_document() -> None:
    torch.manual_seed(42)
    config = _create_yoco_config(2, 2, 16, vocab_size=128)
    model = YOCOForCausalLM(config).to(device=device, dtype=torch.float16).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 12), device=device)
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device=device, dtype=torch.long)
    cu_seqlens = torch.tensor([0, 5, 12], device=device, dtype=torch.int32)

    expected = torch.cat([
        model(input_ids=input_ids[:, start:end], use_cache=False).logits
        for start, end in ((0, 5), (5, 12))
    ], dim=1)
    actual = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        cu_seqlens=cu_seqlens,
    ).logits

    assert_close("logits", expected, actual, 0.005)


@torch.no_grad()
def test_swa_packed_with_mask_empty_cache_matches_per_document() -> None:
    if not HAS_FLASH_ATTN:
        pytest.skip(reason="YOCO swa attention requires flash-attn (`pip install flash-attn --no-build-isolation`).")

    torch.manual_seed(42)
    config = _create_yoco_config(2, 2, 16, vocab_size=128, self_decoder_attn_type='swa')
    config.num_self_decoder_layers = config.num_hidden_layers
    model = YOCOForCausalLM(config).to(device=device, dtype=torch.float16).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 12), device=device)
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device=device, dtype=torch.long)
    cu_seqlens = torch.tensor([0, 5, 12], device=device, dtype=torch.int32)

    expected = torch.cat([
        model(input_ids=input_ids[:, start:end], past_key_values=Cache(), use_cache=True).logits
        for start, end in ((0, 5), (5, 12))
    ], dim=1)
    actual = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=Cache(),
        use_cache=True,
        cu_seqlens=cu_seqlens,
    ).logits

    assert_close("logits", expected, actual, 0.005)
