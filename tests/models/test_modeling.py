# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import (
    ABCConfig,
    BitNetConfig,
    CombaConfig,
    DeltaFormerConfig,
    DeltaNetConfig,
    GSAConfig,
    HGRN2Config,
    HGRNConfig,
    LinearAttentionConfig,
    MesaNetConfig,
    MLAConfig,
    NSAConfig,
    RetNetConfig,
)
from fla.utils import device_name, find_spec_cached

from .utils import check_forward, check_generation

pytestmark = pytest.mark.skipif(device_name == 'cpu', reason='Model kernels require an accelerator.')

REQUIRES_FLASH_ATTN = pytest.mark.skipif(find_spec_cached('flash_attn') is None, reason='Attention requires flash-attn.')

MODEL_CONFIGS = [
    ABCConfig,
    pytest.param(BitNetConfig, marks=REQUIRES_FLASH_ATTN),
    CombaConfig,
    pytest.param(DeltaFormerConfig, marks=REQUIRES_FLASH_ATTN),
    DeltaNetConfig,
    GSAConfig,
    HGRNConfig,
    HGRN2Config,
    LinearAttentionConfig,
    MesaNetConfig,
    pytest.param(MLAConfig, marks=REQUIRES_FLASH_ATTN),
    pytest.param(NSAConfig, marks=REQUIRES_FLASH_ATTN),
    RetNetConfig,
]


@pytest.mark.parametrize('config_class', MODEL_CONFIGS, ids=lambda cls: cls.model_type)
@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.parametrize(
    ('L', 'B', 'T', 'H', 'D', 'use_l2warp', 'attnres_block_size', 'dtype'),
    [
        pytest.param(*case, id='L{}-B{}-T{}-H{}-D{}-l2{}-bs{}-{}'.format(*case))
        for case in [
            (4, 4, 1024, 4, 64, True, None, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, None, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, None, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, 1, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, 4, torch.bfloat16),
        ]
    ],
)
def test_forward(config_class, varlen, L, B, T, H, D, use_l2warp, attnres_block_size, dtype):
    check_forward(
        L=L,
        B=B,
        T=T,
        H=H,
        D=D,
        config_class=config_class,
        use_l2warp=use_l2warp,
        attnres_block_size=attnres_block_size,
        dtype=dtype,
        varlen=varlen,
    )


@pytest.mark.usefixtures('triton_conv')
@pytest.mark.parametrize('config_class', MODEL_CONFIGS, ids=lambda cls: cls.model_type)
def test_generation(config_class):
    check_generation(L=2, B=4, T=2000, H=8, D=64, config_class=config_class, dtype=torch.float16)
