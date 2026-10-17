# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

from fla.layers.kda import KimiDeltaAttention
from fla.models import KDAConfig

from .utils import check_forward, check_generation


@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'attnres_block_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-bs{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, True,  None, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, None, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, 1,    torch.bfloat16),
            (4, 4, 1024, 4, 64, False, 4,    torch.bfloat16),
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
    dtype: torch.dtype,
    varlen: bool,
):
    check_forward(
        L,
        B,
        T,
        H,
        D,
        KDAConfig,
        use_l2warp=use_l2warp,
        attnres_block_size=attnres_block_size,
        dtype=dtype,
        varlen=varlen,
    )


@pytest.mark.usefixtures('triton_conv')
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
    check_generation(L, B, T, H, D, KDAConfig, dtype)


@pytest.mark.parametrize('safe_gate', [False, True])
def test_initialization(init_layout, make_model, safe_gate):
    torch.manual_seed(42)
    model = make_model(KDAConfig)
    with torch.device('meta'):
        layer = KimiDeltaAttention(hidden_size=64, head_dim=16, num_heads=4, use_short_conv=False, safe_gate=safe_gate)
    model._init_weights(layer)
    assert all(parameter.is_meta for parameter in layer.parameters())
    layer.to_empty(device='cpu')
    for name in ('A_log', 'dt_bias'):
        value = torch.full_like(getattr(layer, name), float('nan'))
        if init_layout is not None:
            mesh, placements = init_layout
            value = distribute_tensor(value, mesh, placements)
        setattr(layer, name, nn.Parameter(value))
    parameters = (layer.A_log, layer.dt_bias)
    model._init_weights(layer)
    values = []
    for parameter, actual in zip(parameters, (layer.A_log, layer.dt_bias), strict=True):
        assert actual is parameter
        if init_layout is not None:
            assert isinstance(actual, DTensor)
            assert actual.device_mesh == mesh
            assert actual.placements == placements
            actual = actual.to_local()
        assert torch.isfinite(actual).all()
        values.append(actual)
    if safe_gate:
        assert torch.count_nonzero(values[0]) == 0
    else:
        assert ((values[0].exp() >= 1) & (values[0].exp() <= 16)).all()
    assert layer.dt_bias._is_hf_initialized
    dt = F.softplus(values[1])
    assert ((dt >= 0.001) & (dt <= 0.1)).all()
    for parameter in parameters:
        parameter._is_hf_initialized = True
    expected = [parameter.detach().clone() for parameter in parameters]
    model._init_weights(layer)
    for actual, reference in zip(parameters, expected, strict=True):
        if init_layout is not None:
            actual, reference = actual.to_local(), reference.to_local()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
