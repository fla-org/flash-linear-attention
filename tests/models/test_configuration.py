# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import inspect
import warnings

import pytest

from fla import models

CONFIG_CLASSES = [getattr(models, name) for name in models.__all__ if name.endswith('Config')]
LOSS_CONFIG_CLASSES = [
    cls for cls in CONFIG_CLASSES
    if 'fuse_linear_cross_entropy' in inspect.signature(cls).parameters or cls is models.LogLinearMamba2Config
]
ATTNRES_CONFIG_CLASSES = [cls for cls in CONFIG_CLASSES if 'attnres_block_size' in inspect.signature(cls).parameters]


@pytest.mark.parametrize('config_class', LOSS_CONFIG_CLASSES, ids=lambda cls: cls.model_type)
@pytest.mark.parametrize(('fuse_cross_entropy', 'fuse_linear_cross_entropy'), [(False, False), (True, False), (False, True), (True, True)])
def test_loss(config_class, fuse_cross_entropy, fuse_linear_cross_entropy):
    kwargs = {
        'fuse_cross_entropy': fuse_cross_entropy,
        'fuse_linear_cross_entropy': fuse_linear_cross_entropy,
        'downstream_config': {'enabled': True},
    }
    if fuse_cross_entropy and fuse_linear_cross_entropy:
        with pytest.raises(ValueError, match='cannot.*True'):
            config_class(**kwargs)
        return

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        config = config_class(**kwargs)
        restored = config_class.from_dict(config.to_dict())
    loss_warnings = [record for record in records if 'reduced precision' in str(record.message)]
    assert bool(loss_warnings) == (fuse_linear_cross_entropy and config_class is not models.StickBreakingAttentionConfig)
    assert restored.to_dict() == config.to_dict()
    for name, value in kwargs.items():
        assert getattr(restored, name) == value


@pytest.mark.parametrize('config_class', ATTNRES_CONFIG_CLASSES, ids=lambda cls: cls.model_type)
@pytest.mark.parametrize('block_size', [None, 1, 2, 4, 0, -2, 3])
def test_attnres(config_class, block_size):
    if block_size in (0, -2, 3):
        with pytest.raises(ValueError, match='`attnres_block_size` must be'):
            config_class(attnres_block_size=block_size)
    else:
        config = config_class(attnres_block_size=block_size)
        restored = config_class.from_dict(config.to_dict())
        assert restored.attnres_block_size == block_size


@pytest.mark.parametrize(('config_class', 'kwargs', 'match'), [
    pytest.param(models.KDAConfig, {'safe_gate': True, 'attnres_block_size': 3}, 'lower_bound', id='kda-gate'),
    pytest.param(models.GatedDeltaNetConfig, {'output_gate_activation': 'relu'}, 'output_gate_activation', id='gdn-gate'),
    pytest.param(models.Mamba2Config, {'A_init_range': (0, 16)}, 'A_init_range', id='mamba2-state'),
    pytest.param(models.Mamba2Config, {'dt_min': 0}, 'dt_min', id='mamba2-dt'),
    pytest.param(models.Mamba2Config, {'dt_init_floor': 0}, 'dt_init_floor', id='mamba2-floor'),
    pytest.param(models.Mamba3Config, {'rope_fraction': 0.25}, 'rope_fraction', id='mamba3-rope'),
    pytest.param(models.Mamba3Config, {'A_floor': 0}, 'A_floor', id='mamba3-state'),
    pytest.param(models.Mamba3Config, {'is_mimo': True, 'mimo_rank': 0}, 'mimo_rank', id='mamba3-mimo'),
    pytest.param(models.LogLinearMamba2Config, {'attnres_block_size': 3}, 'attnres_block_size', id='log-linear-attnres'),
])
def test_model_validation(config_class, kwargs, match):
    with pytest.raises(ValueError, match=match):
        config_class(**kwargs, fuse_cross_entropy=True, fuse_linear_cross_entropy=True)
