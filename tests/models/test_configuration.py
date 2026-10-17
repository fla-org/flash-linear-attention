# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import inspect

import pytest

from fla import models

CONFIG_CLASSES = [getattr(models, name) for name in models.__all__ if name.endswith('Config')]
LOSS_CONFIG_CLASSES = [cls for cls in CONFIG_CLASSES if 'fuse_linear_cross_entropy' in inspect.signature(cls).parameters]
ATTNRES_CONFIG_CLASSES = [cls for cls in CONFIG_CLASSES if 'attnres_block_size' in inspect.signature(cls).parameters]


@pytest.mark.parametrize('config_class', LOSS_CONFIG_CLASSES, ids=lambda cls: cls.model_type)
@pytest.mark.parametrize(('fuse_cross_entropy', 'fuse_linear_cross_entropy'), [(False, False), (True, False), (False, True)])
def test_loss_configuration_round_trip(config_class, fuse_cross_entropy, fuse_linear_cross_entropy):
    kwargs = {
        'fuse_cross_entropy': fuse_cross_entropy,
        'fuse_linear_cross_entropy': fuse_linear_cross_entropy,
        'downstream_config': {'enabled': True},
    }
    if fuse_linear_cross_entropy:
        with pytest.warns(UserWarning, match='`fuse_linear_cross_entropy` is enabled'):
            config = config_class(**kwargs)
            restored = config_class.from_dict(config.to_dict())
    else:
        config = config_class(**kwargs)
        restored = config_class.from_dict(config.to_dict())
    for name, value in kwargs.items():
        assert getattr(restored, name) == value


@pytest.mark.parametrize('config_class', LOSS_CONFIG_CLASSES, ids=lambda cls: cls.model_type)
def test_conflicting_loss_configuration(config_class):
    with pytest.raises(ValueError, match='cannot be True at the same time'):
        config_class(fuse_cross_entropy=True, fuse_linear_cross_entropy=True)


@pytest.mark.parametrize('config_class', ATTNRES_CONFIG_CLASSES, ids=lambda cls: cls.model_type)
@pytest.mark.parametrize('block_size', [None, 1, 2, 4, 0, -2, 3])
def test_attention_residual_configuration(config_class, block_size):
    if block_size in (0, -2, 3):
        with pytest.raises(ValueError, match='`attnres_block_size` must be'):
            config_class(attnres_block_size=block_size)
    else:
        config = config_class(attnres_block_size=block_size)
        restored = config_class.from_dict(config.to_dict())
        assert restored.attnres_block_size == block_size
