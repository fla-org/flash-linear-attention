# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
from torch import nn

from fla.modules.residuals import (
    AttentionResidual,
    BaseResidual,
    ManifoldHyperConnection,
    StandardResidual,
    get_residual_class,
    register_residual,
    registry,
)


@pytest.fixture(autouse=True)
def poison_torch_memory():
    """Registry tests allocate no tensors and need neither kernel memory poisoning nor device synchronization."""


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, '_RESIDUAL_CLASSES', registry._RESIDUAL_CLASSES.copy())


@pytest.mark.parametrize(('name', 'residual_cls'), [
    ('standard', StandardResidual), ('attnres', AttentionResidual), ('mhc', ManifoldHyperConnection),
])
def test_builtin_residuals(name, residual_cls):
    assert get_residual_class(name) is residual_cls
    with pytest.raises(ValueError, match='already registered'):
        register_residual(name, StandardResidual)
    assert get_residual_class(name) is residual_cls


def test_register_custom_residual():
    class CustomResidual(StandardResidual):
        def __init__(self, *args, **kwargs):
            raise AssertionError('Registration must not instantiate the class')

    register_residual('test.custom', CustomResidual)
    assert get_residual_class('test.custom') is CustomResidual
    with pytest.raises(ValueError, match='already registered'):
        register_residual('test.custom', CustomResidual)


@pytest.mark.parametrize('residual_cls', [nn.Module, BaseResidual, object, None])
def test_reject_invalid_residual_class(residual_cls):
    with pytest.raises(TypeError):
        register_residual('test.invalid', residual_cls)
    with pytest.raises(ValueError, match='register_residual'):
        get_residual_class('test.invalid')


@pytest.mark.parametrize(('name', 'error'), [
    ('', ValueError), (' ', ValueError), ('test. custom', ValueError), ('test.custom\n', ValueError),
    (None, TypeError), ([], TypeError),
])
def test_reject_invalid_residual_name(name, error):
    with pytest.raises(error):
        register_residual(name, StandardResidual)
    with pytest.raises(error):
        get_residual_class(name)


def test_unknown_residual_requires_registration():
    with pytest.raises(ValueError, match='register_residual'):
        get_residual_class('test.unknown')


def test_custom_residual_requires_hidden_state_accessor():
    class IncompleteResidual(BaseResidual):
        def initialize(self, x):
            return x, x

        def forward(self, branch_output, history):
            return branch_output, history

    with pytest.raises(TypeError, match='concrete'):
        register_residual('test.incomplete', IncompleteResidual)
