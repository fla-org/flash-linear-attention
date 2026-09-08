# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

import fla.backends.registry as registry_module
from fla.backends import BackendRegistry, BaseBackend


@pytest.fixture(autouse=True)
def enable_dispatch(monkeypatch):
    monkeypatch.setattr(registry_module, '_DISPATCH_DISABLED', False)


@pytest.mark.parametrize('route', ['accepted', 'rejected', 'unavailable', 'disabled', 'missing', 'verifier-error'])
def test_dispatch_selection_preserves_outputs_and_gradients(monkeypatch, route):
    class Backend(BaseBackend):
        env_var = 'FLA_TEST_BACKEND'

        def compute(self, x):
            return x * 2

        def compute_verifier(self, x):
            if route == 'verifier-error':
                raise ValueError('unsupported call')
            return route != 'rejected', None

    registry = BackendRegistry('test')
    registry.register(Backend())
    if route == 'unavailable':
        monkeypatch.setattr(Backend, 'is_available', classmethod(lambda cls: False))
    elif route == 'disabled':
        monkeypatch.setenv('FLA_TEST_BACKEND', '0')
    elif route == 'missing':
        monkeypatch.delattr(Backend, 'compute')

    @registry.dispatch
    def compute(x):
        return x * 3

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    result = compute(x)
    factor = 2 if route == 'accepted' else 3
    torch.testing.assert_close(result, x * factor)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, factor))


def test_registry_ownership_and_priority():
    class Backend(BaseBackend):
        def __init__(self, name, priority, value):
            self.backend_type = name
            self.priority = priority
            self.value = value

        def compute(self):
            return self.value

    first = BackendRegistry('same-name')
    second = BackendRegistry('same-name')
    first.register(Backend('first', 5, 1))
    first.register(Backend('second', 5, 2))

    def compute():
        return 0

    wrapped = first.dispatch(compute)
    assert wrapped() == 1
    assert second.dispatch(compute)() == 0
    first.register(Backend('priority', 0, 3))
    assert wrapped() == 3


def test_registration_replaces_backend_and_active_policy_is_current(monkeypatch):
    class Backend(BaseBackend):
        env_var = 'FLA_TEST_BACKEND'

        def __init__(self, value):
            self.value = value

        def compute(self):
            return self.value

    registry = BackendRegistry('test')
    registry.register(Backend(1))
    replacement = Backend(2)
    registry.register(replacement)

    @registry.dispatch
    def compute():
        return 0

    assert compute() == 2
    assert registry.get_active() is replacement
    monkeypatch.setenv('FLA_TEST_BACKEND', '0')
    assert compute() == 0
    assert registry.get_active() is None


def test_deprecated_bound_operation_key():
    registry = BackendRegistry('test')
    with pytest.warns(DeprecationWarning, match='next release after 0.6.0'):
        dispatch = registry.dispatch('test')

    @dispatch
    def compute(x):
        return x + 1

    assert compute(1) == 2
