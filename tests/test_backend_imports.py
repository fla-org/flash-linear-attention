# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_python(source, **env):
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(source)],
        cwd=ROOT,
        env={**os.environ, 'FLA_DISABLE_BACKEND_DISPATCH': '0', **env},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('first_import', ['fla.backends', 'fla.ops.backends'])
def test_legacy_import_shares_registry_and_dispatch(first_import):
    _run_python(
        """
        import importlib
        import os
        import warnings
        import torch

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', DeprecationWarning)
            importlib.import_module(os.environ['FIRST_IMPORT'])
            import fla.backends as current
            import fla.ops.backends as legacy

        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert sum('fla.ops.backends is deprecated' in message for message in messages) == 1
        assert any('next release after 0.6.0' in message for message in messages)
        for name in current.__all__:
            assert getattr(legacy, name) is getattr(current, name)

        class Backend(legacy.BaseBackend):
            def compute(self, x, use_backend):
                return x * 2

            def compute_verifier(self, x, use_backend):
                return use_backend, None

        registry = legacy.BackendRegistry('import_test')
        registry.register(Backend())
        current.BackendRegistry._initialized.add('import_test')
        assert current.BackendRegistry._registries['import_test'] is registry

        @current.dispatch('import_test')
        def compute(x, use_backend):
            return x * 3

        for use_backend, factor in [(True, 2), (False, 3)]:
            x = torch.tensor([1.0, 2.0], requires_grad=True)
            result = compute(x, use_backend=use_backend)
            torch.testing.assert_close(result, x * factor)
            result.sum().backward()
            torch.testing.assert_close(x.grad, torch.full_like(x, factor))
        """,
        FIRST_IMPORT=first_import,
    )


@pytest.mark.parametrize('disabled', ['0', '1'], ids=['dispatch-enabled', 'dispatch-disabled'])
def test_current_imports_and_backend_discovery(disabled):
    _run_python(
        """
        import os
        import sys
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', DeprecationWarning)
            from fla.backends import BackendRegistry, BaseBackend, dispatch
            from fla.modules import RMSNorm, ShortConvolution
            from fla.modules.backends import modules_registry
            BackendRegistry.ensure_initialized('modules')
            BackendRegistry.ensure_initialized('kda')
        assert 'fla.ops.backends' not in sys.modules
        assert not any('fla.ops.backends is deprecated' in str(w.message) for w in caught)
        assert BackendRegistry._registries['modules'] is modules_registry
        assert isinstance(modules_registry._backends['triton_ascend'], BaseBackend)
        assert 'kda' in BackendRegistry._registries

        def compute(x):
            return x + 1

        wrapped = dispatch('import_fallback')(compute)
        assert wrapped(1) == 2
        assert (wrapped is compute) == (os.environ['FLA_DISABLE_BACKEND_DISPATCH'] == '1')
        """,
        FLA_DISABLE_BACKEND_DISPATCH=disabled,
    )
