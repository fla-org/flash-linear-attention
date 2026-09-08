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
def test_legacy_dispatch_uses_directory_registry(first_import):
    _run_python(
        """
        import importlib
        import os
        import warnings
        import torch

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', DeprecationWarning)
            importlib.import_module(os.environ['FIRST_IMPORT'])
            import fla.ops.backends as legacy
        assert any('next release after 0.6.0' in str(w.message) for w in caught)

        from fla.backends import BaseBackend
        from fla.ops.kda.backends import kda_registry
        from fla.modules.backends import modules_registry
        assert legacy.BaseBackend is BaseBackend
        assert legacy.BackendRegistry('kda') is kda_registry
        assert legacy.BackendRegistry('modules') is modules_registry
        legacy.BackendRegistry.ensure_initialized('kda')
        assert legacy.BackendRegistry._registries['kda'] is kda_registry

        class Backend(BaseBackend):
            backend_type = 'import_test'
            priority = -1

            def compute(self, x, use_backend):
                return x * 2

            def compute_verifier(self, x, use_backend):
                return use_backend, None

        legacy.BackendRegistry('kda').register(Backend())

        def compute(x, use_backend):
            return x * 3

        current = kda_registry.dispatch(compute)
        old = legacy.dispatch('kda')(compute)
        for use_backend, factor in [(True, 2), (False, 3)]:
            for implementation in [current, old]:
                x = torch.tensor([1.0, 2.0], requires_grad=True)
                result = implementation(x, use_backend=use_backend)
                torch.testing.assert_close(result, x * factor)
                result.sum().backward()
                torch.testing.assert_close(x.grad, torch.full_like(x, factor))

        custom = legacy.BackendRegistry('custom_import_test')
        custom.register(Backend())
        assert legacy.dispatch('custom_import_test')(compute)(2, use_backend=True) == 4
        """,
        FIRST_IMPORT=first_import,
    )


@pytest.mark.parametrize('disabled', ['0', '1'], ids=['dispatch-enabled', 'dispatch-disabled'])
def test_local_dispatch_policy_and_optional_dependencies(disabled):
    _run_python(
        """
        import importlib
        import os
        import sys
        import warnings

        from fla.backends import BackendRegistry
        from fla.modules import RMSNorm, ShortConvolution
        assert 'fla.ops.backends' not in sys.modules
        enabled = os.environ['FLA_DISABLE_BACKEND_DISPATCH'] != '1'
        for operation in ['attn', 'attnres', 'common', 'gated_delta_rule', 'generalized_delta_rule.dplr',
                          'gla', 'kda', 'rwkv6', 'utils']:
            package = importlib.import_module('fla.ops.' + operation + '.backends')
            registry = package.dispatch.__self__
            assert registry.operation_name == operation
            assert registry._backends

        assert 'tilelang' not in sys.modules
        assert 'flash_kda' not in sys.modules
        assert 'fla.ops.kda.backends.triton_ascend.chunk_intra' not in sys.modules
        registry = BackendRegistry('local_test')

        def compute(x):
            return x + 1

        wrapped = registry.dispatch(compute)
        assert (wrapped is compute) == (not enabled)
        assert wrapped(1) == 2

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            from fla.ops.backends import dispatch
        legacy = dispatch('kda')(compute)
        assert (legacy is compute) == (not enabled)
        assert legacy(1) == 2
        """,
        FLA_DISABLE_BACKEND_DISPATCH=disabled,
    )
