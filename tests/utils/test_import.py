# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import subprocess
import sys
from pathlib import Path

import pytest
import torch


def test_import_with_caching_precompile():
    if not hasattr(torch._dynamo.config, 'caching_precompile'):
        pytest.skip('caching_precompile is unavailable in this PyTorch version')
    result = subprocess.run(
        [sys.executable, '-c', 'import torch; torch._dynamo.config.caching_precompile = True; import fla'],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_utils_import_without_transformers(run_python):
    run_python(
        """
        import importlib.abc
        import sys

        class WithoutTransformers(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'transformers' or fullname.startswith('transformers.'):
                    raise ModuleNotFoundError(f'No module named {fullname!r}', name=fullname)

        sys.meta_path.insert(0, WithoutTransformers())

        from fla.utils import check_transformers_version, supports_transformers_next_sequence_length

        assert 'transformers' not in sys.modules
        assert isinstance(check_transformers_version('4.56'), bool)
        assert not supports_transformers_next_sequence_length()
        """,
    )

@pytest.mark.parametrize('error_name', ['ImportError', 'OSError'])
def test_causal_conv_import_with_unloadable_extension(run_python, error_name):
    run_python(
        f"""
        import importlib
        import importlib.abc
        import importlib.util
        import sys

        class UnloadableCausalConv(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'causal_conv1d':
                    return importlib.util.spec_from_loader(fullname, self, is_package=True)

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                raise {error_name}('optional causal-conv1d extension cannot be loaded')

        sys.meta_path.insert(0, UnloadableCausalConv())

        conv = importlib.import_module('fla.modules.causal_conv1d.ops')
        assert conv.causal_conv1d_fn_cuda is None
        assert conv.causal_conv1d_update_cuda is None

        cuda = importlib.import_module('fla.modules.causal_conv1d.backends.cuda.ops')
        assert cuda.causal_conv1d_fn_cuda is None
        assert cuda.causal_conv1d_bwd_function is None
        """,
    )
