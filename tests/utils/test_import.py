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
