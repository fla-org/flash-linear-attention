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


@pytest.mark.parametrize(
    'module',
    [
        'missing_backend',
        'missing_backend.child',
        'fla_test_backend.missing.child',
        'fla_test_backend.present',
        'fla_test_backend.broken.child',
    ],
)
def test_find_spec_cached(module: str, tmp_path, monkeypatch):
    from fla.utils import find_spec_cached

    package = tmp_path / 'fla_test_backend'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'present.py').write_text('')
    (package / 'broken').mkdir()
    (package / 'broken' / '__init__.py').write_text('import missing_backend_dependency\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    find_spec_cached.cache_clear()
    try:
        if module == 'fla_test_backend.broken.child':
            with pytest.raises(ModuleNotFoundError, match='missing_backend_dependency'):
                find_spec_cached(module)
        else:
            assert (find_spec_cached(module) is not None) == (module == 'fla_test_backend.present')
    finally:
        find_spec_cached.cache_clear()
        for name in tuple(sys.modules):
            if name == 'fla_test_backend' or name.startswith('fla_test_backend.'):
                del sys.modules[name]


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
