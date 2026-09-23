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
