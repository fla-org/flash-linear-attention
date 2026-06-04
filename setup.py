# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import ast
import os
import re
import sys
from pathlib import Path

from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()


def get_package_version():
    init_file = Path(os.path.dirname(os.path.abspath(__file__))) / 'fla' / '__init__.py'
    with open(init_file) as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    if version_match is None:
        raise RuntimeError(f"Could not find `__version__` in the file {init_file}")
    return ast.literal_eval(version_match.group(1))


# Backends fla can resolve dependencies for.
_SUPPORTED_TARGET_DEVICES = ('cuda', 'rocm', 'xpu', 'cpu')

# Each backend pins its own torch + triton flavor. Keep in sync with the
# extras in pyproject.toml.
_BACKEND_REQUIREMENTS = {
    'cuda': ['torch>=2.7.0', 'triton>=3.3'],
    'rocm': ['torch>=2.7.0', 'pytorch-triton-rocm>=3.3'],
    'xpu':  ['torch>=2.7.0', 'pytorch-triton-xpu>=3.3'],
    'cpu':  ['torch>=2.7.0'],
}


def _detect_target_device() -> str:
    """Resolve the backend whose torch/triton pins go in install_requires.

    Precedence:
      1. FLA_TARGET_DEVICE env var (cuda/rocm/xpu/cpu, or "auto" to probe).
      2. If torch is importable, probe torch.version.{hip,xpu,cuda}.
      3. Fall back to "cuda" (matches PyPI default behavior).
    """
    requested = os.environ.get('FLA_TARGET_DEVICE', 'auto').strip().lower()
    if requested and requested != 'auto':
        if requested not in _SUPPORTED_TARGET_DEVICES:
            raise RuntimeError(
                f"FLA_TARGET_DEVICE={requested!r} is not one of {_SUPPORTED_TARGET_DEVICES}"
            )
        return requested

    try:
        import torch
    except ImportError:
        return 'cuda'

    if getattr(torch.version, 'hip', None):
        return 'rocm'
    if getattr(torch.version, 'xpu', None):
        return 'xpu'
    if getattr(torch.version, 'cuda', None):
        return 'cuda'
    return 'cpu'


_target_device = _detect_target_device()
_install_requires = [
    *_BACKEND_REQUIREMENTS[_target_device],
    'transformers>=4.45.0',
    'einops',
]
print(f"fla setup.py: target device = {_target_device}", file=sys.stderr)

setup(
    name='flash-linear-attention',
    version=get_package_version(),
    description='Fast Triton-based implementations of causal linear attention',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Songlin Yang, Yu Zhang',
    author_email='yangsl66@mit.edu, yzhang.cs@outlook.com',
    url='https://github.com/fla-org/flash-linear-attention',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.10',
    install_requires=_install_requires,
    # extras_require lives in pyproject.toml [project.optional-dependencies].
)
