# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import ast
import os
import re
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


setup(
    name='flash-linear-attention',
    version=get_package_version(),
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
)
