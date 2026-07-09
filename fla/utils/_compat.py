# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import functools
import inspect
from importlib.util import find_spec

import triton
from packaging import version as package_version

from ._config import FLA_CACHE_RESULTS

TRITON_ABOVE_3_4_0 = package_version.parse(triton.__version__) >= package_version.parse("3.4.0")
TRITON_ABOVE_3_5_1 = package_version.parse(triton.__version__) >= package_version.parse("3.5.1")
TRITON_ABOVE_3_7_1 = package_version.parse(triton.__version__) >= package_version.parse("3.7.1")

SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}


@functools.cache
def find_spec_cached(name):
    return find_spec(name)
