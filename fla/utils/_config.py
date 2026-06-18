# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'

FLA_DISABLE_TENSOR_CACHE = os.getenv('FLA_DISABLE_TENSOR_CACHE', '0') == '1'
try:
    FLA_TENSOR_CACHE_SIZE = int(os.getenv('FLA_TENSOR_CACHE_SIZE', "4"))
except ValueError:
    FLA_TENSOR_CACHE_SIZE = 4
