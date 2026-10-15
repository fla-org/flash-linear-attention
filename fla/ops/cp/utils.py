# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.utils import IS_TF32_SUPPORTED

DEFAULT_AFFINE_CHAIN_PRECISION = 'tf32x3' if IS_TF32_SUPPORTED else 'ieee'


def get_affine_chain_precision(use_tf32x3_affine_chain: bool = True) -> str | None:
    if use_tf32x3_affine_chain or DEFAULT_AFFINE_CHAIN_PRECISION == 'ieee':
        return DEFAULT_AFFINE_CHAIN_PRECISION
    return None
