# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

_SUPPORTED_INTRA_CHUNK_SIZES = (32, 64)
_SUB_CHUNK = 16


def _chunk_size_from(args, kwargs, default: int = 64) -> int:
    if 'chunk_size' in kwargs and kwargs['chunk_size'] is not None:
        return int(kwargs['chunk_size'])
    return default


def _verify_intra_chunk_size(args, kwargs):
    chunk_size = _chunk_size_from(args, kwargs)
    if chunk_size not in _SUPPORTED_INTRA_CHUNK_SIZES:
        return False, f'KDA Ascend intra only supports chunk_size in {_SUPPORTED_INTRA_CHUNK_SIZES}, got {chunk_size}'
    return True, None
