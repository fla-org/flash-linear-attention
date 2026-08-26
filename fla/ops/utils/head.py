# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


def get_gqa_group_size(num_query_heads: int, num_kv_heads: int) -> int:
    if num_kv_heads == 0 or num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"The number of query heads ({num_query_heads}) must be divisible by "
            f"the number of key/value heads ({num_kv_heads})."
        )
    return num_query_heads // num_kv_heads
