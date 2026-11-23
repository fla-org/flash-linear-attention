# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton
import triton.language as tl


@triton.jit
def compact_chunk_state_index(bos, i_t, i_n, H_CHUNKS: tl.constexpr, BT: tl.constexpr, HAS_INITIAL_STATE: tl.constexpr):
    # Non-first chunk starts are at least BT tokens apart, including across documents.
    index = bos.to(tl.int64) // BT + i_t.to(tl.int64) - 1
    if HAS_INITIAL_STATE:
        index = tl.where((i_n == 0) & (i_t == 0), H_CHUNKS, index)
        valid = (i_t > 0) | (i_n == 0)
    else:
        valid = i_t > 0
    return tl.where(valid, index, -1)


@triton.jit
def compact_chunk_state_gradient_index(bos, eos, i_t, BT: tl.constexpr):
    # Each stored outgoing gradient corresponds to the next chunk's incoming state.
    index = bos.to(tl.int64) // BT + i_t.to(tl.int64)
    return tl.where((i_t + 1) * BT < eos - bos, index, -1)
