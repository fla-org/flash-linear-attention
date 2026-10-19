# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import GluonBackend, TileLangBackend, TritonAscendBackend
from fla.ops.utils.csr import prepare_block_csr
from fla.ops.utils.cumsum import (
    chunk_global_cumsum,
    chunk_global_cumsum_scalar,
    chunk_global_cumsum_vector,
    chunk_local_cumsum,
    chunk_local_cumsum_scalar,
    chunk_local_cumsum_vector,
)
from fla.ops.utils.index import (
    get_max_num_splits,
    prepare_chunk_indices,
    prepare_chunk_indices_static,
    prepare_chunk_offsets,
    prepare_cu_seqlens_from_lens,
    prepare_cu_seqlens_from_mask,
    prepare_lens,
    prepare_lens_from_mask,
    prepare_position_ids,
    prepare_sequence_ids,
    prepare_token_indices,
)
from fla.ops.utils.logsumexp import logsumexp_fwd
from fla.ops.utils.matmul import addmm, matmul
from fla.ops.utils.pack import pack_sequence, unpack_sequence
from fla.ops.utils.pooling import mean_pooling
from fla.ops.utils.sinkhorn import sinkhorn
from fla.ops.utils.softmax import softmax_bwd, softmax_fwd
from fla.ops.utils.softplus import softplus
from fla.ops.utils.solve_tril import solve_tril

__all__ = [
    "addmm",
    "chunk_global_cumsum",
    "chunk_global_cumsum_scalar",
    "chunk_global_cumsum_vector",
    "chunk_local_cumsum",
    "chunk_local_cumsum_scalar",
    "chunk_local_cumsum_vector",
    "get_max_num_splits",
    "logsumexp_fwd",
    "matmul",
    "mean_pooling",
    "pack_sequence",
    "prepare_block_csr",
    "prepare_chunk_indices",
    "prepare_chunk_indices_static",
    "prepare_chunk_offsets",
    "prepare_cu_seqlens_from_lens",
    "prepare_cu_seqlens_from_mask",
    "prepare_lens",
    "prepare_lens_from_mask",
    "prepare_position_ids",
    "prepare_sequence_ids",
    "prepare_token_indices",
    "sinkhorn",
    "softmax_bwd",
    "softmax_fwd",
    "softplus",
    "solve_tril",
    "unpack_sequence",
]


if TritonAscendBackend.is_available():
    from fla.ops.utils import triton_ascend  # noqa: F401

if GluonBackend.is_available():
    from fla.ops.utils import gluon  # noqa: F401

if TileLangBackend.is_available():
    from fla.ops.utils import tilelang  # noqa: F401
