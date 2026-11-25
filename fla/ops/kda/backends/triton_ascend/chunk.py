# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.backends import TritonAscendBackend, register
from fla.ops.cp import FLACPContext
from fla.ops.kda.chunk import _prepare_kda_graph_cpu_seqlens


def _verify_npu_graph_input(q, *args, **kwargs):
    if q.device.type != 'npu':
        return False, 'input device is not NPU'
    return True, None


@register(_prepare_kda_graph_cpu_seqlens, backend=TritonAscendBackend, verifier=_verify_npu_graph_input)
def prepare_kda_graph_cpu_seqlens_npu(
    q: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    cp_context: FLACPContext | None,
) -> None:
    if cp_context is None:
        if cu_seqlens is None:
            raise NotImplementedError("Ascend KDA graph mode currently requires flattened variable-length inputs.")
    else:
        if cp_context.layout not in ('contiguous', 'zigzag'):
            raise NotImplementedError(f"Ascend KDA graph mode does not support CP layout {cp_context.layout!r}.")
        if cp_context.group is None:
            raise ValueError("Ascend KDA CP graph mode requires a process group.")
        zigzag = cp_context.layout == 'zigzag'
        for name in ('pre_num_ranks_dev', 'post_num_ranks_dev'):
            metadata = getattr(cp_context, name)
            valid_shape = metadata is not None and (metadata.shape == (2,) if zigzag else metadata.numel() == 1)
            if not valid_shape or metadata.dtype != torch.int32 or metadata.device != q.device:
                shape = 'shape [2]' if zigzag else 'one element'
                raise ValueError(f"Ascend KDA CP graph mode requires {name} to be int32 with {shape} on {q.device}.")
    # replay may change device boundaries without refreshing the CPU snapshot.
    return None
