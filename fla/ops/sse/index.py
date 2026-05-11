# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
from typing import Tuple

from fla.utils import tensor_cache


@tensor_cache
def prepare_sample_relpos_global_index(
    offsets: torch.Tensor
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    lengths = offsets[1:] - offsets[:-1]
    S = lengths.numel()
    sample_idx_per_token = torch.repeat_interleave(torch.arange(S, device=offsets.device), lengths)  # [L]
    token_global_idx = torch.arange(offsets[-1], device=offsets.device)                              # [L]
    token_start_idx = offsets[:-1].index_select(0, sample_idx_per_token)                             # [L]
    relpos_in_sample = token_global_idx - token_start_idx
    return sample_idx_per_token, relpos_in_sample, token_global_idx, lengths


@tensor_cache
def prepare_sample_relpos_global_index_flat(
    offsets: torch.Tensor,
    K: int
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    sample_idx_per_token, relpos_in_sample, token_global_idx, lengths = prepare_sample_relpos_global_index(offsets)
    sample_idx_flat = sample_idx_per_token[:, None].expand(-1, K).reshape(-1)     # [L*K]
    relpos_flat = relpos_in_sample[:, None].expand(-1, K).reshape(-1)             # [L*K]
    global_idx_flat = token_global_idx[:, None].expand(-1, K).reshape(-1)         # [L*K]
    return sample_idx_flat, relpos_flat, global_idx_flat, lengths
