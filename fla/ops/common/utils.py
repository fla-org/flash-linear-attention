# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Tuple

import torch
import triton

from fla.utils import tensor_cache


@tensor_cache
def prepare_chunk_offsets(
    offsets: torch.Tensor,
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.cat([offsets.new_tensor([0]), triton.cdiv(offsets[1:] - offsets[:-1], chunk_size)]).cumsum(-1)


@tensor_cache
def prepare_chunk_indices(
    offsets: torch.LongTensor,
    chunk_size: int
) -> Tuple[torch.LongTensor]:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
    indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
    return indices
