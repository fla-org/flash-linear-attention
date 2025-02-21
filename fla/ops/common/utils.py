# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from fla.utils import tensor_cache


@tensor_cache
def prepare_lens(offsets: torch.LongTensor) -> torch.LongTensor:
    return offsets[1:] - offsets[:-1]


@tensor_cache
def prepare_token_indices(offsets: torch.LongTensor) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in (prepare_lens(offsets)).tolist()])
    indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
    return indices


@tensor_cache
def prepare_chunk_offsets(
    offsets: torch.Tensor,
    chunk_size: int
) -> torch.LongTensor:
    return torch.cat([offsets.new_tensor([0]), triton.cdiv(prepare_lens(offsets), chunk_size)]).cumsum(-1)


@tensor_cache
def prepare_chunk_indices(
    offsets: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(offsets), chunk_size).tolist()])
    indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
    return indices
