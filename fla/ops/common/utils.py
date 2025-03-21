# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.utils import tensor_cache


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [4, 8, 16, 32]
    ],
    key=['B'],
)
@triton.jit
def prepare_position_ids_kernel(
    y,
    offsets,
    B: tl.constexpr
):
    i_n = tl.program_id(0)
    bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
    T = eos - bos

    o = tl.arange(0, B)
    for i in range(0, tl.cdiv(T, B) * B, B):
        o_i = o + i
        tl.store(y + bos + o_i, o_i, o_i < T)


@tensor_cache
def prepare_lens(offsets: torch.LongTensor) -> torch.LongTensor:
    return offsets[1:] - offsets[:-1]


@tensor_cache
def prepare_position_ids(offsets: torch.LongTensor) -> torch.LongTensor:
    return torch.cat([torch.arange(n, dtype=offsets.dtype, device=offsets.device) for n in prepare_lens(offsets).unbind()])


@tensor_cache
def prepare_sequence_ids(position_ids: torch.LongTensor) -> torch.LongTensor:
    return position_ids.eq(0).cumsum(0) - 1


@tensor_cache
def prepare_token_indices(offsets: torch.LongTensor) -> torch.LongTensor:
    position_ids = prepare_position_ids(offsets)
    return torch.stack([prepare_sequence_ids(position_ids), position_ids], 1).to(offsets)


@tensor_cache
def prepare_chunk_indices(
    offsets: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(offsets), chunk_size).tolist()])
    return torch.stack([prepare_sequence_ids(indices), indices], 1).to(offsets)


@tensor_cache
def prepare_chunk_offsets(
    offsets: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    return torch.cat([offsets.new_tensor([0]), triton.cdiv(prepare_lens(offsets), chunk_size)]).cumsum(-1)
