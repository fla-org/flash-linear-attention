# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.utils import autotune_cache_kwargs, tensor_cache


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [4, 8, 16, 32]
    ],
    key=['B'],
    **autotune_cache_kwargs,
)
@triton.jit
def prepare_position_ids_kernel(
    y,
    cu_seqlens,
    B: tl.constexpr,
):
    i_n = tl.program_id(0)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    o = tl.arange(0, B)
    for i in range(0, tl.cdiv(T, B) * B, B):
        o_i = o + i
        tl.store(y + bos + o_i, o_i, o_i < T)


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_lens_from_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    return mask.sum(dim=-1, dtype=torch.int32)


@tensor_cache
def prepare_cu_seqlens_from_lens(
    lens: torch.LongTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))


@tensor_cache
def prepare_cu_seqlens_from_mask(
    mask: torch.BoolTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    return prepare_cu_seqlens_from_lens(prepare_lens_from_mask(mask), dtype)


@tensor_cache
def prepare_lens_from_cu_seqlens(
    cu_seqlens: torch.LongTensor,
) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_split_cu_seqlens(
    batch_size: int,
    seq_len: int,
    split_size: int,
    cu_seqlens: torch.LongTensor | None = None,
    dtype: torch.dtype | None = torch.int32,
    device: torch.device | None = torch.device('cpu'),
) -> torch.LongTensor:
    if cu_seqlens is None:
        total_tokens = batch_size * seq_len
        cu_seqlens = list(range(0, total_tokens, seq_len)) + [total_tokens]
    else:
        cu_seqlens = cu_seqlens.tolist()
    return torch.tensor(
        [
            i
            for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)
            for i in range(bos, eos, split_size)
        ] + [cu_seqlens[-1]],
        dtype=dtype,
        device=device,
    )


@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.cat([
        torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        for n in prepare_lens(cu_seqlens).unbind()
    ])


@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


@tensor_cache
def prepare_token_indices(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    position_ids = prepare_position_ids(cu_seqlens)
    return torch.stack([prepare_sequence_ids(cu_seqlens), position_ids], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


@tensor_cache
def get_max_num_splits(cu_seqlens: torch.LongTensor, chunk_size: int) -> int:
    return triton.cdiv(int(max(prepare_lens(cu_seqlens))), chunk_size)
