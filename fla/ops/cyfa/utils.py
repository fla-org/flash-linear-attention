# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


# Portions adapted from CyclicFlowAttention, Copyright (c) 2026 Yixiao Chen.
# https://github.com/Chyxx/CyclicFlowAttention

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

CyFAState = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@triton.jit(do_not_specialize=['T'])
def _cyfa_clock_cumsum_kernel(
    delta,
    clock,
    cu_seqlens,
    T,
    H: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BT: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_n * T, (i_n + 1) * T
    length = eos - bos
    total = tl.full((), 0, tl.float32)
    for i_t in range(tl.cdiv(length, BT)):
        offsets = i_t * BT + tl.arange(0, BT)
        indices = (bos + offsets) * H + i_h
        values = tl.load(delta + indices, mask=offsets < length, other=0).to(tl.float32)
        prefix = tl.cumsum(values, 0) + total
        total += tl.sum(values, 0)
        tl.store(clock + indices, prefix, mask=offsets < length)


def cyfa_clock_cumsum(delta: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
    # a fixed scan geometry keeps the cyclic phase identical for dense and packed layouts.
    B, T, H = delta.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    clock = torch.empty_like(delta, dtype=torch.float32)
    _cyfa_clock_cumsum_kernel[(N * H,)](
        delta, clock, cu_seqlens, T=T, H=H, IS_VARLEN=cu_seqlens is not None,
        BT=256, num_warps=4, num_stages=1,
    )
    return clock


@triton.jit
def cyfa_cos(x):
    return libdevice.fast_cosf(x)


@triton.jit
def cyfa_sin(x):
    return libdevice.fast_sinf(x)


@triton.jit
def modal_omega(o_m, M: tl.constexpr):
    pair = (o_m // 2).to(tl.float32)
    return pair * (6.283185307179586 / (M - 1))


@triton.jit
def modal_phase(lam, o_m, M: tl.constexpr):
    # reduce in integer slot coordinates before converting to radians.
    period: tl.constexpr = M - 1
    lam = lam.to(tl.float32)
    pair_i = (o_m // 2).to(tl.int32)
    whole = tl.floor(lam)
    fraction = lam - whole
    whole_mod = whole.to(tl.int32) % period
    whole_mod = tl.where(whole_mod < 0, whole_mod + period, whole_mod)
    coordinate = ((whole_mod * pair_i) % period).to(tl.float32)
    coordinate += fraction * pair_i.to(tl.float32)
    coordinate = tl.where(coordinate > period * 0.5, coordinate - period, coordinate)
    return coordinate * (6.283185307179586 / period)


def validate_num_slots(num_slots: int) -> int:
    num_slots = int(num_slots)
    if num_slots <= 0 or num_slots % 2:
        raise ValueError("`num_slots` must be a positive even integer.")
    return num_slots


def validate_cyfa_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    delta: torch.Tensor,
    beta: torch.Tensor,
    readout: torch.Tensor,
    *,
    require_cuda: bool = False,
) -> int:
    if require_cuda and not q.is_cuda:
        raise ValueError("CyclicFlowAttention optimized operators require CUDA tensors.")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("`q`, `k`, and `v` must have shape [B, T, H, D].")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("`q`, `k`, and `v` must share [B, T, H].")
    if g.shape != q.shape[:3] or delta.shape != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("`g`, `delta`, and `beta` must have shape [B, T, H].")
    if readout.ndim != 3 or readout.shape[0] != q.shape[2] or readout.shape[1] != readout.shape[2]:
        raise ValueError("`readout` must have shape [H, C, C].")
    return validate_num_slots(readout.shape[-1] + 1)


def prepare_cu_seqlens(
    cu_seqlens: torch.LongTensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.LongTensor | None:
    if cu_seqlens is None:
        return None
    if batch_size != 1:
        raise ValueError("Packed variable-length inputs must have batch size 1.")
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long).contiguous()
    if int(cu_seqlens[-1].item()) != seq_len:
        raise ValueError("Packed sequence length does not match `cu_seqlens`.")
    return cu_seqlens


@lru_cache(maxsize=16)
@torch.inference_mode(False)
def _readout_basis(num_slots: int, device: torch.device) -> torch.Tensor:
    period = num_slots - 1
    idx = torch.arange(num_slots, device=device)
    rows = torch.arange(period, device=device, dtype=torch.float32)[:, None]
    omega = (idx // 2) * (2.0 * torch.pi / period)
    basis = torch.where(
        idx.remainder(2)[None, :] == 1,
        torch.sin(rows * omega),
        torch.cos(rows * omega),
    )
    basis[:, 0] *= period**-0.5
    basis[:, 1] = 0.0
    basis[:, 2:] *= (2.0 / period) ** 0.5
    return basis


def build_readout_table(readout: torch.Tensor) -> torch.Tensor:
    num_slots = validate_num_slots(readout.shape[-1] + 1)
    # only the fixed Fourier basis is cached, never the learned readout table.
    basis = _readout_basis(num_slots, readout.device)
    return torch.matmul(readout.float(), basis).contiguous()


def add_segment_initial(
    x: torch.Tensor,
    initial: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    if cu_seqlens is None:
        return x + initial[:, None, :]
    out = x.clone()
    boundaries = cu_seqlens.detach().cpu().tolist()
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        out[:, start:end] += initial[i][None, None, :]
    return out


def segment_sum(x: torch.Tensor, cu_seqlens: torch.Tensor | None) -> torch.Tensor:
    x = x.float()
    if cu_seqlens is None:
        return x.sum(dim=1)
    boundaries = cu_seqlens.detach().cpu().tolist()
    return torch.stack([x[:, start:end].sum(dim=1).squeeze(0) for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)])


def validate_initial_state(
    initial_state: CyFAState | None,
    *,
    batch: int,
    n_heads: int,
    num_slots: int,
    d_k: int,
    d_v: int,
    device: torch.device,
) -> CyFAState | None:
    if initial_state is None:
        return None
    if len(initial_state) != 3:
        raise ValueError("`initial_state` must contain (k, v, lambda).")
    k0, v0, lambda0 = initial_state
    if k0.shape != (batch, n_heads, d_k, num_slots):
        raise ValueError("K initial state must have shape [B, H, Dk, M].")
    if v0.shape != (batch, n_heads, num_slots, d_v):
        raise ValueError("V initial state must have shape [B, H, M, Dv].")
    if lambda0.shape != (batch, n_heads):
        raise ValueError("Lambda initial state must have shape [B, H].")
    return (
        k0.to(device=device, dtype=torch.float32).contiguous(),
        v0.to(device=device, dtype=torch.float32).contiguous(),
        lambda0.to(device=device, dtype=torch.float32).contiguous(),
    )
