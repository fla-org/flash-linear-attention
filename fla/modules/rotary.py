# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import triton
import triton.language as tl
from einops import rearrange, repeat

from fla.ops.utils import prepare_chunk_indices
from fla.utils import get_multiprocessor_count, input_guard, is_amd

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [2, 4, 8, 16, 32]


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), '... d two -> ... (d two)', two=2)


def rotary_embedding_ref(x, cos, sin, interleaved=False):
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, '... d -> ... 1 (2 d)' if not interleaved else '... d -> ... 1 (d 2)')
    sin = repeat(sin, '... d -> ... 1 (2 d)' if not interleaved else '... d -> ... 1 (d 2)')
    return torch.cat([x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]], -1)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=['B', 'H', 'D', 'INTERLEAVED'],
)
@triton.jit(do_not_specialize=['T'])
def rotary_embedding_kernel(
    x,
    cos,
    sin,
    y,
    cu_seqlens,
    chunk_indices,
    seq_offsets,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    TR: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr
):
    i_t, i_b, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
        T = eos - bos
        x = x + bos * H*D + i_h * D
        y = y + bos * H*D + i_h * D
    else:
        i_n = i_b
        x = x + i_n * T*H*D + i_h * D
        y = y + i_n * T*H*D + i_h * D

    if i_t * BT >= T:
        return

    o_t = i_t * BT + tl.arange(0, BT)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        o_cs = o_t + seq_offsets
    else:
        o_cs = o_t + tl.load(seq_offsets + i_n)
    m_t = (o_t >= 0) & (o_t < T) & (o_cs >= 0) & (o_cs < TR)

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of x, do calculation, then store to 1st and 2nd halves of out
        o_r = tl.arange(0, BD // 2)
        p_x = x + o_t[:, None] * H*D + o_r[None, :]
        p_cos = cos + (o_cs[:, None] * R + o_r[None, :])
        p_sin = sin + (o_cs[:, None] * R + o_r[None, :])
        mask = m_t[:, None] & (o_r < R)[None, :]

        b_cos = tl.load(p_cos, mask=mask, other=1.0).to(tl.float32)
        b_sin = tl.load(p_sin, mask=mask, other=0.0).to(tl.float32)
        b_x0 = tl.load(p_x, mask=mask, other=0.0).to(tl.float32)
        b_x1 = tl.load(p_x + R, mask=mask, other=0.0).to(tl.float32)
        if CONJUGATE:
            b_sin = -b_sin
        b_o0 = b_x0 * b_cos - b_x1 * b_sin
        b_o1 = b_x0 * b_sin + b_x1 * b_cos
        # write back result
        p_y = y + (o_t[:, None] * H*D + o_r[None, :])
        tl.store(p_y, b_o0, mask=mask)
        tl.store(p_y + R, b_o1, mask=mask)
    else:
        # We don't want to load x[0, 2, 4, ...] and x[1, 3, 5, ...] separately since both are slow.
        # Instead, we load x0 = x[0, 1, 2, 3, ...] and x1 = x[1, 0, 3, 2, ...].
        # Loading x0 will be fast but x1 will be slow.
        # Then we load cos = cos[0, 0, 1, 1, ...] and sin = sin[0, 0, 1, 1, ...].
        # Then we do the calculation and use tl.where to pick put the right outputs for the even
        # and for the odd indices.
        o_d = tl.arange(0, BD)
        o_d_swap = o_d + ((o_d + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
        o_d_repeat = tl.arange(0, BD) // 2
        p_x0 = x + o_t[:, None] * H*D + o_d[None, :]
        p_x1 = x + o_t[:, None] * H*D + o_d_swap[None, :]
        p_cos = cos + (o_cs[:, None] * R + o_d_repeat[None, :])
        p_sin = sin + (o_cs[:, None] * R + o_d_repeat[None, :])
        mask = m_t[:, None] & (o_d_repeat < R)[None, :]

        b_cos = tl.load(p_cos, mask=mask, other=1.0).to(tl.float32)
        b_sin = tl.load(p_sin, mask=mask, other=0.0).to(tl.float32)
        b_x0 = tl.load(p_x0, mask=mask, other=0.0).to(tl.float32)
        b_x1 = tl.load(p_x1, mask=mask, other=0.0).to(tl.float32)
        if CONJUGATE:
            b_sin = -b_sin
        b_o0 = b_x0 * b_cos
        b_o1 = b_x1 * b_sin
        b_y = tl.where(o_d[None, :] % 2 == 0, b_o0 - b_o1, b_o0 + b_o1)
        p_y = y + (o_t[:, None] * H*D + o_d[None, :])
        tl.store(p_y, b_y, mask=mask)


def rotary_embedding_fwdbwd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False
) -> torch.Tensor:
    """
    Args:
        x: [B, T, H, D].
        cos: [TR, R / 2]
        sin: [TR, R / 2]
        seqlen_offsets: integer or integer tensor of size [N]
        cu_seqlens: [N + 1,] or None

    Returns:
        y: [B, T, H, D]
    """
    is_varlen = cu_seqlens is not None

    B, T, H, D = x.shape
    N = B if not is_varlen else cu_seqlens.shape[0] - 1
    TR, R = cos.shape
    R2 = R * 2

    assert D <= 256, "Only support D <= 256"
    assert TR >= T, f"TR must be >= T, got {TR} and {T}"

    assert cos.dtype == sin.dtype, f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert x.dtype == cos.dtype, f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (N,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
    else:
        assert seqlen_offsets + T <= TR

    y = torch.empty_like(x) if not inplace else x
    if R2 < D and not inplace:
        y[..., R2:].copy_(x[..., R2:])

    BD = triton.next_power_of_2(R2)
    BT = min(128, triton.next_power_of_2(triton.cdiv(T, get_multiprocessor_count(x.device.index))))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if is_varlen else None
    NT = len(chunk_indices) if is_varlen else triton.cdiv(T, BT)

    grid = (NT, B, H)
    rotary_embedding_kernel[grid](
        x,
        cos,
        sin,
        y,
        cu_seqlens,
        chunk_indices,
        seqlen_offsets,
        B=B,
        T=T,
        H=H,
        D=D,
        R=R,
        TR=TR,
        BT=BT,
        BD=BD,
        IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offsets, torch.Tensor),
        IS_VARLEN=is_varlen,
        INTERLEAVED=interleaved,
        CONJUGATE=conjugate
    )
    return y


class RotaryEmbeddingFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        y = rotary_embedding_fwdbwd(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            # Can't save int with save_for_backward
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return y if not inplace else x

    @staticmethod
    @input_guard
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
        # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = rotary_embedding_fwdbwd(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def rotary_embedding(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None
):
    """
    Args:
        x: [B, T, H, D]
        cos, sin: [TR, R//2]
        interleaved:
            If True, rotate pairs of even and odd dimensions (GPT-J style) instead of 1st half and 2nd half (GPT-NeoX style).
        inplace:
            If True, apply rotary embedding in-place.
        seqlen_offsets: [N,] or int.
            Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: [N + 1,] or None

    Returns:
        out: [B, T, H, D]
    """
    return RotaryEmbeddingFunction.apply(
        x,
        cos,
        sin,
        interleaved,
        inplace,
        seqlen_offsets,
        cu_seqlens
    )


class RotaryEmbedding(nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        scale_base: Optional[float] = None,
        interleaved: bool = False,
        pos_idx_in_fp32: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        interleaved:
            If True, rotate pairs of even and odd dimensions (GPT-J style) instead of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32:
            If True, the position indices [0.0, ..., seqlen - 1] are in fp32, otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq.
            In most cases this would be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16, we add this option.
        """
        super().__init__()

        self.dim = dim
        self.base = float(base)
        self.scale_base = scale_base
        self.interleaved = interleaved
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.device = device

        # Generate and save the inverse frequency buffer (non trainable)
        self.register_buffer("inv_freq", torch.empty(-(dim // -2), dtype=torch.float32, device=device), persistent=False)

        scale = None
        if scale_base is not None:
            scale = torch.empty(-(dim // -2), dtype=torch.float32, device=device)
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.inv_freq.copy_(self._compute_inv_freq(device=self.inv_freq.device))
            if self.scale_base is not None:
                self.scale.copy_(self._compute_scale(device=self.scale.device))

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"dim={self.dim}, "
        s += f"base={self.base}, "
        s += f"interleaved={self.interleaved}, "
        if self.scale_base is not None:
            s += f"scale_base={self.scale_base}, "
        s += f"pos_idx_in_fp32={self.pos_idx_in_fp32})"
        return s

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _compute_scale(self, device=None):
        return (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) + 0.4 * self.dim) / (1.4 * self.dim)

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        q: [B, T, H, D]
        k: [B, T, H, D]
        seqlen_offset:
            [N] or int.
            Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: [N + 1] or None
        max_seqlen: int
        """
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=q.device, dtype=q.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(q.shape[1] + seqlen_offset, device=q.device, dtype=q.dtype)
        if self.scale is None:
            q = rotary_embedding(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
            )
            k = rotary_embedding(
                k,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
            )

        else:
            q = rotary_embedding(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
            )
            k = rotary_embedding(
                k,
                self._cos_k_cached,
                self._sin_k_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
            )

        return q, k
