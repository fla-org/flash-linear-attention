# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn as nn
from einops import rearrange

from fla.modules.rotary.ops import rotary_embedding


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
        scale_base: float | None = None,
        interleaved: bool = False,
        pos_idx_in_fp32: bool = True,
        device: torch.device | None = None,
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

        assert dim % 2 == 0, f"Rotary dimension must be even, got {dim}"

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
        seqlen_offset: int | torch.Tensor = 0,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        q: [B, T, H, D]
        k: [B, T, H, D]
        seqlen_offset:
            [N] or int.
            Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: [N + 1] or None
        max_seqlen:
            Cache length used to initialize the rotary tables. Tensor offsets require this on the first call so cache sizing does not
            synchronize the device.
        """
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=q.device, dtype=q.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(q.shape[1] + seqlen_offset, device=q.device, dtype=q.dtype)
        else:
            assert self._cos_cached is not None, "Tensor offsets require an initialized cache; pass max_seqlen on the first call"
        if self.scale is None:
            q = rotary_embedding(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
            k = rotary_embedding(
                k,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )

        else:
            q = rotary_embedding(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
            k = rotary_embedding(
                k,
                self._cos_k_cached,
                self._sin_k_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )

        return q, k
