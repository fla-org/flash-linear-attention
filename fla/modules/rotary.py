# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn as nn
from einops import rearrange

from fla.ops.rotary import rotary_embedding


class RotaryEmbedding(nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    Queries and keys are transformed by rotation matrices that depend on their relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in GPT-NeoX_, which inspired this implementation.

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96.
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
        """Initialize rotary position tables.

        Args:
            dim (int):
                Number of dimensions to rotate in each head.
            base (float, Optional):
                Base for the inverse frequencies. Default: 10000.0.
            interleaved (bool, Optional):
                Whether to rotate adjacent pairs (GPT-J) instead of matching the two halves (GPT-NeoX). Default: `False`.
            scale_base (float, Optional):
                Base for XPos scaling. Default: `None`.
            pos_idx_in_fp32 (bool, Optional):
                Whether to create position indices in fp32. Default: `True`.
                Pure bf16 indices can round distinct positions to the same value (for example, 1995.0 to 2000.0).
                Set to `False` for compatibility with models trained with bf16 position indices.
            device (torch.device, Optional):
                Device for the frequency buffers. Default: `None`.
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
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

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
        """Rotate query and key tensors using cached position tables.

        Args:
            q (torch.Tensor):
                Query tensor of shape `[B, T, H, D]`.
            k (torch.Tensor):
                Key tensor of shape `[B, T, H_k, D]`.
            seqlen_offset (int | torch.Tensor, Optional):
                Shared position offset or per-sequence offsets of shape `[N]`. Default: 0.
            cu_seqlens (torch.Tensor, Optional):
                Cumulative sequence lengths of shape `[N + 1]` for packed inputs. Default: `None`.
            max_seqlen (int, Optional):
                Cache length used to initialize the rotary tables. Default: `None`.
                Tensor offsets require this on the first call so cache sizing does not synchronize the device.
            chunk_indices (torch.LongTensor, Optional):
                Precomputed sequence chunk indices. Default: `None`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                Rotated query and key tensors.
        """
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=q.device, dtype=q.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(q.shape[1] + seqlen_offset, device=q.device, dtype=q.dtype)
        else:
            assert self._cos_cached is not None, (
                "Tensor offsets require an initialized cache; pass max_seqlen on the first call"
            )
        if self.scale is None:
            q = rotary_embedding(
                x=q,
                cos=self._cos_cached,
                sin=self._sin_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
            k = rotary_embedding(
                x=k,
                cos=self._cos_cached,
                sin=self._sin_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )

        else:
            q = rotary_embedding(
                x=q,
                cos=self._cos_cached,
                sin=self._sin_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
            k = rotary_embedding(
                x=k,
                cos=self._cos_k_cached,
                sin=self._sin_k_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )

        return q, k


__all__ = ['RotaryEmbedding', 'rotary_embedding']
