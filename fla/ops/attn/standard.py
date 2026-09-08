# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend-neutral implementation of standard causal attention.

The linear-attention operators in :mod:`fla.ops.attn` have different
semantics from the full attention layer in ``fla.layers.attn``.  This module
therefore provides a small, explicit backend boundary for the latter instead
of routing it through the linear-attention dispatch table.

The public function accepts the layout used by ``fla.layers.attn``:

* dense: ``[B, S, H, D]``;
* packed/varlen: ``[T, H, D]`` plus cumulative sequence lengths.

The fallback implementations intentionally favour semantic correctness.  In
particular, the causal mask is bottom-right aligned when ``S_q != S_kv``;
this is required for decoding with a populated KV cache.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_BACKEND_ENV = "FLA_STANDARD_ATTN_BACKEND"
_VALID_BACKENDS = {"auto", "flash_attn", "npu", "sdpa", "reference"}
_MAX_INT = 2**31 - 1


@lru_cache(maxsize=1)
def _load_flash_attention():
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
    except ImportError:
        return None, None
    return flash_attn_func, flash_attn_varlen_func


@lru_cache(maxsize=1)
def _load_torch_npu():
    try:
        import torch_npu
    except (ImportError, RuntimeError, OSError) as exc:
        logger.debug("Unable to load torch_npu: %s", exc)
        return None
    return torch_npu


def _requested_backend() -> str:
    backend = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"{_BACKEND_ENV}={backend!r} is invalid; expected one of "
            f"{sorted(_VALID_BACKENDS)}"
        )
    return backend


def _has_grad(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


def _can_use_npu(q: torch.Tensor, *, packed: bool, training: bool) -> bool:
    """Return whether the currently installed torch_npu exposes a usable op.

    The NPU implementation deliberately starts with dense BNSD input.  The
    TND/varlen contract differs across CANN and torch_npu releases, so packed
    input stays on the reference/SDPA path until it has a version-specific
    test.  This avoids silently assigning the wrong sequence lengths.
    """
    if not _is_npu_tensor(q) or packed:
        return False
    torch_npu = _load_torch_npu()
    if torch_npu is None:
        return False
    if training:
        return callable(getattr(torch_npu, "npu_fusion_attention", None))
    return callable(getattr(torch_npu, "npu_fused_infer_attention_score", None))


def select_standard_attention_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    packed: bool = False,
    training: bool | None = None,
) -> str:
    """Select and validate a standard-attention backend.

    ``auto`` prefers the native backend for the current device, then CUDA
    FlashAttention, SDPA for dense tensors, and finally the reference path.
    A forced backend fails early with a useful error instead of silently
    changing the requested execution policy.
    """
    requested = _requested_backend()
    if training is None:
        training = _has_grad(q, k, v)
    flash_func, flash_varlen_func = _load_flash_attention()

    if requested == "auto":
        if _can_use_npu(q, packed=packed, training=training):
            return "npu"
        # The initial native path is dense-only.  SDPA support for packed
        # tensors is version-dependent on Ascend, while the reference loop
        # preserves the cu_seqlens contract on every torch_npu release.
        if q.device.type == "npu" and packed:
            return "reference"
        if q.device.type in {"cuda", "hip"} and (flash_varlen_func if packed else flash_func) is not None:
            return "flash_attn"
        if hasattr(F, "scaled_dot_product_attention"):
            return "sdpa"
        return "reference"

    if requested == "flash_attn":
        if q.device.type not in {"cuda", "hip"}:
            raise RuntimeError("flash_attn backend requires a CUDA/HIP tensor")
        if (flash_varlen_func if packed else flash_func) is None:
            raise RuntimeError(
                "FLA_STANDARD_ATTN_BACKEND=flash_attn requires flash-attn; "
                "it is not an Ascend NPU backend"
            )
        return requested

    if requested == "npu":
        if not _is_npu_tensor(q):
            raise RuntimeError("FLA_STANDARD_ATTN_BACKEND=npu requires an NPU tensor")
        if not _can_use_npu(q, packed=packed, training=training):
            mode = "training" if training else "inference"
            raise RuntimeError(
                f"the installed torch_npu does not expose a supported dense {mode} "
                "standard-attention operator for this input"
            )
        return requested

    if requested == "sdpa" and not hasattr(F, "scaled_dot_product_attention"):
        raise RuntimeError("FLA_STANDARD_ATTN_BACKEND=sdpa requires torch SDPA support")

    return requested


def _causal_mask(
    q_len: int,
    kv_len: int,
    *,
    device: torch.device,
    query_start: int | None,
    window_size: int | None,
) -> torch.Tensor:
    """Build a boolean mask where True means masked.

    Query positions are aligned to the end of the KV sequence by default.
    Thus a one-token query over a cached KV sequence can attend to all cached
    tokens instead of only to key position zero.
    """
    if q_len == 0 or kv_len == 0:
        return torch.ones((q_len, kv_len), dtype=torch.bool, device=device)
    if query_start is None:
        query_start = kv_len - q_len
    q_positions = torch.arange(q_len, device=device, dtype=torch.long) + query_start
    k_positions = torch.arange(kv_len, device=device, dtype=torch.long)
    mask = k_positions[None, :] > q_positions[:, None]
    if window_size is not None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        mask |= k_positions[None, :] < q_positions[:, None] - window_size + 1
    return mask


def _expand_kv_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_heads = q.shape[-3]
    kv_heads = k.shape[-3]
    if q_heads % kv_heads != 0:
        raise ValueError(f"query heads ({q_heads}) must be divisible by KV heads ({kv_heads})")
    if q_heads == kv_heads:
        return k, v
    groups = q_heads // kv_heads
    return k.repeat_interleave(groups, dim=-3), v.repeat_interleave(groups, dim=-3)


def _reference_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    window_size: int | None,
    query_start: int | None,
) -> torch.Tensor:
    # [B, S, H, D] -> [B, H, S, D]
    q_bhsd = q.transpose(1, 2)
    k_bhsd = k.transpose(1, 2)
    v_bhsd = v.transpose(1, 2)
    k_bhsd, v_bhsd = _expand_kv_heads(q_bhsd, k_bhsd, v_bhsd)
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * scale
    if causal:
        mask = _causal_mask(
            q.shape[1], k.shape[1], device=q.device, query_start=query_start, window_size=window_size
        )
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    if causal and scores.shape[-1] > 0:
        invalid_rows = mask.all(dim=-1, keepdim=True)
        probs = torch.where(invalid_rows, torch.zeros_like(probs), probs)
    return torch.matmul(probs, v_bhsd).transpose(1, 2)


def _packed_lengths(
    cu_seqlens: torch.Tensor,
    total_length: int,
    name: str,
) -> list[int]:
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"{name} must be a 1-D cumulative sequence-length tensor with at least two entries")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must use int32 or int64 dtype")
    lengths = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
    if lengths[0] != 0:
        raise ValueError(f"{name} must start at 0")
    if any(end < start for start, end in zip(lengths, lengths[1:])):
        raise ValueError(f"{name} must be non-decreasing")
    if lengths[-1] != total_length:
        raise ValueError(f"{name} must end at the packed tensor length ({total_length}), got {lengths[-1]}")
    return lengths


def _validate_packed_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
) -> tuple[list[int], list[int]]:
    q_lens = _packed_lengths(cu_seqlens_q, q.shape[0], "cu_seqlens_q")
    k_lens = _packed_lengths(cu_seqlens_k, k.shape[0], "cu_seqlens_k")
    if len(q_lens) != len(k_lens):
        raise ValueError("query and KV cumulative sequence lengths must have the same batch size")
    if max_seqlen_q is not None and max_seqlen_q < max(end - start for start, end in zip(q_lens, q_lens[1:])):
        raise ValueError("max_seqlen_q is smaller than a packed query sequence")
    if max_seqlen_k is not None and max_seqlen_k < max(end - start for start, end in zip(k_lens, k_lens[1:])):
        raise ValueError("max_seqlen_k is smaller than a packed KV sequence")
    return q_lens, k_lens


def _sdpa_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    window_size: int | None,
    query_start: int | None,
) -> torch.Tensor:
    q_bhsd = q.transpose(1, 2)
    k_bhsd = k.transpose(1, 2)
    v_bhsd = v.transpose(1, 2)
    k_bhsd, v_bhsd = _expand_kv_heads(q_bhsd, k_bhsd, v_bhsd)
    mask = None
    if causal:
        mask = ~_causal_mask(
            q.shape[1], k.shape[1], device=q.device, query_start=query_start, window_size=window_size
        )
    output = F.scaled_dot_product_attention(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return output.transpose(1, 2)


def _reference_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    window_size: int | None,
) -> torch.Tensor:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("packed standard attention expects q, k, v with shape [T, H, D]")
    q_lens = _packed_lengths(cu_seqlens_q, q.shape[0], "cu_seqlens_q")
    k_lens = _packed_lengths(cu_seqlens_k, k.shape[0], "cu_seqlens_k")
    if len(q_lens) != len(k_lens):
        raise ValueError("query and KV cumulative sequence lengths must have the same batch size")
    outputs = []
    for i in range(len(q_lens) - 1):
        q_start, q_end = int(q_lens[i]), int(q_lens[i + 1])
        k_start, k_end = int(k_lens[i]), int(k_lens[i + 1])
        q_i = q[q_start:q_end].unsqueeze(0)
        k_i = k[k_start:k_end].unsqueeze(0)
        v_i = v[k_start:k_end].unsqueeze(0)
        query_start = (k_end - k_start) - (q_end - q_start)
        output = _reference_dense(
            q_i,
            k_i,
            v_i,
            scale=scale,
            causal=causal,
            window_size=window_size,
            query_start=query_start,
        )
        outputs.append(output.squeeze(0))
    if not outputs:
        return q.new_empty((0, q.shape[-2], v.shape[-1]))
    return torch.cat(outputs, dim=0)


def _sdpa_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    window_size: int | None,
) -> torch.Tensor:
    # SDPA does not accept ragged tensors.  The per-sequence loop preserves
    # semantics and is intentionally used only as a correctness fallback.
    q_lens = _packed_lengths(cu_seqlens_q, q.shape[0], "cu_seqlens_q")
    k_lens = _packed_lengths(cu_seqlens_k, k.shape[0], "cu_seqlens_k")
    outputs = []
    for i in range(len(q_lens) - 1):
        q_start, q_end = int(q_lens[i]), int(q_lens[i + 1])
        k_start, k_end = int(k_lens[i]), int(k_lens[i + 1])
        query_start = (k_end - k_start) - (q_end - q_start)
        outputs.append(
            _sdpa_dense(
                q[q_start:q_end].unsqueeze(0),
                k[k_start:k_end].unsqueeze(0),
                v[k_start:k_end].unsqueeze(0),
                scale=scale,
                causal=causal,
                window_size=window_size,
                query_start=query_start,
            ).squeeze(0)
        )
    if not outputs:
        return q.new_empty((0, q.shape[-2], v.shape[-1]))
    return torch.cat(outputs, dim=0)


def _flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    window_size: int | None,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
) -> torch.Tensor:
    flash_func, flash_varlen_func = _load_flash_attention()
    window = (-1, -1) if window_size is None else (window_size - 1, 0)
    kwargs: dict[str, Any] = dict(causal=causal, window_size=window)
    # The layer historically relied on FlashAttention's default scale.  Pass
    # the explicit value so all backends share the same contract.
    kwargs["softmax_scale"] = scale
    if q.ndim == 4:
        if flash_func is None:
            raise RuntimeError("flash-attn is not installed")
        return flash_func(q, k, v, **kwargs)
    if flash_varlen_func is None:
        raise RuntimeError("flash-attn varlen support is not installed")
    if cu_seqlens_q is None or cu_seqlens_k is None or max_seqlen_q is None or max_seqlen_k is None:
        raise ValueError("varlen FlashAttention requires cumulative and maximum sequence lengths")
    return flash_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        **kwargs,
    )


def _npu_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool,
    window_size: int | None,
    query_start: int | None,
) -> torch.Tensor | None:
    if not causal:
        return None
    return _causal_mask(
        q.shape[1], k.shape[1], device=q.device, query_start=query_start, window_size=window_size
    ).contiguous()


def _npu_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    window_size: int | None,
    query_start: int | None,
    training: bool,
) -> torch.Tensor:
    torch_npu = _load_torch_npu()
    if torch_npu is None:
        raise RuntimeError("torch_npu could not be imported; check CANN environment variables")

    q_bnsd = q.transpose(1, 2).contiguous()
    k_bnsd = k.transpose(1, 2).contiguous()
    v_bnsd = v.transpose(1, 2).contiguous()
    q_heads, kv_heads = q.shape[2], k.shape[2]
    mask = _npu_mask(q, k, causal=causal, window_size=window_size, query_start=query_start)
    if mask is not None:
        effective_query_start = k.shape[1] - q.shape[1] if query_start is None else query_start
        if q.shape[1] == 1 and (
            effective_query_start < k.shape[1] - 1
            or (window_size is not None and window_size < k.shape[1])
        ):
            raise RuntimeError(
                "native Ascend inference does not guarantee a masked single-query row; "
                "use the generic backend for this query_start/window"
            )
        if effective_query_start < 0:
            raise RuntimeError(
                "native Ascend standard attention does not guarantee fully masked rows; "
                "use the generic backend when query length exceeds KV length"
            )
    # An explicit mask is needed for bottom-right cache alignment and windows.
    # allMask makes the NPU operator consume exactly this matrix instead of
    # applying a second top-left sparse range from pre/next_tokens.
    sparse_mode = 1 if mask is not None else 0

    if training:
        func = getattr(torch_npu, "npu_fusion_attention", None)
        if not callable(func):
            raise RuntimeError("torch_npu.npu_fusion_attention is unavailable for training")
        result = func(
            q_bnsd,
            k_bnsd,
            v_bnsd,
            q_heads,
            "BNSD",
            atten_mask=mask,
            scale=scale,
            keep_prob=1.0,
            pre_tockens=_MAX_INT,
            next_tockens=_MAX_INT,
            sparse_mode=sparse_mode,
        )
        output = result[0] if isinstance(result, (tuple, list)) else result
    else:
        func = getattr(torch_npu, "npu_fused_infer_attention_score", None)
        if not callable(func):
            raise RuntimeError("torch_npu.npu_fused_infer_attention_score is unavailable for inference")
        result = func(
            q_bnsd,
            k_bnsd,
            v_bnsd,
            atten_mask=mask,
            num_heads=q_heads,
            num_key_value_heads=kv_heads,
            scale=scale,
            pre_tokens=_MAX_INT,
            next_tokens=_MAX_INT,
            input_layout="BNSD",
            sparse_mode=sparse_mode,
        )
        output = result[0] if isinstance(result, (tuple, list)) else result
    return output.transpose(1, 2).contiguous()


def standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    causal: bool = True,
    window_size: int | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    query_start: int | None = None,
    training: bool | None = None,
) -> torch.Tensor:
    """Compute standard attention through the selected backend.

    Args:
        q, k, v: Dense ``[B, S, H, D]`` or packed ``[T, H, D]`` tensors.
        cu_seqlens_q, cu_seqlens_k: Cumulative lengths for packed inputs.
        query_start: Absolute start position of dense queries in the KV
            sequence.  When omitted, bottom-right alignment is inferred.
    """
    if q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError("standard attention expects all inputs to be 3-D packed or 4-D dense tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k and v must have the same dtype")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k head dimensions must match")
    if q.ndim == 4 and (q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]):
        raise ValueError("dense q, k and v must have the same batch size")
    if q.shape[-2] == 0 or k.shape[-2] == 0:
        raise ValueError("q, k and v must have at least one attention head")
    packed = q.ndim == 3
    k_sequence_dim = 0 if packed else 1
    if k.shape[k_sequence_dim] != v.shape[k_sequence_dim] or k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have matching sequence and head dimensions")
    if q.shape[-2] % k.shape[-2] != 0:
        raise ValueError(f"query heads ({q.shape[-2]}) must be divisible by KV heads ({k.shape[-2]})")
    if packed and (cu_seqlens_q is None or cu_seqlens_k is None):
        raise ValueError("packed standard attention requires cu_seqlens_q and cu_seqlens_k")
    if not packed and (cu_seqlens_q is not None or cu_seqlens_k is not None):
        raise ValueError("cumulative sequence lengths are only valid for packed inputs")

    if scale is None:
        scale = q.shape[-1] ** -0.5
    if packed:
        _validate_packed_inputs(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    requested = _requested_backend()
    backend = select_standard_attention_backend(q, k, v, packed=packed, training=training)
    if backend == "flash_attn":
        return _flash_attention(
            q,
            k,
            v,
            scale=scale,
            causal=causal,
            window_size=window_size,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
    if backend == "npu":
        if packed:
            raise RuntimeError("the initial NPU standard-attention path does not accept packed tensors")
        try:
            return _npu_attention(
                q,
                k,
                v,
                scale=scale,
                causal=causal,
                window_size=window_size,
                query_start=query_start,
                training=_has_grad(q, k, v) if training is None else training,
            )
        except (NotImplementedError, RuntimeError):
            if requested != "auto":
                raise
            logger.warning("Ascend native standard attention failed; falling back to a generic backend")
            backend = "sdpa" if hasattr(F, "scaled_dot_product_attention") else "reference"
    if packed:
        if backend == "sdpa":
            return _sdpa_packed(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                scale=scale,
                causal=causal,
                window_size=window_size,
            )
        return _reference_packed(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            scale=scale,
            causal=causal,
            window_size=window_size,
        )
    if backend == "sdpa":
        try:
            return _sdpa_dense(
                q,
                k,
                v,
                scale=scale,
                causal=causal,
                window_size=window_size,
                query_start=query_start,
            )
        except (NotImplementedError, RuntimeError):
            if requested != "auto" or q.device.type != "npu":
                raise
            logger.warning("Ascend SDPA standard attention failed; falling back to reference attention")
    return _reference_dense(
        q,
        k,
        v,
        scale=scale,
        causal=causal,
        window_size=window_size,
        query_start=query_start,
    )


__all__ = ["select_standard_attention_backend", "standard_attention"]
