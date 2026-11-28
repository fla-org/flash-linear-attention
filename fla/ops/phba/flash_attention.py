# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"FlashAttention local branches using the joint PHBA outputs and log normalizers."

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from fla.ops.phba.plans import FA_SEGMENT_BATCH_LIMIT, _local_fa_batches
from fla.ops.phba.utils import fail

if TYPE_CHECKING:
    from fla.ops.phba.routing import RoutingLayout

FA_MIN_COMPUTE_CAPABILITY = (8, 0)

# parameter names of the supported flash-attn2.8.3.post1 low-level operations
# forward has neither num_splits nor deterministic
FORWARD_PARAMS = frozenset(
    {
        "q",
        "k",
        "v",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "dropout_p",
        "softmax_scale",
        "causal",
        "window_size_left",
        "window_size_right",
        "softcap",
        "alibi_slopes",
        "return_softmax",
        "block_table",
        "leftpad_k",
        "seqused_k",
        "zero_tensors",
    },
)
BACKWARD_PARAMS = frozenset(
    {
        "dout",
        "q",
        "k",
        "v",
        "out",
        "softmax_lse",
        "dq",
        "dk",
        "dv",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "dropout_p",
        "softmax_scale",
        "causal",
        "window_size_left",
        "window_size_right",
        "softcap",
        "alibi_slopes",
        "deterministic",
        "rng_state",
        "zero_tensors",
    },
)


@dataclass(frozen=True)
class FAOps:
    forward: Callable[..., Any]
    backward: Callable[..., Any]
    forward_signature_source: Callable[..., Any]
    backward_signature_source: Callable[..., Any]


def _load_ops() -> FAOps:
    try:
        from flash_attn import flash_attn_interface as fai
    except Exception as exc:  # noqa: BLE001
        raise fail("fa_layout_probe_failed", f"Cannot import flash_attn.flash_attn_interface: {type(exc).__name__}: {exc}")
    # torch>=2.4 wraps these operations; validate parameter names against the Python definitions
    fwd_src = getattr(fai, "_flash_attn_varlen_forward", None)
    bwd_src = getattr(fai, "_flash_attn_varlen_backward", None)
    if fwd_src is None or bwd_src is None:
        raise fail(
            "fa_layout_probe_failed",
            "flash_attn_interface lacks _flash_attn_varlen_forward / _flash_attn_varlen_backward",
        )
    fwd = getattr(fai, "_wrapped_flash_attn_varlen_forward", fwd_src)
    bwd = getattr(fai, "_wrapped_flash_attn_varlen_backward", bwd_src)
    return FAOps(forward=fwd, backward=bwd, forward_signature_source=fwd_src, backward_signature_source=bwd_src)


def _param_names(fn: Callable[..., Any]) -> frozenset:
    """Read parameter names from the registered schema or the original Python callable."""
    schema = getattr(fn, "_schema", None)
    if schema is not None and hasattr(schema, "arguments"):
        return frozenset(a.name for a in schema.arguments)
    # torch.ops.<ns>.<op> exposes its schema on the default overload
    default = getattr(fn, "default", None)
    schema = getattr(default, "_schema", None)
    if schema is not None and hasattr(schema, "arguments"):
        return frozenset(a.name for a in schema.arguments)
    target = getattr(fn, "_init_fn", None) or inspect.unwrap(fn)
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        raise fail("fa_layout_probe_failed", f"Cannot inspect the signature of {fn!r}: {exc}")
    kinds = {p.kind for p in sig.parameters.values()}
    if inspect.Parameter.VAR_POSITIONAL in kinds or inspect.Parameter.VAR_KEYWORD in kinds:
        raise fail("fa_layout_probe_failed", f"Cannot validate parameter names in the variadic signature of {fn!r}")
    return frozenset(sig.parameters)


def check_schema(forward_fn: Callable[..., Any] | None = None, backward_fn: Callable[..., Any] | None = None) -> None:
    """Reject FlashAttention interfaces that differ from the supported parameter schema."""
    if forward_fn is None or backward_fn is None:
        ops = _load_ops()
        forward_fn = forward_fn or ops.forward_signature_source
        backward_fn = backward_fn or ops.backward_signature_source
    for name, fn, expected in (("forward", forward_fn, FORWARD_PARAMS), ("backward", backward_fn, BACKWARD_PARAMS)):
        actual = _param_names(fn)
        if actual != expected:
            raise fail(
                "fa_layout_probe_failed",
                f"Unsupported FA varlen {name} schema: extra {sorted(actual - expected)}, missing {sorted(expected - actual)}",
                op=name,
            )


def check_device(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        raise fail("fa_device_unsupported", f"FlashAttention requires CUDA, got {device}")
    cap = torch.cuda.get_device_capability(device)
    if cap < FA_MIN_COMPUTE_CAPABILITY:
        raise fail("fa_device_unsupported", f"FlashAttention2 requires sm80 or newer, got compute capability {cap}")
    return cap


def _ops() -> FAOps:
    global _OPS
    if _OPS is None:
        _OPS = _load_ops()
    return _OPS


_OPS: FAOps | None = None


def varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return storage-dtype output [total_q, H, Dh] and FP32 log-normalizers [H, total_q]."""
    out, lse, _s_dmask, _rng = _ops().forward(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        dropout_p=0.0,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
        block_table=None,
        leftpad_k=None,
        seqused_k=None,
        zero_tensors=False,
    )
    return out, lse


def varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    softmax_scale: float,
    causal: bool,
    deterministic: bool,
) -> torch.Tensor:
    """Use the joint PHBA output and contiguous FP32 log-normalizers [H, total_q] for local backward."""
    if not softmax_lse.is_contiguous():
        raise fail(
            "fa_layout_probe_failed",
            "softmax_lse must be contiguous [H, total_q]; make sliced tensors contiguous first",
        )
    return _ops().backward(
        dout=dout,
        q=q,
        k=k,
        v=v,
        out=out,
        softmax_lse=softmax_lse,
        dq=dq,
        dk=dk,
        dv=dv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        dropout_p=0.0,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        deterministic=bool(deterministic),
        rng_state=None,
        zero_tensors=False,
    )


_READY: set[tuple[str, str, int]] = set()
_READY_LOCK = threading.Lock()
_SCHEMA_OK = False


def ensure_ready(device: torch.device, dtype: torch.dtype, head_dim: int) -> None:
    """Validate the device and low-level interface without running numerical probes."""
    global _SCHEMA_OK
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if dtype not in (torch.float16, torch.bfloat16):
        raise fail("dtype_mismatch", f"FlashAttention requires FP16/BF16, got {dtype}")
    if not 16 <= head_dim <= 128 or head_dim % 8:
        raise fail("invalid_shape", f"FlashAttention head dimension must be a multiple of 8 in [16, 128], got {head_dim}")
    key = (str(device), str(dtype), int(head_dim))
    if key in _READY:
        return
    with _READY_LOCK:
        if key in _READY:
            return
        check_device(device)
        if not _SCHEMA_OK:
            check_schema()
            _SCHEMA_OK = True
        _READY.add(key)


def local_backward(dout, q, k, v, joint_out, joint_lse, block_ptr, batches, *, deterministic=False, softmax_scale=None):
    """Use the same block batches as forward, caller-provided joint O/L, storage gradients.

    C++ mha_varlen_bwd requires dq/dk/dv dtype == q.dtype and stride(-1)==1.
    L[:, t0:t1] must be copied: FA indexes L as contiguous [H, batch_total_q].
    """
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    grads = tuple(torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k, v))
    for batch in batches:
        t0, t1 = batch.t0, batch.t1
        cu = block_ptr[batch.b0: batch.b1 + 1] - t0
        varlen_backward(
            dout[t0:t1],
            q[t0:t1],
            k[t0:t1],
            v[t0:t1],
            joint_out[t0:t1],
            joint_lse[:, t0:t1].contiguous(),
            *(g[t0:t1] for g in grads),
            cu,
            cu,
            batch.max_seqlen,
            batch.max_seqlen,
            softmax_scale=scale,
            causal=True,
            deterministic=deterministic,
        )
    return grads


__all__ = [
    "BACKWARD_PARAMS",
    "FORWARD_PARAMS",
    "check_device",
    "check_schema",
    "ensure_ready",
    "varlen_backward",
    "varlen_forward",
]


def local_attention(
    q_rope,
    k_rope,
    v,
    layout: RoutingLayout,
    *,
    scale: float,
    fa_segment_batch_max: int = FA_SEGMENT_BATCH_LIMIT,
    fa_rows_per_call_max: int = 1 << 20,
):
    """Compute causal attention within each block and return storage output with FP32 log-normalizers."""

    c = layout.chunk_size
    batches = _local_fa_batches(layout.block_ptr_host, c, fa_segment_batch_max, fa_rows_per_call_max)
    if len(batches) == 1:
        return varlen_forward(q_rope, k_rope, v, layout.block_ptr, layout.block_ptr, c, c, softmax_scale=scale, causal=True)
    T, H, D = q_rope.shape
    oloc = torch.empty_like(q_rope)
    lloc = torch.empty((H, T), dtype=torch.float32, device=q_rope.device)
    for b in batches:
        # rebase boundaries on device without a host synchronization
        cu_b = layout.block_ptr[b.b0: b.b1 + 1] - b.t0
        o_b, l_b = varlen_forward(
            q_rope[b.t0: b.t1],
            k_rope[b.t0: b.t1],
            v[b.t0: b.t1],
            cu_b,
            cu_b,
            b.max_seqlen,
            b.max_seqlen,
            softmax_scale=scale,
            causal=True,
        )
        oloc[b.t0: b.t1] = o_b
        lloc[:, b.t0: b.t1] = l_b
    return oloc, lloc
