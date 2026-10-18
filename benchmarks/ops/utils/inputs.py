# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Tensor specifications and input construction for operator benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from benchmarks.ops.utils.registry import OpConfig


RWKV7_W_MIN = -0.6065306597126334


def shape_bthd(B, T, H, D, **kw):
    return (B, T, H, D)


def shape_bth(B, T, H, D, **kw):
    return (B, T, H)


def shape_btd(B, T, H, D, **kw):
    return (B, T, H * D)


def shape_h(B, T, H, D, **kw):
    return (H,)


def shape_hd(B, T, H, D, **kw):
    return (H, D)


def shape_d(B, T, H, D, **kw):
    return (D,)


def shape_lbtd(B, T, H, D, L=None, **kw):
    """AttnRes-style residuals stack: [L, B, T, D] where L is the number of residual sources."""
    if L is None:
        raise ValueError("shape_lbtd requires the 'L' shape config key")
    return (L, B, T, D)


def shape_q_hq(B, T, H, D, HQ=None, **kw):
    """q with HQ query heads (GQA); k/v keep H heads. Used by NSA."""
    if HQ is None:
        raise ValueError("shape_q_hq requires the 'HQ' (query-head) shape config key")
    return (B, T, HQ, D)


def sigmoid_transform(t):
    return t.sigmoid()


def logsigmoid_clamp(t):
    return F.logsigmoid(t).clamp_min(-5)


def rwkv7_w_transform(t):
    w = RWKV7_W_MIN * t.sigmoid()
    return w.clamp(min=RWKV7_W_MIN, max=-1e-6)


@dataclass
class TensorSpec:
    """Specification for a benchmark tensor.

    Args:
        shape_fn (Callable):
            Return the tensor shape from the workload dimensions.
        requires_grad (bool, Optional):
            Whether floating-point inputs require gradients. Default: `True`.
        dtype (str, Optional):
            Use 'float32', 'long', or inherit the workload dtype. Default: 'default'.
        transform (Callable, Optional):
            Transform the sampled tensor before enabling gradients. Default: `None`.
    """
    shape_fn: Callable
    requires_grad: bool = True
    dtype: str = 'default'
    transform: Callable | None = None


def generate_inputs(
    config: OpConfig,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = 'cuda',
    **extra_shape_kw,
) -> dict[str, Any]:
    """Create operator arguments, raising ValueError for unsupported dimensions."""
    if config.dim_constraints:
        shape_vals = {'B': B, 'T': T, 'H': H, 'D': D, **extra_shape_kw}
        for dim_name, allowed in config.dim_constraints.items():
            val = shape_vals.get(dim_name)
            if val is not None and val not in allowed:
                raise ValueError(
                    f"Op '{config.name}' requires {dim_name} in {allowed}, got {val}"
                )

    inputs = {}
    for param_name, spec in config.inputs.items():
        shape = spec.shape_fn(B=B, T=T, H=H, D=D, **extra_shape_kw)

        if spec.dtype == 'default':
            tensor_dtype = dtype
        elif spec.dtype == 'float32':
            tensor_dtype = torch.float32
        elif spec.dtype == 'long':
            tensor_dtype = torch.long
        else:
            tensor_dtype = dtype

        if tensor_dtype == torch.long:
            tensor = torch.randint(0, 10, shape, dtype=tensor_dtype, device=device)
        else:
            tensor = torch.randn(shape, dtype=tensor_dtype, device=device)

        if spec.transform is not None:
            tensor = spec.transform(tensor)

        if spec.requires_grad and tensor.is_floating_point():
            tensor = tensor.requires_grad_(True)

        inputs[param_name] = tensor

    if config.post_init is not None:
        config.post_init(inputs=inputs, B=B, T=T, H=H, D=D, **extra_shape_kw)

    return inputs


def shape_cyfa_readout(B, T, H, D, M=128, **kw):
    return (H, M - 1, M - 1)


def init_cyfa_inputs(inputs, **kwargs):
    with torch.no_grad():
        inputs['q_norm_weight'].fill_(1)
        inputs['k_norm_weight'].fill_(1)
        readout = inputs['readout']
        readout.copy_(torch.eye(readout.shape[-1], dtype=readout.dtype, device=readout.device))


def init_rwkv7_inputs(inputs, B, T, H, D, **kw):
    """RWKV7 needs a/b to be initialized as small positive values."""
    with torch.no_grad():
        inputs['a'] = (torch.randn_like(inputs['a']) * 0.1).requires_grad_(True)
        inputs['b'] = (torch.randn_like(inputs['b']) * 0.1).requires_grad_(True)


def init_attnres_inputs(inputs, B, T, H, D, L=None, **kw):
    # fused_attnres expects a sequence of [B, T, D] tensors, not the stacked [L, B, T, D] buffer.
    res = inputs['residuals']
    if isinstance(res, torch.Tensor) and res.ndim == 4:
        inputs['residuals'] = [
            res[i].detach().clone().requires_grad_(True) for i in range(res.shape[0])
        ]


def shape_nsa_gate(B, T, H, D, HQ=None, **kw):
    if HQ is None:
        raise ValueError("NSA gate shape requires HQ")
    return (B, T, HQ)


def shape_nsa_compressed_k(B, T, H, D, block_size=64, **kw):
    return (B, (T + block_size - 1) // block_size, H, D)


def shape_nsa_compressed_v(B, T, H, D, V=None, block_size=64, **kw):
    if V is None:
        raise ValueError("NSA value shape requires V")
    return (B, (T + block_size - 1) // block_size, H, V)


def shape_nsa_v(B, T, H, D, V=None, **kw):
    if V is None:
        raise ValueError("NSA value shape requires V")
    return (B, T, H, V)


def init_nsa_compression_inputs(inputs, B, T, H, D, block_size=64, **kw):
    inputs['TK'] = T
    inputs['block_size'] = block_size
    inputs['scale'] = D**-0.5


def init_nsa_windowed_inputs(inputs, B, T, H, D, S=16, block_size=64, window_size=512, **kw):
    inputs['block_counts'] = S
    inputs['block_size'] = block_size
    inputs['scale'] = D**-0.5
    inputs['window_size'] = window_size


def init_nsa_inputs(inputs, B, T, H, D, HQ=None, S=16, block_size=64, **kw):
    # build block_indices [B, T, H, S]: for each query t, pick S of the causal
    # blocks (block i is selectable iff i <= t // block_size), -1-padded when
    # fewer than S exist, then sorted — mirroring tests/ops/test_nsa.py.
    device = inputs['q'].device
    NS = (T + block_size - 1) // block_size
    valid = (torch.arange(NS, device=device)[None, None, None, :]
             <= (torch.arange(T, device=device) // block_size)[None, :, None, None])
    scores = torch.rand(B, T, H, NS, device=device).masked_fill(~valid, float('-inf'))
    topv, topi = scores.topk(min(S, NS), dim=-1)
    block_indices = topi.masked_fill(topv == float('-inf'), -1).sort(-1)[0].to(torch.long)
    inputs['block_indices'] = block_indices.contiguous()
    inputs['block_counts'] = S
    inputs['block_size'] = block_size
