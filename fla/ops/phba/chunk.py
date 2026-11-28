# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from fla.ops.phba.flash_attention import ensure_ready
from fla.ops.phba.parallel import BlockMajorConfig, block_major_attention
from fla.ops.phba.plans import CHUNKWISE_VALIDATED_C, MAX_KH, SequenceLengthPlan, _resolve_length_plan
from fla.ops.phba.prefix import chunkwise_prefix_states, gate_algebra
from fla.ops.phba.routing import build_routes
from fla.utils import input_guard


@input_guard
def chunk_phba(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    gate: torch.Tensor,
    chunk_size: int = 128,
    topk: int = 8,
    gate_logit_normalizer: float = 8.0,
    cu_seqlens: torch.Tensor | None = None,
    length_plan: SequenceLengthPlan | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Compute causal block-routed attention with gated prefix-state compensation.

    Routing is nondifferentiable. All six floating inputs support first-order gradients.
    Head dimensions must be multiples of 8 from 16 through 128; slot counts are 1–128.

    Args:
        q (torch.Tensor):
            Raw queries [B, T, H, D] for prefix-state attention, in CUDA FP16/BF16 storage.
        k (torch.Tensor):
            Raw keys [B, T, H, D] used to build prefix states.
        v (torch.Tensor):
            Values [B, T, H, D] shared by local, historical and prefix branches.
        q_rope (torch.Tensor):
            Rotary queries [B, T, H, D] for routing and token attention.
        k_rope (torch.Tensor):
            Rotary keys [B, T, H, D] for routing and token attention.
        gate (torch.Tensor):
            Gate logits [B, T, H, M], with the same device and storage dtype as q.
        chunk_size (int, Optional):
            Block length, one of 32, 64, 128, 256, 512. Default: 128.
        topk (int, Optional):
            Number of selected blocks including the current block, from 1 through 16. Default: 8.
        gate_logit_normalizer (float, Optional):
            Finite positive gate-log normalization factor. Default: 8.0.
        cu_seqlens (torch.Tensor, Optional):
            Int32 packed boundaries on the input device; packed inputs require B=1. Default: `None`.
        length_plan (SequenceLengthPlan, Optional):
            Owned sequence geometry. Dense B>1 requires lengths (T,)*B.
            If cu_seqlens is also supplied, its values must agree with this plan. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return final prefix states. Default: `False`.

    Returns:
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
            Output [B, T, H, D] and optional nondifferentiable FP32 key/value states [Nseq, H, M, D].
    """
    if q.ndim != 4 or any(x.shape != q.shape for x in (k, v, q_rope, k_rope)):
        raise ValueError("PHBA q, k, v and rotary inputs must have the same [B, T, H, D] shape")
    if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("PHBA requires CUDA FP16 or BF16 storage tensors")
    if any(x.device != q.device or x.dtype != q.dtype for x in (k, v, q_rope, k_rope, gate)):
        raise ValueError("PHBA floating-point inputs must have the same device and storage dtype")
    batch, tokens, heads, width = q.shape
    if min(batch, tokens, heads) < 1 or not 16 <= width <= 128 or width % 8:
        raise ValueError("PHBA requires nonempty inputs and head dimensions divisible by 8 from 16 through 128")
    if gate.ndim != 4 or gate.shape[:3] != q.shape[:3] or not 1 <= gate.shape[-1] <= 128:
        raise ValueError("PHBA gate logits must have shape [B, T, H, M], with 1 <= M <= 128")
    if chunk_size not in CHUNKWISE_VALIDATED_C or not 1 <= topk <= MAX_KH + 1:
        raise ValueError("PHBA chunk_size must be 32, 64, 128, 256 or 512; topk must be between 1 and 16")
    if not math.isfinite(gate_logit_normalizer) or gate_logit_normalizer <= 0:
        raise ValueError("gate_logit_normalizer must be finite and positive")
    length_plan = _resolve_length_plan(batch, tokens, q.device, cu_seqlens, length_plan)
    ensure_ready(q.device, q.dtype, width)
    q, k, v, q_rope, k_rope = (x.reshape(-1, heads, width) for x in (q, k, v, q_rope, k_rope))
    gate = gate.reshape(-1, heads, gate.shape[-1])
    layout, token_route, state_route = build_routes(
        q_rope=q_rope,
        k_rope=k_rope,
        length_plan=length_plan,
        chunk_size=chunk_size,
        topk=topk,
    )
    g_log, s = gate_algebra(F.logsigmoid(gate.float()), q.dtype, gate_logit_normalizer)
    prefix_k, prefix_v, final_k, final_v = chunkwise_prefix_states(
        s_val=s,
        k_raw=k,
        v=v,
        g_log=g_log,
        layout=length_plan.block_layout(chunk_size),
        output_final_state=output_final_state,
    )
    output, _ = block_major_attention(
        qr=q_rope,
        qw=q,
        kr=k_rope,
        v=v,
        pk=prefix_k,
        pv=prefix_v,
        rtok=token_route,
        rstate=state_route,
        layout=layout,
        config=BlockMajorConfig(),
    )
    final = (final_k, final_v) if output_final_state else None
    return output.reshape(batch, tokens, heads, width), final
