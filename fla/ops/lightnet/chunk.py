# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.gla import chunk_gla
from fla.ops.lightnet.gate import fused_lightnet_gate


@torch.compiler.disable
def chunk_lightnet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor | None, torch.Tensor]]:
    """Compute chunkwise LightNet attention with cumulative key normalization.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Key logits of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        scale (float, Optional):
            Attention score scale, defaulting to `1 / sqrt(K)`. Default: `None`.
        initial_state (tuple, Optional):
            FP32 `(recurrent_state, log_normalizer)` with shapes `[N, H, K, V]` and `[N, 1, H, K]`.
            Either component may be `None`. `N` is the number of sequences, equal to `B` for dense inputs. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final recurrent state. Default: `False`.
        state_v_first (bool, Optional):
            Use `[N, H, V, K]` for recurrent states instead of `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for packed inputs with `B=1`. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (tuple):
            `(recurrent_state, log_normalizer)` in the same layouts as `initial_state`.
            The recurrent state is `None` when `output_final_state=False`; the final log-normalizer is always returned.
    """
    h0, z0 = (None, None) if initial_state is None else initial_state
    k, g, zt = fused_lightnet_gate(x=k, initial_state=z0, cu_seqlens=cu_seqlens)
    o, ht = chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=h0,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
    )
    return o, (ht, zt)
