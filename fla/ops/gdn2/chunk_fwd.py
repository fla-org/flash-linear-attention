# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Forward orchestration for GDN-2 chunkwise training.

import torch

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.gdn2.chunk_intra import chunk_gdn2_fwd_intra
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2


def chunk_gdn2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
):
    """Top-level GDN-2 forward pipeline.

    The pipeline is:
      1. Compute the base-2 log-decay cumsum within each chunk
         (``kda_gate_chunk_cumsum`` if ``use_gate_in_kernel`` else
         ``chunk_local_cumsum``).
      2. Build the intra-chunk score matrices (Aqk, Akk_inv) and the WY
         auxiliaries (w_wy, u_wy, qg, kg) via ``chunk_gdn2_fwd_intra``.
      3. Run the inter-chunk state recurrence (shared with KDA / GDN v1).
      4. Compose the output via the GLA-style ``chunk_gla_fwd_o_gk``.

    Returns ``(o, final_state, g_cumsum, Aqk, Akk, w_wy, u_wy, qg, kg, v_new,
    h, initial_state)``.
    """
    if use_gate_in_kernel:
        g = kda_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )
    else:
        g = chunk_local_cumsum(
            g=g,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    w_wy, u_wy, qg, kg, Aqk, Akk = chunk_gdn2_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        b=b,
        w_gate=w_gate,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute,
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w_wy,
        u=u_wy,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        state_v_first=state_v_first,
    )

    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )

    if disable_recompute is False:
        # Free intermediates that the backward will recompute.
        w_wy, u_wy, qg, kg, v_new = None, None, None, None, None
        if not return_intermediate_states:
            h = None
        if use_gate_in_kernel:
            g = None
    return o, final_state, g, Aqk, Akk, w_wy, u_wy, qg, kg, v_new, h, initial_state
