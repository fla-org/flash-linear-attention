# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Precond-KDA fused-recurrent entry adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch

from fla.ops.precond_kda.fused_recurrent import fused_recurrent_precond_kda_fwd


def fused_recurrent_precond_kda_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_atk: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor = None,
    initial_A_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
    **kwargs,
):
    # Mirrors the mainline public entry: scale default + varlen validation.
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    # Mirrors the mainline entry; the undecorated stage is called directly
    # because the @dispatch-wrapped public entry would recurse.
    return fused_recurrent_precond_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        g_atk=g_atk,
        beta_atk=beta_atk,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        initial_A_state=initial_A_state,
        inplace_final_state=False,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        transpose_state_layout=transpose_state_layout,
        x=x,
        eps=eps,
        log_atk_scale=log_atk_scale,
    )
