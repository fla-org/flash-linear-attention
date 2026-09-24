# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Precond-KDA chunk entry adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import warnings

import torch

from fla.ops.precond_kda.chunk import ChunkPrecondKDAFunction


def chunk_precond_kda_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_atk: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_A_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    cp_context=None,
    transpose_state_layout: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
    solve_tril_precision: str | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    **kwargs,
):
    # Mirrors the validation in the mainline public entry (cp not supported).
    if cp_context is not None:
        raise NotImplementedError(
            "Context parallelism is not yet supported for chunk_precond_kda: "
            "the preconditioned path lacks the CP state compression/expansion plumbing. "
            "Please run without `cp_context`."
        )

    # Ascend-specific precision mapping (mainline stays platform-neutral).
    # tf32/tf32x3 are NVIDIA-only; triton-ascend accepts 'ieee'/'hf32' only.
    # Follow the intracard precedent: warn, then fall back — don't silently
    # rewrite, and don't swallow unrelated typos (those pass through and
    # surface at compile time).
    if solve_tril_precision in ('tf32', 'tf32x3'):
        warnings.warn(
            f"solve_tril_precision '{solve_tril_precision}' is NVIDIA-only; "
            "falling back to 'hf32' on Ascend NPU",
            stacklevel=2,
        )
        solve_tril_precision = 'hf32'

    if scale is None:
        scale = k.shape[-1] ** -0.5

    # Prepare log_atk_scale with default if needed
    H = k.shape[2]
    if log_atk_scale is None:
        log_atk_scale = torch.full((H,), -0.2, device=k.device, dtype=torch.float32)

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")

    # Call the undecorated autograd Function directly: the @dispatch-wrapped
    # public entry would re-enter dispatch and recurse.
    results = ChunkPrecondKDAFunction.apply(
        q, k, v, g, g_atk, beta_atk, beta,
        scale,
        initial_state,
        initial_A_state,
        output_final_state,
        True,   # use_qk_l2norm_in_kernel (always True)
        use_gate_in_kernel,
        A_log,
        dt_bias,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_indices,
        safe_gate,
        cp_context,
        transpose_state_layout,
        x,
        eps,
        log_atk_scale,
        lower_bound,
        solve_tril_precision,
        disable_recompute,
        return_intermediate_states,
    )

    if return_intermediate_states:
        # returns (o, final_state, at, h)
        return results

    o, h_final, a_final = results
    return o, h_final, a_final
