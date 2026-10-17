# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.modules.l2norm import l2norm
from fla.ops.diag_kdn.chunk import MIN_LOG_DECAY
from fla.ops.diag_kdn.gain import diag_kdn_gain
from fla.ops.generalized_delta_rule.dplr import fused_recurrent_dplr_delta_rule
from fla.ops.kda.gate import fused_kda_gate


def fused_recurrent_diag_kdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    scale: float | None = None,
    initial_state: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    output_final_state: bool = False,
    info_scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    r"""
    Forward-only recurrent DiagKDN for inference.
    The arguments and returns follow :func:`fla.ops.diag_kdn.chunk_diag_kdn`.

    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, H, V]``.
        g (torch.Tensor):
            Forget gates (in log space) of shape ``[B, T, H, K]``,
            or the raw gate input if ``use_gate_in_kernel=True``.
        omega (torch.Tensor):
            Positive process noise of shape ``[B, T, H, K]``.
        r (torch.Tensor):
            Positive observation noise of shape ``[B, T, H]``.
        scale (float, Optional):
            Scale factor of the attention scores. Default: `None`, i.e., ``1 / sqrt(K)``.
        initial_state (tuple[torch.Tensor, torch.Tensor], Optional):
            Initial memory of shape ``[N, H, K, V]`` and initial precision of shape ``[N, H, K]``,
            both in ``float32``. Either entry may be `None`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final memory and precision. Default: `False`.
        info_scale (float, Optional):
            Information contributed by one unit-norm key. Default: `None`, i.e., ``K``.
        use_qk_l2norm_in_kernel (bool, Optional):
            Whether to L2-normalize queries and keys before the gain and the memory update. Default: `False`.
        use_gate_in_kernel (bool, Optional):
            Whether to compute the log-space decay ``-exp(A_log) * softplus(g + dt_bias)`` internally.
            Default: `False`.
        A_log (torch.Tensor, Optional):
            Log decay rates of shape ``[H]``, required if ``use_gate_in_kernel=True``. Default: `None`.
        dt_bias (torch.Tensor, Optional):
            Bias of shape ``[H * K]`` added to ``g`` before the activation if ``use_gate_in_kernel=True``.
            Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape ``[N+1]`` used for variable-length inputs. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape ``[B, T, H, V]``.
        final_state (tuple[torch.Tensor, torch.Tensor]):
            Final memory of shape ``[N, H, K, V]`` and precision of shape ``[N, H, K]``
            if ``output_final_state=True`` else `None`.
    """
    # the recurrent kernel does not need the CPU copy of the sequence lengths
    kwargs.pop('cu_seqlens_cpu', None)
    if kwargs:
        raise TypeError(f"Unexpected arguments for DiagKDN: {', '.join(kwargs)}.")
    if use_gate_in_kernel and A_log is None:
        raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True`.")
    assert v.shape[2] == q.shape[2], "DiagKDN does not support grouped value attention."
    h0, c0 = initial_state if initial_state is not None else (None, None)
    if c0 is not None:
        assert c0.dtype == torch.float32, "The initial precision must be in float32."
    if use_qk_l2norm_in_kernel:
        q, k = l2norm(q), l2norm(k, output_dtype=torch.float32)
    if use_gate_in_kernel:
        g = fused_kda_gate(g, A_log, dt_bias)
    kappa, ct = diag_kdn_gain(
        k=k,
        g=g,
        omega=omega,
        r=r,
        initial_state=c0,
        output_final_state=output_final_state,
        info_scale=info_scale,
        cu_seqlens=cu_seqlens,
    )
    # S_t = (Diag(exp(g_t)) - kappa_t (exp(g_t) * k_t)^T) S_{t-1} + kappa_t v_t^T,
    # the recurrent kernel computes in float32, so the gain is passed at full precision
    g = g.clamp_min(MIN_LOG_DECAY)
    o, ht = fused_recurrent_dplr_delta_rule(
        q=q,
        k=kappa,
        v=v,
        a=g.exp() * k,
        b=-kappa,
        gk=g,
        scale=scale,
        initial_state=h0,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return o, ((ht, ct) if output_final_state else None)
