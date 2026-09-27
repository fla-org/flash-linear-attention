# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.iso_kdn.gain import iso_kdn_gain
from fla.ops.kda import fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate


def fused_recurrent_iso_kdn(
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
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    r"""
    Forward-only recurrent IsoKDN for inference.
    The arguments and returns follow :func:`fla.ops.iso_kdn.chunk_iso_kdn`.

    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, HV, V]``. GVA is applied if ``HV > H``.
        g (torch.Tensor):
            Forget gates (in log space) of shape ``[B, T, HV, K]``,
            or the raw gate input if ``use_gate_in_kernel=True``.
        omega (torch.Tensor):
            Positive process noise of shape ``[B, T, HV]``.
        r (torch.Tensor):
            Positive observation noise of shape ``[B, T, HV]``.
        scale (float, Optional):
            Scale factor of the attention scores. Default: `None`, i.e., ``1 / sqrt(K)``.
        initial_state (tuple[torch.Tensor, torch.Tensor], Optional):
            Initial memory of shape ``[N, HV, K, V]`` (``[N, HV, V, K]`` if ``state_v_first=True``)
            and initial precision of shape ``[N, HV]`` in ``float32``. Either entry may be `None`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final memory and precision. Default: `False`.
        info_scale (float, Optional):
            Information contributed by one unit-norm key, relative to ``K``. Default: `None`, i.e., ``K``.
        use_qk_l2norm_in_kernel (bool, Optional):
            Whether to L2-normalize queries and keys in the kernel. Default: `False`.
        use_gate_in_kernel (bool, Optional):
            Whether to compute the log-space decay ``-exp(A_log) * softplus(g + dt_bias)`` internally.
            Default: `False`.
        A_log (torch.Tensor, Optional):
            Log decay rates of shape ``[HV]``, required if ``use_gate_in_kernel=True``. Default: `None`.
        dt_bias (torch.Tensor, Optional):
            Bias of shape ``[HV * K]`` added to ``g`` before the activation if ``use_gate_in_kernel=True``.
            Default: `None`.
        state_v_first (bool, Optional):
            Store the memory in V-first ``[V, K]`` layout instead of ``[K, V]``. Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape ``[N+1]`` used for variable-length inputs. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape ``[B, T, HV, V]``.
        final_state (tuple[torch.Tensor, torch.Tensor]):
            Final memory of shape ``[N, HV, K, V]`` and precision of shape ``[N, HV]``
            if ``output_final_state=True`` else `None`.
    """
    for key in ('use_beta_sigmoid_in_kernel', 'allow_neg_eigval', 'safe_gate', 'lower_bound'):
        if kwargs.pop(key, False):
            raise ValueError(f"`{key}` is not supported by IsoKDN.")
    if use_gate_in_kernel and A_log is None:
        raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True`.")
    h0, c0 = initial_state if initial_state is not None else (None, None)
    if c0 is not None:
        assert c0.dtype == torch.float32, "The initial precision must be in float32."
    # the gain takes the activated decay, which the KDA kernels recompute from the raw input
    beta, ct = iso_kdn_gain(
        g=fused_kda_gate(g, A_log, dt_bias) if use_gate_in_kernel else g,
        omega=omega,
        r=r,
        initial_state=c0,
        output_final_state=output_final_state,
        info_scale=info_scale,
        cu_seqlens=cu_seqlens,
    )
    o, ht = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        **kwargs,
    )
    return o, ((ht, ct) if output_final_state else None)
