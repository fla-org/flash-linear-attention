# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.iso_kdn.gain import iso_kdn_gain
from fla.ops.kda import chunk_kda
from fla.ops.kda.gate import fused_kda_gate


def chunk_iso_kdn(
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
    cu_seqlens_cpu: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    r"""
    Isotropic Kalman Delta Network (IsoKDN).
    The memory follows the KDA recurrence, with the write strength ``beta`` given by the isotropic Kalman gain,
    see :func:`fla.ops.iso_kdn.gain.iso_kdn_gain`.
    The gain assumes unit-norm keys, so pass L2-normalized keys or set ``use_qk_l2norm_in_kernel=True``.

    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, HV, V]``.
            GVA (Grouped Value Attention) is applied if ``HV > H``, where ``HV`` must be divisible by ``H``.
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
            and initial precision of shape ``[N, HV]``, both in ``float32``, for ``N`` input sequences.
            Either entry may be `None`, in which case the memory starts at zero and the precision at one.
            Default: `None`.
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
            Cumulative sequence lengths of shape ``[N+1]`` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.
        cu_seqlens_cpu (torch.LongTensor, Optional):
            CPU copy of ``cu_seqlens`` that avoids a device synchronization. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape ``[B, T, HV, V]``.
        final_state (tuple[torch.Tensor, torch.Tensor]):
            Final memory of shape ``[N, HV, K, V]`` and precision of shape ``[N, HV]``
            if ``output_final_state=True`` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.iso_kdn import chunk_iso_kdn
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 128, 128
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, dtype=torch.float, device='cuda'))
        >>> omega = F.softplus(torch.randn(B, T, H, dtype=torch.float, device='cuda'))
        >>> r = F.softplus(torch.randn(B, T, H, dtype=torch.float, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.float, device='cuda')
        >>> c0 = torch.ones(B, H, dtype=torch.float, device='cuda')
        >>> o, (ht, ct) = chunk_iso_kdn(
            q, k, v, g, omega, r,
            initial_state=(h0, c0),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, omega, r = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, omega, r))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, (ht, ct) = chunk_iso_kdn(
            q, k, v, g, omega, r,
            initial_state=(h0, c0),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
    """
    if kwargs.get('cp_context') is not None:
        raise NotImplementedError("Context parallelism is not supported for IsoKDN yet.")
    for key in ('use_beta_sigmoid_in_kernel', 'allow_neg_eigval', 'safe_gate', 'lower_bound',
                'return_intermediate_states'):
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
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    o, ht = chunk_kda(
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
        cu_seqlens_cpu=cu_seqlens_cpu,
        **kwargs,
    )
    return o, ((ht, ct) if output_final_state else None)
