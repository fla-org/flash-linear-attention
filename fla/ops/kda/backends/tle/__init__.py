# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 FlagOS Contributors

"""TLE inference backend for KDA."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fla.ops.backends import BaseBackend

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


class KDATLEBackend(BaseBackend):
    """TLE forward backend for inference-only chunk_kda.

    The kernel fuses q/k L2 norm, beta sigmoid, and the KDA safe gate. Compared
    with the FlashKDA CUDA backend, this path also supports fp16, GVA, flexible
    state layout, K in {64, 128, 192, 256}, and arbitrary positive V.
    """

    backend_type = "tle"
    package_name = "triton"
    env_var = "FLA_TLE"
    default_enable = True
    # Higher priority than FlashKDA (3); the verifier falls back automatically
    # when a call does not match TLE's inference-only contract.
    priority = 2

    @classmethod
    def is_available(cls) -> bool:
        try:
            import triton.experimental.tle.language  # noqa: F401
            return True
        except ImportError:
            return False

    def chunk_kda_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if torch.is_grad_enabled():
            return False, "TLE backend only supports inference mode"
        if return_intermediate_states:
            return False, "TLE backend does not support return_intermediate_states"
        chunk_size = kwargs.get("chunk_size", 16)
        if chunk_size != 16:
            return False, f"TLE backend requires chunk_size=16, got {chunk_size}"
        supported_dtypes = (torch.bfloat16, torch.float16)
        for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
            if tensor.dtype not in supported_dtypes:
                return False, f"TLE backend requires {name} dtype to be bf16 or fp16, got {tensor.dtype}"
        K = q.shape[-1]
        H = q.shape[2]
        HV = v.shape[2]
        V = v.shape[-1]
        if K not in (64, 128, 192, 256):
            return False, f"TLE backend requires K in {{64, 128, 192, 256}}, got {K}"
        if H <= 0:
            return False, f"TLE backend requires H > 0, got {H}"
        if V <= 0:
            return False, f"TLE backend requires V > 0, got {V}"
        if k.shape[2] != H:
            return False, f"TLE backend requires k heads to match q heads, got H={H}, HK={k.shape[2]}"
        if k.shape[-1] != K or g.shape[-1] != K:
            return False, f"TLE backend requires q/k/g to share K={K}, got HK={k.shape[-1]}, KG={g.shape[-1]}"
        if g.shape[2] != HV or beta.shape[2] != HV:
            return False, f"TLE backend requires v/g/beta to share HV={HV}, got HG={g.shape[2]}, HB={beta.shape[2]}"
        if HV < H or HV % H != 0:
            return False, f"TLE backend requires HV % H == 0 for GVA, got H={H}, HV={HV}"
        if initial_state is not None:
            N = len(cu_seqlens) - 1 if cu_seqlens is not None else q.shape[0]
            expected_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
            if tuple(initial_state.shape) != expected_shape:
                return False, f"TLE backend requires initial_state shape {expected_shape}, got {tuple(initial_state.shape)}"
        if cp_context is not None:
            return False, "TLE backend does not support context parallel"
        if not use_qk_l2norm_in_kernel:
            return False, "TLE backend requires use_qk_l2norm_in_kernel=True"
        if not use_gate_in_kernel:
            return False, "TLE backend requires use_gate_in_kernel=True"
        A_log = kwargs.get("A_log")
        dt_bias = kwargs.get("dt_bias")
        if A_log is None:
            return False, "TLE backend requires A_log"
        if A_log.numel() != HV:
            return False, f"TLE backend requires A_log.numel() == HV={HV}, got {A_log.numel()}"
        if dt_bias is None:
            return False, "TLE backend requires dt_bias"
        if dt_bias.numel() != HV * K:
            return False, f"TLE backend requires dt_bias.numel() == HV*K={HV * K}, got {dt_bias.numel()}"
        if not use_beta_sigmoid_in_kernel:
            return False, "TLE backend requires use_beta_sigmoid_in_kernel=True"
        if allow_neg_eigval:
            return False, "TLE backend does not support allow_neg_eigval=True"
        if lower_bound is None:
            return False, "TLE backend requires lower_bound"
        return True, None

    def chunk_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla.ops.kda.backends.tle.chunk_fwd import chunk_kda_fwd_infer
        return chunk_kda_fwd_infer(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            A_log=kwargs.get("A_log"),
            dt_bias=kwargs.get("dt_bias"),
        )
