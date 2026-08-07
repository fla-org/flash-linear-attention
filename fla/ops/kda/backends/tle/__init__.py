# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Copyright 2026, The FlagOS Contributors.

"""TLE backend for KDA chunk inference (BT=16, TMA + warp-specialized)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from fla.ops.backends import BaseBackend
from fla.utils import IS_NVIDIA_HOPPER

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


def _has_tle() -> bool:
    try:
        import triton
        from packaging.version import Version
        ver = Version(triton.__version__.split("+")[0])
        if ver < Version("3.6.0"):
            return False
        import triton.experimental.tle.language  # noqa: F401
        return True
    except Exception:
        return False


_TLE_AVAILABLE = _has_tle()


def _tle_input_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    state_v_first: bool,
    cu_seqlens: torch.Tensor | None,
    safe_gate: bool,
    lower_bound: float | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
) -> str | None:
    inputs = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    if any(x.dtype != torch.bfloat16 for x in inputs.values()):
        actual = ", ".join(f"{name}={x.dtype}" for name, x in inputs.items())
        return f"TLE KDA requires bfloat16 inputs, got {actual}"
    if any(not x.is_cuda for x in inputs.values()):
        return "TLE KDA requires CUDA inputs"
    if any(x.device != q.device for x in inputs.values()):
        return "TLE KDA requires all inputs on the same device"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or g.ndim != 4 or beta.ndim != 3:
        return "TLE KDA expects q/k/v/g with rank 4 and beta with rank 3"

    B, T, H, D = q.shape
    if T == 0:
        return "TLE KDA requires a non-empty sequence"
    if D != 128:
        return f"TLE KDA requires K=128, got {D}"
    if k.shape != q.shape or v.shape != q.shape or g.shape != q.shape:
        return "TLE KDA requires q, k, v, and g to share shape [B, T, H, 128]"
    if beta.shape != (B, T, H):
        return f"TLE KDA requires beta shape {(B, T, H)}, got {tuple(beta.shape)}"

    if not state_v_first:
        return "TLE KDA requires state_v_first=True"
    if not safe_gate:
        return "TLE KDA requires safe_gate=True"
    if lower_bound is None or not -5 <= lower_bound < 0:
        return f"TLE KDA requires -5 <= lower_bound < 0, got {lower_bound}"

    if A_log is None or A_log.dtype != torch.float32 or A_log.shape != (H,):
        actual = None if A_log is None else (tuple(A_log.shape), A_log.dtype)
        return f"TLE KDA requires float32 A_log with shape {(H,)}, got {actual}"
    if A_log.device != q.device:
        return "TLE KDA requires A_log on the input device"
    if dt_bias is not None:
        if dt_bias.dtype != torch.float32 or dt_bias.shape not in ((H * D,), (H, D)):
            actual = (tuple(dt_bias.shape), dt_bias.dtype)
            return f"TLE KDA requires float32 dt_bias with shape {(H * D,)} or {(H, D)}, got {actual}"
        if dt_bias.device != q.device:
            return "TLE KDA requires dt_bias on the input device"

    N = B
    if cu_seqlens is not None:
        if B != 1:
            return "TLE KDA requires B=1 when cu_seqlens is provided"
        if (
            cu_seqlens.device != q.device
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.ndim != 1
        ):
            return "TLE KDA requires a 1D int32 or int64 cu_seqlens tensor on the input device"
        if cu_seqlens.numel() < 2:
            return "TLE KDA requires cu_seqlens to contain at least two elements"
        N = cu_seqlens.numel() - 1

    if initial_state is not None:
        expected = (N, H, D, D)
        if initial_state.dtype != torch.float32 or initial_state.shape != expected:
            return f"TLE KDA requires float32 initial_state with shape {expected}"
        if initial_state.device != q.device:
            return "TLE KDA requires initial_state on the input device"
    return None


class KDATLEBackend(BaseBackend):
    """TLE-accelerated KDA chunk forward (inference only).

    Uses TMA + warp-specialized fused kernels with BT=16.
    Requires Triton >= 3.6.0 with TLE extension.
    """

    backend_type = "tle"
    package_name = None
    env_var = "FLA_TLE_KDA"
    default_enable = True
    priority = 2

    @classmethod
    def is_available(cls) -> bool:
        return _TLE_AVAILABLE and IS_NVIDIA_HOPPER

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
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if torch.is_grad_enabled():
            return False, "TLE KDA only supports inference mode"
        if not use_gate_in_kernel:
            return False, "TLE KDA requires use_gate_in_kernel=True"
        if not use_qk_l2norm_in_kernel:
            return False, "TLE KDA requires use_qk_l2norm_in_kernel=True"
        if not use_beta_sigmoid_in_kernel:
            return False, "TLE KDA requires use_beta_sigmoid_in_kernel=True"
        if allow_neg_eigval:
            return False, "TLE KDA does not support allow_neg_eigval=True"
        if cp_context is not None:
            return False, "TLE KDA does not support context parallel"
        if return_intermediate_states:
            return False, "TLE KDA does not support return_intermediate_states"
        if "transpose_state_layout" in kwargs:
            if state_v_first:
                return False, "Cannot pass both state_v_first and transpose_state_layout"
            state_v_first = kwargs["transpose_state_layout"]
        chunk_size = kwargs.get("chunk_size", 64)
        if chunk_size not in (32, 64):
            return False, f"chunk_size must be either 32 or 64, got {chunk_size}"
        reason = _tle_input_error(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            A_log=kwargs.get("A_log"),
            dt_bias=kwargs.get("dt_bias"),
        )
        return reason is None, reason

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
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ):
        from fla.ops.kda.backends.tle.chunk_kda import chunk_kda_fwd_infer

        if "transpose_state_layout" in kwargs:
            if state_v_first:
                raise ValueError("Cannot pass both `state_v_first` and the deprecated `transpose_state_layout`.")
            warnings.warn(
                "`transpose_state_layout` is deprecated and renamed to `state_v_first`.",
                DeprecationWarning,
                stacklevel=2,
            )
            state_v_first = kwargs.pop("transpose_state_layout")

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
