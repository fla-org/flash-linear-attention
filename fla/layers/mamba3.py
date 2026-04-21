# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Port of Mamba-3 (Dao AI Lab, Goombalab) into the FLA layout.
# Reference: mamba_ssm/modules/mamba3.py in https://github.com/state-spaces/mamba

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import (
    get_layer_cache,
    get_unpad_data,
    index_first_axis,
    pad_input,
    update_layer_cache,
)
from fla.modules.layernorm_gated import RMSNormGated


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    except ImportError:
        mamba3_siso_combined = None
    try:
        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as mamba3_mimo_combined
    except ImportError:
        mamba3_mimo_combined = None
    try:
        from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import apply_rotary_qk_inference_fwd
    except ImportError:
        apply_rotary_qk_inference_fwd = None
    try:
        from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn
    except ImportError:
        mamba3_step_fn = None

is_fast_path_available = mamba3_siso_combined is not None

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)


class Mamba3(nn.Module):
    """
    Mamba-3 selective state-space layer.

    Compared to Mamba-2, Mamba-3 drops the causal 1D convolution, introduces
    input-independent per-head B/C biases, normalises B and C with RMSNorm, and
    applies a blockwise rotary transform to Q/K before the SSM scan. It also
    optionally supports a low-rank MIMO projection for V and the output gate.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        state_size: int = 128,
        expand: int = 2,
        head_dim: int = 64,
        n_groups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_outproj_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        use_bias: bool = False,
        norm_eps: float = 1e-5,
        layer_idx: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Mamba3:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.hidden_size = hidden_size
        self.ssm_state_size = state_size
        self.expand = expand
        self.head_dim = head_dim
        self.n_groups = n_groups
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        self.A_floor = A_floor
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.norm_eps = norm_eps

        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1

        self.intermediate_size = int(self.expand * self.hidden_size)
        if self.intermediate_size % head_dim != 0:
            raise ValueError(
                f"`expand * hidden_size` ({self.intermediate_size}) must be divisible by `head_dim` ({head_dim})."
            )
        self.num_heads = self.intermediate_size // head_dim

        if self.is_mimo and mamba3_mimo_combined is None:
            logger.warning_once(
                "Mamba-3 MIMO kernels are unavailable. Please install TileLang "
                "to enable `is_mimo=True`. Falling back to SISO is not possible "
                "once the layer is constructed in MIMO mode."
            )

        if rope_fraction not in (0.5, 1.0):
            raise ValueError("`rope_fraction` must be either 0.5 or 1.0.")
        self.rope_fraction = rope_fraction
        self.rotary_dim_divisor = int(2 / rope_fraction)
        split_tensor_size = int(state_size * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        self.split_tensor_size = split_tensor_size
        self.num_rope_angles = split_tensor_size // 2
        if self.num_rope_angles <= 0:
            raise ValueError("`state_size * rope_fraction` is too small to produce any rotary angle.")

        # projection order: [z, x, B, C, dd_dt, dd_A, trap, angles]
        self.d_in_proj = (
            2 * self.intermediate_size
            + 2 * self.ssm_state_size * self.n_groups * self.mimo_rank
            + 3 * self.num_heads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(self.hidden_size, self.d_in_proj, bias=use_bias, **factory_kwargs)

        # dt bias parameterisation (inverse of softplus on sampled dt)
        dt = torch.exp(
            torch.rand(self.num_heads, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # input-independent per-head B and C biases (initialised to 1)
        self.B_bias = nn.Parameter(
            torch.ones(self.num_heads, self.mimo_rank, self.ssm_state_size, device=device, dtype=torch.float32)
        )
        self.C_bias = nn.Parameter(
            torch.ones(self.num_heads, self.mimo_rank, self.ssm_state_size, device=device, dtype=torch.float32)
        )

        # RMSNorm applied to B and C before the SSM scan
        self.B_norm = RMSNormGated(self.ssm_state_size, eps=norm_eps, **factory_kwargs)
        self.C_norm = RMSNormGated(self.ssm_state_size, eps=norm_eps, **factory_kwargs)

        if self.is_mimo:
            mimo_x = torch.ones(self.num_heads, self.mimo_rank, self.head_dim, device=device) / self.mimo_rank
            mimo_z = torch.ones(self.num_heads, self.mimo_rank, self.head_dim, device=device)
            mimo_o = torch.ones(self.num_heads, self.mimo_rank, self.head_dim, device=device) / self.mimo_rank
            self.mimo_x = nn.Parameter(mimo_x)
            self.mimo_z = nn.Parameter(mimo_z)
            self.mimo_o = nn.Parameter(mimo_o)

        self.D = nn.Parameter(torch.ones(self.num_heads, device=device))
        self.D._no_weight_decay = True

        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.intermediate_size,
                eps=norm_eps,
                norm_before_gate=True,
                group_size=self.head_dim,
                **factory_kwargs,
            )

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=use_bias, **factory_kwargs)
        self.use_bias = use_bias

        if not is_fast_path_available:
            logger.warning_once(
                "Mamba-3 fast path is not available because `mamba3_siso_combined` is None. "
                "Install Mamba-3 kernels from https://github.com/state-spaces/mamba to enable it."
            )

    def _project_and_split(self, hidden_states: torch.Tensor):
        zxBCdtAtrap = self.in_proj(hidden_states)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                self.intermediate_size,
                self.intermediate_size,
                self.ssm_state_size * self.n_groups * self.mimo_rank,
                self.ssm_state_size * self.n_groups * self.mimo_rank,
                self.num_heads,
                self.num_heads,
                self.num_heads,
                self.num_rope_angles,
            ],
            dim=-1,
        )
        return z, x, B, C, dd_dt, dd_A, trap, angles

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        last_state: dict | None = None,
        use_cache: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ):
        # Gate kernel availability per active mode so the SISO path works even
        # without MIMO kernels (and vice versa).
        if self.is_mimo:
            if mamba3_mimo_combined is None:
                raise RuntimeError(
                    "Mamba-3 MIMO kernels are unavailable. Install TileLang to enable `is_mimo=True`."
                )
        else:
            if mamba3_siso_combined is None:
                raise RuntimeError(
                    "Mamba-3 SISO kernels are unavailable. Please install `mamba_ssm` with Mamba-3 support."
                )

        if last_state is not None and hidden_states.shape[1] != 1:
            raise ValueError("Mamba-3 cached decoding only supports a single new token per step.")

        if last_state is not None:
            angle_state, ssm_state, k_state, v_state = last_state['recurrent_state']
            out = self.step(
                hidden_states,
                angle_state=angle_state,
                ssm_state=ssm_state,
                k_state=k_state,
                v_state=v_state,
            )
            return out, (angle_state, ssm_state, k_state, v_state)

        z, x, B, C, dd_dt, dd_A, trap, angles = self._project_and_split(hidden_states)
        z = rearrange(z, "b l (h p) -> b l h p", p=self.head_dim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.head_dim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.n_groups)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.n_groups)
        trap = rearrange(trap, "b l h -> b h l")

        A = -F.softplus(dd_A.to(torch.float32))
        A = torch.clamp(A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        ADT = A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        angles = angles.unsqueeze(-2).expand(-1, -1, self.num_heads, -1).to(torch.float32)

        B = self.B_norm(B)
        C = self.C_norm(C)

        return_state = use_cache

        if self.is_mimo:
            y = mamba3_mimo_combined(
                Q=C,
                K=B,
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias,
                K_bias=self.B_bias,
                MIMO_V=self.mimo_x,
                MIMO_Z=self.mimo_z,
                MIMO_Out=self.mimo_o if not self.is_outproj_norm else None,
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                rotary_dim_divisor=self.rotary_dim_divisor,
                dtype=x.dtype,
                return_state=return_state,
                cu_seqlens=cu_seqlens,
            )
            last_angle = last_ssm = last_k = last_v = None
            if return_state:
                y, last_angle, last_ssm, last_k, last_v, *_ = y
            if self.is_outproj_norm:
                z_r = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z)
                z_r = rearrange(z_r, "b l r h p -> b l r (h p)")
                y = rearrange(y, "b l r h p -> b l r (h p)").float()
                y = self.norm(y, z_r)
                y = rearrange(y, "b l r (h p) -> b l r h p", p=self.head_dim)
                y = torch.einsum("blrhp,hrp->blhp", y, self.mimo_o)
            y = rearrange(y, "b l h p -> b l (h p)")
        else:
            y = mamba3_siso_combined(
                Q=C.squeeze(2),
                K=B.squeeze(2),
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                Input_States=None,
                return_final_states=return_state,
                cu_seqlens=cu_seqlens,
            )
            last_angle = last_ssm = last_k = last_v = None
            if return_state:
                y, last_angle, last_ssm, last_k, last_v, *_ = y
                last_k = last_k.unsqueeze(1)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.is_outproj_norm:
                z_flat = rearrange(z, "b l h p -> b l (h p)")
                y = self.norm(y, z_flat)

        out = self.out_proj(y.to(x.dtype))

        new_state = None
        if return_state:
            new_state = (last_angle, last_ssm, last_k, last_v)
        return out, new_state

    def _preprocess_step(self, dd_A, dd_dt, B, C, x, z, trap_proj, angle_proj):
        A = -F.softplus(dd_A.to(torch.float32))
        A = torch.clamp(A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        trap = torch.sigmoid(trap_proj)

        B = rearrange(B, "b (r g s) -> b r g s", g=self.n_groups, r=self.mimo_rank)
        C = rearrange(C, "b (r g s) -> b r g s", g=self.n_groups, r=self.mimo_rank)
        B = self.B_norm(B)
        C = self.C_norm(C)
        B = B.expand(-1, -1, self.num_heads, -1)
        C = C.expand(-1, -1, self.num_heads, -1)

        x = rearrange(x, "b (h p) -> b h p", p=self.head_dim)
        z = rearrange(z, "b (h p) -> b h p", p=self.head_dim)

        angles = angle_proj.unsqueeze(-2).expand(-1, self.num_heads, -1)
        return DT, B, C, x, z, trap, A, angles

    def _postprocess_mimo_outproj_norm(self, y, z, zproj, outproj):
        z_r = torch.einsum("bhp,rhp->brhp", z.float(), zproj)
        z_r = rearrange(z_r, "b r h p -> b r (h p)")
        y = rearrange(y, "b r h p -> b r (h p)").float()
        y = self.norm(y, z_r)
        y = rearrange(y, "b r (h p) -> b r h p", p=self.head_dim)
        y = torch.einsum("brhp,rhp->bhp", y, outproj)
        return y

    def step(
        self,
        hidden_states: torch.Tensor,
        angle_state: torch.Tensor,
        ssm_state: torch.Tensor,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
    ) -> torch.Tensor:
        if mamba3_step_fn is None or apply_rotary_qk_inference_fwd is None:
            raise RuntimeError(
                "Mamba-3 decode kernels are not available. "
                "Please install `nvidia-cutlass-dsl` and `quack-kernels`."
            )
        if hidden_states.shape[1] != 1:
            raise ValueError("Mamba-3 cached decoding only supports a single new token per step.")

        u = hidden_states.squeeze(1)
        z, x, B, C, dd_dt, dd_A, trap, angles = self._project_and_split(u)

        DT, B, C, x, z, trap, A, angles = self._preprocess_step(
            dd_A, dd_dt, B, C, x, z, trap, angles,
        )

        bias_q = rearrange(self.C_bias, "h r n -> r h n")
        bias_k = rearrange(self.B_bias, "h r n -> r h n")

        rotate_pairwise = not self.is_mimo
        C, B, nxt_angle_state = apply_rotary_qk_inference_fwd(
            q=C, k=B, angle_state=angle_state,
            angle_proj=angles, dt=DT, bias_q=bias_q, bias_k=bias_k,
            conjugate=False, inplace=False,
            rotate_pairwise=rotate_pairwise,
        )

        nxt_v_state = x
        nxt_k_state = B

        if self.is_mimo:
            xpj = rearrange(self.mimo_x, "h r p -> r h p", p=self.head_dim).contiguous()
            zpj = rearrange(self.mimo_z, "h r p -> r h p", p=self.head_dim).contiguous()
            outpj = rearrange(self.mimo_o, "h r p -> r h p", p=self.head_dim).contiguous()
        else:
            xpj = torch.ones(self.mimo_rank, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
            zpj = torch.ones(self.mimo_rank, self.num_heads, self.head_dim, device=z.device, dtype=z.dtype)
            outpj = torch.ones(self.mimo_rank, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)

        if self.is_outproj_norm:
            batch = x.shape[0]
            y = torch.empty(batch, self.mimo_rank, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
            mamba3_step_fn(
                ssm_state, k_state, v_state, A, B, C, self.D, x, DT, trap, xpj,
                outproj=None, state_out=None, out=y, z=None, zproj=None,
                tile_D=64, num_warps=4,
            )
            y = self._postprocess_mimo_outproj_norm(y, z, zpj, outpj)
        else:
            y = torch.empty_like(x)
            mamba3_step_fn(
                ssm_state, k_state, v_state, A, B, C, self.D, x, DT, trap, xpj,
                outproj=outpj, state_out=None, out=y, z=z, zproj=zpj,
                tile_D=64, num_warps=4,
            )

        out = rearrange(y, "b h p -> b (h p)")
        out = self.out_proj(out.to(x.dtype))

        angle_state.copy_(nxt_angle_state)
        k_state.copy_(nxt_k_state)
        v_state.copy_(nxt_v_state)
        return out.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        last_state = get_layer_cache(self, past_key_values)

        if "cuda" not in self.in_proj.weight.device.type:
            raise NotImplementedError("Mamba-3 currently requires a CUDA device.")

        if attention_mask is not None:
            assert attention_mask.dim() == 2, (
                "Expected attention_mask as a 0/1 tensor with shape [batch_size, seq_len] "
                "for padding purposes. Arbitrary 3-D masks are not supported."
            )

        batch_size, q_len, _ = hidden_states.shape

        # Decode step (q_len == 1) is handled inside cuda_kernels_forward and does
        # not need unpadding; only prefill takes the varlen packing path.
        indices_q = None
        if (
            last_state is None
            and cu_seqlens is None
            and attention_mask is not None
            and q_len > 1
        ):
            # Upstream SISO/MIMO kernels require batch=1 plus cu_seqlens in
            # varlen mode; pack [B, T, D] -> [1, total_valid_tokens, D].
            indices_q, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."),
                indices_q,
            ).unsqueeze(0)

        output, new_state = self.cuda_kernels_forward(
            hidden_states,
            last_state,
            use_cache=bool(use_cache) or last_state is not None,
            cu_seqlens=cu_seqlens,
        )

        if new_state is not None:
            update_layer_cache(
                self,
                past_key_values,
                recurrent_state=new_state,
                offset=q_len,
            )

        if indices_q is not None:
            output = pad_input(output.squeeze(0), indices_q, batch_size, q_len)

        return output, None, past_key_values

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = self.in_proj.weight.device if device is None else device
        dtype = self.in_proj.weight.dtype if dtype is None else dtype

        angle_state = torch.zeros(
            (batch_size, self.num_heads, self.num_rope_angles), device=device, dtype=torch.float32,
        )
        ssm_state = torch.zeros(
            (batch_size, self.num_heads, self.head_dim, self.ssm_state_size), device=device, dtype=torch.float32,
        )
        k_state = torch.zeros(
            (batch_size, self.mimo_rank, self.num_heads, self.ssm_state_size), device=device, dtype=dtype,
        )
        v_state = torch.zeros(
            (batch_size, self.num_heads, self.head_dim), device=device, dtype=dtype,
        )
        return angle_state, ssm_state, k_state, v_state
