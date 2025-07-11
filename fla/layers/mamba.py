# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn
from transformers.utils import logging

from fla.modules.activations import ACT2FN

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    except ImportError:
        selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except ImportError:
        causal_conv1d_update, causal_conv1d_fn = None, None
    is_fast_path_available = all((
        selective_state_update,
        selective_scan_fn,
        mamba_inner_fn
    ))
if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.mamba.modeling_mamba import MambaCache

logger = logging.get_logger(__name__)


class Mamba(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        state_size: int = 16,
        conv_kernel: int = 4,
        use_conv_bias: bool = True,
        intermediate_size: int = 2048,
        time_step_rank: int = 256,
        use_bias: bool = True,
        hidden_act: str = "silu",
        layer_idx: int = None,
        backend: str = "cuda",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.ssm_state_size = state_size
        self.conv_kernel_size = conv_kernel
        self.use_conv_bias = use_conv_bias
        self.intermediate_size = intermediate_size
        self.time_step_rank = time_step_rank
        self.use_bias = use_bias

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=use_conv_bias,
            kernel_size=conv_kernel,
            groups=self.intermediate_size,
            padding=conv_kernel - 1,
        )

        self.activation = hidden_act
        self.act = ACT2FN[hidden_act]

        self.layer_idx = layer_idx

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=use_bias)

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of "
                "`(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. Falling back to the naive implementation. "
                "To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )
        import os
        backend = os.environ.get('FLA_CONV_BACKEND', backend)
        assert backend in ['cuda', 'triton'], f"Unsupported backend: {backend}"
        if backend == 'cuda' and causal_conv1d_fn is None:
            logger.warning_once(
                "The CUDA backend is not available because `causal_conv1d` is None. "
                "Falling back to the Triton backend. "
                "To install follow https://github.com/Dao-AILab/causal-conv1d"
            )
            backend = 'triton'
        if backend == 'triton':
            from fla.modules.convolution import causal_conv1d as causal_conv1d_triton
            from fla.modules.convolution import causal_conv1d_update as causal_conv1d_update_triton
            self.causal_conv1d_fn = causal_conv1d_triton
            self.causal_conv1d_update = causal_conv1d_update_triton
        else:
            self.causal_conv1d_fn = causal_conv1d_fn
            self.causal_conv1d_update = causal_conv1d_update
        self.backend = backend

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[MambaCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[Dict]
    ):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        if self.training and cache_params is None:  # Doesn't support outputting the states -> used for training
            contextualized_states = mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.float() if self.use_bias else None,
                -torch.exp(self.A_log.float()),
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

        else:
            hidden_states, gate = projected_states.chunk(2, dim=1)

            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(1)

            # 2. Convolution sequence transformation
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if cache_params is not None and cache_position[0] > 0:
                hidden_states = self.causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.conv_states[self.layer_idx],
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                )
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(
                        hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                    )
                    cache_params.update_conv_state(self.layer_idx, conv_states, cache_position)
                hidden_states = self.causal_conv1d_fn(
                    hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
                )

            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(1)

            # 3. State Space Model sequence transformation
            # 3.a. input varying initialization of time_step, B and C
            ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(
                ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
            )
            discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

            A = -torch.exp(self.A_log.float())
            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
            if cache_params is not None and cache_position[0] > 0:
                scan_outputs = selective_state_update(
                    cache_params.ssm_states[self.layer_idx],
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D,
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose(1, 2),
                    C.transpose(1, 2),
                    self.D.float(),
                    gate,
                    time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.update_ssm_state(self.layer_idx, ssm_state)

            # 4. Final linear projection
            contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    def slow_forward(
        self,
        input_states,
        cache_params: Optional[MambaCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[Dict]
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        # [batch, 2 * intermediate_size, seq_len]
        projected_states = self.in_proj(input_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            ssm_state = ssm_state.to(hidden_states.device)
            # use `cache_position.shape[0]` to check whether we are in prefill
            # stage, it's equivalent to check `cache_position[0] == 0`, which
            # breaks dynamo fullgraph constraints
            if cache_position.shape[0] == self.conv_kernel_size:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )

                cache_params.update_conv_state(self.layer_idx, conv_state, cache_position)
                # [batch, intermediate_size, seq_len]
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
            else:
                conv_state = cache_params.update_conv_state(self.layer_idx, hidden_states, cache_position)
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                # [batch, intermediate_size, 1] : decoding
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            # [batch, intermediate_size, seq_len]
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        # [batch, seq_len, intermediate_size]
        discrete_time_step = self.dt_proj(time_step)
        # [batch, intermediate_size, seq_len]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2)

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        # [intermediate_size, ssm_state_size]
        A = -torch.exp(self.A_log.float())
        # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
        # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            # [batch, intermediade_size, ssm_state]
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            # [batch, intermediade_size, 1]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
            scan_outputs.append(scan_output[:, :, 0])
        # [batch, seq_len, intermediade_size]
        scan_output = torch.stack(scan_outputs, dim=-1)
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        # [batch, seq_len, hidden_size]
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states,
        cache_params: Optional[MambaCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[Dict]
    ):
        if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask, **kwargs)
        return self.slow_forward(hidden_states, cache_params, cache_position, attention_mask, **kwargs)
