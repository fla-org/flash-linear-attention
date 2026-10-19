# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""All-to-all forward adapters and FP32 gradient accumulation for layer benchmarks."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from benchmarks.utils.distributed import maybe_wait, qkvo_all_to_all
from fla.modules.convolution import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.kda import chunk_kda, fused_recurrent_kda

if TYPE_CHECKING:
    from fla.models.cache_utils import Cache

try:
    import fused_weight_gradient_mlp_cuda
    APEX_WEIGHT_GRADIENT = None
    if hasattr(fused_weight_gradient_mlp_cuda, "wgrad_gemm_accum_fp32"):
        APEX_WEIGHT_GRADIENT = fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32
    if hasattr(fused_weight_gradient_mlp_cuda, "fused_wgrad_gemm_accum_fp32"):
        APEX_WEIGHT_GRADIENT = fused_weight_gradient_mlp_cuda.fused_wgrad_gemm_accum_fp32
    if APEX_WEIGHT_GRADIENT is None:
        HAVE_APEX = False
    else:
        HAVE_APEX = True
except Exception:
    HAVE_APEX = False


class _LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        inp_shape = inp.shape
        out = torch.matmul(inp.view(-1, inp_shape[-1]), weight.t())
        if bias is not None:
            out += bias
        ctx.save_for_backward(inp, weight, bias)
        return out.view(*inp_shape[:-1], -1)

    @staticmethod
    def backward(ctx, dout):
        inp, weight, bias = ctx.saved_tensors
        inp_shape = inp.shape
        inp = inp.view(-1, inp_shape[-1])
        dout = dout.view(-1, dout.size(-1))
        dgrad = torch.matmul(dout, weight).view(inp_shape)
        main_grad = getattr(weight, "main_grad", None)
        if main_grad is not None:
            if HAVE_APEX:
                wgrad = APEX_WEIGHT_GRADIENT(inp, dout, main_grad)
            else:
                wgrad = torch.matmul(dout.t(), inp)
                main_grad += wgrad
            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                wgrad = torch.empty(
                    main_grad.shape,
                    dtype=inp.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
                weight.grad_added_to_main_grad = True
            else:
                wgrad = None
        else:
            wgrad = torch.matmul(dout.t(), inp)
        if bias is not None:
            dbias = dout.sum(0)
        else:
            dbias = None
        return dgrad, wgrad, dbias


def fp32_grad_linear_forward(input, self):
    out = _LinearFunction.apply(input, self.weight, self.bias)
    return out


def patch_fp32_grad_linear_forward(model: torch.nn.Module):
    if HAVE_APEX:
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                m.forward = partial(fp32_grad_linear_forward, self=m)
                m.weight.main_grad = torch.zeros_like(m.weight, dtype=torch.float32)


def gdn_forward(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    use_cache: bool | None = False,
    output_attentions: bool | None = False,
    cp_group=None,
    cp_size=None,
    cp_rank=None,
    self=None,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
    if attention_mask is not None:
        assert len(attention_mask.shape) == 2, (
            "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
            "for padding purposes (0 indicating padding). "
            "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
        )

    batch_size, q_len, _ = hidden_states.shape
    # change to inference mode.
    mode = 'fused_recurrent' if q_len <= 64 else self.mode
    if self.training:
        assert mode == 'chunk', "Only chunk mode is supported in training."

    last_state = None
    if past_key_values is not None and len(past_key_values) > self.layer_idx:
        last_state = past_key_values[self.layer_idx]

    cu_seqlens = kwargs.get('cu_seqlens')

    async_op = False
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    a = self.a_proj(hidden_states)
    b = self.b_proj(hidden_states)
    if self.use_gate:
        gate = self.g_proj(hidden_states)
    # set async_op=True and one proj one comm can overlap
    q, handle_q = qkvo_all_to_all(q.view(-1, self.num_heads, self.head_k_dim), is_qkv=True, group=cp_group, async_op=async_op)
    k, handle_k = qkvo_all_to_all(k.view(-1, self.num_heads, self.head_k_dim), is_qkv=True, group=cp_group, async_op=async_op)
    v, handle_v = qkvo_all_to_all(v.view(-1, self.num_v_heads, self.head_v_dim),
                                  is_qkv=True, group=cp_group, async_op=async_op)
    a, handle_a = qkvo_all_to_all(a.view(-1, self.num_v_heads, 1), is_qkv=True, group=cp_group, async_op=async_op)
    b, handle_b = qkvo_all_to_all(b.view(-1, self.num_v_heads, 1), is_qkv=True, group=cp_group, async_op=async_op)
    if self.use_gate:
        gate, handle_gate = qkvo_all_to_all(gate.view(-1, self.num_v_heads, self.head_v_dim),
                                            is_qkv=True, group=cp_group, async_op=async_op)
    q, k, v, a, b = [t.flatten(-2, -1).unsqueeze(0) for t in [q, k, v, a, b]]

    if self.use_short_conv:
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
        maybe_wait(handle_q)
        q, conv_state_q = self.q_conv1d(
            x=q,
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        maybe_wait(handle_k)
        k, conv_state_k = self.k_conv1d(
            x=k,
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        maybe_wait(handle_v)
        v, conv_state_v = self.v_conv1d(
            x=v,
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
    else:
        maybe_wait(handle_q)
        q = F.silu(q)
        maybe_wait(handle_k)
        k = F.silu(k)
        maybe_wait(handle_v)
        v = F.silu(v)

    q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
    v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

    if self.num_v_heads > self.num_heads:
        q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

    maybe_wait(handle_b)
    beta = b.sigmoid()
    if self.allow_neg_eigval:
        beta = beta * 2.

    maybe_wait(handle_a)
    A_log = self.A_log.chunk(cp_size)[cp_rank]
    dt_bias = self.dt_bias.chunk(cp_size)[cp_rank]
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

    recurrent_state = last_state['recurrent_state'] if last_state is not None else None
    if mode == 'chunk':
        o, recurrent_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
    elif mode == 'fused_recurrent':
        o, recurrent_state = fused_recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        raise NotImplementedError(f"Not supported mode `{mode}`.")

    if past_key_values is not None:
        past_key_values.update(
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            layer_idx=self.layer_idx,
            offset=q_len
        )

    if self.use_gate:
        maybe_wait(handle_gate)
        g = rearrange(gate.flatten(-2, -1).unsqueeze(0), '... (h d) -> ... h d', d=self.head_v_dim)
        o = self.o_norm(o, g)
    else:
        o = self.o_norm(o)
    # THD
    o = o.flatten(0, 1)
    o, _ = qkvo_all_to_all(o, is_qkv=False, group=cp_group)

    o = rearrange(o.unsqueeze(0), 'b t h d -> b t (h d)')
    o = self.o_proj(o)

    return o, None, past_key_values


def kda_forward(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    use_cache: bool | None = False,
    output_attentions: bool | None = False,
    cp_group=None,
    cp_size=None,
    cp_rank=None,
    self=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
    if attention_mask is not None:
        assert len(attention_mask.shape) == 2, (
            "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
            "for padding purposes (0 indicating padding). "
            "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
        )

    batch_size, q_len, _ = hidden_states.shape
    # change to inference mode.
    mode = 'fused_recurrent' if q_len <= 64 else self.mode
    if self.training:
        assert mode == 'chunk', "Only chunk mode is supported in training."

    last_state = None
    if past_key_values is not None and len(past_key_values) > self.layer_idx:
        last_state = past_key_values[self.layer_idx]

    cu_seqlens = kwargs.get('cu_seqlens')

    async_op = False
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    g = self.f_proj(hidden_states)
    b = self.b_proj(hidden_states)
    gate = self.g_proj(hidden_states)
    q, handle_q = qkvo_all_to_all(q.view(-1, self.num_heads, self.head_k_dim), is_qkv=True, group=cp_group, async_op=async_op)
    k, handle_k = qkvo_all_to_all(k.view(-1, self.num_heads, self.head_k_dim), is_qkv=True, group=cp_group, async_op=async_op)
    v, handle_v = qkvo_all_to_all(v.view(-1, self.num_v_heads, self.head_v_dim),
                                  is_qkv=True, group=cp_group, async_op=async_op)
    g, handle_g = qkvo_all_to_all(g.view(-1, self.num_heads, self.head_k_dim), is_qkv=True, group=cp_group, async_op=async_op)
    b, handle_b = qkvo_all_to_all(b.view(-1, self.num_heads, 1), is_qkv=True, group=cp_group, async_op=async_op)
    gate, handle_gate = qkvo_all_to_all(gate.view(-1, self.num_v_heads, self.head_v_dim),
                                        is_qkv=True, group=cp_group, async_op=async_op)
    q, k, v, g, b, gate = [t.flatten(-2, -1).unsqueeze(0) for t in [q, k, v, g, b, gate]]

    if self.use_short_conv:
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
        maybe_wait(handle_q)
        q, conv_state_q = self.q_conv1d(
            x=q,
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        maybe_wait(handle_k)
        k, conv_state_k = self.k_conv1d(
            x=k,
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        maybe_wait(handle_v)
        v, conv_state_v = self.v_conv1d(
            x=v,
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
    else:
        maybe_wait(handle_q)
        q = F.silu(q)
        maybe_wait(handle_k)
        k = F.silu(k)
        maybe_wait(handle_v)
        v = F.silu(v)

    A_log = self.A_log.chunk(cp_size, 0)[cp_rank]
    dt_bias = self.dt_bias.chunk(cp_size, 0)[cp_rank]
    maybe_wait(handle_g)
    maybe_wait(handle_b)
    beta = b.sigmoid()

    q, k, g = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, g))
    v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

    if self.num_v_heads > self.num_heads:
        q, k, g = (repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads) for x in (q, k, g))
        beta = repeat(beta, '... h -> ... (h g)', g=self.num_v_heads // self.num_heads)

    if self.allow_neg_eigval:
        beta = beta * 2.
    recurrent_state = last_state['recurrent_state'] if last_state is not None else None
    if mode == 'chunk':
        o, recurrent_state = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            A_log=A_log,
            dt_bias=dt_bias,
            use_gate_in_kernel=True,
        )
    elif mode == 'fused_recurrent':
        o, recurrent_state = fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
    else:
        raise NotImplementedError(f"Not supported mode `{mode}`.")

    if past_key_values is not None:
        past_key_values.update(
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            layer_idx=self.layer_idx,
            offset=q_len,
        )
    maybe_wait(handle_gate)
    o = self.o_norm(o, rearrange(gate, '... (h d) -> ... h d', d=self.head_v_dim))

    # THD
    o, _ = qkvo_all_to_all(o.flatten(0, 1), is_qkv=False, group=cp_group)
    o = rearrange(o.unsqueeze(0), 'b t h d -> b t (h d)')
    o = self.o_proj(o)

    return o, None, past_key_values


def short_conv_forward(
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    cache: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cp_size=None,
    cp_rank=None,
    self=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x (`torch.Tensor`):
            Tensor of shape `[B, T, D]`. `B` must be 1 if `seq_idx` is provided.
        residual (`Optional[torch.Tensor]`):
            Residual tensor of shape `[B, T, D]`. Default: `None`.
        mask (`Optional[torch.Tensor]`):
            Attention mask dealing with padded positions.
        cache (`Optional[torch.Tensor]`):
            Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
            If provided, the cache is updated **inplace**.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, D, W]`. Default: `False`.
        cu_seqlens (Optional[torch.LongTensor]):
            Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
            Shape: [B+1]

    Returns:
        Tensor of shape `[B, T, D]`.
    """

    B, T, *_ = x.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if mask is not None:
        if cu_seqlens is not None:
            raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
        x = x.mul_(mask.unsqueeze(-1))

    # in decoding phase, the cache (if provided) is updated inplace
    if B * T == N:
        y, cache = self.step(
            x=x,
            residual=residual,
            cache=cache,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens
        )
        return y, cache

    # cuda backend do not support:
    # 1. both `cu_seqlens` and `cache` being provided
    # 2. both `cu_seqlens` and `output_final_state` being provided
    if self.backend == 'cuda' and (
        (cu_seqlens is not None and cache is not None) or
        (cu_seqlens is not None and output_final_state)
    ):
        self.backend = 'triton'
    # hidden_size, 1, kernel_size
    weight = self.weight.chunk(cp_size, 0)[cp_rank].squeeze(1)

    return causal_conv1d(
        x=x,
        weight=weight,
        bias=self.bias,
        residual=residual,
        initial_state=cache,
        output_final_state=output_final_state,
        activation=self.activation,
        backend=self.backend,
        cu_seqlens=cu_seqlens,
        **kwargs
    )
