# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from benchmarks.utils.distributed import add_cp_arguments, maybe_wait, qkvo_all_to_all, run_distributed
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

SCENARIOS = ('cp',)


if TYPE_CHECKING:
    from fla.models.cache_utils import Cache


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


def add_arguments(parser, scenario):
    add_cp_arguments(parser)
    parser.add_argument('--use-cp2hp', action='store_true')


def run(args):
    from benchmarks.utils.layers import benchmark_layer

    args.kda = False
    return run_distributed(args=args, measure=benchmark_layer)
