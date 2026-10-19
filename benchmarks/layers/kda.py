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
from fla.ops.kda import chunk_kda, fused_recurrent_kda

SCENARIOS = ('cp',)


if TYPE_CHECKING:
    from fla.models.cache_utils import Cache


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


def add_arguments(parser, scenario):
    add_cp_arguments(parser)
    parser.add_argument('--use-cp2hp', action='store_true')


def run(args):
    from benchmarks.utils.layers import benchmark_layer

    args.kda = True
    return run_distributed(args=args, measure=benchmark_layer)
