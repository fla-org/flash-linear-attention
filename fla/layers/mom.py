# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from fla.layers.utils import (
    get_layer_cache,
    get_unpad_indices_and_cu,
    index_first_axis,
    pad_input,
    repad_hidden_states,
    unpad_hidden_states,
    unpad_input,
    update_layer_cache,
)
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def transform(
    x: torch.Tensor,
    routing_mask: torch.Tensor,
    num_memories: int,
    selected_memories: torch.Tensor,
    attention_mask: torch.Tensor | None,
):
    """
    Reorganize token embeddings into memory-aligned chunks, padded to the largest memory sequence.

    Padding positions must be excluded using the returned masks; their embeddings are not necessarily zero.

    Args:
        x (torch.Tensor):
            Input embeddings. Shape: (batch, seq, hidden).
        routing_mask (torch.Tensor):
            Binary routing mask. Shape: (batch, seq, num_memories).
            Entries for tokens excluded by `attention_mask` are zeroed in place.
        num_memories (int):
            Number of routed memories.
        selected_memories (torch.Tensor):
            Memory indices per token. Shape: (batch, seq, topk).
            Shape (batch, seq) is also accepted for top-1 routing without an attention mask.
        attention_mask (torch.Tensor | None):
            Boolean valid-token mask. Shape: (batch, seq), or a longer mask whose trailing `seq` entries are used.
            `None` includes all tokens.

    Returns:
        transformed_x (torch.Tensor):
            Grouped token embeddings. Shape: (num_memories, batch, max_len, hidden).
        indices (torch.Tensor):
            Gather indices into the sorted token slots. Shape: (batch * num_memories, max_len).
            Invalid positions have index zero and are identified by `mask`.
        sorted_indices (torch.Tensor):
            Permutation from flattened token slots to (batch, memory) order. Shape: (batch * seq * topk,).
        max_len (torch.Tensor):
            Scalar maximum number of valid tokens in any (batch, memory) sequence.
        mask (torch.Tensor):
            Boolean validity mask in (batch, memory) order. Shape: (batch * num_memories, max_len).
        memory_mask (torch.Tensor):
            The same validity mask in output order. Shape: (num_memories, batch, max_len).
    """
    if selected_memories.dim() == 3:
        topk = selected_memories.shape[2]
        x = x.repeat_interleave(topk, dim=1)  # [batch, seq * topk, hidden]
        selected_memories = selected_memories.reshape(selected_memories.shape[0], -1)  # [batch, seq * topk]

    if attention_mask is not None:
        attention_mask = attention_mask[:, -routing_mask.shape[1]:]
        routing_mask[attention_mask.bitwise_not().unsqueeze(-1).expand(-1, -1, num_memories)] = 0

    batch_size, num_slots, hidden_size = x.shape
    x_flat = x.reshape(batch_size * num_slots, hidden_size)
    with torch.no_grad():
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(-1)
        batch_indices = batch_indices.repeat(1, num_slots).reshape(-1)
        if attention_mask is not None:
            # masked slots sort after every valid batch row
            batch_indices[attention_mask.repeat_interleave(topk, dim=1).bitwise_not().flatten()] = batch_size
        memories = selected_memories.reshape(-1)
        sorted_indices = (batch_indices * (memories.max() + 1) + memories).argsort()

    x_sorted = x_flat[sorted_indices]
    with torch.no_grad():
        lengths = routing_mask.sum(dim=1)  # [batch, num_memories]
        offsets = lengths.flatten().cumsum(dim=0)
        max_len = lengths.max()
        indices = (
            torch.arange(max_len, device=offsets.device).unsqueeze(0).expand(batch_size * num_memories, -1)
            + torch.cat([torch.tensor([0], device=offsets.device), offsets[:-1]], dim=0).unsqueeze(1)
        )
        mask = indices < offsets.unsqueeze(-1)
        indices = torch.where(mask, indices, torch.zeros_like(indices))

    gathered_x = torch.gather(x_sorted, dim=0, index=indices.reshape(-1).unsqueeze(-1).expand(-1, hidden_size))
    transformed_x = gathered_x.reshape(batch_size, num_memories, max_len, hidden_size).transpose(0, 1)
    memory_mask = mask.reshape(batch_size, num_memories, max_len).transpose(0, 1)
    return transformed_x, indices, sorted_indices, max_len, mask, memory_mask


def reconstruct(
    transformed_x: torch.Tensor,
    indices: torch.Tensor,
    sorted_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    topk: int,
    routing_weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct and mix memory outputs back into the original input sequence shape.

    The indices and mask must come from the matching `transform` call.
    Padded outputs are masked before the top-k routing weights are applied.

    Args:
        transformed_x (torch.Tensor):
            Outputs of the memory units. Shape: (num_memories, batch_size, max_len, hidden_size).
        indices (torch.Tensor):
            Gather indices returned by `transform`. Shape: (batch_size * num_memories, max_len).
        sorted_indices (torch.Tensor):
            Slot permutation returned by `transform`. Shape: (batch_size * seq_len * topk,).
        batch_size (int):
            Batch size of the original input.
        seq_len (int):
            Sequence length of the original input.
        topk (int):
            Number of selected memories per token.
        routing_weights (torch.Tensor):
            Weights for the selected memories, in the original top-k order. Shape: (batch_size, seq_len, topk).
        mask (torch.Tensor):
            Boolean validity mask returned by `transform`. Shape: (batch_size * num_memories, max_len).

    Returns:
        restored_x (torch.Tensor):
            Weighted memory outputs in the original token order. Shape: (batch_size, seq_len, hidden_size).
    """
    hidden_size = transformed_x.shape[-1]
    transformed_x = transformed_x.transpose(0, 1).reshape(-1, transformed_x.shape[2], hidden_size)
    memory_outputs = transformed_x.reshape(-1, hidden_size)
    memory_outputs = memory_outputs * mask.reshape(-1).unsqueeze(-1).expand_as(memory_outputs)

    assert (indices >= 0).all(), "Indices should be non-negative"
    sorted_outputs = torch.zeros(
        (batch_size * seq_len * topk, hidden_size),
        device=memory_outputs.device,
        dtype=memory_outputs.dtype,
    ).scatter_add_(dim=0, index=indices.reshape(-1).unsqueeze(-1).expand(-1, hidden_size), src=memory_outputs)
    assert (indices < sorted_outputs.size(0)).all(), "Indices should be less than resorted_x size"

    outputs = sorted_outputs[sorted_indices.argsort()].reshape(batch_size, seq_len, topk, hidden_size)
    outputs = outputs * routing_weights.reshape(batch_size, seq_len, topk).unsqueeze(-1)
    return outputs.sum(dim=2)


def _pack_slots(
    selected_memories: torch.Tensor,
    num_memories: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Group slots by memory and batch row while preserving the padded implementation's token order.

    Args:
        selected_memories (torch.Tensor):
            Memory indices per token. Shape: (batch, seq, topk).
        num_memories (int):
            Number of routed memories, including those that receive no tokens.

    Returns:
        sorted_indices (torch.Tensor):
            Indices of flattened token slots in (memory, batch) order. Shape: (batch * seq * topk,).
            Floor division by `topk` gives the corresponding flattened input token indices.
        inverse_indices (torch.Tensor):
            Inverse permutation restoring the original token and top-k order. Shape: (batch * seq * topk,).
        cu_seqlens (torch.Tensor):
            Int32 cumulative slot counts in (memory, batch) order. Shape: (num_memories * batch + 1,).
            Empty memory sequences have repeated boundaries.
    """
    batch_size = selected_memories.shape[0]
    batch_indices = torch.arange(batch_size, device=selected_memories.device)[:, None, None].expand_as(selected_memories)
    memories = selected_memories.flatten()
    batch_indices = batch_indices.flatten()
    # the padded path's unstable sort can reorder equal keys on short inputs
    sorted_indices = (batch_indices * (memories.max() + 1) + memories).argsort()
    sequence_ids = memories * batch_size + batch_indices
    sorted_indices = sorted_indices[sequence_ids[sorted_indices].argsort(stable=True)]
    lengths = torch.bincount(sequence_ids, minlength=num_memories * batch_size)
    cu_seqlens = F.pad(lengths.cumsum(dim=0, dtype=torch.int32), (1, 0))
    return sorted_indices, sorted_indices.argsort(), cu_seqlens


class MomAttention(nn.Module):
    """
    The layer implementation for [MoM: Linear Sequence Modeling with Mixture-of-Memories](https://arxiv.org/abs/2502.13685).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 256,
        num_heads: int = 4,
        expand_v: float = 2,
        mode: str = 'chunk',
        use_output_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        num_memories: int = 8,
        topk: int = 2,
        capacity: float = 1.0,
        shared_mem: bool = False,
        single_kv_proj: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_memories = num_memories
        self.topk = topk
        self.capacity = capacity
        self.shared_mem = shared_mem
        self.single_kv_proj = single_kv_proj

        self.mode = mode

        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_output_gate = use_output_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads

        self.key_dim = int(self.num_heads * self.head_dim)
        value_dim, head_v_dim = self.key_dim * self.expand_v, head_dim * self.expand_v
        self.value_dim, self.head_v_dim = round(value_dim), round(head_v_dim)
        assert math.isclose(value_dim, self.value_dim), f"`key_dim * expand_v` must be an integer, got {value_dim}."
        assert math.isclose(head_v_dim, self.head_v_dim), f"`head_dim * expand_v` must be an integer, got {head_v_dim}."
        self.head_qk_dim = head_dim
        self.layer_idx = layer_idx
        self.silu = nn.SiLU()

        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(in_features=hidden_size, out_features=self.key_dim, bias=False)
        self.gate = nn.Linear(in_features=self.hidden_size, out_features=self.num_memories, bias=False)
        if self.single_kv_proj:
            self.shared_k = nn.Linear(in_features=hidden_size, out_features=self.key_dim, bias=False)
            self.shared_v = nn.Linear(in_features=hidden_size, out_features=self.value_dim, bias=False)
            self.shared_b = nn.Linear(in_features=hidden_size, out_features=self.num_heads, bias=False)
            self.shared_a = nn.Linear(in_features=hidden_size, out_features=self.num_heads, bias=False)
        else:
            self.k_proj = nn.ModuleList([
                nn.Linear(in_features=self.hidden_size, out_features=self.key_dim, bias=False)
                for _ in range(self.num_memories)
            ])
            self.v_proj = nn.ModuleList([
                nn.Linear(in_features=self.hidden_size, out_features=self.value_dim, bias=False)
                for _ in range(self.num_memories)
            ])
            self.b_proj = nn.ModuleList([
                nn.Linear(in_features=self.hidden_size, out_features=self.num_heads, bias=False)
                for _ in range(self.num_memories)
            ])
            self.a_proj = nn.ModuleList([
                nn.Linear(in_features=self.hidden_size, out_features=self.num_heads, bias=False)
                for _ in range(self.num_memories)
            ])
            if self.shared_mem:
                self.shared_k = nn.Linear(in_features=hidden_size, out_features=self.key_dim, bias=False)
                self.shared_v = nn.Linear(in_features=hidden_size, out_features=self.value_dim, bias=False)
                self.shared_b = nn.Linear(in_features=hidden_size, out_features=self.num_heads, bias=False)
                self.shared_a = nn.Linear(in_features=hidden_size, out_features=self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
        else:
            raise UserWarning(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing.",
            )
        if use_output_gate:
            self.g_proj = nn.Linear(in_features=hidden_size, out_features=self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)
        self.o_proj = nn.Linear(in_features=self.value_dim, out_features=hidden_size, bias=False)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, None, Cache | None, torch.Tensor]:
        if attention_mask is not None:
            attention_mask = (attention_mask == 1)
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        origin_cu_seqlens = kwargs.get('cu_seqlens')
        if origin_cu_seqlens is not None:
            hidden_states, attention_mask = self.cu2pad(x=hidden_states, cu_seqlens=origin_cu_seqlens)

        if torch.is_grad_enabled():
            mode = 'chunk'
        elif hidden_states.shape[1] <= 64 and not self.training:
            mode = 'fused_recurrent'
        else:
            mode = self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)
        router_logits = self.gate(hidden_states)  # [B, T, M]
        scores = F.softmax(router_logits, dim=2, dtype=torch.float)
        routing_weights, selected_memories = torch.topk(scores, self.topk, dim=-1)  # [B, T, K]
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        routing_weights_full = torch.zeros(
            routing_weights.shape[0],
            routing_weights.shape[1],
            self.num_memories,
            dtype=routing_weights.dtype,
            device=routing_weights.device,
        ).scatter(-1, selected_memories, routing_weights)
        routing_mask = routing_weights_full.bool().int()

        batch_size, seq_len = hidden_states.shape[:2]
        shared_hidden_states = hidden_states
        packed = (
            not use_cache
            and past_key_values is None
            and attention_mask is None
            and origin_cu_seqlens is None
            and (routing_weights != 0).all().item()
        )
        memory_lengths = None
        if packed:
            sorted_indices, inverse_indices, cu_seqlens_all = _pack_slots(
                selected_memories=selected_memories,
                num_memories=self.num_memories,
            )
            hidden_states = hidden_states.flatten(0, 1)[sorted_indices // self.topk].unsqueeze(0)
            memory_lengths = (cu_seqlens_all[batch_size::batch_size] - cu_seqlens_all[:-1:batch_size]).tolist()
        else:
            hidden_states, indices, sorted_indices, max_len, mask, memory_mask = transform(
                x=hidden_states,
                routing_mask=routing_mask,
                num_memories=self.num_memories,
                selected_memories=selected_memories,
                attention_mask=attention_mask,
            )

        q, k, v, g, beta = self._project_memories(hidden_states=hidden_states, memory_lengths=memory_lengths)
        if packed:
            cu_q, cu_k, cu_v, cu_g, cu_beta = q, k, v, g, beta
        else:
            q, k, v, g, beta, memory_mask = (
                rearrange(x, 'e b l ... -> (e b) l ...') for x in (q, k, v, g, beta, memory_mask)
            )
            cu_q, (cu_k, cu_v, cu_g, cu_beta), indices_q, sequence_lengths, _ = unpad_input(
                q=q,
                states=(k, v, g, beta),
                attention_mask=memory_mask,
                q_len=q.shape[1],
            )
            cu_seqlens_all = sequence_lengths[0]
            cu_q, cu_k, cu_v, cu_g, cu_beta = (x.unsqueeze(0).contiguous() for x in (cu_q, cu_k, cu_v, cu_g, cu_beta))
        cu_seqlens, reverse_indices = cu_seqlens_all.to(torch.long).unique(return_inverse=True)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = [None, None], [None, None], [None, None]
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

            conv_cu_seqlens = cu_seqlens
            padded = False
            if self.training:
                # preserve the existing convolution across concatenated memories during training
                conv_cu_seqlens = None
            elif seq_len != 1 and (cu_seqlens[1:] - cu_seqlens[:-1]).min().item() < self.conv_size:
                padded = True
                conv_cu_seqlens, cu_q, cu_k, cu_v, pad_lengths = self.pad_for_conv(
                    cu_seqlens=cu_seqlens,
                    cu_q=cu_q,
                    cu_k=cu_k,
                    cu_v=cu_v,
                )

            convolved = []
            for x, conv, state in zip(
                (cu_q, cu_k, cu_v),
                (self.q_conv1d, self.k_conv1d, self.v_conv1d),
                (conv_state_q, conv_state_k, conv_state_v),
            ):
                conv_cache = self.prepare_recurrent_state(
                    recurrent_state=state[0],
                    cu_seqlens=conv_cu_seqlens,
                    cu_seqlen_all=cu_seqlens_all,
                    reverse_indices=reverse_indices,
                    batch_size=batch_size,
                )
                x, conv_cache = conv(x=x, cache=conv_cache, output_final_state=use_cache, cu_seqlens=conv_cu_seqlens)
                state[0] = self.handle_recurrent_state(
                    recurrent_state=state[0],
                    recurrent_state_new=conv_cache,
                    cu_seqlens=conv_cu_seqlens,
                    cu_seqlen_all=cu_seqlens_all,
                    reverse_indices=reverse_indices,
                )
                convolved.append(x)
            cu_q, cu_k, cu_v = convolved

            if padded:
                cu_q, cu_k, cu_v = self.unpad_after_conv(
                    conv_cu_seqlens=conv_cu_seqlens,
                    cu_seqlens=cu_seqlens,
                    cu_q=cu_q,
                    cu_k=cu_k,
                    cu_v=cu_v,
                    pad_lengths=pad_lengths,
                )

        else:
            q, k, v = self.silu(q), self.silu(k), self.silu(v)

        cu_q, cu_k, cu_v = (rearrange(x, 'b t (h d) -> b t h d', h=self.num_heads) for x in (cu_q, cu_k, cu_v))

        recurrent_state = last_state['recurrent_state'] if last_state is not None else [None] * (1 + self.shared_mem)
        initial_state = recurrent_state[0]
        if mode == 'fused_recurrent':
            initial_state = self.prepare_recurrent_state(
                recurrent_state=initial_state,
                cu_seqlens=cu_seqlens,
                cu_seqlen_all=cu_seqlens_all,
                reverse_indices=reverse_indices,
                batch_size=batch_size,
            )
        attention = chunk_gated_delta_rule if mode == 'chunk' else fused_recurrent_gated_delta_rule
        o, final_state = attention(
            q=cu_q,
            k=cu_k,
            v=cu_v,
            g=cu_g,
            beta=cu_beta,
            initial_state=initial_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )
        recurrent_state[0] = self.handle_recurrent_state(
            recurrent_state=recurrent_state[0],
            recurrent_state_new=final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlen_all=cu_seqlens_all,
            reverse_indices=reverse_indices,
        )

        o = o.squeeze(0).contiguous()
        if packed:
            o = o[inverse_indices].unflatten(dim=0, sizes=(batch_size, seq_len, self.topk))
            o = (o * routing_weights[..., None, None]).sum(dim=2)
        else:
            o = pad_input(hidden_states=o, indices=indices_q, batch_size=batch_size * self.num_memories, seq_len=max_len)
            o = rearrange(o, '(e b) l h d -> e b l (h d)', b=batch_size)
            o = reconstruct(
                transformed_x=o,
                indices=indices,
                sorted_indices=sorted_indices,
                batch_size=batch_size,
                seq_len=seq_len,
                topk=self.topk,
                routing_weights=routing_weights,
                mask=mask,
            )
            o = rearrange(o, 'b l (h d) -> b l h d', h=self.num_heads)

        if self.shared_mem:
            shared_o = self.shared_o(
                hidden_states=shared_hidden_states,
                attention_mask=attention_mask,
                recurrent_state=recurrent_state,
                use_cache=use_cache,
                conv_state_q=conv_state_q,
                conv_state_k=conv_state_k,
                conv_state_v=conv_state_v,
            )
            o += shared_o

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q.shape[2],
        )

        if self.use_output_gate:
            g = rearrange(self.g_proj(shared_hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        if origin_cu_seqlens is not None:
            indices, _ = get_unpad_indices_and_cu(attention_mask, seq_len)
            o = index_first_axis(rearrange(o, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        return o, None, past_key_values, router_logits.view(-1, self.num_memories)

    def _project_memories(
        self,
        hidden_states: torch.Tensor,
        memory_lengths: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project routed inputs while preserving their packed or padded memory layout.

        Args:
            hidden_states (torch.Tensor):
                Packed inputs of shape (1, total_slots, hidden_size), ordered by memory then batch,
                or padded inputs of shape (num_memories, batch, max_len, hidden_size).
            memory_lengths (list[int], Optional):
                Slot counts for each memory in packed inputs, summed over batch rows; empty memories have count zero.
                Counts must sum to `total_slots`. `None` selects the padded layout. Default: `None`.

        Returns:
            q (torch.Tensor):
                Queries with the input's leading dimensions and final dimension `key_dim`.
            k (torch.Tensor):
                Keys with the input's leading dimensions and final dimension `key_dim`.
            v (torch.Tensor):
                Values with the input's leading dimensions and final dimension `value_dim`.
            g (torch.Tensor):
                Log-decay gates with the input's leading dimensions and final dimension `num_heads`.
            beta (torch.Tensor):
                Sigmoid update weights with the input's leading dimensions and final dimension `num_heads`.
        """
        q = self.q_proj(hidden_states)
        if self.single_kv_proj:
            k = self.shared_k(hidden_states)
            v = self.shared_v(hidden_states)
            beta = self.shared_b(hidden_states).sigmoid()
            g = -self.A_log.float().exp() * F.softplus(self.shared_a(hidden_states).float() + self.dt_bias)
        else:
            if memory_lengths is None:
                memories = hidden_states.unbind(dim=0)
            else:
                memories = hidden_states.split(memory_lengths, dim=1)
            k = [projection(x) for x, projection in zip(memories, self.k_proj)]
            v = [projection(x) for x, projection in zip(memories, self.v_proj)]
            beta = [projection(x).sigmoid() for x, projection in zip(memories, self.b_proj)]
            g = [
                -self.A_log.float().exp() * F.softplus(projection(x).float() + self.dt_bias)
                for x, projection in zip(memories, self.a_proj)
            ]
            if memory_lengths is None:
                k, v, beta, g = (torch.stack(x, dim=0) for x in (k, v, beta, g))
            else:
                k, v, beta, g = (torch.cat(x, dim=1) for x in (k, v, beta, g))
        return q, k, v, g, beta

    def shared_o(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        recurrent_state=None,
        use_cache: bool | None = False,
        conv_state_q=[None, None],
        conv_state_k=[None, None],
        conv_state_v=[None, None],
        **kwargs,
    ) -> torch.Tensor:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        if torch.is_grad_enabled():
            mode = 'chunk'
        elif hidden_states.shape[1] <= 64 and not self.training:
            mode = 'fused_recurrent'
        else:
            mode = self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        cu_seqlens = kwargs.get('cu_seqlens')
        batch_size, q_len = hidden_states.shape[:2]
        hidden_states, indices, cu_seqlens = unpad_hidden_states(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            q_len=q_len,
        )

        if self.use_short_conv:
            q, conv_state_q[1] = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q[1],
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k[1] = self.k_conv1d(
                x=self.shared_k(hidden_states),
                cache=conv_state_k[1],
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v[1] = self.v_conv1d(
                x=self.shared_v(hidden_states),
                cache=conv_state_v[1],
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = self.silu(self.q_proj(hidden_states))
            k = self.silu(self.shared_k(hidden_states))
            v = self.silu(self.shared_v(hidden_states))

        q, k, v = (rearrange(x, 'b t (h d) -> b t h d', h=self.num_heads) for x in (q, k, v))
        beta = self.shared_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.shared_a(hidden_states).float() + self.dt_bias)

        if mode not in ('chunk', 'fused_recurrent'):
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        attention = chunk_gated_delta_rule if mode == 'chunk' else fused_recurrent_gated_delta_rule
        o, recurrent_state[-1] = attention(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state[-1],
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )

        o = repad_hidden_states(hidden_states=o, indices=indices, batch_size=batch_size, q_len=q_len)
        return o

    def cu2pad(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = cu_seqlens.shape[0] - 1
        max_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        indices = torch.tensor([], dtype=torch.long, device=x.device)
        attention_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            seq_len = cu_seqlens[i+1] - cu_seqlens[i]
            pad_len = max_len - seq_len
            batch_indices = torch.arange(pad_len, max_len, device=x.device)
            batch_indices = batch_indices + i * max_len
            indices = torch.cat([indices, batch_indices])
            attention_mask[i, :pad_len] = False
        x = pad_input(hidden_states=x.squeeze(0), indices=indices, batch_size=batch_size, seq_len=max_len)
        return x, attention_mask

    def pad_for_conv(
        self,
        cu_seqlens: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        cu_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Left-pad each memory sequence to at least the convolution width."""
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        pad_lengths = torch.clamp(self.conv_size - lengths, min=0)
        new_lengths = lengths + pad_lengths
        new_cu_seqlens = torch.cat([
            torch.tensor([0], device=cu_seqlens.device, dtype=cu_seqlens.dtype),
            torch.cumsum(new_lengths, dim=0),
        ])
        final_total_len = new_cu_seqlens[-1].item()
        new_q = torch.zeros((1, final_total_len, cu_q.shape[-1]), dtype=cu_q.dtype, device=cu_q.device)
        new_k = torch.zeros((1, final_total_len, cu_k.shape[-1]), dtype=cu_k.dtype, device=cu_k.device)
        new_v = torch.zeros((1, final_total_len, cu_v.shape[-1]), dtype=cu_v.dtype, device=cu_v.device)
        num_sequences = len(lengths)
        for i in range(num_sequences):
            src_start = cu_seqlens[i]
            src_end = cu_seqlens[i+1]
            dest_start = new_cu_seqlens[i] + pad_lengths[i]
            dest_end = new_cu_seqlens[i+1]
            new_q[:, dest_start:dest_end, ...] = cu_q[:, src_start:src_end, ...]
            new_k[:, dest_start:dest_end, ...] = cu_k[:, src_start:src_end, ...]
            new_v[:, dest_start:dest_end, ...] = cu_v[:, src_start:src_end, ...]

        return new_cu_seqlens, new_q, new_k, new_v, pad_lengths

    def unpad_after_conv(
        self,
        conv_cu_seqlens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        cu_v: torch.Tensor,
        pad_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_total_len = cu_seqlens[-1].item()
        orig_q = torch.empty((1, original_total_len, cu_q.shape[-1]), dtype=cu_q.dtype, device=cu_q.device)
        orig_k = torch.empty((1, original_total_len, cu_k.shape[-1]), dtype=cu_k.dtype, device=cu_k.device)
        orig_v = torch.empty((1, original_total_len, cu_v.shape[-1]), dtype=cu_v.dtype, device=cu_v.device)

        num_sequences = len(pad_lengths)
        for i in range(num_sequences):
            dest_start = cu_seqlens[i]
            dest_end = cu_seqlens[i+1]
            src_start = conv_cu_seqlens[i] + pad_lengths[i]
            src_end = conv_cu_seqlens[i+1]

            orig_q[:, dest_start:dest_end, ...] = cu_q[:, src_start:src_end, ...]
            orig_k[:, dest_start:dest_end, ...] = cu_k[:, src_start:src_end, ...]
            orig_v[:, dest_start:dest_end, ...] = cu_v[:, src_start:src_end, ...]
        return orig_q, orig_k, orig_v

    def prepare_recurrent_state(
        self,
        recurrent_state: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        cu_seqlen_all: torch.Tensor,
        reverse_indices: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor | None:
        if recurrent_state is None:
            return None

        if cu_seqlens is None:
            return recurrent_state

        total_len = len(cu_seqlen_all)
        if len(cu_seqlens) != total_len:
            memories = torch.zeros_like(recurrent_state[:self.topk*batch_size])
            mem_id = 0
            for i in range(total_len-1):
                if cu_seqlen_all[i] != cu_seqlen_all[i+1]:
                    memories[mem_id] = recurrent_state[i]
                    mem_id += 1
            assert mem_id == self.topk * batch_size, f"The number of memories {mem_id} is not correct."
        else:
            memories = recurrent_state

        return memories

    def handle_recurrent_state(
        self,
        recurrent_state: torch.Tensor | None,
        recurrent_state_new: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        cu_seqlen_all: torch.Tensor,
        reverse_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        if recurrent_state_new is None:
            return None
        if cu_seqlens is None:
            return recurrent_state_new
        if recurrent_state is None:
            recurrent_state = torch.zeros_like(recurrent_state_new[reverse_indices[1:]-1])
        total_len = len(cu_seqlen_all)
        if len(cu_seqlens) != total_len:
            for i in range(total_len-1):
                if cu_seqlen_all[i] != cu_seqlen_all[i+1]:
                    recurrent_state[i] = recurrent_state_new[reverse_indices[i+1]-1]
        else:
            recurrent_state = recurrent_state_new
        return recurrent_state
