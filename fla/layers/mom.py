# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input, unpad_input


def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    gate_layer: torch.Tensor,
    beta_layer: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to
    different batches.

    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation
    of the same intermediary
    tensors for query, key, value tensors.

    Arguments:
        query_layer (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length (`int`):
            Target length.

    Return:
        query_layer (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index
            into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query,
            `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    query_length = query_layer.shape[1]
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = get_unpad_data(attention_mask)
    batch_size, kv_seq_len, dim = key_layer.shape
    v_dim = value_layer.shape[-1]

    key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, dim), indices_k)
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, v_dim), indices_k
    )
    gate_layer = index_first_axis(gate_layer.reshape(batch_size * kv_seq_len, -1), indices_k)
    beta_layer = index_first_axis(beta_layer.reshape(batch_size * kv_seq_len, -1), indices_k)
    if query_length == kv_seq_len:
        query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, dim), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        gate_layer,
        beta_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def transform(
    x: torch.Tensor,
    routing_mask: torch.Tensor,
    num_memories: int,
    selected_memories: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """
    Reorganize token embeddings into memory-aligned chunks.

    Steps:
        - Expand for top-k routing if needed.
        - Mask out padded tokens via `attention_mask`.
        - Sort tokens by (batch, memory).
        - Gather and pad tokens per memory slot.

    Args:
        x: (batch, seq, hidden) input embeddings.
        routing_mask: (batch, seq, num_memories) binary routing mask.
        num_memories: number of memory slots.
        selected_memories: memory indices per token,
            (batch, seq) if k=1 else (batch, seq, topk).
        attention_mask: (batch, seq) valid-token mask.

    Returns:
        transformed_x: (num_memories, batch, max_len, hidden) reorganized tokens.
        truncation_indices: (batch*num_memories, max_len) gather indices.
        sorted_indices: (batch*seq*topk,) global sort order.
        max_len: int, max tokens per memory.
        mask: (batch*num_memories, max_len) validity mask.
        mask_2: (num_memories, batch, max_len) validity mask reshaped.
    """
    if selected_memories.dim() == 3:
        # (batch, seq, topk)
        topk = selected_memories.shape[2]
        # x (batch, seq, hidden)
        x = x.repeat_interleave(topk, dim=1)
        # x (batch, seq * topk, hidden)
        # (batch, seq, topk)
        selected_memories = selected_memories.reshape(selected_memories.shape[0], -1)
        # (batch, seq * topk)

    if attention_mask is not None:
        attention_mask = attention_mask[:, -routing_mask.shape[1]:]
        # mask out the masked tokens
        routing_mask[attention_mask.bitwise_not().unsqueeze(-1).expand(-1, -1, num_memories)] = 0

    b, s, d = x.shape
    x_flat = x.reshape(b * s, d)  # [b*s, d]

    with torch.no_grad():
        batch_indices = torch.arange(b, device=x.device).unsqueeze(-1)
        batch_indices = batch_indices.repeat(1, s).reshape(-1)
        if attention_mask is not None:
            # sort the masked tokens to the end
            batch_indices[attention_mask.repeat_interleave(topk, dim=1).bitwise_not().flatten()] = b
        # (b * s)
        memories_flat = selected_memories.reshape(-1)  # [b*s]

        combined = batch_indices * (memories_flat.max() + 1) + memories_flat
        sorted_indices = combined.argsort()

    x_sorted = x_flat[sorted_indices]  # [b*s, d]
    # (b*s, hidden) -> (b, s, hidd)
    with torch.no_grad():
        # routing_mask (b, s, num_memories)
        batch_memory_tokens = routing_mask.sum(dim=1)
        # (b, num_memories)
        flatten_offset = batch_memory_tokens.flatten().cumsum(dim=0)
        max_len = batch_memory_tokens.max()
        indices = (
            torch.arange(max_len, device=flatten_offset.device).unsqueeze(0).expand(b * num_memories, -1)
            + torch.cat([torch.tensor([0], device=flatten_offset.device), flatten_offset[:-1]], dim=0).unsqueeze(1)
        )
        mask = indices < flatten_offset.unsqueeze(-1)
        truncation_indices = torch.where(mask, indices, torch.zeros_like(indices))

    gathered_x = torch.gather(x_sorted, 0, truncation_indices.reshape(-1).unsqueeze(-1).expand(-1, d))
    transformed_x = gathered_x.reshape(b * num_memories, -1, d).reshape((b, num_memories, max_len, d)).transpose(0, 1)
    # transformed_x = transformed_x * mask.unsqueeze(-1).expand_as(transformed_x)
    # pad_x = torch.zeros((b * num_memories, capacity_len-max_len, d), dtype=transformed_x.dtype, device=transformed_x.device)
    # pad_mask = torch.zeros((b * num_memories, capacity_len-max_len), dtype=transformed_x.dtype, device=transformed_x.device)
    # left pad
    # transformed_x = torch.cat((pad_x, transformed_x), dim=1).reshape((b, num_memories, capacity_len, d)).transpose(0, 1)
    mask_2 = mask.reshape((b, num_memories, max_len)).transpose(0, 1)
    # truncation_indices += capacity_len-max_len
    # if attention_mask is not None:
    #     mask_2

    return transformed_x, truncation_indices, sorted_indices, max_len, mask, mask_2


def reconstruct(
    transformed_x,
    indices: torch.Tensor,
    sorted_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    topk: int,
    routing_weights: torch.Tensor,
    mask: torch.Tensor
):
    '''
    Reconstruct and mix transformed outputs back into the original input sequence shape.

    Key operations:
    1. Reshapes and transposes `transformed_x` to prepare for scattering.
    2. Applies the `mask` to zero out invalid positions.
    3. Uses `torch.scatter_add_` to scatter and sum the transformed outputs back to their original positions
        based on `indices`.
    4. Rearranges the scattered outputs using `sorted_indices` to ensure correct ordering.
    5. Applies the `routing_weights` to weight the outputs.
    6. Sums over the `topk` dimension to produce the final reconstructed output.

    Args:
        transformed_x (torch.Tensor):
            The transformed output tensor from memory units or experts.
            Shape: (num_memories, batch_size, capacity_len, hidden_size)
        indices (torch.Tensor):
            Indices used for scattering the transformed outputs back to their corresponding positions.
            Shape: (batch*num_memories, max_len)
        sorted_indices (torch.Tensor):
            Sorting indices used to rearrange the scattered outputs back into the original sequence order.
            Shape: (batch_size*seq_len*topk)
        batch_size (int):
            The size of the batch.
        seq_len (int):
            The length of the input sequence.
        topk (int):
            The number of top elements selected (`topk`) per token during the selection process.
        routing_weights (torch.Tensor):
            Routing weights assigned to the top-k selected outputs when reconstructing the final output.
            Shape: (batch_size, seq_len, topk)
        mask (torch.Tensor):
            Boolean mask indicating valid positions in the sequence.
            Shape: (batch*num_memories, max_len)

    Returns:
        restored_x (torch.Tensor):
            The reconstructed output tensor in the original input sequence shape.
            Shape: (batch_size, seq_len, hidden_size)
    '''
    transformed_x = transformed_x.transpose(0, 1).reshape(
        (-1, transformed_x.shape[2], transformed_x.shape[3]))
    b, s, k, d = batch_size, seq_len, topk, transformed_x.shape[2]
    gathered_x = transformed_x.reshape(
        (transformed_x.shape[0] * transformed_x.shape[1], transformed_x.shape[2]))
    mask_expanded = mask.reshape(-1).unsqueeze(-1).expand_as(gathered_x)
    gathered_x = gathered_x * mask_expanded

    assert (indices >= 0).all(), "Indices should be non-negative"

    resortd_x = torch.zeros((b * s * k, d), device=gathered_x.device, dtype=gathered_x.dtype).scatter_add_(
        0,
        indices.reshape(-1).unsqueeze(-1).expand(-1, d),
        gathered_x,
    )
    assert (indices < resortd_x.size(0)).all(), "Indices should be less than resortd_x size"

    inverse_indices = sorted_indices.argsort()
    rearranged_x_flat = resortd_x[inverse_indices]
    restored_x = rearranged_x_flat.reshape((b, s * k, d))
    restored_x = restored_x.reshape(b, s, k, d) * routing_weights.reshape(b, s, k).unsqueeze(-1)
    restored_x = restored_x.sum(dim=2)
    return restored_x


class MomAttention(nn.Module):
    """
    The layer implementaion for [MoM: Linear Sequence Modeling with Mixture-of-Memories](https://arxiv.org/abs/2502.13685).
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
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        num_memories: int = 8,
        topk: int = 2,
        capacity: float = 1.0,
        shared_mem: bool = False,
        single_kv_proj: bool = False,
        **kwargs
    ) -> MomAttention:
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
        self.value_dim = int(self.key_dim * self.expand_v)
        self.head_qk_dim = head_dim
        self.head_v_dim = int(head_dim * self.expand_v)
        self.layer_idx = layer_idx
        self.silu = nn.SiLU()

        assert mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.gate = nn.Linear(self.hidden_size, self.num_memories, bias=False)
        if self.single_kv_proj:
            self.shared_k = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.shared_v = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.shared_b = nn.Linear(hidden_size, self.num_heads, bias=False)
            self.shared_a = nn.Linear(hidden_size, self.num_heads, bias=False)
        else:
            self.k_proj = nn.ModuleList([
                nn.Linear(self.hidden_size, self.key_dim, bias=False)
                for _ in range(self.num_memories)
            ])
            self.v_proj = nn.ModuleList([
                nn.Linear(self.hidden_size, self.value_dim, bias=False)
                for _ in range(self.num_memories)
            ])
            self.b_proj = nn.ModuleList([
                nn.Linear(self.hidden_size, self.num_heads, bias=False)
                for _ in range(self.num_memories)
            ])
            self.a_proj = nn.ModuleList([
                nn.Linear(self.hidden_size, self.num_heads, bias=False)
                for _ in range(self.num_memories)
            ])
            if self.shared_mem:
                self.shared_k = nn.Linear(hidden_size, self.key_dim, bias=False)
                self.shared_v = nn.Linear(hidden_size, self.value_dim, bias=False)
                self.shared_b = nn.Linear(hidden_size, self.num_heads, bias=False)
                self.shared_a = nn.Linear(hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
        else:
            raise UserWarning(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
            )
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
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
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            attention_mask = (attention_mask == 1)
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        origin_cu_seqlens = kwargs.get('cu_seqlens', None)
        if origin_cu_seqlens is not None:
            hidden_states, attention_mask = self.cu2pad(hidden_states, origin_cu_seqlens)

        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        # _, q_len = hidden_states.shape[0], hidden_states.shape[1]
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # 🔍 topk gating
        router_logits = self.gate(hidden_states)  # (bsz, q_len, num_memories)
        scores = F.softmax(router_logits, dim=2, dtype=torch.float)
        routing_weights, selected_memories = torch.topk(scores, self.topk, dim=-1)  # (bsz, seq, topk)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)  # we cast back to the input dtype
        routing_weights_full = torch.zeros(
            routing_weights.shape[0],
            routing_weights.shape[1],
            self.num_memories,
            dtype=routing_weights.dtype,
            device=routing_weights.device
        ).scatter(-1, selected_memories, routing_weights)
        routing_mask = routing_weights_full.bool().int()

        # if self.use_output_gate:
        #     o_g = self.g_proj(hidden_states)

        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]

        shared_hidden_states = hidden_states
        hidden_states, indices, sorted_indices, max_len, mask, mask_2 = transform(
            hidden_states, routing_mask, self.num_memories, selected_memories, attention_mask)

        q = self.q_proj(hidden_states)
        if self.single_kv_proj:
            k = self.shared_k(hidden_states)
            v = self.shared_v(hidden_states)
            beta = self.shared_b(hidden_states).sigmoid()
            g = -self.A_log.float().exp() * F.softplus(self.shared_a(hidden_states).float() + self.dt_bias)
        else:
            k = torch.stack([k_expert(hidden_states[i]) for i, k_expert in enumerate(self.k_proj)], dim=0)
            v = torch.stack([v_expert(hidden_states[i]) for i, v_expert in enumerate(self.v_proj)], dim=0)
            beta = torch.stack([b_expert(hidden_states[i]).sigmoid() for i, b_expert in enumerate(self.b_proj)], dim=0)
            g = torch.stack([-self.A_log.float().exp() * F.softplus(a_expert(hidden_states[i]).float() + self.dt_bias)
                            for i, a_expert in enumerate(self.a_proj)], dim=0)

        q, k, v, g, beta, mask_2 = (rearrange(x, 'e b l ... ->  (e b) l ...') for x in (q, k, v, g, beta, mask_2))
        cu_q, cu_k, cu_v, cu_g, cu_beta, indices_q, cu_seqlen_all, max_seq_lens = _upad_input(q, k, v, g, beta, mask_2)
        cu_seqlens, reverse_indices = cu_seqlen_all[0].to(torch.long).unique(return_inverse=True)
        cu_q, cu_k, cu_v, cu_g, cu_beta = (x.unsqueeze(0).contiguous() for x in (cu_q, cu_k, cu_v, cu_g, cu_beta))

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = [None, None], [None, None], [None, None]
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

            conv_cu_seqlens = cu_seqlens
            padded = False
            if self.training:
                conv_cu_seqlens = None
            elif seq_len != 1 and (cu_seqlens[1:] - cu_seqlens[:-1]).min().item() < self.conv_size:
                padded = True
                conv_cu_seqlens, cu_q, cu_k, cu_v, pad_lengths = self.pad_for_conv(cu_seqlens, cu_q, cu_k, cu_v)

            conv_q = self.prepare_recurrent_state(
                conv_state_q[0],
                conv_cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices,
                batch_size
            )
            cu_q, conv_q_new = self.q_conv1d(
                x=cu_q,
                cache=conv_q,
                output_final_state=use_cache,
                cu_seqlens=conv_cu_seqlens,
            )
            conv_state_q[0] = self.handle_recurrent_state(
                conv_state_q[0],
                conv_q_new,
                conv_cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices
            )
            conv_k = self.prepare_recurrent_state(
                conv_state_k[0],
                conv_cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices,
                batch_size
            )
            cu_k, conv_k_new = self.k_conv1d(
                x=cu_k,
                cache=conv_k,
                output_final_state=use_cache,
                cu_seqlens=conv_cu_seqlens,
            )
            conv_state_k[0] = self.handle_recurrent_state(
                conv_state_k[0],
                conv_k_new,
                conv_cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices
            )
            conv_v = self.prepare_recurrent_state(
                conv_state_v[0],
                conv_cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices,
                batch_size
            )
            cu_v, conv_v_new = self.v_conv1d(
                x=cu_v,
                cache=conv_v,
                output_final_state=use_cache,
                cu_seqlens=conv_cu_seqlens,
            )
            conv_state_v[0] = self.handle_recurrent_state(
                conv_state_v[0],
                conv_v_new, conv_cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices
            )

            if padded:
                cu_q, cu_k, cu_v = self.unpad_after_conv(conv_cu_seqlens, cu_seqlens, cu_q, cu_k, cu_v, pad_lengths)

        else:
            q, k, v = self.silu(q), self.silu(k), self.silu(v),

        cu_q, cu_k, cu_v = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', h=self.num_heads), (cu_q, cu_k, cu_v))

        recurrent_state = last_state['recurrent_state'] if last_state is not None else [
            None for _ in range(1 + self.shared_mem)]
        if mode == 'chunk':
            o, recurrent_state_ = chunk_gated_delta_rule(
                q=cu_q,
                k=cu_k,
                v=cu_v,
                g=cu_g,
                beta=cu_beta,
                initial_state=recurrent_state[0],
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
            recurrent_state[0] = self.handle_recurrent_state(
                recurrent_state[0],
                recurrent_state_,
                cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices,
            )

        elif mode == 'fused_recurrent':
            memories = self.prepare_recurrent_state(
                recurrent_state[0],
                cu_seqlens, cu_seqlen_all[0],
                reverse_indices, batch_size
            )
            o, recurrent_state_ = fused_recurrent_gated_delta_rule(
                q=cu_q,
                k=cu_k,
                v=cu_v,
                g=cu_g,
                beta=cu_beta,
                initial_state=memories,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
            recurrent_state[0] = self.handle_recurrent_state(
                recurrent_state[0],
                recurrent_state_,
                cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices,
            )

        o = o.squeeze(0).contiguous()
        o = pad_input(o, indices_q, batch_size*self.num_memories, max_len)
        o = rearrange(o, '(e b) l h d -> e b l (h d)', b=batch_size)
        o = reconstruct(o, indices=indices, sorted_indices=sorted_indices, batch_size=batch_size,
                        seq_len=seq_len, topk=self.topk, routing_weights=routing_weights, mask=mask)
        o = rearrange(o, 'b l (h d) -> b l h d', h=self.num_heads)

        if self.shared_mem:
            shared_o = self.shared_o(shared_hidden_states, attention_mask, recurrent_state,
                                     use_cache, conv_state_q, conv_state_k, conv_state_v)
            o += shared_o

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[2]
            )

        if self.use_output_gate:
            g = rearrange(self.g_proj(shared_hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        if origin_cu_seqlens is not None:
            indices, _, _ = get_unpad_data(attention_mask[:, -seq_len:])
            o = index_first_axis(rearrange(o, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        return o, None, past_key_values, router_logits.view(-1, self.num_memories)

    def shared_o(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        recurrent_state=None,
        use_cache: Optional[bool] = False,
        conv_state_q=[None, None],
        conv_state_k=[None, None],
        conv_state_v=[None, None],
        **kwargs
    ) -> torch.Tensor:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        cu_seqlens = None
        if attention_mask is not None:
            batch_size, q_len = hidden_states.shape[0], hidden_states.shape[1]
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            q, conv_state_q[1] = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q[1],
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k[1] = self.k_conv1d(
                x=self.shared_k(hidden_states),
                cache=conv_state_k[1],
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v[1] = self.v_conv1d(
                x=self.shared_v(hidden_states),
                cache=conv_state_v[1],
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            q = self.silu(self.q_proj(hidden_states))
            k = self.silu(self.shared_k(hidden_states))
            v = self.silu(self.shared_v(hidden_states))

        q, k, v = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', h=self.num_heads), (q, k, v))
        beta = self.shared_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.shared_a(hidden_states).float() + self.dt_bias)

        if mode == 'chunk':
            o, recurrent_state[-1] = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state[-1],
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state[-1] = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state[-1],
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o

    def cu2pad(self, x, cu_seqlens):
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
        x = pad_input(x.squeeze(0), indices, batch_size, max_len)
        return x, attention_mask

    def pad_for_conv(self, cu_seqlens, cu_q, cu_k, cu_v):
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        pad_lengths = torch.clamp(self.conv_size - lengths, min=0)
        new_lengths = lengths + pad_lengths
        new_cu_seqlens = torch.cat([
            torch.tensor([0], device=cu_seqlens.device, dtype=cu_seqlens.dtype),
            torch.cumsum(new_lengths, dim=0)
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

    def unpad_after_conv(self, conv_cu_seqlens, cu_seqlens, cu_q, cu_k, cu_v, pad_lengths):
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

    def prepare_recurrent_state(self, recurrent_state, cu_seqlens, cu_seqlen_all, reverse_indices, batch_size):
        if recurrent_state is None:
            return None

        if cu_seqlens is None:
            return recurrent_state

        total_len = len(cu_seqlen_all)
        if len(cu_seqlens) != total_len:
            # select memories that are activated
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

    def handle_recurrent_state(self, recurrent_state, recurrent_state_new, cu_seqlens, cu_seqlen_all, reverse_indices):
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
