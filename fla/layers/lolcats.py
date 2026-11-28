# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import RotaryEmbedding
from fla.ops.linear_attn import chunk_linear_attn
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.cache_utils import Cache


class LoLCATsFeatureMap(nn.Module):
    """Hedgehog feature map with untied heads, `[softmax(x W_h), softmax(-x W_h)]` for every head `h`."""

    def __init__(self, num_heads: int, head_dim: int, feature_dim: int, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.layer = nn.Parameter(torch.empty(num_heads, head_dim, feature_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum('...hd,hdf->...hf', x, self.layer)
        return torch.cat([x.softmax(-1), (-x).softmax(-1)], -1).clamp(min=self.eps)


def window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    valid: torch.Tensor | None = None,
):
    """
    Unnormalized softmax attention of every query over the last `window_size` keys up to itself, computed block by block.
    Returns the numerator `sum_j exp(s_ij - m_i) v_j` and the denominator `sum_j exp(s_ij - m_i)` with the row maximum
    `m_i` of the window, in fp32. Keys with `valid` `[B, T]` false are skipped, except by their own queries.
    """
    B, T, H, D = q.shape
    W = window_size
    N = -(-T // W)
    # queries in blocks of W attend to their own block and the previous one
    q = F.pad(q.float(), (0, 0, 0, 0, 0, N * W - T)).view(B, N, W, H, D)
    k, v = (F.pad(x.float(), (0, 0, 0, 0, W, N * W - T)) for x in (k, v))
    k, v = (x.unfold(1, 2 * W, W).permute(0, 1, 4, 2, 3) for x in (k, v))
    s = torch.einsum('bnihd,bnjhd->bnhij', q, k) * D ** -0.5
    i = torch.arange(W, device=q.device)[:, None] + W
    j = torch.arange(2 * W, device=q.device)[None, :]
    # keys before the start of the sequence are padding
    pos = torch.arange(N, device=q.device)[:, None, None] * W + j - W
    mask = ((j <= i) & (j > i - W) & (pos >= 0))[None, :, None]
    if valid is not None:
        valid = F.pad(valid, (W, N * W - T)).unfold(1, 2 * W, W)
        mask = mask & (valid[:, :, None, None, :] | (j == i))
    s = s.masked_fill(~mask, float('-inf'))
    a = (s - s.amax(-1, keepdim=True)).exp()
    num = torch.einsum('bnhij,bnjhd->bnihd', a, v).reshape(B, N * W, H, D)[:, :T]
    den = a.sum(-1).transpose(2, 3).reshape(B, N * W, H)[:, :T]
    return num, den


class LoLCATsAttention(nn.Module):
    """
    The linearized attention of LoLCATs (https://arxiv.org/abs/2410.10254): softmax attention over a sliding window of
    `window_size` tokens and linear attention with Hedgehog feature maps over all earlier tokens, normalized jointly,

        o_i = (w * sum_{window} exp(s_ij - m_i) v_j + l * sum_{earlier} phi(q_i)^T phi(k_j) v_j)
            / (w * sum_{window} exp(s_ij - m_i) + l * sum_{earlier} phi(q_i)^T phi(k_j)),

    where `m_i` is the maximum score in the window, `w = sigmoid(window_factors)` per head, and `l = 1 - w` with
    `affine_attention_factors` or 1 otherwise. The projections and rotary embeddings follow Llama, so pretrained
    attention weights can be loaded directly before the feature maps are trained.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        num_heads (int, Optional):
            The number of query heads. Default: 32.
        num_kv_heads (int, Optional):
            The number of key/value heads for grouped-query attention. Default: `None`, i.e., `num_heads`.
        feature_dim (int, Optional):
            The output size of each half of the feature maps. Default: 64.
        window_size (int, Optional):
            The size of the softmax attention window. Default: 64.
        affine_attention_factors (bool, Optional):
            Whether to weight the linear attention by `1 - w` instead of 1. Default: `False`.
        init_window_factor (float, Optional):
            The initial value of `window_factors` before the sigmoid. Default: 0.
        rope_theta (float, Optional):
            The base of the rotary embeddings. Default: 10000.
        max_position_embeddings (int, Optional):
            The maximum number of positions of the rotary embeddings. Default: `None`.
        layer_idx (int, Optional):
            The index of the layer. Default: `None`.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        feature_dim: int = 64,
        window_size: int = 64,
        affine_attention_factors: bool = False,
        init_window_factor: float = 0.,
        rope_theta: float = 10000.,
        max_position_embeddings: int | None = None,
        layer_idx: int | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads}).")
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.affine_attention_factors = affine_attention_factors
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=rope_theta)
        # the key feature maps act on the keys repeated for every query head
        self.feature_map_q = LoLCATsFeatureMap(num_heads, self.head_dim, feature_dim)
        self.feature_map_k = LoLCATsFeatureMap(num_heads, self.head_dim, feature_dim)
        self.init_window_factor = init_window_factor
        self.window_factors = nn.Parameter(torch.empty(num_heads))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.window_factors, self.init_window_factor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, Cache | None]:
        if attention_mask is not None:
            assert attention_mask.dim() == 2, "Expected a 0-1 padding mask of shape [batch_size, seq_len]."
        if kwargs.get('cu_seqlens') is not None:
            raise NotImplementedError("LoLCATsAttention does not support variable-length inputs yet.")
        B, T, _ = hidden_states.shape
        last_state = get_layer_cache(self, past_key_values)
        if last_state is not None and T != 1:
            raise NotImplementedError("LoLCATsAttention only supports one token per step once the cache has content.")

        seqlen_offset = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        max_seqlen = T + seqlen_offset
        valid = None
        if attention_mask is not None:
            # with left padding, positions start at the first token of every sequence
            seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
            valid = attention_mask.bool()
        q, k, v = self._qkv(hidden_states, seqlen_offset, max_seqlen)
        w, l = self._factors()
        if last_state is None:
            o, state = self._prefill(
                q, k, v, w, l, use_cache and past_key_values is not None, valid=valid[:, -T:] if valid is not None else None,
            )
            if state is not None:
                update_layer_cache(
                    self,
                    past_key_values,
                    recurrent_state=state,
                    attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                    offset=T,
                    cache_kwargs=dict(window_size=self.window_size),
                )
        else:
            o = self._decode(q, k, v, w, l, last_state, past_key_values, valid)
        o = self.o_proj(o.to(hidden_states.dtype).flatten(-2, -1))
        return o, None, past_key_values

    def attention_transfer_loss(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        The mean squared error between the attention outputs (before `o_proj`) of this layer and of causal softmax
        attention with the same projections, which trains the feature maps and window factors in the first stage of
        LoLCATs.
        """
        q, k, v = self._qkv(hidden_states, 0, hidden_states.shape[1])
        k, v = self._repeat(k), self._repeat(v)
        with torch.no_grad():
            target = F.scaled_dot_product_attention(
                q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(), is_causal=True,
            ).transpose(1, 2)
        o, _ = self._prefill(q, k, v, *self._factors(), False, repeated=True)
        return F.mse_loss(o, target)

    def _qkv(self, hidden_states: torch.Tensor, seqlen_offset: int | torch.Tensor, max_seqlen: int):
        q, k, v = (
            rearrange(proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        max_seqlen = max(max_seqlen, self.max_position_embeddings or 0)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen)
        return q, k, v

    def _factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The weights `[H]` of the softmax and linear attention terms."""
        w = torch.sigmoid(self.window_factors.float())
        return w, (1 - w if self.affine_attention_factors else torch.ones_like(w))

    def _repeat(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(self.num_kv_groups, dim=-2)

    def _prefill(self, q, k, v, w, l, output_final_state, repeated=False, valid=None):
        if not repeated:
            k, v = self._repeat(k), self._repeat(v)
        num, den = window_attention(q, k, v, self.window_size, valid)
        fq, fk = self.feature_map_q(q), self.feature_map_k(k)
        if valid is not None:
            fk = fk * valid[..., None, None]
        # every query reads the keys that have left its window, i.e., the keys shifted right by `window_size`
        W = min(self.window_size, k.shape[1])
        fk_far, v_far = (F.pad(x, (0, 0, 0, 0, W, 0))[:, :k.shape[1]] for x in (fk, v))
        o_far, kv_state = chunk_linear_attn(
            fq, fk_far, v_far.to(fq.dtype), scale=1., normalize=False, output_final_state=output_final_state,
        )
        k_sum = fk_far.float().cumsum(1)
        den_far = (fq.float() * k_sum).sum(-1)
        o = (w[:, None] * num + l[:, None] * o_far.float()) / (w * den + l * den_far)[..., None]
        state = (kv_state, k_sum[:, -1]) if output_final_state else None
        return o, state

    def _decode(self, q, k, v, w, l, last_state, past_key_values, valid=None):
        kv_state, k_state = last_state['recurrent_state']
        k_win, v_win = (rearrange(x, '... (h d) -> ... h d', d=self.head_dim) for x in last_state['attn_state'])
        if k_win.shape[1] == self.window_size:
            # the oldest key leaves the window and joins the linear attention state
            fk_old = self.feature_map_k(self._repeat(k_win[:, :1])).float()[:, 0]
            if valid is not None:
                fk_old = fk_old * valid[:, -self.window_size - 1, None, None]
            v_old = self._repeat(v_win[:, :1]).float()[:, 0]
            kv_state = kv_state + torch.einsum('bhf,bhd->bhfd', fk_old, v_old)
            k_state = k_state + fk_old
        k_win, v_win = (
            rearrange(x, '... (h d) -> ... h d', d=self.head_dim)
            for x in update_layer_cache(
                self,
                past_key_values,
                recurrent_state=(kv_state, k_state),
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                offset=1,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
        )
        k_win, v_win = self._repeat(k_win).float(), self._repeat(v_win).float()
        s = torch.einsum('bqhd,bkhd->bhqk', q.float(), k_win) * self.head_dim ** -0.5
        if valid is not None:
            s = s.masked_fill(~valid[:, None, None, -k_win.shape[1]:], float('-inf'))
        a = (s - s.amax(-1, keepdim=True)).exp()
        num = torch.einsum('bhqk,bkhd->bqhd', a, v_win)
        den = a.sum(-1).transpose(1, 2)
        fq = self.feature_map_q(q).float()
        o_far = torch.einsum('bqhf,bhfd->bqhd', fq, kv_state.float())
        den_far = torch.einsum('bqhf,bhf->bqh', fq, k_state.float())
        return (w[:, None] * num + l[:, None] * o_far) / (w * den + l * den_far)[..., None]
