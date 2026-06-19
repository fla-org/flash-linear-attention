# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F


def naive_parallel_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    causal: bool = True,
    *,
    g: torch.Tensor | None = None,
    sink_bias: torch.Tensor | None = None,
):
    """
    Reference PyTorch implementation of parallel attention that returns both output and max_logits.

    Args:
        q: [B, T, HQ, D]
        k: [B, T, H, D]
        v: [B, T, H, D]
        scale: float, optional. If None, defaults to 1 / sqrt(D)
        window_size: int, optional. If provided, each query at position i only attends to
            keys in [i - window_size + 1, i]. If None, full causal attention is used.
        causal: bool, default True
        g: [B, T, HQ], optional per-query-head gating logits. Any scaling
            (e.g. matching `attn_decoding_one_step(do_gate_scale=True)`) should
            be applied by the caller before passing `g` in.
        sink_bias: [HQ], optional per-query-head attention-sink bias logits
            (GPT-OSS style). One scalar per query head added to the softmax
            denominator without a corresponding key/value — absorbs probability
            mass but does not contribute to the output.

    Returns:
        output: [B, T, HQ, D]
        max_logits: [B, T, HQ]
    """
    B, T, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H

    if scale is None:
        scale = D ** -0.5

    # reshape q to separate groups: [B, T, HQ, D] -> [B, T, H, G, D]
    q = q.reshape(B, T, H, G, D)

    # compute attention scores via einsum: [B, H, G, T, T]
    # k is [B, T, H, D] — no group dim, so each group shares the same k
    scores = torch.einsum('bqhgd,bkhd->bhgqk', q, k) * scale

    # apply causal mask
    if causal:
        row_idx = torch.arange(T, device=q.device)[None, :, None]
        col_idx = torch.arange(T, device=q.device)[None, None, :]
        mask = col_idx > row_idx
        if window_size is not None:
            mask = mask | (row_idx - col_idx >= window_size)
        scores = scores.masked_fill(mask[:, None, None], float('-inf'))

    if g is not None:
        assert g.shape == (B, T, HQ), "g must have shape [B, T, HQ]"
        g_cumsum = g.float().cumsum(1).reshape(B, T, H, G).permute(0, 2, 3, 1)
        scores = scores + (g_cumsum[..., :, None] - g_cumsum[..., None, :])

    # max_logits: [B, H, G, T] -> [B, T, HQ]
    max_logits = scores.max(dim=-1).values
    if sink_bias is not None:
        assert sink_bias.shape == (HQ,), "sink_bias must have shape [HQ]"
        sink_bias_logits = sink_bias.reshape(H, G)[None, :, :, None]
        max_logits = torch.maximum(max_logits, sink_bias_logits)

    if sink_bias is None:
        # compute output via einsum: [B, H, G, T, T] x [B, T, H, D] -> [B, T, H, G, D]
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhgqk,bkhd->bqhgd', attn_weights, v).reshape(B, T, HQ, D)
    else:
        probs_unnorm = torch.exp(scores - max_logits[..., None])
        sink_bias_unnorm = torch.exp(sink_bias_logits - max_logits)
        denom = probs_unnorm.sum(dim=-1) + sink_bias_unnorm
        output = torch.einsum('bhgqk,bkhd->bqhgd', probs_unnorm, v)
        output = (output / denom.permute(0, 3, 1, 2)[..., None]).reshape(B, T, HQ, D)

    return output, max_logits.permute(0, 3, 1, 2).reshape(B, T, HQ)


def naive_attn_decoding(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    do_gate_scale: bool = False,
    *,
    sink_bias: torch.Tensor | None = None,
):
    """
    Reference PyTorch implementation of packed-varlen decoding attention,
    mirroring `attn_decoding_one_step`. A single query per sequence attends
    to all of its KV (no causal mask needed — query is at the last position).

    Args:
        q: [1, B, HQ, D] — one query token per sequence
        k: [1, T_total, H, D]
        v: [1, T_total, H, D]
        g: [1, T_total, HQ], optional log decay factors
        scale: float, defaults to 1/sqrt(D)
        cu_seqlens: [B+1]
        do_gate_scale: bool, if True scales `g` by `scale` before use
            (matches PaTH / Forgetting Transformer convention).
        sink_bias: [HQ], optional GPT-OSS-style sink bias logits

    Returns:
        o: [1, B, HQ, V]
    """
    assert cu_seqlens is not None, "cu_seqlens must be provided for varlen decoding"
    HQ, D = q.shape[-2], q.shape[-1]
    V = v.shape[-1]
    H = k.shape[2]
    G = HQ // H
    if scale is None:
        scale = D ** -0.5
    if sink_bias is not None:
        assert sink_bias.shape == (HQ,), "sink_bias must have shape [HQ]"

    outputs = []
    for i in range(len(cu_seqlens) - 1):
        bos, eos = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        T_i = eos - bos
        qi = q[:, i:i + 1]  # [1, 1, HQ, D]

        if T_i == 0:
            # no KV for this sequence → output zeros (sink_bias has no value to contribute)
            outputs.append(torch.zeros((1, 1, HQ, V), dtype=q.dtype, device=q.device))
            continue

        ki = k[:, bos:eos]  # [1, T_i, H, D]
        vi = v[:, bos:eos]  # [1, T_i, H, D]

        # scores: [1, H, G, 1, T_i]
        qi_g = qi.reshape(1, 1, H, G, D)
        scores = torch.einsum('bqhgd,bkhd->bhgqk', qi_g, ki) * scale

        if g is not None:
            gi = g[:, bos:eos].float()
            if do_gate_scale:
                gi = gi * scale
            # query is at position T_i - 1 (end of sequence)
            gi_cumsum = gi.cumsum(1).reshape(1, T_i, H, G).permute(0, 2, 3, 1)  # [1, H, G, T_i]
            g_q = gi_cumsum[..., -1:].unsqueeze(-1)  # [1, H, G, 1, 1]
            scores = scores + (g_q - gi_cumsum[..., None, :])

        if sink_bias is None:
            attn_weights = F.softmax(scores, dim=-1)
            oi = torch.einsum('bhgqk,bkhd->bqhgd', attn_weights, vi).reshape(1, 1, HQ, V)
        else:
            sink_bias_logits = sink_bias.reshape(H, G)[None, :, :, None]  # [1, H, G, 1]
            max_logits = torch.maximum(scores.max(dim=-1).values, sink_bias_logits)  # [1, H, G, 1]
            probs_unnorm = torch.exp(scores - max_logits[..., None])
            sink_bias_unnorm = torch.exp(sink_bias_logits - max_logits)
            denom = probs_unnorm.sum(dim=-1) + sink_bias_unnorm  # [1, H, G, 1]
            oi_pack = torch.einsum('bhgqk,bkhd->bqhgd', probs_unnorm, vi)
            oi = (oi_pack / denom.permute(0, 3, 1, 2)[..., None]).reshape(1, 1, HQ, V)

        outputs.append(oi)

    return torch.cat(outputs, dim=1)  # [1, B, HQ, V]
