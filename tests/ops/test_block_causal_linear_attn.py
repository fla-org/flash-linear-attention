# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.block_causal_linear_attn import chunk_block_causal_linear_attn, naive_block_causal_linear_attn
from fla.ops.linear_attn.naive import naive_chunk_linear_attn
from fla.utils import assert_close, device


def make_inputs(B, T, H, D, dtype, use_g, use_h0, N=None):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_()
    k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_()
    v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_()
    g = F.logsigmoid(torch.randn(B, T, H, dtype=dtype, device=device)).requires_grad_() if use_g else None
    h0 = torch.randn(N or B, H, D, D, dtype=torch.float, device=device).requires_grad_() if use_h0 else None
    return q, k, v, g, h0


def grads(o, ht, do, dht, *inputs):
    ((o * do).sum() + (ht * dht).sum()).backward()
    out = []
    for x in inputs:
        if x is not None:
            out.append(x.grad)
            x.grad = None
    return out


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'block_size', 'use_g', 'use_h0', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-bs{}-g{}-h0{}-{}".format(*test))
        for test in [
            (2, 64, 2, 32, 64, False, False, torch.float),
            (2, 100, 2, 32, 32, True, True, torch.float),
            (1, 300, 3, 64, 128, True, True, torch.float),
            (2, 256, 2, 32, 48, True, False, torch.float),
            (2, 200, 2, 64, 256, False, True, torch.float),
            (1, 130, 2, 32, 16, True, True, torch.float),
            (2, 256, 2, 64, 128, True, True, torch.bfloat16),
        ]
    ],
)
def test_chunk(B, T, H, D, block_size, use_g, use_h0, dtype):
    q, k, v, g, h0 = make_inputs(B, T, H, D, dtype, use_g, use_h0)
    do = torch.randn_like(v)
    dht = torch.randn(B, H, D, D, dtype=torch.float, device=device)
    inputs = (q, k, v, g, h0)

    ref, ref_ht = naive_block_causal_linear_attn(q, k, v, g, block_size, initial_state=h0, output_final_state=True)
    ref_grads = grads(ref, ref_ht, do, dht, *inputs)
    tri, tri_ht = chunk_block_causal_linear_attn(q, k, v, g, block_size, initial_state=h0, output_final_state=True)
    tri_grads = grads(tri, tri_ht, do, dht, *inputs)

    tol = 0.005 if dtype == torch.float else 0.02
    assert_close('o', ref, tri, tol)
    assert_close('ht', ref_ht, tri_ht, tol)
    names = ['dq', 'dk', 'dv'] + (['dg'] if use_g else []) + (['dh0'] if use_h0 else [])
    for name, r, t in zip(names, ref_grads, tri_grads):
        assert_close(name, r, t, tol if name != 'dg' else 2 * tol)


def test_mask_matches_linear_attention():
    """Without gates, the reference is plain linear attention under the block-causal mask of the issue."""
    B, T, H, D, block_size = 2, 96, 2, 16, 32
    q, k, v, _, _ = make_inputs(B, T, H, D, torch.float, False, False)
    for bs in [block_size, 1]:
        # block size 1 is ordinary causal linear attention
        blocks = torch.arange(T, device=device) // bs
        mask = blocks[None, :] <= blocks[:, None]
        ref = torch.einsum('bhts,bshv->bthv', torch.einsum('bthk,bshk->bhts', q, k).masked_fill(~mask, 0), v) * D ** -0.5
        assert_close('o', ref, naive_block_causal_linear_attn(q, k, v, block_size=bs)[0], 1e-5)
    assert_close('o', naive_chunk_linear_attn(q[:, :64], k[:, :64], v[:, :64]), naive_block_causal_linear_attn(
        q[:, :64], k[:, :64], v[:, :64], block_size=1)[0], 1e-5)


@pytest.mark.parametrize('use_g', [False, True])
def test_chunk_varlen(use_g):
    H, D, block_size = 2, 32, 64
    cu_seqlens = [0, 15, 100, 128, 300]
    N, T = len(cu_seqlens) - 1, cu_seqlens[-1]
    q, k, v, g, h0 = make_inputs(1, T, H, D, torch.float, use_g, True, N=N)
    do = torch.randn_like(v)
    dht = torch.randn(N, H, D, D, dtype=torch.float, device=device)
    inputs = (q, k, v, g, h0)

    refs, ref_hts = [], []
    for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        o, ht = naive_block_causal_linear_attn(
            q[:, bos:eos], k[:, bos:eos], v[:, bos:eos], g[:, bos:eos] if use_g else None, block_size,
            initial_state=h0[i:i+1], output_final_state=True,
        )
        refs.append(o)
        ref_hts.append(ht)
    ref_grads = grads(torch.cat(refs, 1), torch.cat(ref_hts), do, dht, *inputs)

    tri, tri_ht = chunk_block_causal_linear_attn(
        q, k, v, g, block_size, initial_state=h0, output_final_state=True,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.long, device=device),
    )
    tri_grads = grads(tri, tri_ht, do, dht, *inputs)

    assert_close('o', torch.cat(refs, 1), tri, 0.005)
    assert_close('ht', torch.cat(ref_hts), tri_ht, 0.005)
    names = ['dq', 'dk', 'dv'] + (['dg'] if use_g else []) + ['dh0']
    for name, r, t in zip(names, ref_grads, tri_grads):
        assert_close(name, r, t, 0.005)


def test_invalid_block_size():
    q, k, v, _, _ = make_inputs(1, 32, 1, 16, torch.float, False, False)
    with pytest.raises(ValueError, match="multiple of 16"):
        chunk_block_causal_linear_attn(q, k, v, block_size=24)
