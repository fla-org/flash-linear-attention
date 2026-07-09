# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.lightning_attn import chunk_lightning_attn, fused_recurrent_lightning_attn
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'dtype', 'chunk_size'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-{}-chunk{}".format(*test))
        for chunk_size in [16, 32, 64]
        for test in [
            (1, 64, 2, 64, torch.float16, chunk_size),
        ]
    ],
)
def test_chunk_with_chunk_size(
    B: int,
    T: int,
    H: int,
    K: int,
    dtype: torch.dtype,
    chunk_size: int,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, K, dtype=dtype, device=device)
    h0 = torch.randn(B, H, K, K, dtype=dtype, device=device)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def run_ref():
        q_, k_, v_, h0_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v, h0))
        o, ht = fused_recurrent_lightning_attn(
            q=q_,
            k=k_,
            v=v_,
            layer_idx=1,
            num_layers=4,
            initial_state=h0_,
            output_final_state=True,
        )
        ((o * do).sum() + (ht * dht).sum()).backward()
        return o, ht, q_.grad, k_.grad, v_.grad, h0_.grad

    def run_tri(chunk_size: int):
        q_, k_, v_, h0_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v, h0))
        o, ht = chunk_lightning_attn(
            q=q_,
            k=k_,
            v=v_,
            layer_idx=1,
            num_layers=4,
            initial_state=h0_,
            output_final_state=True,
            chunk_size=chunk_size,
        )
        ((o * do).sum() + (ht * dht).sum()).backward()
        return o, ht, q_.grad, k_.grad, v_.grad, h0_.grad

    ref_o, ref_ht, ref_dq, ref_dk, ref_dv, ref_dh0 = run_ref()
    tri_o, tri_ht, tri_dq, tri_dk, tri_dv, tri_dh0 = run_tri(chunk_size)

    assert_close(f'o@{chunk_size}', ref_o, tri_o, 0.005)
    assert_close(f'ht@{chunk_size}', ref_ht, tri_ht, 0.005)
    assert_close(f'dq@{chunk_size}', ref_dq, tri_dq, 0.005)
    assert_close(f'dk@{chunk_size}', ref_dk, tri_dk, 0.005)
    assert_close(f'dv@{chunk_size}', ref_dv, tri_dv, 0.005)
    assert_close(f'dh0@{chunk_size}', ref_dh0, tri_dh0, 0.005)
