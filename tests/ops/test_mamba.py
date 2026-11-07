# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.mamba import fused_recurrent_mamba, naive_recurrent_mamba
from fla.utils import assert_close, device


def make_inputs(B, T, D, N, dtype, use_D=True, use_dt_bias=True, use_h0=True, N_seq=None):
    torch.manual_seed(42)
    x = torch.randn(B, T, D, dtype=dtype, device=device)
    dt = torch.randn(B, T, D, dtype=dtype, device=device) - 2
    # S4D-real initialization
    A = -torch.arange(1, N + 1, dtype=torch.float, device=device).repeat(D, 1) * torch.rand(D, 1, device=device)
    Bm = torch.randn(B, T, N, dtype=dtype, device=device)
    Cm = torch.randn(B, T, N, dtype=dtype, device=device)
    D_skip = torch.randn(D, device=device) if use_D else None
    dt_bias = torch.randn(D, device=device) if use_dt_bias else None
    h0 = torch.randn(N_seq or B, D, N, device=device) if use_h0 else None
    return [t.requires_grad_() if t is not None else None for t in (x, dt, A, Bm, Cm, D_skip, dt_bias, h0)]


def run(fn, inputs, dy, dht, dt_softplus=True, **kwargs):
    x, dt, A, B, C, D, dt_bias, h0 = inputs
    y, ht = fn(x, dt, A, B, C, D, dt_bias, dt_softplus, initial_state=h0, output_final_state=True, **kwargs)
    ((y * dy).sum() + (ht * dht).sum()).backward()
    grads = []
    for t in inputs:
        grads.append(None if t is None else t.grad)
        if t is not None:
            t.grad = None
    return y, ht, grads


NAMES = ['dx', 'ddt', 'dA', 'dB', 'dC', 'dD', 'ddt_bias', 'dh0']


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'N', 'use_D', 'use_dt_bias', 'use_h0', 'dt_softplus', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-D{}-N{}-D{}-bias{}-h0{}-softplus{}-{}".format(*test))
        for test in [
            (1, 1, 32, 16, True, True, True, True, torch.float),
            (2, 63, 40, 16, True, True, True, True, torch.float),
            (2, 150, 64, 12, False, True, False, True, torch.float),
            (1, 200, 70, 16, True, False, True, False, torch.float),
            (3, 129, 16, 4, True, True, True, True, torch.float),
            (2, 256, 128, 16, True, True, True, True, torch.bfloat16),
        ]
    ],
)
def test_fused_recurrent(B, T, D, N, use_D, use_dt_bias, use_h0, dt_softplus, dtype):
    inputs = make_inputs(B, T, D, N, dtype, use_D, use_dt_bias, use_h0)
    dy = torch.randn(B, T, D, dtype=dtype, device=device)
    dht = torch.randn(B, D, N, device=device)
    if not dt_softplus:
        # keep the raw time steps positive
        with torch.no_grad():
            inputs[1].abs_().mul_(0.1)
            if inputs[6] is not None:
                inputs[6].abs_()

    ref, ref_ht, ref_grads = run(naive_recurrent_mamba, inputs, dy, dht, dt_softplus)
    tri, tri_ht, tri_grads = run(fused_recurrent_mamba, inputs, dy, dht, dt_softplus)

    tol = 1e-3 if dtype == torch.float else 0.01
    assert_close('y', ref, tri, tol)
    assert_close('ht', ref_ht, tri_ht, tol)
    for name, r, t in zip(NAMES, ref_grads, tri_grads):
        if r is None:
            assert t is None, name
        else:
            assert_close(name, r, t, tol)


def test_fused_recurrent_varlen():
    D, N = 48, 16
    cu_seqlens = [0, 1, 70, 100, 230]
    N_seq, T = len(cu_seqlens) - 1, cu_seqlens[-1]
    inputs = make_inputs(1, T, D, N, torch.float, N_seq=N_seq)
    x, dt, A, Bm, Cm, D_skip, dt_bias, h0 = inputs
    dy = torch.randn(1, T, D, device=device)
    dht = torch.randn(N_seq, D, N, device=device)

    def naive_varlen(x, dt, A, B, C, D, dt_bias, dt_softplus, initial_state, output_final_state):
        ys, hts = [], []
        for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
            y, ht = naive_recurrent_mamba(
                x[:, bos:eos], dt[:, bos:eos], A, B[:, bos:eos], C[:, bos:eos], D, dt_bias, dt_softplus,
                initial_state[i:i+1], output_final_state,
            )
            ys.append(y)
            hts.append(ht)
        return torch.cat(ys, 1), torch.cat(hts)

    ref, ref_ht, ref_grads = run(naive_varlen, inputs, dy, dht)
    tri, tri_ht, tri_grads = run(
        fused_recurrent_mamba, inputs, dy, dht, cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.long, device=device),
    )
    assert_close('y', ref, tri, 1e-3)
    assert_close('ht', ref_ht, tri_ht, 1e-3)
    for name, r, t in zip(NAMES, ref_grads, tri_grads):
        assert_close(name, r, t, 1e-3)


def test_state_passing():
    """Scanning two halves with the state passed between them reproduces a single scan."""
    x, dt, A, Bm, Cm, D_skip, dt_bias, _ = make_inputs(2, 100, 32, 16, torch.float, use_h0=False)
    with torch.no_grad():
        ref, ref_ht = fused_recurrent_mamba(x, dt, A, Bm, Cm, D_skip, dt_bias, output_final_state=True)
        o1, h1 = fused_recurrent_mamba(
            x[:, :37], dt[:, :37], A, Bm[:, :37], Cm[:, :37], D_skip, dt_bias, output_final_state=True,
        )
        o2, h2 = fused_recurrent_mamba(
            x[:, 37:], dt[:, 37:], A, Bm[:, 37:], Cm[:, 37:], D_skip, dt_bias, initial_state=h1, output_final_state=True,
        )
    assert_close('y', ref, torch.cat([o1, o2], 1), 1e-5)
    assert_close('ht', ref_ht, h2, 1e-5)
