# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.slstm import fused_recurrent_slstm, naive_recurrent_slstm
from fla.utils import assert_close, device


def make_inputs(B, T, H, D, dtype, use_h0):
    torch.manual_seed(42)
    x = torch.randn(B, T, 4, H, D, dtype=dtype, device=device).requires_grad_()
    R = (torch.randn(H, 4, D, D, device=device) / D ** 0.5).requires_grad_()
    h0 = None
    if use_h0:
        h0 = (
            torch.randn(B, H, D, device=device).tanh(),
            torch.randn(B, H, D, device=device),
            torch.rand(B, H, D, device=device) + 0.5,
            torch.randn(B, H, D, device=device),
        )
        h0 = tuple(s.requires_grad_() for s in h0[:3]) + (h0[3],)
    return x, R, h0


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'use_h0', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-h0{}-{}".format(*test))
        for test in [
            (1, 1, 1, 16, False, torch.float),
            (2, 63, 2, 32, True, torch.float),
            (2, 100, 4, 24, True, torch.float),
            (3, 50, 2, 64, False, torch.float),
            (2, 256, 4, 64, True, torch.bfloat16),
        ]
    ],
)
def test_fused_recurrent(B, T, H, D, use_h0, dtype):
    x, R, h0 = make_inputs(B, T, H, D, dtype, use_h0)
    dy = torch.randn(B, T, H, D, dtype=dtype, device=device)
    leaves = [x, R] + (list(h0[:3]) if use_h0 else [])

    def run(fn):
        y, ht = fn(x, R, initial_state=h0, output_final_state=True)
        (y * dy).sum().backward()
        grads = [t.grad for t in leaves]
        for t in leaves:
            t.grad = None
        return y, ht, grads

    ref, ref_ht, ref_grads = run(naive_recurrent_slstm)
    tri, tri_ht, tri_grads = run(fused_recurrent_slstm)
    tol = 1e-4 if dtype == torch.float else 0.01
    assert_close('y', ref, tri, tol)
    for name, r, t in zip('hcnm', ref_ht, tri_ht):
        assert_close(f'h{name}', r, t, tol)
    for name, r, t in zip(['dx', 'dR', 'dh0', 'dc0', 'dn0'], ref_grads, tri_grads):
        assert_close(name, r, t, tol)


def test_state_passing():
    x, R, _ = make_inputs(2, 80, 2, 32, torch.float, False)
    with torch.no_grad():
        ref, ref_ht = fused_recurrent_slstm(x, R, output_final_state=True)
        y1, h1 = fused_recurrent_slstm(x[:, :33], R, output_final_state=True)
        y2, h2 = fused_recurrent_slstm(x[:, 33:], R, initial_state=h1, output_final_state=True)
    assert_close('y', ref, torch.cat([y1, y2], 1), 1e-5)
    for name, r, t in zip('hcnm', ref_ht, h2):
        assert_close(f'h{name}', r, t, 1e-5)


def official_slstm(x, R):
    """The vanilla sLSTM loop of the xLSTM reference code, with its [H, 4 * D, D] recurrent kernel."""
    B, T, _, H, D = x.shape
    R_official = R.permute(0, 1, 3, 2).reshape(H, 4 * D, D)
    y, c, n, m = (x.new_zeros(B, H * D) for _ in range(4))
    ys = []
    for t in range(T):
        Ry = y.view(B, H, 1, D).matmul(R_official.transpose(1, 2).view(1, H, D, 4 * D))
        Ry = Ry.view(B, H, 4, D).transpose(1, 2).reshape(B, -1)
        iraw, fraw, zraw, oraw = (x[:, t].reshape(B, -1) + Ry).view(B, 4, -1).unbind(1)
        logfplusm = m + F.logsigmoid(fraw)
        mnew = iraw if torch.all(n == 0.0) else torch.max(iraw, logfplusm)
        igate = torch.minimum(torch.exp(iraw - mnew), torch.ones_like(iraw))
        fgate = torch.minimum(torch.exp(logfplusm - mnew), torch.ones_like(iraw))
        c = fgate * c + igate * torch.tanh(zraw)
        n = fgate * n + igate
        m = mnew
        y = torch.sigmoid(oraw) * c / n
        ys.append(y.view(B, H, D))
    return torch.stack(ys, 1)


def test_official_reference():
    x, R, _ = make_inputs(2, 40, 2, 16, torch.float, False)
    with torch.no_grad():
        assert_close('y', official_slstm(x, R), fused_recurrent_slstm(x, R)[0], 1e-5)
