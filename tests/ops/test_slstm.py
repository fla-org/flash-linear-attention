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


def make_inputs(B, T, H, D, dtype, use_h0, scale=1.0):
    torch.manual_seed(42)
    x = (torch.randn(B, T, 4, H, D, dtype=dtype, device=device) * scale).detach().requires_grad_()
    recurrent_weight = (torch.randn(H, 4, D, D, dtype=dtype, device=device) / D ** 0.5).detach().requires_grad_()
    h0 = None
    if use_h0:
        h0 = (
            torch.randn(B, H, D, device=device).tanh(),
            torch.randn(B, H, D, device=device),
            torch.rand(B, H, D, device=device) + 0.5,
            torch.randn(B, H, D, device=device),
        )
        h0 = tuple(s.requires_grad_() for s in h0[:3]) + (h0[3],)
    return x, recurrent_weight, h0


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'use_h0', 'output_final_state', 'dtype'),
    [
        pytest.param(1, 1, 1, 16, False, True, torch.float, id='B1-T1-H1-D16-h0False-ofsTrue-fp32', marks=pytest.mark.smoke),
        pytest.param(2, 63, 2, 32, True, True, torch.float, id='B2-T63-H2-D32-h0True-ofsTrue-fp32', marks=pytest.mark.smoke),
        pytest.param(2, 50, 2, 25, True, False, torch.float, id='B2-T50-H2-D25-h0True-ofsFalse-fp32'),
        pytest.param(2, 100, 4, 24, True, True, torch.float, id='B2-T100-H4-D24-h0True-ofsTrue-fp32'),
        pytest.param(3, 50, 2, 64, False, False, torch.float, id='B3-T50-H2-D64-h0False-ofsFalse-fp32'),
        pytest.param(2, 256, 4, 64, True, True, torch.bfloat16, id='B2-T256-H4-D64-h0True-ofsTrue-bf16'),
    ],
)
def test_fused_recurrent(B, T, H, D, use_h0, output_final_state, dtype):
    x, recurrent_weight, h0 = make_inputs(B, T, H, D, dtype, use_h0)
    dy = torch.randn(B, T, H, D, dtype=dtype, device=device)
    leaves = [x, recurrent_weight] + (list(h0[:3]) if use_h0 else [])

    def run(fn):
        y, ht = fn(x, recurrent_weight, initial_state=h0, output_final_state=output_final_state)
        (y * dy).sum().backward()
        grads = [t.grad for t in leaves]
        for t in leaves:
            t.grad = None
        return y, ht, grads

    ref, ref_ht, ref_grads = run(naive_recurrent_slstm)
    tri, tri_ht, tri_grads = run(fused_recurrent_slstm)
    tol = 1e-4 if dtype == torch.float else 0.01
    assert_close('y', ref, tri, tol)
    if output_final_state:
        for name, r, t in zip('hcnm', ref_ht, tri_ht):
            assert_close(f'h{name}', r, t, tol)
    else:
        assert tri_ht is None
    for name, r, t in zip(['dx', 'd_recurrent_weight', 'dh0', 'dc0', 'dn0'], ref_grads, tri_grads):
        assert_close(name, r, t, tol)


def test_saturated_gates():
    x, recurrent_weight, h0 = make_inputs(2, 48, 2, 19, torch.float, True, scale=10.0)
    dy = torch.randn(2, 48, 2, 19, device=device)
    leaves = [x, recurrent_weight] + list(h0[:3])

    def run(fn):
        y, ht = fn(x, recurrent_weight, initial_state=h0, output_final_state=True)
        (y * dy).sum().backward()
        grads = [t.grad for t in leaves]
        for t in leaves:
            t.grad = None
        return y, ht, grads

    ref, ref_ht, ref_grads = run(naive_recurrent_slstm)
    tri, tri_ht, tri_grads = run(fused_recurrent_slstm)
    assert_close('y', ref, tri, 1e-4)
    for name, r, t in zip('hcnm', ref_ht, tri_ht):
        assert_close(f'h{name}', r, t, 1e-4)
    for name, r, t in zip(['dx', 'd_recurrent_weight', 'dh0', 'dc0', 'dn0'], ref_grads, tri_grads):
        assert_close(name, r, t, 1e-4)


@pytest.mark.parametrize(
    ('cu_seqlens', 'H', 'D', 'use_h0', 'dtype'),
    [
        pytest.param([0, 1, 17, 40, 65], 2, 25, True, torch.float, id='varlen-h0True-D25-fp32', marks=pytest.mark.smoke),
        pytest.param([0, 1, 12, 13, 45], 2, 32, False, torch.float, id='varlen-h0False-D32-fp32'),
        pytest.param([0, 1, 32, 64, 128], 2, 32, True, torch.bfloat16, id='varlen-h0True-D32-bf16'),
    ],
)
def test_fused_recurrent_varlen(cu_seqlens, H, D, use_h0, dtype):
    N, T = len(cu_seqlens) - 1, cu_seqlens[-1]
    _, recurrent_weight, h0 = make_inputs(N, 1, H, D, dtype, use_h0)
    x = torch.randn(1, T, 4, H, D, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn(1, T, H, D, dtype=dtype, device=device)
    cu = torch.tensor(cu_seqlens, dtype=torch.long, device=device)
    leaves = [x, recurrent_weight] + (list(h0[:3]) if use_h0 else [])

    def run(fn, **kwargs):
        y, ht = fn(x, recurrent_weight, initial_state=h0, output_final_state=True, **kwargs)
        (y * dy).sum().backward()
        grads = [t.grad for t in leaves]
        for t in leaves:
            t.grad = None
        return y, ht, grads

    def run_independent():
        ys, hts = [], []
        for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
            h0_i = tuple(s[i:i + 1] for s in h0) if use_h0 else None
            y_i, ht_i = naive_recurrent_slstm(x[:, bos:eos], recurrent_weight, initial_state=h0_i, output_final_state=True)
            ys.append(y_i)
            hts.append(ht_i)
        y = torch.cat(ys, dim=1)
        ht = tuple(torch.cat(s, dim=0) for s in zip(*hts))
        (y * dy).sum().backward()
        grads = [t.grad for t in leaves]
        for t in leaves:
            t.grad = None
        return y, ht, grads

    ref, ref_ht, ref_grads = run_independent()
    naive_var, naive_var_ht, naive_var_grads = run(naive_recurrent_slstm, cu_seqlens=cu)
    tri, tri_ht, tri_grads = run(fused_recurrent_slstm, cu_seqlens=cu)
    tol = 1e-4 if dtype == torch.float else 0.01
    assert_close('naive_var_y', ref, naive_var, 1e-6)
    for name, r, nv in zip('hcnm', ref_ht, naive_var_ht):
        assert_close(f'naive_var_h{name}', r, nv, 1e-6)
    for name, r, nv in zip(['dx', 'd_recurrent_weight', 'dh0', 'dc0', 'dn0'], ref_grads, naive_var_grads):
        assert_close(f'naive_var_{name}', r, nv, 1e-6)

    assert_close('y', ref, tri, tol)
    for name, r, t in zip('hcnm', ref_ht, tri_ht):
        assert_close(f'h{name}', r, t, tol)
    for name, r, t in zip(['dx', 'd_recurrent_weight', 'dh0', 'dc0', 'dn0'], ref_grads, tri_grads):
        assert_close(name, r, t, tol)


def test_state_passing():
    x, recurrent_weight, _ = make_inputs(2, 80, 2, 32, torch.float, False)
    with torch.no_grad():
        ref, ref_ht = fused_recurrent_slstm(x, recurrent_weight, output_final_state=True)
        y1, h1 = fused_recurrent_slstm(x[:, :33], recurrent_weight, output_final_state=True)
        y2, h2 = fused_recurrent_slstm(x[:, 33:], recurrent_weight, initial_state=h1, output_final_state=True)
    assert_close('y', ref, torch.cat([y1, y2], 1), 1e-5)
    for name, r, t in zip('hcnm', ref_ht, h2):
        assert_close(f'h{name}', r, t, 1e-5)


def test_input_validation():
    x, rw, h0 = make_inputs(2, 8, 2, 16, torch.float, True)
    with pytest.raises(ValueError, match='shape'):
        fused_recurrent_slstm(x[:, :, :3], rw)
    with pytest.raises(ValueError, match='recurrent_weight'):
        fused_recurrent_slstm(x, rw[:, :, :8, :8])
    with pytest.raises(ValueError, match='4-tuple'):
        fused_recurrent_slstm(x, rw, initial_state=h0[:3])
    with pytest.raises(ValueError, match='torch.Tensor'):
        fused_recurrent_slstm(x, rw, initial_state=(None, h0[1], h0[2], h0[3]))
    with pytest.raises(ValueError, match='initial_state'):
        fused_recurrent_slstm(x, rw, initial_state=tuple(s[:1] for s in h0))
    cu = torch.tensor([0, 4, 8], dtype=torch.long, device=device)
    with pytest.raises(ValueError, match='batch size'):
        fused_recurrent_slstm(x, rw, cu_seqlens=cu)
    with pytest.raises(ValueError, match='initial_state'):
        fused_recurrent_slstm(x[:1], rw, initial_state=tuple(s[:1] for s in h0), cu_seqlens=cu)


def official_slstm(x, recurrent_weight):
    """The vanilla sLSTM loop of the xLSTM reference code, with its [H, 4 * D, D] recurrent kernel."""
    B, T, _, H, D = x.shape
    R_official = recurrent_weight.permute(0, 1, 3, 2).reshape(H, 4 * D, D)
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
    x, recurrent_weight, _ = make_inputs(2, 40, 2, 16, torch.float, False)
    with torch.no_grad():
        assert_close('y', official_slstm(x, recurrent_weight), fused_recurrent_slstm(x, recurrent_weight)[0], 1e-5)
