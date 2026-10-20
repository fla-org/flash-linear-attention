# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Temporary native lowering checks for the GDN2 diagonal backward helper."""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from fla.ops.gdn2.chunk_intra_diagonal import gdn2_bwd_diagonal
from fla.utils import device


@triton.jit
def gather_probe(X, Y, MODE: tl.constexpr):
    i = tl.arange(0, 16)
    k = tl.arange(0, 16)
    if MODE == 0:
        x = tl.load(X + i[:, None].to(tl.int64) * 16 + k[None, :])
        rows = tl.broadcast_to((8 + tl.arange(0, 2))[:, None], (2, 16))
        result = tl.gather(x, rows, axis=0)
        tl.store(Y + tl.arange(0, 2)[:, None].to(tl.int64) * 16 + k[None, :], result)
    elif MODE == 1:
        x = tl.load(X + tl.arange(0, 2)[:, None].to(tl.int64) * 16 + k[None, :])
        indices = tl.broadcast_to((i % 2)[:, None], (16, 16))
        result = tl.gather(x, indices, axis=0)
        tl.store(Y + i[:, None].to(tl.int64) * 16 + k[None, :], result)
    else:
        x = tl.load(X + i[:, None].to(tl.int64) * 16 + k[None, :])
        result = tl.zeros((16, 16), tl.float32)
        indices = tl.broadcast_to((i % 2)[:, None], (16, 16))
        for block in range(8):
            rows = tl.broadcast_to((block * 2 + tl.arange(0, 2))[:, None], (2, 16))
            values = tl.gather(x, rows, axis=0)
            result = tl.where((i // 2 == block)[:, None], tl.gather(values, indices, axis=0), result)
        tl.store(Y + i[:, None].to(tl.int64) * 16 + k[None, :], result)


@triton.jit
def scan_probe(X, Y):
    i = tl.arange(0, 16)
    k = tl.arange(0, 16)
    x = tl.load(X + i[:, None].to(tl.int64) * 16 + k[None, :])
    for block in range(8):
        r = block * 2 + tl.arange(0, 2)
        interval = tl.where(r[:, None, None] > i[None, :, None], x[None, :, :], 0.)
        result = tl.cumsum(interval, axis=1, reverse=True)
        offsets = r[:, None, None].to(tl.int64) * 256 + i[None, :, None] * 16 + k[None, None, :]
        tl.store(Y + offsets, result)


@triton.jit
def diagonal_probe(Q, K, B, G, AQ, AK, OQ, OK, OT):
    i = tl.arange(0, 16)
    channels = 16 + tl.arange(0, 16).to(tl.int64)
    dq, dk, dt = gdn2_bwd_diagonal(Q + 48, K + 48, B + 48, G + 48, AQ + 64, AK + 64,
                                   16, 33, channels, 2, 48, 64, 16)
    offsets = i[:, None].to(tl.int64) * 16 + tl.arange(0, 16)[None, :]
    tl.store(OQ + offsets, dq)
    tl.store(OK + offsets, dk)
    tl.store(OT + offsets, dt)


def main():
    report = {}
    path = Path('gdn2-paired/diagonal-lowering.json')
    path.parent.mkdir(exist_ok=True)

    def compare(name, actual, expected):
        actual = actual.float().cpu()
        expected = expected.float().cpu()
        diff = (actual - expected).abs()
        entry = dict(max_abs=diff.max().item(), passed=torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))
        if actual.ndim >= 2:
            entry['row_max_abs'] = diff.reshape(diff.shape[0], -1).amax(dim=1).tolist()
        if not entry['passed']:
            entry['actual'] = actual.flatten()[:64].tolist()
            entry['expected'] = expected.flatten()[:64].tolist()
        report[name] = entry
        path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({name: entry}), flush=True)

    x = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 16
    for mode, expected in enumerate((x[8:10], x[:2].repeat(8, 1), x)):
        out = torch.full(expected.shape, float('nan'), device=device)
        gather_probe[(1,)](x.to(device), out, mode)
        compare(f'gather_{mode}', out, expected)
    out = torch.full((16, 16, 16), float('nan'), device=device)
    scan_probe[(1,)](x.to(device), out)
    r = torch.arange(16)[:, None, None]
    j = torch.arange(16)[None, :, None]
    expected = torch.where(r > j, x[None], 0).flip(1).cumsum(1).flip(1)
    compare('reverse_scan', out, expected)

    torch.manual_seed(32)
    q, k, b = [torch.randn(33, 2, 48) for _ in range(3)]
    g = -torch.rand_like(q)
    g[16, :, 0] = -1e30
    aq, ak = [torch.randn(33, 2, 64) for _ in range(2)]
    outputs = [torch.full((16, 16), float('nan'), device=device) for _ in range(3)]
    diagonal_probe[(1,)](*[v.to(device) for v in (q, k, b, g, aq, ak)], *outputs)
    expected = [torch.zeros(16, 16, dtype=torch.float64) for _ in range(3)]
    for i in range(16, 32):
        for j in range(16, i + 1):
            decay = g[j + 1:i + 1, 1, 16:32].double().sum(0).exp()
            a, c = aq[i, 1, j].double(), ak[i, 1, j].double()
            expected[0][i - 16] += a * k[j, 1, 16:32].double() * decay
            expected[1][i - 16] += c * k[j, 1, 16:32].double() * decay
            expected[2][j - 16] += (a * q[i, 1, 16:32].double() +
                                    c * k[i, 1, 16:32].double() * b[i, 1, 16:32].double()) * decay
    for name, got, want in zip(('dq', 'dk', 'dkt'), outputs, expected):
        compare(name, got, want)
    path.write_text(json.dumps(report, indent=2) + '\n')
    assert all(row['passed'] for row in report.values()), 'Native lowering probe failed; see diagonal-lowering.json'


if __name__ == '__main__':
    main()
