# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

r"""
Benchmark of the gated delta rule forward, backward and forward-mode derivative (JVP).

    python benchmarks/ops/benchmark_gdn_jvp.py [--B 1] [--H 16] [--K 128] [--T 2048] [--D 8] [--dtype fp32]

Rows:
  fwd             chunk_gated_delta_rule (library) and fla::chunk_gated_delta_rule (custom op)
  fwd + 20 bwd    one forward, then 20 torch.autograd.grad calls on the same graph (retain_graph=True)
  jvp D           fla::chunk_gated_delta_rule_jvp along D directions (precision and fusion variants; and with
                  reverse mode with respect to the tangents), against torch.func.vmap(torch.func.jvp) of a
                  plain-PyTorch chunked implementation, compiled (and eager)
Time: median of triton.testing.do_bench; memory: peak allocated during the call minus what was allocated before.
"""

import argparse

import torch
import torch.nn.functional as F
import triton

from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.chunk_jvp import _chunk, _unchunk
from fla.ops.gated_delta_rule.custom_ops import chunk_gated_delta_rule_jvp_op, chunk_gated_delta_rule_op


def torch_chunk_gdn(q, k, v, g, beta, scale, C=64):
    r"""Plain-PyTorch chunked gated delta rule (no initial state): batched WY solve, loop over chunks."""
    T = q.shape[1]
    qc, kc, vc = (_chunk(x, C, 0) for x in (q, k, v))
    bc, gc = _chunk(beta, C, 0), _chunk(g, C, 0)
    G = gc.cumsum(-1)
    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    dG = G[..., :, None] - G[..., None, :]
    M = bc[..., :, None] * (kc @ kc.transpose(-1, -2)) * torch.where(strict, dG, float('-inf')).exp()
    eye = torch.eye(C, dtype=q.dtype, device=q.device)
    A = torch.linalg.solve_triangular(eye + M, eye.expand_as(M), upper=False, unitriangular=True)
    u = A @ (bc[..., None] * vc)
    w = A @ ((bc * G.exp())[..., None] * kc)
    P = (qc @ kc.transpose(-1, -2)) * torch.where(causal, dG, float('-inf')).exp()
    B, H, N = qc.shape[:3]
    h = q.new_zeros(B, H, k.shape[-1], v.shape[-1])
    os = []
    for n in range(N):
        nu = u[:, :, n] - w[:, :, n] @ h
        os.append(scale * (G[:, :, n].exp()[..., None] * (qc[:, :, n] @ h) + P[:, :, n] @ nu))
        GL = G[:, :, n, -1]
        decayed_k = kc[:, :, n] * (GL[..., None] - G[:, :, n]).exp()[..., None]
        h = GL.exp()[..., None, None] * h + decayed_k.transpose(-1, -2) @ nu
    return _unchunk(torch.stack(os, 2), T, 0)


def measure(fn, warmup=5, rep=50):
    torch.cuda.synchronize()
    fn()  # compile / autotune
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode='median')
    return ms, peak / 2**20


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--B', type=int, default=1)
    p.add_argument('--H', type=int, default=16)
    p.add_argument('--K', type=int, default=128)
    p.add_argument('--T', type=int, default=2048)
    p.add_argument('--D', type=int, default=8)
    p.add_argument('--dtype', default='fp32', choices=['fp32', 'bf16'])
    p.add_argument('--skip-torch', action='store_true', help='skip the plain-PyTorch JVP rows')
    a = p.parse_args()
    dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[a.dtype]
    B, H, K, T, D = a.B, a.H, a.K, a.T, a.D
    dev = 'cuda'
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device=dev, dtype=dtype)
    k = F.normalize(torch.randn(B, T, H, K, device=dev), dim=-1).to(dtype)
    v = torch.randn(B, T, H, K, device=dev, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=dev)).to(dtype)
    beta = torch.rand(B, T, H, device=dev).to(dtype)
    tans = [torch.randn(D, *x.shape, device=dev, dtype=dtype) for x in (q, k, v, g, beta)]
    scale = K ** -0.5
    rows = []

    rows.append(('fwd: library chunk_gated_delta_rule', *measure(lambda: chunk_gated_delta_rule(q, k, v, g, beta)[0])))
    rows.append(('fwd: fla::chunk_gated_delta_rule op', *measure(lambda: chunk_gated_delta_rule_op(q, k, v, g, beta))))

    def fwd_bwd(n=20):
        xs = [x.detach().requires_grad_() for x in (q, k, v, g, beta)]
        o = chunk_gated_delta_rule_op(*xs)
        do = torch.ones_like(o)
        for _ in range(n):
            torch.autograd.grad(o, xs, do, retain_graph=True)
    rows.append(('fwd + 20 bwd: op (retain_graph)', *measure(fwd_bwd, warmup=2, rep=20)))
    rows.append(('fwd + 1 bwd: op', *measure(lambda: fwd_bwd(1))))

    rows.append((f'jvp D={D}: fla::chunk_gated_delta_rule_jvp (tf32 dots, fused)',
                 *measure(lambda: chunk_gated_delta_rule_jvp_op(q, k, v, g, beta, *tans))))
    rows.append((f'jvp D={D}: fla::chunk_gated_delta_rule_jvp (tf32, fuse_output=False)',
                 *measure(lambda: chunk_gated_delta_rule_jvp_op(q, k, v, g, beta, *tans, fuse_output=False))))
    rows.append((f'jvp D={D}: fla::chunk_gated_delta_rule_jvp (tf32x3 dots)',
                 *measure(lambda: chunk_gated_delta_rule_jvp_op(q, k, v, g, beta, *tans, input_precision='tf32x3'))))
    rows.append((f'jvp D={D}: fla::chunk_gated_delta_rule_jvp (ieee dots)',
                 *measure(lambda: chunk_gated_delta_rule_jvp_op(q, k, v, g, beta, *tans, input_precision='ieee'))))

    def jvp_bwd_tangents():
        ts = [t.detach().requires_grad_() for t in tans]
        _, o_dot = chunk_gated_delta_rule_jvp_op(q, k, v, g, beta, *ts)
        torch.autograd.grad(o_dot, ts, torch.ones_like(o_dot))
    rows.append((f'jvp D={D} + reverse mode wrt the tangents', *measure(jvp_bwd_tangents, warmup=2, rep=20)))

    if not a.skip_torch:
        def torch_jvp(q, k, v, g, beta, qd, kd, vd, gd, bd):
            def f(q, k, v, g, beta):
                return torch_chunk_gdn(q, k, v, g, beta, scale)
            return torch.func.vmap(lambda *t: torch.func.jvp(f, (q, k, v, g, beta), t),
                                   in_dims=0, out_dims=0)(qd, kd, vd, gd, bd)
        rows.append((f'jvp D={D}: vmap(jvp) plain PyTorch chunked, eager',
                     *measure(lambda: torch_jvp(q, k, v, g, beta, *tans), warmup=1, rep=5)))
        compiled = torch.compile(torch_jvp, fullgraph=True, dynamic=False)
        rows.append((f'jvp D={D}: vmap(jvp) plain PyTorch chunked, compiled',
                     *measure(lambda: compiled(q, k, v, g, beta, *tans), warmup=2, rep=10)))

    print(f'B={B} H={H} K=V={K} T={T} D={D} dtype={a.dtype} GPU={torch.cuda.get_device_name()}')
    print(f"| {'case':<62} | {'ms':>9} | {'peak MiB':>9} |")
    print(f"|{'-' * 64}|{'-' * 11}|{'-' * 11}|")
    for name, ms, mib in rows:
        print(f'| {name:<62} | {ms:9.3f} | {mib:9.1f} |')


if __name__ == '__main__':
    main()
