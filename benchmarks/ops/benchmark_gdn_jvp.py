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
from fla.utils import device, device_torch_lib


def torch_chunk_gdn(q, k, v, g, beta, scale, C=64):
    r"""Plain-PyTorch chunked gated delta rule (no initial state): batched WY solve, loop over chunks."""
    T = q.shape[1]
    qc, kc, vc, bc, gc = (_chunk(x, C, lead=0) for x in (q, k, v, beta, g))
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
    return _unchunk(torch.stack(os, 2), T, lead=0)


def measure(fn, warmup=5, rep=50):
    device_torch_lib.synchronize()
    # compile and autotune, then reach a steady state: the peak right after autotuning undercounts
    fn()
    fn()
    device_torch_lib.synchronize()
    device_torch_lib.reset_peak_memory_stats()
    base = device_torch_lib.memory_allocated()
    fn()
    device_torch_lib.synchronize()
    peak = device_torch_lib.max_memory_allocated() - base
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
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1).to(dtype)
    v = torch.randn(B, T, H, K, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device)).to(dtype)
    beta = torch.rand(B, T, H, device=device).to(dtype)
    primals = {'q': q, 'k': k, 'v': v, 'g': g, 'beta': beta}
    tans = [torch.randn(D, *x.shape, device=device, dtype=dtype) for x in (q, k, v, g, beta)]
    tangents = dict(zip(('q_dot', 'k_dot', 'v_dot', 'g_dot', 'beta_dot'), tans))
    scale = K ** -0.5
    rows = []

    rows.append(('fwd: library chunk_gated_delta_rule', *measure(lambda: chunk_gated_delta_rule(**primals)[0])))
    rows.append(('fwd: fla::chunk_gated_delta_rule op', *measure(lambda: chunk_gated_delta_rule_op(**primals))))

    def fwd_bwd(n=20):
        xs = {name: x.detach().requires_grad_() for name, x in primals.items()}
        o = chunk_gated_delta_rule_op(**xs)
        do = torch.ones_like(o)
        for _ in range(n):
            torch.autograd.grad(o, list(xs.values()), do, retain_graph=True)
    rows.append(('fwd + 20 bwd: op (retain_graph)', *measure(fwd_bwd, warmup=2, rep=20)))
    rows.append(('fwd + 1 bwd: op', *measure(lambda: fwd_bwd(1))))

    rows.append((f'jvp D={D}: fla::chunk_gated_delta_rule_jvp (tf32 dots, fused)',
                 *measure(lambda: chunk_gated_delta_rule_jvp_op(**primals, **tangents))))
    rows.append((f'jvp D={D}: fla::chunk_gated_delta_rule_jvp (tf32, fuse_output=False)',
                 *measure(lambda: chunk_gated_delta_rule_jvp_op(**primals, **tangents, fuse_output=False))))
    rows.append((f'jvp D={D}: fla::chunk_gated_delta_rule_jvp (tf32x3 dots)',
                 *measure(lambda: chunk_gated_delta_rule_jvp_op(**primals, **tangents, input_precision='tf32x3'))))
    rows.append((f'jvp D={D}: fla::chunk_gated_delta_rule_jvp (ieee dots)',
                 *measure(lambda: chunk_gated_delta_rule_jvp_op(**primals, **tangents, input_precision='ieee'))))

    def jvp_bwd_tangents():
        ts = {name: t.detach().requires_grad_() for name, t in tangents.items()}
        _, o_dot = chunk_gated_delta_rule_jvp_op(**primals, **ts)
        torch.autograd.grad(o_dot, list(ts.values()), torch.ones_like(o_dot))
    rows.append((f'jvp D={D} + reverse mode wrt the tangents', *measure(jvp_bwd_tangents, warmup=2, rep=20)))

    def jvp_bwd_all():
        xs = {name: x.detach().requires_grad_() for name, x in primals.items()}
        ts = {name: t.detach().requires_grad_() for name, t in tangents.items()}
        o, o_dot = chunk_gated_delta_rule_jvp_op(**xs, **ts)
        torch.autograd.grad((o.sum(), o_dot.sum()), [*xs.values(), *ts.values()])
    rows.append((f'jvp D={D} + reverse mode wrt all inputs', *measure(jvp_bwd_all, warmup=1, rep=5)))

    if not a.skip_torch:
        def torch_jvp(q, k, v, g, beta, qd, kd, vd, gd, bd):
            def f(q, k, v, g, beta):
                return torch_chunk_gdn(q=q, k=k, v=v, g=g, beta=beta, scale=scale)
            return torch.func.vmap(lambda *t: torch.func.jvp(f, (q, k, v, g, beta), t),
                                   in_dims=0, out_dims=0)(qd, kd, vd, gd, bd)
        rows.append((f'jvp D={D}: vmap(jvp) plain PyTorch chunked, eager',
                     *measure(lambda: torch_jvp(q, k, v, g, beta, *tans), warmup=1, rep=5)))
        compiled = torch.compile(torch_jvp, fullgraph=True, dynamic=False)
        rows.append((f'jvp D={D}: vmap(jvp) plain PyTorch chunked, compiled',
                     *measure(lambda: compiled(q, k, v, g, beta, *tans), warmup=2, rep=10)))

    print(f'B={B} H={H} K=V={K} T={T} D={D} dtype={a.dtype} device={device_torch_lib.get_device_name()}')
    print(f"| {'case':<62} | {'ms':>9} | {'peak MiB':>9} |")
    print(f"|{'-' * 64}|{'-' * 11}|{'-' * 11}|")
    for name, ms, mib in rows:
        print(f'| {name:<62} | {ms:9.3f} | {mib:9.1f} |')


if __name__ == '__main__':
    main()
