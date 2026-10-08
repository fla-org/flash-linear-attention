# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare fused convolution/L2 with convolution followed by the existing L2 kernel."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
import triton
from triton.testing import do_bench_cudagraph

from fla.modules.convolution import causal_conv1d
from fla.modules.l2norm import l2_norm
from fla.ops.convolution import fused_short_conv
from fla.utils import assert_close


def benchmark(B, T, H, head_dim, packed, modes, repeats, rep_ms):
    torch.manual_seed(42)
    channels = H * head_dim
    x = torch.randn(B, T, channels, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(channels, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(channels, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, 1, T // 4 + 3, T // 2, T], device=x.device) if packed else None
    kwargs = dict(x=x, weight=weight, bias=bias, activation='silu', cu_seqlens=cu_seqlens)

    def separate():
        y, _ = causal_conv1d(**kwargs, backend='triton')
        return l2_norm(x=y.reshape(B, T, H, head_dim), eps=1e-6).reshape_as(x)

    def fused():
        return fused_short_conv(**kwargs, use_norm=True, norm_eps=1e-6, head_dim=head_dim)[0]

    functions = {'separate': separate, 'fused': fused}
    inputs = (x, weight, bias)
    ref = separate()
    ref_grads = torch.autograd.grad(outputs=ref, inputs=inputs, grad_outputs=dy)
    out = fused()
    grads = torch.autograd.grad(outputs=out, inputs=inputs, grad_outputs=dy)
    assert_close('y', ref, out, 1e-3)
    for name, reference, result in zip(('dx', 'dw', 'db'), ref_grads, grads):
        assert_close(name, reference, result, 1e-3)

    results = []
    for mode in modes:
        samples = {provider: [] for provider in functions}

        def run(provider):
            if mode == 'fwd':
                with torch.no_grad():
                    return functions[provider]()
            return torch.autograd.grad(outputs=functions[provider](), inputs=inputs, grad_outputs=dy)

        for provider in functions:
            run(provider)
        torch.cuda.synchronize()
        for repeat in range(repeats):
            providers = ('separate', 'fused') if repeat % 2 == 0 else ('fused', 'separate')
            for provider in providers:
                samples[provider].append(do_bench_cudagraph(lambda: run(provider), rep=rep_ms))
        medians = {provider: statistics.median(times) for provider, times in samples.items()}
        result = dict(
            B=B, T=T, H=H, head_dim=head_dim, packed=packed, mode=mode,
            separate_ms=medians['separate'], fused_ms=medians['fused'],
            speedup=medians['separate'] / medians['fused'], samples_ms=samples,
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', nargs='+', choices=['fwd', 'fwdbwd'], default=['fwd', 'fwdbwd'])
    parser.add_argument('--lengths', nargs='+', type=int, default=[128, 1024, 8192])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-heads', type=int, default=16)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--layouts', nargs='+', choices=['dense', 'varlen'], default=['dense', 'varlen'])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--rep-ms', type=int, default=100)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('--repeats must be at least 3 for alternating paired measurements.')
    if min(args.lengths) < 16 or min(args.batch_size, args.num_heads, args.head_dim, args.rep_ms) < 1:
        parser.error('Sequence lengths must be at least 16; dimensions and timing duration must be positive.')
    if args.batch_size != 1 and 'varlen' in args.layouts:
        parser.error('Packed variable-length inputs require --batch-size 1.')
    if not torch.cuda.is_available():
        parser.error('This benchmark requires an NVIDIA GPU.')

    source_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ['git', '-c', f'safe.directory={source_root}', 'rev-parse', 'HEAD'],
        cwd=source_root, capture_output=True, text=True, check=False,
    ).stdout.strip()
    sources = [
        'fla/ops/convolution/fused_short_conv.py', 'fla/modules/conv/triton/kernels.py',
        'fla/modules/conv/triton/ops.py', 'fla/modules/l2norm.py',
    ]
    metadata = dict(
        commit=commit, gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
        input_dtype='bfloat16', weight_dtype='bfloat16', width=4, activation='silu', bias=True,
        eps=1e-6, seed=42, repeats=args.repeats, rep_ms=args.rep_ms, timing='CUDA graph',
        source_sha256={name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in sources},
        environment={name: os.environ.get(name) for name in (
            'CUDA_VISIBLE_DEVICES', 'FLA_CI_ENV', 'FLA_DISABLE_BACKEND_DISPATCH', 'FLA_GLUON', 'FLA_CONV_GLUON',
            'TRITON_F32_DEFAULT',
        )},
    )
    print(json.dumps(metadata), flush=True)
    results = []
    for layout in args.layouts:
        for length in args.lengths:
            results.extend(benchmark(
                B=args.batch_size,
                T=length,
                H=args.num_heads,
                head_dim=args.head_dim,
                packed=layout == 'varlen',
                modes=args.modes,
                repeats=args.repeats,
                rep_ms=args.rep_ms,
            ))
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(dict(metadata=metadata, results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
