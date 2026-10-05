# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare opt-in convolution dispatch against the default path on identical inputs."""

import argparse
import json
import os
import statistics

import torch
import triton
from triton.testing import do_bench_cudagraph

from fla.modules.conv.causal_conv1d import causal_conv1d
from fla.ops.utils import prepare_chunk_indices
from fla.utils import assert_close, device


def benchmark(args):
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)
    weight_dtype = getattr(torch, args.weight_dtype)
    shapes = [(1, 128, 1024), (1, 256, 1024), (1, 257, 1024), (1, 1024, 1024), (1, 1025, 1024), (1, 2048, 2048),
              (1, 8192, 2048), (4, 8192, 4096), (1, 32768, 4096)]
    if args.quick:
        shapes = [(1, 128, 1024), (1, 2048, 2048), (1, 8192, 2048)]
    rows = []
    for packed in [False, True]:
        for B, T, D in shapes:
            if packed and B != 1:
                continue
            x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
            weight = torch.randn(D, 4, device=device, dtype=weight_dtype, requires_grad=True)
            bias = torch.randn(D, device=device, dtype=weight_dtype, requires_grad=True) if args.bias else None
            dy = torch.randn_like(x)
            cu = torch.tensor([0, 1, T // 4 + 3, T // 2, T], device=device) if packed else None
            indices = prepare_chunk_indices(cu, 64) if packed else None
            inputs = (x, weight, bias) if bias is not None else (x, weight)

            def forward():
                return causal_conv1d(
                    x=x,
                    weight=weight,
                    bias=bias,
                    activation='silu',
                    cu_seqlens=cu,
                    chunk_indices=indices,
                )[0]

            outputs, gradients = [], []
            for enabled in ['0', '1']:
                os.environ['FLA_GLUON'] = enabled
                y = forward()
                outputs.append(y.detach())
                gradients.append(torch.autograd.grad(y, inputs, dy))
            assert_close('y', outputs[0], outputs[1], 1e-3)
            for name, expected, actual in zip(('dx', 'dw', 'db'), gradients[0], gradients[1]):
                assert_close(name, expected, actual, 1e-3)
            del y, outputs, gradients

            for mode in ['fwd', 'fwdbwd']:
                def run():
                    y = forward()
                    if mode == 'fwdbwd':
                        torch.autograd.grad(y, inputs, dy)

                samples = {'0': [], '1': []}
                for repeat in range(args.repeats):
                    for enabled in (['0', '1'] if repeat % 2 == 0 else ['1', '0']):
                        os.environ['FLA_GLUON'] = enabled
                        samples[enabled].append(do_bench_cudagraph(run, rep=100) * 1000)
                times = {key: statistics.median(value) for key, value in samples.items()}
                row = dict(
                    B=B,
                    T=T,
                    D=D,
                    packed=packed,
                    mode=mode,
                    triton_us=times['0'],
                    gluon_us=times['1'],
                    speedup=times['0'] / times['1'],
                    samples_us=samples,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--weight-dtype', choices=['float16', 'bfloat16', 'float32'], default='float32')
    parser.add_argument('--bias', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--json')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    old_backends = {name: os.environ.get(name) for name in ['FLA_GLUON', 'FLA_CONV_GLUON']}
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        os.environ['FLA_CONV_GLUON'] = '0'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        metadata = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__, **vars(args))
        print(json.dumps(metadata), flush=True)
        rows = benchmark(args)
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(dict(environment=metadata, results=rows), f, indent=2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        for name, value in old_backends.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == '__main__':
    main()
