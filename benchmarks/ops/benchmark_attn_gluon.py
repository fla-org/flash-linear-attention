# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare opt-in attention backends after running the attention correctness tests."""

import argparse
import json
import os
import statistics

import torch
import triton
from triton.runtime.errors import OutOfResources

from fla.ops.attn.backends.gluon import AttnGluonBackend
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.attn.parallel import parallel_attn
from fla.ops.backends import _DISPATCH_DISABLED
from fla.utils import IS_TMA_SUPPORTED, get_device_capability


def measure(fn, repeats, duration):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        fn()
    end.record()
    end.synchronize()
    iterations = min(256, max(1, int(duration * 5 / max(start.elapsed_time(end), 0.001))))
    graph = torch.cuda.CUDAGraph()
    # backward capture must use the stream that created the retained forward graph.
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        for _ in range(iterations):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    quantiles = torch.tensor(samples, dtype=torch.float64).quantile(torch.tensor([0.1, 0.9], dtype=torch.float64)).tolist()
    return dict(
        median_us=statistics.median(samples),
        mean_us=statistics.mean(samples),
        std_us=statistics.pstdev(samples),
        min_us=min(samples),
        p10_us=quantiles[0],
        p90_us=quantiles[1],
        samples_us=samples,
    )


def workloads(quick):
    if quick:
        for d in (64, 128, 256, 512):
            yield dict(kind='parallel', B=1, T=2048, H=8, HQ=8, K=d, V=d, varlen=False, features=False)
        yield dict(kind='parallel', B=1, T=2048, H=8, HQ=8, K=128, V=128, varlen=True, features=False)
        yield dict(kind='decode', B=1, T=8192, H=4, HQ=32, K=128, V=128, varlen=True, features=False)
        return
    for t in (2048, 8192):
        for varlen in (False, True):
            yield dict(kind='parallel', B=1, T=t, H=32, HQ=32, K=128, V=128, varlen=varlen, features=False)
    for k, v in ((64, 64), (256, 256), (256, 512), (512, 256), (512, 512)):
        for varlen in (False, True):
            yield dict(kind='parallel', B=1, T=4096, H=8, HQ=8, K=k, V=v, varlen=varlen, features=False)
    for varlen in (False, True):
        yield dict(kind='parallel', B=1, T=4096, H=4, HQ=32, K=128, V=128, varlen=varlen, features=True)
    for t in (128, 2048, 8192):
        for d in (128, 256, 512):
            for features in (False, True):
                yield dict(kind='decode', B=4, T=t, H=4, HQ=32, K=d, V=d, varlen=True, features=features)


def inputs(case, dtype):
    b, t, h, hq, kdim, vdim = (case[key] for key in ('B', 'T', 'H', 'HQ', 'K', 'V'))
    decode = case['kind'] == 'decode'
    torch.manual_seed(42)
    q = torch.randn((1, b, hq, kdim) if decode else (b, t, hq, kdim), device='cuda', dtype=dtype)
    k = torch.randn((1, b * t, h, kdim) if decode else (b, t, h, kdim), device='cuda', dtype=dtype)
    v = torch.randn(*k.shape[:-1], vdim, device='cuda', dtype=dtype)
    kwargs = dict(q=q, k=k, v=v)
    if case['varlen']:
        lengths = [t] * b if decode else [t // 8, 3 * t // 8, t // 2]
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device='cuda', dtype=torch.int32)
        kwargs['cu_seqlens'] = cu
    if case['features']:
        kwargs['g'] = torch.empty(*k.shape[:2], hq, device='cuda', dtype=torch.float32).uniform_(-0.1, -0.01)
        kwargs['sink_bias'] = torch.randn(hq, device='cuda')
        kwargs['window_size'] = 512
    grad_inputs = []
    if not decode:
        for name in ('q', 'k', 'v', 'g', 'sink_bias'):
            if name in kwargs:
                grad_inputs.append(kwargs[name].requires_grad_())
    return kwargs, grad_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--duration', type=int, default=100, help='Milliseconds per timing sample.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if _DISPATCH_DISABLED:
        parser.error('Unset FLA_DISABLE_BACKEND_DISPATCH before starting the backend comparison.')
    if not AttnGluonBackend.is_available() or get_device_capability()[0] not in (9, 10):
        raise RuntimeError('This benchmark requires a supported Gluon attention device and Triton version.')
    os.environ['FLA_TILELANG'] = '0'
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(
        torch=torch.__version__,
        triton=triton.__version__,
        cuda=torch.version.cuda,
        timer='CUDA graph',
        use_tma=IS_TMA_SUPPORTED,
        results=[],
    )
    with torch.cuda.stream(torch.cuda.Stream()):
        for case in workloads(quick=args.quick):
            kwargs, grad_inputs = inputs(case=case, dtype=getattr(torch, args.dtype))
            op = attn_decoding_one_step if case['kind'] == 'decode' else parallel_attn
            modes = ('fwd',) if case['kind'] == 'decode' else ('fwd', 'bwd', 'fwdbwd')
            do = torch.randn(*kwargs['q'].shape[:-1], case['V'], device='cuda', dtype=getattr(torch, args.dtype))
            for backend in ('triton', 'gluon'):
                os.environ['FLA_ATTN_GLUON'] = str(int(backend == 'gluon'))
                if case['kind'] == 'parallel' and case['K'] > 256 and backend == 'triton':
                    report['results'].append(dict(**case, dtype=args.dtype, backend=backend, status='unsupported K > 256'))
                    continue
                try:
                    output = op(**kwargs)
                except OutOfResources as exc:
                    if backend != 'triton':
                        raise
                    result = dict(**case, dtype=args.dtype, backend=backend, status=str(exc))
                    report['results'].append(result)
                    print(json.dumps(result), flush=True)
                    continue

                def forward():
                    return op(**kwargs)

                def backward():
                    return torch.autograd.grad(output, grad_inputs, do, retain_graph=True)

                def forward_backward():
                    result = op(**kwargs)
                    return torch.autograd.grad(result, grad_inputs, do)

                for mode, fn in zip(modes, (forward, backward, forward_backward)):
                    try:
                        timing = measure(fn=fn, repeats=args.repeats, duration=args.duration)
                    except OutOfResources as exc:
                        if backend != 'triton':
                            raise
                        timing = dict(status=str(exc))
                    result = dict(**case, dtype=args.dtype, backend=backend, mode=mode, **timing)
                    report['results'].append(result)
                    print(json.dumps(result), flush=True)
                    with open(args.output, 'w') as f:
                        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
