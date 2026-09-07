# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare eager outputs, gradients, and latency across checkouts on the same GPU.

Run this script with --repo pointing to each checkout. The baseline writes a
tensor snapshot; --reference checks exact eager parity before candidate timing.
Optional graph timing is replay-only and does not include input updates.
"""

import argparse
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


def make_case(op, layout, tokens, heads, dim):
    torch.manual_seed(42)

    def tensor(shape, dtype=torch.bfloat16):
        return (torch.randn(shape, device='cuda', dtype=dtype) * 0.2).requires_grad_()

    cu = None
    sequences = 1
    if layout == 'varlen':
        cu = torch.tensor([0, 63, tokens // 3, tokens - 17, tokens], device='cuda', dtype=torch.long)
        sequences = 4
    if op == 'conv':
        from fla.modules.conv.causal_conv1d import causal_conv1d

        inputs = dict(
            x=tensor((1, tokens, heads * dim)),
            weight=tensor((heads * dim, 4)),
            bias=tensor((heads * dim,)),
            initial_state=tensor((sequences, heads * dim, 4)),
        )
        return causal_conv1d, inputs, dict(
            cu_seqlens=cu, backend='triton', activation='silu', output_final_state=True,
        )

    module = importlib.import_module('fla.ops.gated_delta_rule' if op == 'gdn' else 'fla.ops.kda')
    fn = getattr(module, 'chunk_gated_delta_rule' if op == 'gdn' else 'chunk_kda')
    gate_shape = (1, tokens, heads) if op == 'gdn' else (1, tokens, heads, dim)
    inputs = dict(
        q=tensor((1, tokens, heads, dim)),
        k=tensor((1, tokens, heads, dim)),
        v=tensor((1, tokens, heads, dim)),
        g=tensor(gate_shape, torch.float32),
        beta=tensor((1, tokens, heads)),
        A_log=tensor((heads,), torch.float32),
        dt_bias=tensor((heads if op == 'gdn' else heads * dim,), torch.float32),
        initial_state=tensor((sequences, heads, dim, dim), torch.float32),
    )
    kwargs = dict(
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        output_final_state=True,
        chunk_size=64,
    )
    if op == 'kda':
        kwargs.update(safe_gate=True, lower_bound=-5)
    return fn, inputs, kwargs


def snapshot(outputs, inputs):
    values = {'output': outputs[0], 'state': outputs[1]}
    values.update({f'd_{name}': value.grad for name, value in inputs.items()})
    for name, value in values.items():
        if value is None or not torch.isfinite(value).all():
            raise AssertionError(f'Missing or non-finite result: {name}')
    return {name: value.detach().cpu().clone() for name, value in values.items()}


def measure(fn, iterations, repeats):
    samples = []
    gpu_samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append((time.perf_counter() - begin) * 1000 / iterations)
        gpu_samples.append(start.elapsed_time(end) / iterations)
    return dict(
        wall_ms=statistics.median(samples),
        gpu_ms=statistics.median(gpu_samples),
        wall_samples_ms=samples,
        gpu_samples_ms=gpu_samples,
    )


def run_case(args, op, layout, tokens, reference):
    fn, inputs, kwargs = make_case(op, layout, tokens, args.heads, args.dim)
    graph_kwargs = {'use_graph': True}
    if op == 'conv' and layout == 'varlen':
        graph_kwargs['graph_nt_max'] = (tokens + 63) // 64 + len(kwargs['cu_seqlens']) - 2
    with torch.no_grad():
        outputs = fn(**inputs, **kwargs)
    gradients = tuple(torch.randn_like(value) for value in outputs)

    def step(backward=False, graph=False):
        if backward:
            for value in inputs.values():
                if value.grad is not None:
                    value.grad.zero_()
        with torch.set_grad_enabled(backward):
            result = fn(**inputs, **kwargs, **(graph_kwargs if graph else {}))
            if backward:
                torch.autograd.backward(result, gradients)
        return result

    for _ in range(args.warmup):
        outputs = step(backward=True)
    expected = snapshot(outputs, inputs)
    # retained eager outputs keep AccumulateGrad bound to the default stream
    del outputs
    key = f'{op}/{layout}/T{tokens}/H{args.heads}/D{args.dim}'
    if reference is not None:
        if expected.keys() != reference[key].keys():
            raise AssertionError(f'Result keys differ for {key}')
        for name, value in expected.items():
            torch.testing.assert_close(value, reference[key][name], rtol=0, atol=0, msg=f'{key}: {name}')

    if args.profile_only:
        step()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        step()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        return [], {key: expected}

    rows = []
    for backward in (False, True):
        mode = 'fwdbwd' if backward else 'fwd'

        def eager():
            return step(backward=backward)
        for _ in range(args.warmup):
            eager()
        measurements = [('eager', measure(eager, args.iterations, args.repeats))]
        if args.graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(args.warmup):
                    step(backward=backward, graph=True)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = step(backward=backward, graph=True)
            graph.replay()
            torch.cuda.synchronize()
            # graph correctness is gated against the same eager call before timing
            from fla.utils import assert_close

            actual = snapshot(captured, inputs) if backward else {
                'output': captured[0].detach().cpu(), 'state': captured[1].detach().cpu(),
            }
            for name, value in actual.items():
                if not torch.isfinite(value).all():
                    raise AssertionError(f'Non-finite graph result: {key}/{name}')
                assert_close(f'{key}/{mode}/{name}', expected[name], value, 0.005)
            measurements.append(('graph_replay', measure(graph.replay, args.iterations, args.repeats)))
        for path, timing in measurements:
            rows.append(dict(case=key, op=op, layout=layout, tokens=tokens, mode=mode, path=path, **timing))
            print(f'{key} {mode} {path}: {timing["wall_ms"]:.4f} ms', flush=True)
    return rows, {key: expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--ops', nargs='+', choices=['gdn', 'kda', 'conv'], default=['gdn', 'kda', 'conv'])
    parser.add_argument('--layouts', nargs='+', choices=['dense', 'varlen'], default=['dense', 'varlen'])
    parser.add_argument('--tokens', nargs='+', type=int, default=[512, 2048])
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()
    if min(args.tokens) < 256 or min(args.heads, args.dim, args.warmup, args.iterations, args.repeats) < 1:
        parser.error('token counts must be >= 256; other sizes and iteration counts must be positive')
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    import triton

    import fla

    if not Path(fla.__file__).resolve().is_relative_to(repo):
        raise RuntimeError(f'Wrong checkout imported: {fla.__file__}')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = torch.load(args.reference, weights_only=True) if args.reference is not None else None
    metadata = dict(
        commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        repo=str(repo), device=torch.cuda.get_device_name(), visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
        pytorch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
        dispatch_disabled=os.getenv('FLA_DISABLE_BACKEND_DISPATCH', '0'),
        triton_f32_default=os.getenv('TRITON_F32_DEFAULT', 'unset'),
        settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    rows, tensors = [], {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for op in args.ops:
        for layout in args.layouts:
            for tokens in args.tokens:
                print(f'Checking {op}/{layout}/T{tokens}', flush=True)
                result, values = run_case(args, op, layout, tokens, reference)
                rows.extend(result)
                tensors.update(values)
                args.output.with_suffix('.json').write_text(
                    json.dumps(dict(metadata=metadata, results=rows), indent=2) + '\n',
                )
                torch.save(tensors, args.output.with_suffix('.pt'))


if __name__ == '__main__':
    main()
