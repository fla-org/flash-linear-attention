# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Temporary same-device diagnostic for the GDN2 backward regression in PR #1404."""

import argparse
import hashlib
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))

    import torch
    import torch_npu
    import triton

    from benchmarks.ops.registry import generate_inputs, get_op
    from fla.ops.gdn2 import chunk_bwd, chunk_fwd, chunk_gdn2
    from fla.utils import IS_NPU

    assert IS_NPU and torch.npu.is_available()
    assert Path(sys.modules['fla'].__file__).resolve().is_relative_to(source)
    args.output.mkdir(parents=True, exist_ok=True)
    config = get_op('chunk_gdn2')

    def make_workload(B, T, H, D, packed):
        torch.manual_seed(1404)
        inputs = generate_inputs(config, B=B, T=T, H=H, D=D, dtype=torch.bfloat16, device='npu')
        do = torch.randn_like(inputs['v'])
        kwargs = dict(config.extra_kwargs)
        if packed:
            kwargs['cu_seqlens'] = torch.tensor([0, 17, 1023, T], device='npu', dtype=torch.int32)

        def workload():
            out = chunk_gdn2(**inputs, **kwargs)[0]
            out.backward(do)
            return out

        return inputs, do, workload

    def time_stages(workload):
        stages = {}

        def measure(fn, name):
            def measured(*args, **kwargs):
                torch.npu.synchronize()
                started = time.perf_counter()
                result = fn(*args, **kwargs)
                torch.npu.synchronize()
                stages.setdefault(name, []).append((time.perf_counter() - started) * 1000)
                return result
            return measured

        targets = (
            (chunk_fwd, 'chunk_local_cumsum', 'forward.log_prefix'),
            (chunk_fwd, 'chunk_gdn2_fwd_intra', 'forward.intra'),
            (chunk_fwd, 'chunk_gated_delta_rule_fwd_h', 'forward.recurrence'),
            (chunk_fwd, 'chunk_gla_fwd_o_gk', 'forward.output'),
            (chunk_bwd, 'recompute_w_u_fwd_gdn2', 'backward.recompute_wy'),
            (chunk_bwd, 'chunk_gated_delta_rule_fwd_h', 'backward.recompute_state'),
            (chunk_bwd, 'chunk_kda_bwd_dAv', 'backward.value'),
            (chunk_bwd, 'chunk_gated_delta_rule_bwd_dhu', 'backward.recurrence'),
            (chunk_bwd, 'chunk_gdn2_bwd_wy_dqkg_fused', 'backward.wy'),
            (chunk_bwd, 'chunk_gdn2_bwd_intra', 'backward.intra'),
            (chunk_bwd, 'chunk_local_cumsum', 'backward.gate_scan'),
            (chunk_bwd, 'gdn2_chunk_gate', 'backward.gate_scan'),
        )
        totals = []
        with ExitStack() as stack:
            for module, attr, name in targets:
                if hasattr(module, attr):
                    stack.enter_context(patch.object(module, attr, measure(getattr(module, attr), name)))
            for _ in range(3):
                torch.npu.synchronize()
                started = time.perf_counter()
                workload()
                torch.npu.synchronize()
                totals.append((time.perf_counter() - started) * 1000)
        assert all(len(stages[name]) == 3 for name in ('backward.intra', 'backward.wy', 'backward.recompute_wy'))
        return stages, totals

    shapes = ((4, 2048, 16, 128, False), (8, 2048, 32, 256, False), (1, 2048, 4, 128, True))
    records = []
    for B, T, H, D, packed in shapes:
        inputs, do, workload = make_workload(B, T, H, D, packed)
        hashes = {
            name: hashlib.sha256(tensor.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, tensor in {**inputs, 'do': do}.items()
        }
        for _ in range(3):
            out = workload()
        torch.npu.synchronize()
        assert torch.isfinite(out).all().item()
        for tensor in inputs.values():
            if tensor.requires_grad:
                assert tensor.grad is not None and torch.isfinite(tensor.grad).all().item()
        samples = []
        for _ in range(5):
            torch.npu.synchronize()
            started = time.perf_counter()
            out = workload()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - started) * 1000)
        assert torch.isfinite(out).all().item()
        for tensor in inputs.values():
            if tensor.requires_grad:
                assert torch.isfinite(tensor.grad).all().item()

        stages, stage_totals = time_stages(workload)
        for tensor in inputs.values():
            if tensor.requires_grad:
                assert torch.isfinite(tensor.grad).all().item()
        record = dict(
            B=B,
            T=T,
            H=H,
            D=D,
            packed=packed,
            samples_ms=samples,
            stage_samples_ms=stages,
            instrumented_total_ms=stage_totals,
            input_hashes=hashes,
        )
        print(json.dumps({'label': args.label, **record}), flush=True)
        records.append(record)
        del workload, inputs, out, do
        torch.npu.empty_cache()

    modules = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'fla' or name.startswith('fla.'):
            filename = getattr(module, '__file__', None)
            if filename:
                path = Path(filename).resolve()
                assert path.is_relative_to(source), (name, path, source)
                modules[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(
        label=args.label,
        commit=args.commit,
        source=str(source),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        device=torch.npu.get_device_name(0),
        timing='Five synchronized wall-clock samples after three warmups; gradients accumulate as in CI benchmark',
        stage_timing='Three separate calls with synchronization around each pipeline stage; excludes no work from samples_ms',
        records=records,
        imported_sources=modules,
    )
    (args.output / f'{args.label}.json').write_text(json.dumps(result, indent=2) + '\n')

    if args.profile:
        collector = Path(__file__).resolve().parents[2] / '.agents/skills/fla-ascend-performance/scripts'
        sys.path.insert(0, str(collector))
        from profile_npu import profile_callable

        for B, T, H, D, packed in shapes:
            inputs, do, workload = make_workload(B, T, H, D, packed)
            for _ in range(3):
                workload()
            torch.npu.synchronize()
            for metric in ('PipeUtilization', 'MemoryUB'):
                profile_callable(
                    workload,
                    name=f'{args.label}-D{D}-{metric}',
                    out_dir=str(args.output),
                    aic_metrics=metric,
                )
            del workload, inputs, do
            torch.npu.empty_cache()


if __name__ == '__main__':
    main()
