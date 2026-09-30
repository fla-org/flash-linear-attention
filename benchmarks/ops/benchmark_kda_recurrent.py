# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare Ascend recurrent KDA with automatic multi-buffering enabled and disabled."""

import argparse
import json
import math
from importlib import import_module
from importlib.metadata import version
from itertools import accumulate, product
from pathlib import Path
from statistics import median
from unittest.mock import patch

import torch
import torch.nn.functional as F
import triton

from benchmarks.ops.run import _do_bench_kw, _get_machine_info
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import FLA_CI_ENV, IS_NPU, assert_close, device, device_torch_lib

SEED = 42
ROUNDS = 7


def prepare_cases():
    torch.manual_seed(SEED)
    cases = []
    workloads = [(False, (1,) * 8), (False, (4, 4)), (True, (1,) * 8), (True, (2, 6))]
    for (packed, lengths), dimension, state_v_first in product(workloads, (64, 128), (False, True)):
        inputs = {
            name: F.normalize(torch.randn(1, sum(lengths), 16, dimension, dtype=torch.float32, device=device), dim=-1)
            for name in ('q', 'k')
        }
        inputs['v'] = torch.randn_like(inputs['q'])
        inputs['g'] = F.logsigmoid(torch.randn_like(inputs['q']))
        inputs['beta'] = torch.rand(1, sum(lengths), 16, dtype=torch.float32, device=device)
        initial_state = torch.randn(len(lengths), 16, dimension, dimension, dtype=torch.float32, device=device)
        outputs, states = [], []
        offset = 0
        for sequence, length in enumerate(lengths):
            output, state = naive_recurrent_kda(
                **{name: tensor[:, offset:offset + length] for name, tensor in inputs.items()},
                initial_state=initial_state[sequence:sequence + 1],
                output_final_state=True,
            )
            outputs.append(output)
            states.append(state)
            offset += length
        if packed:
            inputs['cu_seqlens'] = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)
        else:
            inputs = {name: tensor.reshape(len(lengths), lengths[0], *tensor.shape[2:]) for name, tensor in inputs.items()}
        pool = initial_state.transpose(-1, -2).contiguous() if state_v_first else initial_state.clone()
        inputs.update(initial_state=pool, output_final_state=True, inplace_final_state=False, state_v_first=state_v_first)
        cases.append((
            inputs, torch.cat(outputs, dim=1), torch.cat(states), pool.clone(),
            {'layout': 'packed' if packed else 'dense', 'lengths': lengths, 'H': 16, 'D': dimension,
             'state_v_first': state_v_first, 'multibuffer_true_batch_medians_ms': [], 'multibuffer_false_batch_medians_ms': []},
        ))
    return cases


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', required=True, type=Path)
    args = parser.parse_args()
    if not IS_NPU or FLA_CI_ENV:
        raise RuntimeError('Run on Ascend with FLA_CI_ENV=0 to enforce the numerical gate')
    backend = import_module('fla.ops.kda.backends.triton_ascend.fused_recurrent')
    compile_kwargs = backend.ascend_compile_kwargs()
    machine = _get_machine_info()
    machine.update(triton_ascend_version=version('triton-ascend'), torch_npu_version=version('torch-npu'))
    timing_kwargs = _do_bench_kw()
    numerical_ratio = 0.002
    report = {
        'machine_info': machine, 'results': [], 'status': 'failed', 'dtype': 'float32',
        'seed': SEED, 'rounds': ROUNDS, 'timing_kwargs': timing_kwargs,
        'additional_warmup_iterations': 3, 'numerical_ratio': numerical_ratio,
        'baseline_compile_kwargs': {**compile_kwargs, 'multibuffer': True},
        'candidate_compile_kwargs': {**compile_kwargs, 'multibuffer': False},
    }
    try:
        cases = prepare_cases()
        report['results'] = [row for *_, row in cases]
        for inputs, expected_output, expected_state, original_pool, row in cases:
            for multibuffer in (True, False):
                with patch.object(backend, 'ascend_compile_kwargs', lambda: {**compile_kwargs, 'multibuffer': multibuffer}):
                    output, state = backend.fused_recurrent_kda_fwd_npu(**inputs)
                    state_k_first = state.transpose(-1, -2) if inputs['state_v_first'] else state
                    output = output.reshape_as(expected_output)
                    assert_close('recurrent output', expected_output, output, numerical_ratio)
                    assert_close('recurrent state', expected_state, state_k_first, numerical_ratio)
                    if not torch.equal(original_pool, inputs['initial_state']):
                        raise RuntimeError(f'Initial state changed: {row}')
                    row[f'multibuffer_{str(multibuffer).lower()}_correctness'] = 'passed'
                    for _ in range(report['additional_warmup_iterations']):
                        backend.fused_recurrent_kda_fwd_npu(**inputs)
                    device_torch_lib.synchronize()
            print(f'Correctness passed: {row}', flush=True)
        for round_index in range(ROUNDS):
            for inputs, _, _, _, row in cases:
                for multibuffer in ((True, False) if round_index % 2 == 0 else (False, True)):
                    with patch.object(backend, 'ascend_compile_kwargs', lambda: {**compile_kwargs, 'multibuffer': multibuffer}):
                        sample = triton.testing.do_bench(
                            lambda: backend.fused_recurrent_kda_fwd_npu(**inputs),
                            quantiles=[0.5, 0.2, 0.8],
                            **timing_kwargs,
                        )
                    if not math.isfinite(sample[0]) or sample[0] <= 0:
                        raise RuntimeError(f'Invalid latency sample: {sample}')
                    row[f'multibuffer_{str(multibuffer).lower()}_batch_medians_ms'].append(sample[0])
        rows = [row for *_, row in cases]
        for row in rows:
            row['baseline_median_ms'] = median(row['multibuffer_true_batch_medians_ms'])
            row['candidate_median_ms'] = median(row['multibuffer_false_batch_medians_ms'])
            print(json.dumps(row), flush=True)
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
