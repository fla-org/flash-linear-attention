# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib.util
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch
import torch_npu
import triton

from fla.modules import fused_bitlinear as candidate
from fla.modules.norm import layernorm_quant as candidate_norm
from fla.utils import assert_close

root = Path(__file__).resolve().parent


def load_baseline():
    path = Path(__file__).with_name('quant_baseline_reference.py')
    spec = importlib.util.spec_from_file_location('quant_baseline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = load_baseline()
modules = {'baseline': baseline, 'candidate': candidate}
print(json.dumps({'device': torch.npu.get_device_name(), 'torch': torch.__version__,
                  'torch_npu': torch_npu.__version__, 'triton': triton.__version__,
                  'baseline': '2517f5f', 'candidate_quant': '63882398', 'candidate_grpo': '19311fd5',
                  'timing': 'triton.testing.do_bench, 3 alternating rounds, 200ms each'}), flush=True)

cases = [
    (1, 64, False, True, False, False, False, True),
    (7, 50, True, True, False, False, False, True),
    (33, 128, False, True, True, True, False, True),
    (257, 128, True, True, True, True, True, True),
    (17, 64, False, False, True, False, True, True),
    (32, 128, True, False, False, True, True, True),
    (65, 257, False, True, True, True, True, False),
    (129, 2048, True, True, False, False, False, False),
]
dtypes = [(torch.float32, None), (torch.float16, None), (torch.bfloat16, None),
          (torch.float32, torch.float16), (torch.float32, torch.bfloat16)]
bitwise = 0
count = 0
for case, (dtype, amp_dtype) in itertools.product(cases, dtypes):
    T, D, rms, affine, residual, prenorm, fp32_res, linear_bias = case
    torch.manual_seed(42)
    values = [torch.randn(T, D, device='npu', dtype=dtype),
              torch.randn(D, device='npu', dtype=dtype) if affine else None,
              torch.randn(D, device='npu', dtype=dtype) if affine else None,
              torch.randn(32, D, device='npu', dtype=dtype),
              torch.randn(32, device='npu', dtype=dtype) if linear_bias else None,
              torch.randn(T, D, device='npu', dtype=torch.float32 if fp32_res else dtype) if residual else None]
    weights = None
    results = []
    for module in (baseline, candidate):
        tensors = [value.detach().clone().requires_grad_() if value is not None else None for value in values]
        with torch.autocast(device_type='npu', dtype=amp_dtype, enabled=amp_dtype is not None):
            out = module.layer_norm_linear_quant(*tensors, eps=1e-6, prenorm=prenorm,
                                                 residual_in_fp32=fp32_res, is_rms_norm=rms)
        out = out if prenorm else (out,)
        if weights is None:
            weights = tuple(torch.randn_like(value) for value in out)
        grads = torch.autograd.grad(out, [value for value in tensors if value is not None], weights)
        results.append((*out, *grads))
    for index, (expected, actual) in enumerate(zip(*results)):
        assert torch.isfinite(actual).all()
        assert_close(f'parity-{case}-{dtype}-{amp_dtype}-{index}', expected, actual, 0.01)
        bitwise += torch.equal(expected, actual)
        count += 1
print(f'PARITY: 40 cases passed; {bitwise}/{count} tensors bitwise identical', flush=True)

records = []
for T, D, rms, residual in [(1, 2048, True, False), (128, 2048, True, False), (2048, 2048, True, False), (65, 257, False, True)]:
    torch.manual_seed(42)
    dtype = torch.bfloat16
    x = torch.randn(T, D, device='npu', dtype=dtype, requires_grad=True)
    w = torch.randn(D, device='npu', dtype=dtype, requires_grad=True)
    b = torch.randn(D, device='npu', dtype=dtype, requires_grad=True)
    res = torch.randn_like(x, requires_grad=True) if residual else None
    lw = torch.randn(D, D, device='npu', dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)
    inputs = (x, w, b, lw) if res is None else (x, w, b, lw, res)
    work = {}
    for name, module in modules.items():
        norm_module = candidate_norm if module is candidate else module

        def fwd(module=norm_module):
            return module.layer_norm_quant_fwd(x=x, weight=w, bias=b, eps=1e-6, residual=res, is_rms_norm=rms)
        _, mean, rstd, residual_out = fwd()

        def bwd(module=norm_module, mean=mean, rstd=rstd, residual_out=residual_out):
            return module.layer_norm_quant_bwd(dy=dy, x=residual_out, weight=w, bias=b, mean=mean, rstd=rstd,
                                               has_residual=residual, is_rms_norm=rms, x_dtype=x.dtype, recompute_output=True,
                                               **({} if module is candidate_norm else {'eps': 1e-6}))

        def linear(module=module):
            return module.layer_norm_linear_quant(x=x, norm_weight=w, norm_bias=b, linear_weight=lw, linear_bias=None,
                                                  residual=res, is_rms_norm=rms)

        def fwdbwd(linear=linear):
            return torch.autograd.grad(linear(), inputs, dy)
        work[name] = {'norm_fwd': fwd, 'norm_bwd_recompute': bwd, 'linear_fwdbwd': fwdbwd}
        for fn in work[name].values():
            fn()
    for stage in ['norm_fwd', 'norm_bwd_recompute', 'linear_fwdbwd']:
        samples = {name: [] for name in modules}
        for repeat in range(3):
            order = list(modules) if repeat % 2 == 0 else list(reversed(modules))
            for name in order:
                samples[name].append(triton.testing.do_bench(work[name][stage], rep=200, return_mode='median') * 1000)
        row = {'T': T, 'D': D, 'rms': rms, 'residual': residual, 'stage': stage,
               **{name + '_us': statistics.median(values) for name, values in samples.items()}}
        records.append(row)
        print('BENCH: ' + json.dumps(row), flush=True)
Path('quant-validation-benchmark.json').write_text(json.dumps(records, indent=2) + '\n')
