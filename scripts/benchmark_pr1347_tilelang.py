# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import json
import logging
import os
import subprocess
from pathlib import Path

import tilelang
import torch
import triton

import fla
from fla.ops.attn import parallel_attn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

logging.basicConfig(level=logging.INFO)
assert Path(fla.__file__).resolve().parent.parent == Path.cwd()
torch.manual_seed(42)
records = []
for layout in ['dense', 'varlen']:
    batch, length, heads, dim = (2, 512, 4, 64) if layout == 'dense' else (1, 1024, 4, 64)
    q, k, v = [
        torch.randn(batch, length, heads, dim, device='cuda', dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    ]
    grad = torch.randn_like(v)
    cu = None if layout == 'dense' else torch.tensor([0, 257, 512, 1024], device='cuda', dtype=torch.int32)

    def forward():
        with torch.no_grad():
            return parallel_attn(q=q, k=k, v=v, cu_seqlens=cu)

    def forward_backward():
        q.grad = k.grad = v.grad = None
        parallel_attn(q=q, k=k, v=v, cu_seqlens=cu).backward(grad)

    for mode, function in [('fwd', forward), ('fwdbwd', forward_backward)]:
        for _ in range(5):
            function()
        torch.cuda.synchronize()
        samples = [triton.testing.do_bench(function, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8]) for _ in range(3)]
        record = {
            'op': 'parallel_attn', 'layout': layout, 'mode': mode, 'dtype': str(q.dtype),
            'shape': [batch, length, heads, dim], 'samples_ms_q50_q20_q80': samples,
        }
        records.append(record)
        print(json.dumps(record), flush=True)

for layout in ['dense', 'varlen']:
    batch, length, heads, dim = (2, 512, 4, 64) if layout == 'dense' else (1, 1024, 4, 64)
    q, k = [torch.nn.functional.normalize(
        torch.randn(batch, length, heads, dim, device='cuda', dtype=torch.bfloat16), dim=-1,
    ).requires_grad_() for _ in range(2)]
    v = torch.randn_like(q, requires_grad=True)
    g = torch.nn.functional.logsigmoid(torch.randn(batch, length, heads, device='cuda')).requires_grad_()
    beta = torch.randn(batch, length, heads, device='cuda', dtype=torch.bfloat16).sigmoid().requires_grad_()
    grad = torch.randn_like(v)
    cu = None if layout == 'dense' else torch.tensor([0, 257, 512, 1024], device='cuda', dtype=torch.int32)

    def gdn_forward_backward():
        q.grad = k.grad = v.grad = g.grad = beta.grad = None
        output, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu)
        output.backward(grad)

    for _ in range(5):
        gdn_forward_backward()
    torch.cuda.synchronize()
    samples = [
        triton.testing.do_bench(gdn_forward_backward, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
        for _ in range(3)
    ]
    record = {
        'op': 'chunk_gated_delta_rule', 'layout': layout, 'mode': 'fwdbwd', 'dtype': str(q.dtype),
        'shape': [batch, length, heads, dim], 'samples_ms_q50_q20_q80': samples,
    }
    records.append(record)
    print(json.dumps(record), flush=True)
Path(os.environ['FLA_BENCH_RESULT']).write_text(json.dumps({
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'gpu': torch.cuda.get_device_name(),
    'torch': torch.__version__,
    'triton': triton.__version__,
    'tilelang': tilelang.__version__,
    'records': records,
}, indent=2) + '\n')
