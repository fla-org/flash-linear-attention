# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.simple_gla import chunk_simple_gla
from fla.utils import assert_close, device


@pytest.mark.skipif(device != 'cuda', reason='CUDA graph capture requires CUDA')
@pytest.mark.parametrize('state_v_first', [False, True])
def test_chunk_varlen_cudagraph(state_v_first: bool):
    torch.manual_seed(42)
    cu_seqlens_cpu = torch.tensor([0, 63, 129, 256], dtype=torch.int32)
    cu_seqlens = cu_seqlens_cpu.to(device)
    q, k, v = [torch.randn(1, 256, 4, 64, device=device, dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
    g = F.logsigmoid(torch.randn(1, 256, 4, device=device, dtype=torch.bfloat16)).requires_grad_()
    h0 = torch.randn(3, 4, 64, 64, device=device, requires_grad=True)
    do, dht = torch.randn_like(v), torch.randn_like(h0)

    def step():
        o, ht = chunk_simple_gla(
            q=q,
            k=k,
            v=v,
            g=g,
            initial_state=h0,
            output_final_state=True,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
        grads = torch.autograd.grad((o, ht), (q, k, v, g, h0), (do, dht))
        return o, ht, *grads

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = step()
    for _ in range(2):
        with torch.no_grad():
            q.add_(0.125)
        expected = step()
        graph.replay()
        for name, ref, result in zip(('o', 'ht', 'dq', 'dk', 'dv', 'dg', 'dh0'), expected, actual, strict=True):
            assert_close(name, ref, result, 0.005)
