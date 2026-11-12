# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gdn2 import chunk_gdn2, naive_recurrent_gdn2
from fla.ops.kda.gate import naive_kda_gate
from fla.utils import IS_AMD, IS_NVIDIA, assert_close, device


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD), reason="CUDA graph capture requires a CUDA/HIP device")
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16], ids=['bf16', 'fp16'])
@pytest.mark.parametrize('disable_recompute', [False, True], ids=['recompute', 'retain'])
@pytest.mark.parametrize('use_gate_in_kernel', [False, True], ids=['log-gates', 'fused-gates'])
def test_chunk_large_decay_graph(dtype, disable_recompute, use_gate_in_kernel):
    """Captured forward and backward read changed gates and initial states on every replay."""
    torch.manual_seed(42)
    shape = (1, 65, 2, 64)
    q, k = [F.normalize(torch.randn(shape, device=device), dim=-1).to(dtype) for _ in range(2)]
    v = torch.randn(1, 65, 2, 32, dtype=dtype, device=device) * 0.5
    b = torch.rand_like(k)
    w = torch.rand_like(v)
    g = torch.full(shape, -1., device=device)
    h0 = torch.randn(1, 2, 64, 32, device=device)
    h0_seed = h0.clone()
    A_log = torch.zeros(2, device=device) if use_gate_in_kernel else None
    dt_bias = torch.zeros(128, device=device) if use_gate_in_kernel else None
    leaves = [q, k, v, g, b, w, h0] + ([A_log, dt_bias] if use_gate_in_kernel else [])
    for x in leaves:
        x.requires_grad_()
    do, dht = torch.randn_like(v), torch.randn_like(h0)

    def step():
        o, ht = chunk_gdn2(
            q=q, k=k, v=v, g=g, b=b, w=w, initial_state=h0, output_final_state=True,
            use_gate_in_kernel=use_gate_in_kernel, A_log=A_log, dt_bias=dt_bias,
            disable_recompute=disable_recompute,
        )
        return o.detach(), ht.detach(), *torch.autograd.grad((o, ht), leaves, (do, dht))

    # finish autotuning on a side stream before capture
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()

    for replay, spike in enumerate((None, 200., 1e8, None), start=1):
        expected_h0 = h0_seed * replay
        with torch.no_grad():
            g.fill_(-1.)
            if spike is not None:
                g[:, [5, 31, 63], :, ::3] = spike if use_gate_in_kernel else -spike
            h0.copy_(expected_h0)
        graph.replay()

        g_ref = naive_kda_gate(g, A_log, dt_bias) if use_gate_in_kernel else g
        ref, ref_ht = naive_recurrent_gdn2(
            q=q, k=k, v=v, g=g_ref, b=b, w=w, initial_state=h0, output_final_state=True,
        )
        expected = (ref, ref_ht, *torch.autograd.grad((ref, ref_ht), leaves, (do, dht)))
        for name, want, got, tolerance in zip(
            ('o', 'ht', 'dq', 'dk', 'dv', 'dg', 'db', 'dw', 'dh0', 'dA_log', 'dt_bias'), expected, actual,
            (0.005, 0.005, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.01, 0.02, 0.02),
        ):
            assert torch.isfinite(got).all(), name
            assert_close(name, want.float(), got.float(), tolerance)
        assert torch.equal(h0, expected_h0)
