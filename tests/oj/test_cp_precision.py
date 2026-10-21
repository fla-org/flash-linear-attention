# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch_npu  # noqa: F401

from fla.ops.common import intracard_cp
from fla.ops.cp import chunk_bwd, chunk_fwd, chunk_merge
from fla.utils import IS_NPU, assert_close, device, testing
from tests.cp.test_state import test_chunk_bwd_gate_head as check_backward_gate_head


def test_cp_ieee(monkeypatch, tmp_path):
    assert IS_NPU
    monkeypatch.setattr(testing, 'FLA_CI_ENV', False)
    assert chunk_fwd.CP_DOT_PRECISION.value == 'ieee'
    assert chunk_bwd.CP_DOT_PRECISION.value == 'ieee'
    assert chunk_merge.CP_DOT_PRECISION.value == 'ieee'
    torch.manual_seed(42)
    T, H, HV, K, V = 256, 1, 2, 64, 64
    k = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16) * 0.05
    w = torch.randn(1, T, HV, K, device=device, dtype=torch.bfloat16) * 0.05
    u = torch.randn(1, T, HV, V, device=device, dtype=torch.bfloat16)
    decay = -0.01 * (torch.arange(T, device=device) % 64 + 1)
    g = decay.view(1, T, 1).expand(1, T, HV).contiguous()
    h0 = torch.randn(1, HV, K, V, device=device, dtype=torch.float32) * 0.1
    hm = intracard_cp.intracard_pre_scan(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=None,
        cu_seqlens=torch.tensor([0, 128, T], device=device, dtype=torch.int32),
        num_subseqs=2,
    )
    states, count = intracard_cp.intracard_merge(
        hm=hm,
        split_info=intracard_cp.SplitSeqInfo(split_seq_ids=[0], start_subseq_idx=[0], num_subseqs=[2]),
        num_non_first=1,
        merge_seq_offsets=[0, 2],
        merge_init_offsets=[0, 1],
        device=k.device,
        initial_state=h0,
    )
    ref_h, _, _ = intracard_cp._raw_chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=h0,
        cu_seqlens=torch.tensor([0, T], device=device, dtype=torch.int32),
    )
    assert count == 1
    assert torch.isfinite(hm).all()
    assert_close('h', ref_h[0, [2]], states, 0.006)
    check_backward_gate_head(monkeypatch=monkeypatch, tmp_path=tmp_path, T=256, H=4, K=64, V=64)
