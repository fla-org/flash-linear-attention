# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


from unittest.mock import MagicMock, Mock

import pytest
import torch

from fla.ops.common import intracard_cp
from fla.ops.common.intracard_cp import SplitSeqInfo, _raw_chunk_gated_delta_rule_fwd_h, intracard_merge, intracard_pre_scan
from fla.ops.cp import chunk_bwd, chunk_delta_h, chunk_fwd, chunk_merge, context
from fla.ops.cp.chunk_bwd import chunk_cp_bwd_dh
from fla.ops.cp.chunk_delta_h import compress_h0, expand_h0
from fla.ops.cp.chunk_merge import chunk_cp_merge
from fla.ops.cp.context import FLACPContext
from fla.utils import IS_NPU, IS_TF32_SUPPORTED, assert_close, device


@pytest.mark.parametrize(
    ('layout', 'cu_seqlens', 'front_num_seqs', 'state_indices'),
    [
        pytest.param('contiguous', [0, 8], None, [0], id='contiguous-single'),
        pytest.param('contiguous', [0, 3, 8], None, [0], id='contiguous-varlen'),
        pytest.param('zigzag', [0, 4, 8], 1, [0, 1], id='zigzag-single'),
        pytest.param('zigzag', [0, 3, 4, 5, 8], 2, [0, 2], id='zigzag-varlen'),
    ],
)
@pytest.mark.parametrize('state_v_first', [False, True])
def test_compress_expand_h0(layout, cu_seqlens, front_num_seqs, state_indices, state_v_first):
    torch.manual_seed(42)
    context = FLACPContext(cu_seqlens=torch.tensor(cu_seqlens), layout=layout, front_num_seqs=front_num_seqs)
    state_shape = (2, 5, 3) if state_v_first else (2, 3, 5)
    h0 = torch.zeros(context.num_seqs, *state_shape)
    h0[state_indices] = torch.randn(len(state_indices), *state_shape)

    compressed = compress_h0(h0=h0, context=context)
    assert torch.equal(compressed, h0[state_indices])
    assert compressed.untyped_storage().nbytes() == compressed.numel() * compressed.element_size()
    assert torch.equal(expand_h0(h0=compressed, context=context), h0)
    assert compress_h0(h0=None, context=context) is None
    assert expand_h0(h0=None, context=context) is None


@pytest.mark.parametrize('constructor', ['context', 'get', 'build'])
@pytest.mark.parametrize('layout', ['contiguous', 'zigzag'])
def test_affine_chain_precision(monkeypatch, constructor, layout):
    """All context constructors route forward, backward, and merge through the same precision policy."""
    group = object()
    monkeypatch.setattr(context.dist, 'get_rank', Mock(return_value=1))
    monkeypatch.setattr(context.dist, 'get_world_size', Mock(return_value=3))
    cu_seqlens = torch.tensor([0, 12])
    if constructor == 'build':
        cp_context = context.build_cp_context(cu_seqlens=cu_seqlens, group=group, layout=layout)
    else:
        cp_context = context.get_cp_cu_seqlens(cu_seqlens=cu_seqlens, group=group, layout=layout)
        if constructor == 'context':
            cp_context = context.FLACPContext(**vars(cp_context))

    monkeypatch.setattr(chunk_delta_h, '_check_cp_device', Mock())
    launches = []
    for module, name in (
        (chunk_fwd, 'chunk_cp_fwd_kernel_h'),
        (chunk_bwd, 'chunk_cp_bwd_kernel_dh'),
        (chunk_merge, 'chunk_cp_fwd_bwd_kernel_merge'),
    ):
        monkeypatch.setattr(module, '_check_cp_device', Mock())
        kernel = MagicMock()
        monkeypatch.setattr(module, name, kernel)
        launches.append(kernel.__getitem__.return_value)
    fwd, bwd, merge = launches
    monkeypatch.setattr(
        chunk_delta_h,
        'all_gather_into_tensor',
        lambda hm, group: (hm.unsqueeze(0).expand(3, *hm.shape), None),
    )
    k = torch.zeros(1, 4, 1, 2)
    v = torch.zeros(1, 4, 1, 3)
    chunk_delta_h.chunk_gated_delta_rule_fwd_h_pre_process(
        k=k,
        w=k,
        u=v,
        cu_seqlens=cp_context.cu_seqlens,
        context=cp_context,
    )
    chunk_delta_h.chunk_gated_delta_rule_bwd_dhu_pre_process(
        q=k,
        k=k,
        w=k,
        do=v,
        dv=v,
        cu_seqlens=cp_context.cu_seqlens,
        context=cp_context,
    )
    num_parts = 2 if layout == 'zigzag' else 1
    assert fwd.call_count == bwd.call_count == num_parts
    assert merge.call_count == 2 * num_parts
    for call in fwd.call_args_list + bwd.call_args_list + merge.call_args_list:
        assert 'AFFINE_CHAIN_PRECISION' not in call.kwargs


@pytest.mark.skipif(not (IS_TF32_SUPPORTED or IS_NPU), reason='requires NVIDIA Ampere or newer, or Ascend')
@pytest.mark.parametrize('H', [2, 4], ids=['gva', 'mha'])
@pytest.mark.parametrize('vector_gate', [False, True], ids=['scalar-gate', 'vector-gate'])
@pytest.mark.parametrize('state_v_first', [False, True])
@pytest.mark.parametrize(('K', 'V'), [(64, 96), (96, 80), (192, 64), (256, 80)])
def test_intracard_pre_scan_merge(H: int, vector_gate: bool, state_v_first: bool, K: int, V: int):
    torch.manual_seed(42)
    T, HV = 319, 4
    k = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16) * 0.05
    w = torch.randn(1, T, HV, K, device=device, dtype=torch.bfloat16) * 0.05
    u = torch.randn(1, T, HV, V, device=device, dtype=torch.bfloat16)
    decay = -0.01 * (torch.arange(T, device=device) % 64 + 1)
    if vector_gate:
        g, gk = None, decay.view(1, T, 1, 1).expand(1, T, HV, K).contiguous()
    else:
        g, gk = decay.view(1, T, 1).expand(1, T, HV).contiguous(), None
    state_shape = (1, HV, V, K) if state_v_first else (1, HV, K, V)
    h0 = torch.randn(state_shape, device=device, dtype=torch.float32) * 0.1
    cu_seqlens_split = torch.tensor([0, 128, 256, T], device=device, dtype=torch.int32)
    if vector_gate:
        # scalar and vector gates must not share an autotuned pipeline configuration
        intracard_pre_scan(
            k=k,
            w=w,
            u=u,
            g=decay.view(1, T, 1).expand(1, T, HV).contiguous(),
            gk=None,
            cu_seqlens=cu_seqlens_split,
            num_subseqs=3,
        )
    hm = intracard_pre_scan(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        cu_seqlens=cu_seqlens_split,
        num_subseqs=3,
    )
    states, count = intracard_merge(
        hm=hm,
        split_info=SplitSeqInfo(split_seq_ids=[0], start_subseq_idx=[0], num_subseqs=[3]),
        num_non_first=2,
        merge_seq_offsets=[0, 3],
        merge_init_offsets=[0, 2],
        device=k.device,
        initial_state=h0,
        state_v_first=state_v_first,
    )
    ref_h, _, _ = _raw_chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=h0,
        state_v_first=state_v_first,
        cu_seqlens=torch.tensor([0, T], device=device, dtype=torch.int32),
    )
    assert count == 2
    assert torch.isfinite(hm).all()
    assert_close('h', ref_h[0, [2, 4]], states, 0.006)


def test_intracard_affine_chain_precision(monkeypatch):
    monkeypatch.setattr(intracard_cp, '_check_cp_device', Mock())
    fwd, merge = MagicMock(), MagicMock()
    monkeypatch.setattr(chunk_fwd, '_check_cp_device', Mock())
    monkeypatch.setattr(chunk_fwd, 'chunk_cp_fwd_kernel_h', fwd)
    monkeypatch.setattr(intracard_cp, 'chunk_cp_fwd_bwd_kernel_merge', merge)
    k = torch.zeros(1, 4, 1, 2)
    v = torch.zeros(1, 4, 1, 3)
    hm = intracard_pre_scan(
        k=k,
        w=k,
        u=v,
        g=None,
        gk=None,
        cu_seqlens=torch.tensor([0, 2, 4]),
        num_subseqs=2,
    )
    intracard_merge(
        hm=hm,
        split_info=SplitSeqInfo(split_seq_ids=[0], start_subseq_idx=[0], num_subseqs=[2]),
        num_non_first=1,
        merge_seq_offsets=[0, 2],
        merge_init_offsets=[0, 1],
        device=k.device,
    )
    for kernel in (fwd, merge):
        launch = kernel.__getitem__.return_value
        launch.assert_called_once()
        assert 'AFFINE_CHAIN_PRECISION' not in launch.call_args.kwargs


@pytest.mark.skipif(not (IS_TF32_SUPPORTED or IS_NPU), reason='requires NVIDIA Ampere or newer, or Ascend')
@pytest.mark.parametrize(
    ("T", "H", "K", "V"),
    [
        pytest.param(*p, id=f"T{p[0]}-H{p[1]}-K{p[2]}-V{p[3]}")
        for p in [
            (256, 4, 128, 128),
            (256, 8, 64, 64),
            (512, 4, 128, 128),
            (256, 4, 192, 64),
            (256, 4, 256, 80),
            (319, 4, 96, 80),
            (319, 4, 192, 64),
            (319, 4, 256, 80),
        ]
    ],
)
def test_chunk_bwd_gate_head(T: int, H: int, K: int, V: int):
    """Changing one head's vector gate affects only that head's backward summary."""
    torch.manual_seed(42)
    B = 1
    BT = 64

    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
    do = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16) * 0.1
    dv = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16) * 0.1

    gk_zero = torch.zeros(B, T, H, K, device=device, dtype=torch.float32)

    gk_diff = torch.zeros(B, T, H, K, device=device, dtype=torch.float32)
    gk_diff[:, :, 1:, :] = -10.0

    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.long)

    dhm_a = torch.zeros(H, K, V + K, dtype=torch.float32, device=device)
    chunk_cp_bwd_dh(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        dhm=dhm_a,
        g=None,
        gk=gk_zero,
        scale=1.0,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
    )

    dhm_b = torch.zeros(H, K, V + K, dtype=torch.float32, device=device)
    chunk_cp_bwd_dh(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        dhm=dhm_b,
        g=None,
        gk=gk_diff,
        scale=1.0,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
    )

    # head 0 keeps its gate values and must retain the same summary
    dh_h0_diff = (dhm_a[0, :, :V] - dhm_b[0, :, :V]).norm().item()
    assert dh_h0_diff == 0.0, f"Head 0 dh should be identical but diff={dh_h0_diff}"

    # the other heads must respond to their own gates, rather than reusing head 0
    for h in range(1, H):
        dh_diff = (dhm_a[h, :, :V] - dhm_b[h, :, :V]).norm().item()
        dh_norm = dhm_a[h, :, :V].norm().item()
        assert dh_diff > 0, (
            f"Head {h} dh should differ when gk changes from 0 to -10, but diff=0. "
            f"This indicates the kernel is reading head 0's gk for all heads "
            f"(missing i_h * K offset in gk load address)."
        )
        ratio = dh_diff / (dh_norm + 1e-8)
        assert ratio > 0.1, (
            f"Head {h} dh ratio={ratio:.4f} is too small. "
            f"Expected large difference due to exp2(-10) ≈ 0.001 decay."
        )


@pytest.mark.skipif(not IS_TF32_SUPPORTED, reason='requires NVIDIA Ampere or newer')
@pytest.mark.parametrize('K', [96, 192, 256])
@pytest.mark.parametrize('state_v_first', [False, True])
@pytest.mark.parametrize('forward', [False, True], ids=['backward', 'forward'])
def test_chunk_merge(K: int, state_v_first: bool, forward: bool):
    torch.manual_seed(42)
    HV, V = 2, 80
    summaries = torch.randn(3, HV, K, V + K, device=device, dtype=torch.float32) * 0.05
    expected = torch.zeros(HV, K, V, device=device, dtype=torch.float64)
    for rank in ([0, 1] if forward else [2, 1]):
        expected = summaries[rank, :, :, V:].double() @ expected + summaries[rank, :, :, :V].double()
    if state_v_first:
        expected = expected.transpose(-1, -2)
    actual = torch.empty(expected.shape, device=device, dtype=torch.float32)
    chunk_cp_merge(
        h=actual,
        ag_hm=summaries,
        num_ranks=2,
        rank=2 if forward else 0,
        state_v_first=state_v_first,
        forward=forward,
    )
    assert torch.isfinite(actual).all()
    assert_close('h', expected.float(), actual, 0.006)
