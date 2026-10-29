# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F
import triton

from fla.utils import IS_NPU, assert_close, device


@pytest.mark.skipif(not IS_NPU, reason='Ascend compiler options')
def test_chunk_compile_options():
    """Only pass launch controls that the installed compiler applies."""
    from triton.backends.ascend.compiler import AscendBackend
    from triton.runtime import driver

    from fla.ops.gdn2.backends.triton_ascend.chunk_intra import _INTER_COMPILE_KWARGS

    backend = AscendBackend(driver.active.get_current_target())
    for kwargs in (_INTER_COMPILE_KWARGS,):
        options = backend.parse_options(kwargs.copy())
        for name, value in kwargs.items():
            assert getattr(options, name) == value


@pytest.mark.skipif(not IS_NPU, reason='Ascend kernel regression')
@pytest.mark.parametrize(
    ('K', 'dtype', 'varlen'),
    [
        pytest.param(32, torch.float32, False, id='dense-K32-fp32'),
        pytest.param(48, torch.float16, True, id='varlen-K48-fp16'),
        pytest.param(128, torch.bfloat16, False, id='dense-K128-bf16'),
        pytest.param(256, torch.bfloat16, True, id='varlen-K256-bf16'),
    ],
)
def test_chunk_fwd_intra_grouped(K, dtype, varlen):
    """Check causal score rows and structural zeros across ragged chunk tails."""
    from fla.ops.gdn2.backends.triton_ascend.chunk_intra import (
        _launch_diag_kernel,
        chunk_gdn2_fwd_kernel_intra_grouped_npu,
    )
    from fla.ops.utils import prepare_chunk_indices

    torch.manual_seed(42)
    T, H, BT, BC = 145, 16, 64, 16
    bounds = [0, 15, 80, T] if varlen else [0, T]
    q = F.normalize(torch.randn(1, T, H, K, device=device), dim=-1).to(dtype)
    k = F.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1).to(dtype)
    b = torch.rand_like(q)
    g = -torch.rand_like(q, dtype=torch.float32)
    for bos, eos in zip(bounds[:-1], bounds[1:]):
        for start in range(bos, eos, BT):
            end = min(start + BT, eos)
            g[:, start:end] = g[:, start:end].cumsum(dim=1)
    cu_seqlens = torch.tensor(bounds, device=device, dtype=torch.int64) if varlen else None
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if varlen else None
    nt = len(chunk_indices) if varlen else (T + BT - 1) // BT
    Aqk = torch.full((1, T, H, BT), float('nan'), device=device, dtype=dtype)
    Akk = torch.full((1, T, H, BC), float('nan'), device=device, dtype=torch.float32)
    inputs = [q, k, g, b] if varlen else [x.transpose(1, 2).contiguous() for x in (q, k, g, b)]
    scale = K ** -0.5
    kwargs = dict(
        q=inputs[0],
        k=inputs[1],
        g=inputs[2],
        b=inputs[3],
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=triton.next_power_of_2(K),
        BR=8,
        IS_HEAD_MAJOR=not varlen,
        IS_VARLEN=varlen,
    )
    for row_group in range(BC // 8):
        _launch_diag_kernel(
            chunk_gdn2_fwd_kernel_intra_grouped_npu,
            nt=nt,
            nc=BT // BC,
            bh_total=H,
            kernel_kwargs={**kwargs, 'ROW_GROUP': row_group},
        )

    q_ref, k_ref, g_ref, b_ref = [x[0].detach().float().cpu() for x in (q, k, g, b)]
    ref_qk = torch.zeros(1, T, H, BT)
    ref_kk = torch.zeros(1, T, H, BC)
    diagonal_rows = torch.zeros_like(ref_qk, dtype=torch.bool)
    for bos, eos in zip(bounds[:-1], bounds[1:]):
        for start in range(bos, eos, BC):
            end = min(start + BC, eos)
            qi, ki, gi, bi = [x[start:end].transpose(0, 1) for x in (q_ref, k_ref, g_ref, b_ref)]
            decay = torch.exp2(gi[:, :, None, :] - gi[:, None, :, :])
            qk = (qi[:, :, None, :] * ki[:, None, :, :] * decay).sum(dim=-1) * scale
            kk = ((ki * bi)[:, :, None, :] * ki[:, None, :, :] * decay).sum(dim=-1)
            col = (start - bos) % BT
            diagonal_rows[0, start:end, :, col:col + BC] = True
            ref_qk[0, start:end, :, col:col + end - start] = qk.tril().transpose(0, 1)
            ref_kk[0, start:end, :, :end - start] = kk.tril(diagonal=-1).transpose(0, 1)
    actual_qk = Aqk.cpu()
    assert torch.isfinite(actual_qk[diagonal_rows]).all()
    assert torch.isfinite(Akk).all()
    assert_close('Aqk', ref_qk[diagonal_rows].to(dtype), actual_qk[diagonal_rows], 0.005)
    assert_close('Akk', ref_kk, Akk.cpu(), 0.005)


@pytest.mark.skipif(not IS_NPU, reason='Ascend kernel regression')
@pytest.mark.parametrize(
    ('K', 'V', 'dtype'),
    [(128, 64, torch.bfloat16), (64, 128, torch.float16), (256, 128, torch.bfloat16)],
    ids=['K128-V64-bf16', 'K64-V128-fp16', 'K256-V128-bf16'],
)
def test_chunk_asymmetric_head_dims(K, V, dtype):
    """Preserve outputs and gradients when key and value tiles split independently."""
    from test_gdn2 import test_chunk as check_chunk

    check_chunk(
        B=1,
        T=100,
        H=2,
        K=K,
        V=V,
        scale=1.0,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        safe_gate=False,
        dtype=dtype,
    )


@pytest.mark.skipif(not IS_NPU, reason='Ascend kernel regression')
@pytest.mark.parametrize(('K', 'V'), [(128, 256), (256, 512)], ids=['K128-V256', 'K256-V512'])
def test_chunk_varlen_saved_state_v_first(K, V):
    """Check saved WY buffers and transposed states with expanded value heads."""
    from test_gdn2 import test_chunk_varlen as check_chunk_varlen

    check_chunk_varlen(
        cu_seqlens=[0, 15, 80, 145],
        H=2,
        K=K,
        V=V,
        use_gate_in_kernel=True,
        dtype=torch.bfloat16,
        state_v_first=True,
        disable_recompute=True,
    )
