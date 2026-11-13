# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Forward/backward affine summaries and per-head vector-gate addressing."""

from types import SimpleNamespace

import pytest
import torch
import triton

from fla.ops.cp.chunk_bwd import chunk_cp_bwd_dh, chunk_cp_bwd_kernel_dh
from fla.ops.cp.chunk_fwd import chunk_cp_fwd_h
from fla.utils import IS_NPU, IS_TF32_SUPPORTED, assert_close, device

pytestmark = pytest.mark.skipif(not (IS_TF32_SUPPORTED or IS_NPU), reason="requires NVIDIA Ampere or newer, or Ascend")


@pytest.mark.parametrize(
    ("T", "H", "K", "V"),
    [
        pytest.param(*p, id=f"T{p[0]}-H{p[1]}-K{p[2]}-V{p[3]}")
        for p in [
            (256, 4, 128, 128),
            (256, 8, 64, 64),
            (512, 4, 128, 128),
        ]
    ],
)
def test_chunk_bwd_gate_head(monkeypatch, tmp_path, T: int, H: int, K: int, V: int):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    torch.manual_seed(42)
    B = 1
    BT = 64

    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
    do = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16) * 0.1
    dv = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16) * 0.1

    # zero gates leave every head undecayed, providing the control for the per-head comparison
    gk_zero = torch.zeros(B, T, H, K, device=device, dtype=torch.float32)

    # keep head 0 unchanged and apply strong decay to heads 1+ to expose wrong-head addressing
    gk_diff = torch.zeros(B, T, H, K, device=device, dtype=torch.float32)
    gk_diff[:, :, 1:, :] = -10.0

    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.long)

    BK = triton.next_power_of_2(K)
    BV = 32 if K <= 64 else 64
    grid = (triton.cdiv(V, BV) + triton.cdiv(K, BV), H)

    # the control run uses zero gates for every head
    dhm_a = torch.zeros(H, K, V + K, dtype=torch.float32, device=device)
    chunk_cp_bwd_kernel_dh[grid](
        q=q,
        k=k,
        w=w,
        g=None,
        gk=gk_zero,
        do=do,
        dhm=dhm_a,
        dv=dv,
        cu_seqlens=cu_seqlens,
        scale=1.0,
        T=T,
        H=H,
        HV=H,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        BK=BK,
        USE_BG=False,
    )

    # the second run changes only the gates for heads 1+
    dhm_b = torch.zeros(H, K, V + K, dtype=torch.float32, device=device)
    chunk_cp_bwd_kernel_dh[grid](
        q=q,
        k=k,
        w=w,
        g=None,
        gk=gk_diff,
        do=do,
        dhm=dhm_b,
        dv=dv,
        cu_seqlens=cu_seqlens,
        scale=1.0,
        T=T,
        H=H,
        HV=H,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        BK=BK,
        USE_BG=False,
    )

    # head 0 sees identical gates in both runs, so its backward state must match exactly
    dh_h0_diff = (dhm_a[0, :, :V] - dhm_b[0, :, :V]).norm().item()
    assert dh_h0_diff == 0.0, f"Head 0 dh should be identical but diff={dh_h0_diff}"

    # heads 1+ must react to their own decay;
    # reading head 0's gate for every head would incorrectly leave their backward states unchanged
    for h in range(1, H):
        dh_diff = (dhm_a[h, :, :V] - dhm_b[h, :, :V]).norm().item()
        dh_norm = dhm_a[h, :, :V].norm().item()
        assert dh_diff > 0, (
            f"Head {h} dh should differ when gk changes from 0 to -10, but diff=0. "
            f"This indicates the kernel is reading head 0's gk for all heads "
            f"(missing i_h * K offset in gk load address)."
        )
        # exp2(-10) is strong enough that the response must exceed numerical noise
        ratio = dh_diff / (dh_norm + 1e-8)
        assert ratio > 0.1, (
            f"Head {h} dh ratio={ratio:.4f} is too small. "
            f"Expected large difference due to exp2(-10) ≈ 0.001 decay."
        )


SUMMARY_CASES = [
    pytest.param(1, torch.float16, 'none', id='k1-fp16-none'),
    pytest.param(60, torch.bfloat16, 'scalar', id='k60-bf16-scalar'),
    pytest.param(63, torch.float16, 'dplr-vector', id='k63-fp16-dplr-vector'),
    pytest.param(64, torch.bfloat16, 'vector', id='k64-bf16-vector'),
    pytest.param(65, torch.bfloat16, 'vector', id='k65-bf16-vector'),
    pytest.param(80, torch.float16, 'none', id='k80-fp16-none'),
    pytest.param(100, torch.float16, 'dplr', id='k100-fp16-dplr'),
    pytest.param(112, torch.bfloat16, 'dplr-vector', id='k112-bf16-dplr-vector'),
    pytest.param(127, torch.float16, 'scalar', id='k127-fp16-scalar'),
    pytest.param(128, torch.bfloat16, 'vector', id='k128-bf16-vector'),
    pytest.param(129, torch.float16, 'vector', id='k129-fp16-vector'),
    pytest.param(160, torch.bfloat16, 'none', id='k160-bf16-none'),
    pytest.param(192, torch.float16, 'dplr-vector', id='k192-fp16-dplr-vector'),
    pytest.param(193, torch.bfloat16, 'scalar', id='k193-bf16-scalar'),
    pytest.param(255, torch.float16, 'vector', id='k255-fp16-vector'),
    pytest.param(256, torch.bfloat16, 'dplr', id='k256-bf16-dplr'),
]


def _check_chunk_summary(K, dtype, gate, launch, forward=False, chunk_size=32):
    torch.manual_seed(42)
    T, H, V, BT = 96, 2, 80, chunk_size
    is_dplr = gate.startswith('dplr')
    HV = H if is_dplr else 2 * H
    q = torch.randn(1, T, H, K, device=device, dtype=dtype) * 0.1
    k = torch.randn_like(q) * 0.1
    w = torch.randn(1, T, HV, K, device=device, dtype=dtype) * 0.05
    do = torch.randn(1, T, HV, V, device=device, dtype=dtype) * 0.1
    dv = torch.randn_like(do) * 0.1
    g = -torch.rand(1, T, HV, device=device).cumsum(1) * 0.01 if gate == 'scalar' else None
    gk = -torch.rand(1, T, HV, K, device=device).cumsum(1) * 0.01 if gate in ('vector', 'dplr-vector') else None
    bg = torch.randn_like(k) * 0.1 if is_dplr else None
    cu = torch.tensor([7, 84], dtype=torch.long, device=device)
    actual = torch.empty(HV, K, V + K, dtype=torch.float32, device=device)
    scale = K ** -0.5
    launch(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        dhm=actual,
        g=g,
        gk=gk,
        bg=bg,
        scale=scale,
        cu_seqlens=cu,
        chunk_size=BT,
    )

    q, k = (tensor[0].cpu().float().repeat_interleave(HV // H, dim=1) for tensor in (q, k))
    bg = bg[0].cpu().float() if bg is not None else None
    w, do, dv = (tensor[0].cpu().float() for tensor in (w, do, dv))
    g = g[0].cpu() if g is not None else None
    gk = gk[0].cpu() if gk is not None else None
    response = torch.zeros(HV, K, V)
    transform = torch.eye(K).expand(HV, K, K).clone()
    chunks = range(7, 84, BT)
    for start in chunks if forward else reversed(chunks):
        end = min(start + BT, 84)
        qi, ki, wi, doi, dvi = (tensor[start:end].transpose(0, 1) for tensor in (q, k, w, do, dv))
        bgi = bg[start:end].transpose(0, 1) if bg is not None else None
        if is_dplr and not forward:
            ki = bgi
        diagonal = torch.ones(HV, K)
        if g is not None:
            qi = qi * g[start:end].T.exp2()[..., None]
            ki = ki * (g[end - 1] - g[start:end]).T.exp2()[..., None]
            diagonal *= g[end - 1].exp2()[:, None]
        elif gk is not None:
            diagonal = gk[end - 1].exp2()
        sign = 1 if is_dplr else -1
        if forward:
            transition_key = bgi if is_dplr else ki
            step = torch.diag_embed(diagonal) + sign * (transition_key.transpose(-1, -2) @ wi)
            local = ki.transpose(-1, -2) @ doi
            if is_dplr:
                local = ki.transpose(-1, -2) @ dvi + bgi.transpose(-1, -2) @ doi
        else:
            step = torch.diag_embed(diagonal) + sign * (wi.transpose(-1, -2) @ ki)
            local = qi.transpose(-1, -2) @ doi * (1 if is_dplr else scale) + sign * (wi.transpose(-1, -2) @ dvi)
        response = step @ response + local
        transform = step @ transform
    expected = torch.cat((response, transform), dim=-1)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    assert_close('partial-key dh', expected[..., :V], actual.cpu()[..., :V], ratio=8e-3)
    assert_close('partial-key M', expected[..., V:], actual.cpu()[..., V:], ratio=8e-3)


def _forward_summary(**kwargs):
    kwargs.pop('q')
    kwargs.pop('scale')
    kwargs['u'] = kwargs.pop('do')
    value = kwargs.pop('dv')
    kwargs['v'] = value if kwargs['bg'] is not None else kwargs['u']
    kwargs['hm'] = kwargs.pop('dhm')
    chunk_cp_fwd_h(**kwargs)


def _native_summary(monkeypatch, forward=False):
    from fla.ops.cp.backends.triton_ascend import chunk_delta_h

    context = SimpleNamespace(group=object(), layout='contiguous', is_first_rank=forward, is_last_rank=not forward)
    monkeypatch.setattr(chunk_delta_h.dist, 'get_rank', lambda *args, **kwargs: 0)

    def launch(**kwargs):
        target = kwargs.pop('dhm')

        def capture(summary, group):
            target.copy_(summary)
            return summary.unsqueeze(0), None

        # compare the actual local kernel against the oracle before any cross-rank merge.
        monkeypatch.setattr(chunk_delta_h, 'all_gather_into_tensor', capture)
        if forward:
            kwargs.pop('q')
            kwargs.pop('scale')
            kwargs['u'] = kwargs.pop('do')
            value = kwargs.pop('dv')
            kwargs['v'] = value if kwargs['bg'] is not None else kwargs['u']
            chunk_delta_h.chunk_gated_delta_rule_fwd_h_pre_process_npu(**kwargs, context=context)
        else:
            chunk_delta_h.chunk_gated_delta_rule_bwd_dhu_pre_process_npu(**kwargs, context=context)
    return launch


@pytest.mark.skipif(not IS_NPU, reason="requires Ascend summary kernels")
@pytest.mark.parametrize(('K', 'dtype', 'gate'), SUMMARY_CASES)
def test_chunk_bwd_partial_key_summary(K, dtype, gate):
    _check_chunk_summary(K, dtype, gate, chunk_cp_bwd_dh)


@pytest.mark.skipif(not IS_NPU, reason="requires Ascend summary kernels")
@pytest.mark.parametrize(('K', 'dtype', 'gate'), SUMMARY_CASES)
def test_chunk_bwd_npu_summary(monkeypatch, K, dtype, gate):
    _check_chunk_summary(K, dtype, gate, _native_summary(monkeypatch), chunk_size=64)


@pytest.mark.skipif(not IS_NPU, reason="requires Ascend summary kernels")
@pytest.mark.parametrize(('K', 'dtype', 'gate'), SUMMARY_CASES)
def test_chunk_fwd_summary(K, dtype, gate):
    _check_chunk_summary(K, dtype, gate, _forward_summary, forward=True)


@pytest.mark.skipif(not IS_NPU, reason="requires Ascend summary kernels")
@pytest.mark.parametrize(('K', 'dtype', 'gate'), SUMMARY_CASES)
def test_chunk_fwd_npu_summary(monkeypatch, K, dtype, gate):
    _check_chunk_summary(K, dtype, gate, _native_summary(monkeypatch, forward=True), forward=True, chunk_size=64)
