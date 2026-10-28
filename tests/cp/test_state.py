# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Check backward CP state summaries against per-head controls and FP64 references."""

import pytest
import torch
import triton

from fla.ops.cp.chunk_bwd import chunk_cp_bwd_dh, chunk_cp_bwd_kernel_dh
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


@pytest.mark.parametrize(
    ('T', 'H', 'HV', 'K', 'V', 'gate', 'use_bg'),
    [
        (65, 2, 4, 32, 48, 'scalar', False),
        (129, 2, 2, 64, 64, None, False),
        (8192, 2, 4, 128, 80, 'vector', False),
        (257, 2, 4, 192, 64, 'scalar', True),
        (65, 1, 2, 256, 80, 'vector', False),
    ],
    ids=['scalar-grouped-tail', 'ungated-tail', 'long-vector', 'dplr-wide', 'vector-wide-tail'],
)
def test_chunk_bwd_summary(T, H, HV, K, V, gate, use_bg):
    torch.manual_seed(42)
    BT = 64
    # a prefix exercises nonzero varlen offsets; weak updates keep the long M chain observable
    bos = 17
    shape = (1, T + bos, H, K)
    q, k = [torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.02 for _ in range(2)]
    w = torch.randn(1, T + bos, H if use_bg else HV, K, device=device, dtype=torch.bfloat16) * 0.02
    bg = torch.randn_like(k) * 0.02 if use_bg else None
    do, dv = [torch.randn(1, T + bos, HV, V, device=device, dtype=torch.bfloat16) * 0.1 for _ in range(2)]
    cumulative = -1e-5 * (torch.arange(T, device=device) % BT + 1).float()
    g = torch.zeros(1, T + bos, HV, device=device) if gate == 'scalar' else None
    gk = torch.zeros(1, T + bos, HV, K, device=device) if gate == 'vector' else None
    if g is not None:
        g[:, bos:] = cumulative[:, None]
    if gk is not None:
        gk[:, bos:] = cumulative[:, None, None]
    cu_seqlens = torch.tensor([bos, bos + T], device=device, dtype=torch.int32)
    actual = torch.empty(HV, K, V + K, device=device, dtype=torch.float32)
    chunk_cp_bwd_dh(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        dhm=actual,
        g=g,
        gk=gk,
        bg=bg,
        scale=K ** -0.5,
        cu_seqlens=cu_seqlens,
    )

    reference = torch.empty_like(actual, dtype=torch.float64)
    for head in range(HV):
        q_head = head // (HV // H)
        dh = torch.zeros(K, V, device=device, dtype=torch.float64)
        matrix = torch.eye(K, device=device, dtype=torch.float64)
        for start in range((T - 1) // BT * BT, -1, -BT):
            end = min(start + BT, T)
            indices = slice(bos + start, bos + end)
            qc = q[0, indices, q_head].double()
            kc = (bg if use_bg else k)[0, indices, q_head].double()
            wc = w[0, indices, q_head if use_bg else head].double()
            dc = dv[0, indices, head].double()
            dc = dc + kc @ dh.to(k.dtype).double()
            last = bos + end - 1
            if g is not None:
                factors = torch.exp2(g[0, last, head] - g[0, indices, head]).double()
                dc = dc - dv[0, indices, head].double()
                dc = dc * factors[:, None] + dv[0, indices, head].double()
                qc = qc * torch.exp2(g[0, indices, head]).double()[:, None]
                kc = (kc * factors[:, None]).to(k.dtype).double()
                decay = torch.exp2(g[0, last, head]).double().expand(K)
            elif gk is not None:
                decay = torch.exp2(gk[0, last, head]).double()
            else:
                decay = torch.ones(K, device=device, dtype=torch.float64)
            local = qc.T @ do[0, indices, head].double()
            update = wc.T @ dc.to(w.dtype).double()
            dh = decay[:, None] * dh + (local + update if use_bg else local * K ** -0.5 - update)
            transform = torch.diag(decay) + (wc.T @ kc if use_bg else -(wc.T @ kc))
            matrix = transform @ matrix
        reference[head] = torch.cat((dh, matrix), dim=-1)

    assert torch.isfinite(actual).all()
    assert_close('dh', reference[..., :V].float(), actual[..., :V], 0.006)
    assert_close('dM', reference[..., V:].float(), actual[..., V:], 0.006)
