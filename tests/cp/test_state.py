# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Check CP state summaries for per-head gate addressing and against FP64 recurrences."""

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
    BV = 32
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


def _summary_reference(q, k, w, u, v, do, dv, g, gk, bg, scale, bos, T, BT):
    """Return FP64 [h_local | M] and [dh_local | dM] on the host, rounding operands to bf16 where the kernels do."""
    H, K = k.shape[2:]
    HV, V = u.shape[2:]

    def rows(x, head, start, end):
        return x[0, bos + start:bos + end, head].cpu().double()

    def bf16(x):
        return x.to(torch.bfloat16).double()

    hm, dhm = (torch.empty(HV, K, V + K, dtype=torch.float64) for _ in range(2))
    chunks = [(start, min(start + BT, T)) for start in range(0, T, BT)]
    for i_h in range(HV):
        i_k = i_h // (HV // H)
        i_w = i_k if bg is not None else i_h
        h, m = torch.zeros(K, V, dtype=torch.float64), torch.eye(K, dtype=torch.float64)
        dh, dm = torch.zeros_like(h), torch.eye(K, dtype=torch.float64)
        decays = {}
        for start, end in chunks:
            factor = torch.ones(end - start, dtype=torch.float64)
            diag = torch.ones(K, dtype=torch.float64)
            if g is not None:
                g_c = rows(g, i_h, start, end)
                factor, diag = torch.exp2(g_c[-1] - g_c), torch.exp2(g_c[-1]).expand(K)
            elif gk is not None:
                diag = torch.exp2(rows(gk, i_h, start, end)[-1])
            decays[start] = (factor, diag)
            k_c, w_c, u_c = rows(k, i_k, start, end), rows(w, i_w, start, end), rows(u, i_h, start, end)
            if bg is None:
                h = diag[:, None] * h + k_c.T @ bf16((u_c - w_c @ bf16(h)) * factor[:, None])
                m = (torch.diag(diag) - (k_c * factor[:, None]).T @ w_c) @ m
            else:
                bg_c = rows(bg, i_k, start, end)
                v_new = bf16((w_c @ bf16(h) + u_c) * factor[:, None])
                h = diag[:, None] * h + k_c.T @ rows(v, i_h, start, end) + bg_c.T @ v_new
                m = (torch.diag(diag) + (bg_c * factor[:, None]).T @ w_c) @ m
        for start, end in reversed(chunks):
            factor, diag = decays[start]
            # the backward summary reads bg in place of k for DPLR
            q_c, k_c, w_c = rows(q, i_k, start, end), rows(k if bg is None else bg, i_k, start, end), rows(w, i_w, start, end)
            if g is not None:
                q_c = q_c * torch.exp2(rows(g, i_h, start, end))[:, None]
            dv_c = (k_c @ bf16(dh)) * factor[:, None] + rows(dv, i_h, start, end)
            local, update = q_c.T @ rows(do, i_h, start, end), w_c.T @ bf16(dv_c)
            if bg is None:
                dh = diag[:, None] * dh + local * scale - update
                dm = (torch.diag(diag) - w_c.T @ (k_c * factor[:, None])) @ dm
            else:
                dh = diag[:, None] * dh + local + update
                dm = (torch.diag(diag) + w_c.T @ (k_c * factor[:, None])) @ dm
        hm[i_h], dhm[i_h] = torch.cat((h, m), dim=-1), torch.cat((dh, dm), dim=-1)
    return hm, dhm


@pytest.mark.parametrize(
    ('T', 'H', 'HV', 'K', 'V', 'gate', 'use_bg'),
    [
        pytest.param(*p, id='T{}-H{}-HV{}-K{}-V{}-{}-{}'.format(*p[:6], 'dplr' if p[6] else 'delta'))
        for p in [
            # four full 64-row strips with the KDA vector gate
            (300, 2, 2, 256, 128, 'vector', False),
            # three strips with the GDN scalar gate and grouped value heads
            (200, 1, 2, 192, 64, 'scalar', False),
            # DPLR with a partial fourth strip and a partial value block
            (150, 2, 2, 200, 80, 'vector', True),
            # two ungated strips ending in a one-token chunk
            (129, 2, 2, 128, 128, None, False),
            # narrow states keep the [K, K] chunk transform
            (100, 2, 4, 64, 64, 'vector', False),
        ]
    ],
)
def test_chunk_summary(T: int, H: int, HV: int, K: int, V: int, gate: str | None, use_bg: bool):
    """Forward and backward summaries match FP64 recurrences across state widths, gates, and DPLR."""
    torch.manual_seed(42)
    BT, bos = 64, 37
    # the prefix gives the summarized sequence a nonzero varlen offset
    q, k = (torch.randn(1, bos + T, H, K, device=device, dtype=torch.bfloat16) * 0.05 for _ in range(2))
    w = torch.randn(1, bos + T, H if use_bg else HV, K, device=device, dtype=torch.bfloat16) * 0.05
    bg = torch.randn_like(k) * 0.05 if use_bg else None
    u, v, do, dv = (torch.randn(1, bos + T, HV, V, device=device, dtype=torch.bfloat16) for _ in range(4))
    # chunk-local cumulative log2 decays, aligned with the chunks of the offset sequence
    NT = triton.cdiv(T, BT)
    decay = torch.zeros(1, NT * BT, HV, K if gate == 'vector' else 1, device=device)
    decay[:, :T] = -0.05 * torch.rand(1, T, HV, decay.shape[-1], device=device)
    decay = decay.view(1, NT, BT, HV, -1).cumsum(2).view(1, NT * BT, HV, -1)[:, :T]
    gates = torch.zeros(1, bos + T, HV, decay.shape[-1], device=device)
    gates[:, bos:] = decay
    g = gates[..., 0] if gate == 'scalar' else None
    gk = gates if gate == 'vector' else None
    cu_seqlens = torch.tensor([bos, bos + T], device=device, dtype=torch.long)
    scale = K ** -0.5

    # NaN-filled outputs expose any summary element the kernels leave unwritten
    hm = torch.full((HV, K, V + K), float('nan'), device=device)
    dhm = torch.full_like(hm, float('nan'))
    chunk_cp_fwd_h(k=k, w=w, u=u, hm=hm, g=g, gk=gk, bg=bg, v=v if use_bg else None, cu_seqlens=cu_seqlens, chunk_size=BT)
    chunk_cp_bwd_dh(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        dhm=dhm,
        g=g,
        gk=gk,
        bg=bg,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
    )
    ref_hm, ref_dhm = _summary_reference(
        q=q,
        k=k,
        w=w,
        u=u,
        v=v,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        bg=bg,
        scale=scale,
        bos=bos,
        T=T,
        BT=BT,
    )

    hm, dhm = hm.cpu(), dhm.cpu()
    assert hm.isfinite().all() and dhm.isfinite().all()
    assert_close('h', ref_hm[..., :V], hm[..., :V], 0.006)
    assert_close('M', ref_hm[..., V:], hm[..., V:], 1e-4)
    assert_close('dh', ref_dhm[..., :V], dhm[..., :V], 0.006)
    assert_close('dM', ref_dhm[..., V:], dhm[..., V:], 1e-4)
