# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Changing one head's vector gate must affect only that head's backward state summary."""

import pytest
import torch
import triton

from fla.ops.cp.chunk_bwd import chunk_cp_bwd_kernel_dh
from fla.utils import IS_TF32_SUPPORTED, device

pytestmark = pytest.mark.skipif(not IS_TF32_SUPPORTED, reason="requires NVIDIA Ampere or newer")


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
    BLOCK_SIZE = 32 if K <= 64 else 64
    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), H)

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
        BLOCK_SIZE=BLOCK_SIZE,
        BK1=BK,
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
        BLOCK_SIZE=BLOCK_SIZE,
        BK1=BK,
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
