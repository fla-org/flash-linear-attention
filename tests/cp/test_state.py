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

from fla.ops.cp.chunk_delta_h import pre_process_bwd_kernel_merged
from fla.utils import device


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

    # Case A: all gk = 0 (no decay for any head)
    gk_zero = torch.zeros(B, T, H, K, device=device, dtype=torch.float32)

    # Case B: head 0 = 0, heads 1+ = -10 (massive decay)
    gk_diff = torch.zeros(B, T, H, K, device=device, dtype=torch.float32)
    gk_diff[:, :, 1:, :] = -10.0

    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.long)

    BK = triton.next_power_of_2(K)
    BLOCK_SIZE = 32 if K <= 64 else 64
    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), H)

    # Run A: all gk = 0
    dhm_a = torch.zeros(H, K, V + K, dtype=torch.float32, device=device)
    pre_process_bwd_kernel_merged[grid](
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
        BK1=BK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_BG=False,
    )

    # Run B: head 0 = 0, heads 1+ = -10
    dhm_b = torch.zeros(H, K, V + K, dtype=torch.float32, device=device)
    pre_process_bwd_kernel_merged[grid](
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
        BK1=BK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_BG=False,
    )

    # Head 0: same gk (=0) in both runs → dh must be identical
    dh_h0_diff = (dhm_a[0, :, :V] - dhm_b[0, :, :V]).norm().item()
    assert dh_h0_diff == 0.0, f"Head 0 dh should be identical but diff={dh_h0_diff}"

    # Heads 1+: gk changed from 0 to -10 → dh MUST differ if kernel reads per-head gk
    # With the bug (reading head 0's gk=0 for all heads), both runs produce identical dh
    for h in range(1, H):
        dh_diff = (dhm_a[h, :, :V] - dhm_b[h, :, :V]).norm().item()
        dh_norm = dhm_a[h, :, :V].norm().item()
        assert dh_diff > 0, (
            f"Head {h} dh should differ when gk changes from 0 to -10, but diff=0. "
            f"This indicates the kernel is reading head 0's gk for all heads "
            f"(missing i_h * K offset in gk load address)."
        )
        # The diff should be substantial (not just numerical noise)
        ratio = dh_diff / (dh_norm + 1e-8)
        assert ratio > 0.1, (
            f"Head {h} dh ratio={ratio:.4f} is too small. "
            f"Expected large difference due to exp2(-10) ≈ 0.001 decay."
        )
