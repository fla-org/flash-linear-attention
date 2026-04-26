# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Regression test for CP backward gk offset bug in pre_process_bwd_kernel_merged.

Bug Description:
================
In fla/ops/cp/chunk_delta_h.py, the backward pre-processing kernel
pre_process_bwd_kernel_merged (stage 1) loaded gk using:
    gk + last_idx * H*K + o_k
instead of the correct:
    gk + (bos + last_idx) * H*K + i_h * K + o_k

This caused heads with i_h > 0 to incorrectly read head 0's gate values
during backward dh computation (stage 1).

The fix adds proper bos and i_h offsets to all gk loads in the backward kernels.

Test Strategy:
==============
Run the kernel with two gk tensors:
  A) gk_zero: all heads have gk = 0 (exp2(0) = 1, no decay)
  B) gk_diff: head 0 = 0, heads 1+ = large negative (exp2(-10) ≈ 0.001, massive decay)

With the bug:  all heads read head 0's gk (=0 in both A and B) → dh identical
With the fix:  heads 1+ read their own gk → dh very different between A and B
"""

import logging
import os
import tempfile

import pytest
import torch
import triton

from fla.ops.cp.chunk_delta_h import pre_process_bwd_kernel_merged
from fla.utils import device

logging.basicConfig(level=logging.INFO, format='%(message)s')


class TestBwdGkOffset:
    """Test that backward pre-processing kernel correctly addresses per-head gk values."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Use a fresh triton cache for each test to avoid stale compilations."""
        self._original_cache = os.environ.get('TRITON_CACHE_DIR')
        os.environ['TRITON_CACHE_DIR'] = tempfile.mkdtemp()
        yield
        if self._original_cache is not None:
            os.environ['TRITON_CACHE_DIR'] = self._original_cache
        else:
            os.environ.pop('TRITON_CACHE_DIR', None)

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
    def test_stage1_gk_per_head_sensitivity(self, T: int, H: int, K: int, V: int):
        """
        Verify stage 1 (dh computation) is sensitive to per-head gk values.

        The backward pre-processing kernel computes:
            dh = exp2(gk_last) * dh + (q^T @ do * scale - w @ dv)

        If gk is read correctly per-head, changing heads 1+ gk values from 0 to -10
        MUST change their dh output (exp2(-10) ≈ 0.001 vs exp2(0) = 1).

        With the bug:  all heads read head 0 (gk=0), so dh is identical for both runs
        With the fix:  heads 1+ read gk=-10 in run B, producing massively decayed dh
        """
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
            q=q, k=k, w=w, g=None, gk=gk_zero, do=do, dhm=dhm_a, dv=dv,
            cu_seqlens=cu_seqlens, scale=1.0, T=T, H=H, HV=H, K=K, V=V,
            BT=BT, BK1=BK, USE_EXP2=True, BLOCK_SIZE=BLOCK_SIZE,
            USE_BG=False,
        )

        # Run B: head 0 = 0, heads 1+ = -10
        dhm_b = torch.zeros(H, K, V + K, dtype=torch.float32, device=device)
        pre_process_bwd_kernel_merged[grid](
            q=q, k=k, w=w, g=None, gk=gk_diff, do=do, dhm=dhm_b, dv=dv,
            cu_seqlens=cu_seqlens, scale=1.0, T=T, H=H, HV=H, K=K, V=V,
            BT=BT, BK1=BK, USE_EXP2=True, BLOCK_SIZE=BLOCK_SIZE,
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
