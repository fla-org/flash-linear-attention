# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Verify that CP backward summaries use each head's own vector gate.

Both runs leave head 0's gate at zero; the second run strongly decays every other head.
Head 0 must remain unchanged while the other heads change, exposing a missing per-head gate offset.
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
                (256, 4, 192, 64),
                (256, 4, 256, 80),
            ]
        ],
    )
    def test_stage1_gk_per_head_sensitivity(self, T: int, H: int, K: int, V: int):
        """Changing one head's gate must affect only that head's backward state."""
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

        BK = triton.next_power_of_2(K)
        BLOCK_SIZE = 32 if K <= 64 else 64
        grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), H)

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

        # head 0: same gk (=0) in both runs → dh must be identical
        dh_h0_diff = (dhm_a[0, :, :V] - dhm_b[0, :, :V]).norm().item()
        assert dh_h0_diff == 0.0, f"Head 0 dh should be identical but diff={dh_h0_diff}"

        # reading head 0's gate for every head would make both runs identical
        for h in range(1, H):
            dh_diff = (dhm_a[h, :, :V] - dhm_b[h, :, :V]).norm().item()
            dh_norm = dhm_a[h, :, :V].norm().item()
            assert dh_diff > 0, (
                f"Head {h} dh should differ when gk changes from 0 to -10, but diff=0. "
                f"This indicates the kernel is reading head 0's gk for all heads "
                f"(missing i_h * K offset in gk load address)."
            )
            # the diff should be substantial (not just numerical noise)
            ratio = dh_diff / (dh_norm + 1e-8)
            assert ratio > 0.1, (
                f"Head {h} dh ratio={ratio:.4f} is too small. "
                f"Expected large difference due to exp2(-10) ≈ 0.001 decay."
            )
