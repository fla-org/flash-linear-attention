# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda import fused_recurrent_kda
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import IS_NVIDIA, assert_close, device

HAS_FP8 = hasattr(torch, 'float8_e4m3fn')


def setup_inputs(B=2, T=256, H=4, HV=4, K=128, V=128, dtype=torch.bfloat16):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, HV, V, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, HV, K, dtype=torch.float))
    beta = torch.randn(B, T, HV, dtype=dtype).sigmoid()
    h0 = torch.randn(B, HV, K, V, dtype=torch.float32)
    q, k, v, g, beta, h0 = map(lambda x: x.to(device), (q, k, v, g, beta, h0))
    return q, k, v, g, beta, h0


# ========== Test 1: KDA state en FP16 ==========


@pytest.mark.parametrize("T", [256, 1024, 4096])
@pytest.mark.parametrize("H, HV", [(4, 4), (8, 8)])
def test_fp16_state_precision(T, H, HV):
    B, K, V = 2, 128, 128
    q, k, v, g, beta, h0 = setup_inputs(B, T, H, HV, K, V)
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    _, ref_ht = fused_recurrent_kda(
        q, k, v, g, beta,
        initial_state=h0, output_final_state=True, store_dtype=torch.float32,
    )
    _, fp16_ht = fused_recurrent_kda(
        q, k, v, g, beta,
        initial_state=h0, output_final_state=True, store_dtype=torch.float16,
    )

    fp16_ht = fp16_ht.float()
    max_err = (ref_ht - fp16_ht).abs().max().item()
    rel_err = ((ref_ht - fp16_ht).abs() / (ref_ht.abs() + 1e-8)).mean().item()

    assert max_err < 0.01, f"FP16 max_err={max_err:.6f} > 0.01"
    assert rel_err < 0.001, f"FP16 rel_err={rel_err:.6f} > 0.001"
    assert fp16_ht.dtype == torch.float16, f"Expected float16, got {fp16_ht.dtype}"


# ========== Test 2: KDA state en FP8 ==========


@pytest.mark.skipif(not HAS_FP8, reason="FP8 not available in this PyTorch version")
@pytest.mark.parametrize("T", [256, 512])
def test_fp8_state_precision(T):
    B, H, HV, K, V = 2, 4, 4, 128, 128
    q, k, v, g, beta, h0 = setup_inputs(B, T, H, HV, K, V)
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    _, ref_ht = fused_recurrent_kda(
        q, k, v, g, beta,
        initial_state=h0, output_final_state=True, store_dtype=torch.float32,
    )
    _, fp8_ht = fused_recurrent_kda(
        q, k, v, g, beta,
        initial_state=h0, output_final_state=True, store_dtype=torch.float8_e4m3fn,
    )

    fp8_ht = fp8_ht.float()
    max_err = (ref_ht - fp8_ht).abs().max().item()
    rel_err = ((ref_ht - fp8_ht).abs() / (ref_ht.abs() + 1e-8)).mean().item()

    assert max_err < 0.05, f"FP8 max_err={max_err:.6f} > 0.05"
    assert rel_err < 0.01, f"FP8 rel_err={rel_err:.6f} > 0.01"
    assert fp8_ht.element_size() == 1, f"Expected 1-byte elements, got {fp8_ht.element_size()}"


# ========== Test 3: Output equivalence ==========
# L'output `o` ne devrait pas changer significativement meme si le state
# stocke est a precision reduite, car la forward pass interne reste en FP32.


def test_output_equivalence_fp16():
    B, T, H, HV, K, V = 2, 512, 4, 4, 128, 128
    q, k, v, g, beta, h0 = setup_inputs(B, T, H, HV, K, V)
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    o_fp32, _ = fused_recurrent_kda(
        q, k, v, g, beta,
        initial_state=h0, output_final_state=True, store_dtype=torch.float32,
    )
    o_fp16, _ = fused_recurrent_kda(
        q, k, v, g, beta,
        initial_state=h0, output_final_state=True, store_dtype=torch.float16,
    )

    diff = (o_fp32 - o_fp16).abs().max().item()
    assert diff < 0.005, f"Output FP16 max_err={diff:.6f} > 0.005"


@pytest.mark.skipif(not HAS_FP8, reason="FP8 not available in this PyTorch version")
def test_output_equivalence_fp8():
    B, T, H, HV, K, V = 2, 256, 4, 4, 128, 128
    q, k, v, g, beta, h0 = setup_inputs(B, T, H, HV, K, V)
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    o_fp32, _ = fused_recurrent_kda(
        q, k, v, g, beta,
        initial_state=h0, output_final_state=True, store_dtype=torch.float32,
    )
    o_fp8, _ = fused_recurrent_kda(
        q, k, v, g, beta,
        initial_state=h0, output_final_state=True, store_dtype=torch.float8_e4m3fn,
    )

    diff = (o_fp32 - o_fp8).abs().max().item()
    assert diff < 0.01, f"Output FP8 max_err={diff:.6f} > 0.01"


# ========== Test 4: Accumulation d'erreur sur sequence longue ==========


@pytest.mark.parametrize("T", [4096, 16384])
@pytest.mark.parametrize("store_dtype", [
    torch.float16,
    pytest.param(torch.float8_e4m3fn, marks=pytest.mark.skipif(
        not HAS_FP8, reason="FP8 not available")),
])
def test_long_context_accumulation(T, store_dtype):
    B, H, HV, K, V = 1, 4, 4, 128, 128
    chunk = 64
    q, k, v, g, beta, _ = setup_inputs(B, T, H, HV, K, V)
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    state_fp32 = torch.zeros(B, HV, K, V, device=device, dtype=torch.float32)
    state_low = torch.zeros(B, HV, K, V, device=device, dtype=torch.float32)

    metrics = []
    for i in range(0, T, chunk):
        sl = slice(i, min(i + chunk, T))

        _, state_fp32 = fused_recurrent_kda(
            q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl],
            initial_state=state_fp32, output_final_state=True,
            store_dtype=torch.float32)

        _, state_low_raw = fused_recurrent_kda(
            q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl],
            initial_state=state_low, output_final_state=True,
            store_dtype=store_dtype)

        state_low = state_low_raw.float()

        if (i // chunk) % 8 == 0:
            err = (state_fp32 - state_low).abs().max().item()
            metrics.append((i + chunk, err))

    assert len(metrics) > 0, "No metrics collected"
    max_err = max(m[1] for m in metrics)
    assert max_err < 0.1, f"Long context max_err={max_err:.6f} > 0.1"

    # Verifier que la divergence ne s'accelere pas
    if len(metrics) >= 4:
        m0, m2 = metrics[-4], metrics[-2]
        last_slope = (m2[1] - m0[1]) / (m2[0] - m0[0]) if m2[0] != m0[0] else 0
        assert last_slope < 1e-5, f"Divergence accelerates: slope={last_slope:.2e}"


# ========== Test 5: Delta encoding (unitaire) ==========


def test_delta_checkpoint():
    B, T, H, HV, K, V = 2, 2048, 4, 4, 128, 128
    q, k, v, g, beta, _ = setup_inputs(B, T, H, HV, K, V)
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    checkpoint_interval = 512
    compute_interval = 64

    # Reference: continuous forward in FP32
    ref_state = torch.zeros(B, HV, K, V, device=device, dtype=torch.float32)
    ref_states = []
    for i in range(0, T, compute_interval):
        sl = slice(i, min(i + compute_interval, T))
        _, ref_state = fused_recurrent_kda(
            q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl],
            initial_state=ref_state, output_final_state=True,
            store_dtype=torch.float32)
        if (i + compute_interval) % checkpoint_interval == 0:
            ref_states.append(ref_state.clone())

    # Delta-encoded version
    approx_state = torch.zeros(B, HV, K, V, device=device, dtype=torch.float32)
    full_state_fp16 = None
    checkpoints = []

    for i in range(0, T, compute_interval):
        sl = slice(i, min(i + compute_interval, T))
        _, approx_state = fused_recurrent_kda(
            q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl],
            initial_state=approx_state, output_final_state=True,
            store_dtype=torch.float32)
        ckpt_id = (i + compute_interval) // checkpoint_interval

        if (i + compute_interval) % checkpoint_interval == 0:
            full_state_fp16 = approx_state.half()
            checkpoints.append((ckpt_id, full_state_fp16.float(), None))
        elif full_state_fp16 is not None and (i + compute_interval) % compute_interval == 0:
            delta = (approx_state - full_state_fp16.float()).to(torch.float8_e4m3fn) if HAS_FP8 else (approx_state - full_state_fp16.float()).half()
            checkpoints.append((ckpt_id, full_state_fp16.float(), delta))

    # Recomposer et comparer
    for ckpt_id, full_state, delta in checkpoints:
        if delta is not None:
            recomposed = full_state + delta.float()
        else:
            recomposed = full_state
        err = (ref_states[ckpt_id - 1] - recomposed).abs().max().item()
        assert err < 0.02, f"Delta ckpt {ckpt_id}: err={err:.6f}"


# ========== Test 6: Low-rank approximation ==========


@pytest.mark.parametrize("rank", [16, 32, 64])
def test_lowrank_approximation(rank):
    B, HV, K, V = 2, 4, 128, 128
    S = torch.randn(B, HV, K, V, device=device)

    U, Sigma, Vt = torch.linalg.svd(S.float(), full_matrices=False)

    U_r = U[..., :rank]
    Sigma_r = Sigma[..., :rank]
    Vt_r = Vt[..., :rank, :]

    S_approx = (U_r * Sigma_r.unsqueeze(-2)) @ Vt_r
    rel_err = (S.float() - S_approx).norm() / S.float().norm()
    assert rel_err < 0.2, f"Rank {rank}: rel_err={rel_err:.4f} > 0.2"

    # Verifier la matmul simplifiee S * q
    q = torch.randn(B, HV, K, device=device)
    y_direct = S.float() @ q.unsqueeze(-1)
    y_low = (U_r * Sigma_r.unsqueeze(-2)) @ (Vt_r @ q.unsqueeze(-1))
    err = (y_direct - y_low).abs().max().item()
    assert err < 1e-3, f"Rank {rank}: matmul err={err:.6f}"


def test_lowrank_exact_reconstruction():
    B, HV, K, V = 2, 4, 128, 128
    rank_true = 16
    U_true = torch.randn(B, HV, K, rank_true, device=device)
    V_true = torch.randn(B, HV, V, rank_true, device=device)
    S_low = U_true @ V_true.transpose(-2, -1)

    U, Sigma, Vt = torch.linalg.svd(S_low.float(), full_matrices=False)
    for rank in [rank_true, rank_true // 2]:
        U_r = U[..., :rank]
        Sigma_r = Sigma[..., :rank]
        Vt_r = Vt[..., :rank, :]
        S_reconstructed = (U_r * Sigma_r.unsqueeze(-2)) @ Vt_r
        err = (S_low.float() - S_reconstructed).norm().item()
        if rank >= rank_true:
            assert err < 1e-5, f"Expected exact reconstruction, got {err:.2e}"
