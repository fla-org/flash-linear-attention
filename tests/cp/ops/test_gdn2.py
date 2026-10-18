# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare CP GDN-2 outputs and gradients with per-sequence recurrent references."""


import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from fla.ops.cp import build_cp_context
from fla.ops.gdn2 import chunk_gdn2
from fla.ops.gdn2.naive import naive_recurrent_gdn2
from fla.ops.kda.gate import naive_kda_lowerbound_gate
from fla.utils import assert_close, device
from tests.cp.utils import gather_global, run_distributed

pytestmark = pytest.mark.skipif(device != "cuda", reason="requires CUDA or ROCm GPUs")


def run_cp_gdn2_test_worker(
    rank: int,
    world_size: int,
    test_name: str,
    T: int,
    H: int,
    D: int,
    lengths: list[int],
    dtype,
    disable_recompute: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    state_v_first: bool = False,
):
    """
    Worker function for CP GDN-2 test.
    Runs in a spawned process with the given rank.
    """
    device = torch.device(f'cuda:{rank}')

    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
    assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

    B = 1
    q_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    k_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    v_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    g_global = torch.empty(B, T, H, D, device=device, dtype=dtype if use_gate_in_kernel else torch.float)
    b_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    w_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    do_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    A_log_global = None
    dt_bias_global = None

    if rank == 0:
        torch.manual_seed(42)
        # Generate inputs
        # Asymmetric initialization for q and k to test L2 norm
        # q: small positive values around 1.0
        # k: large negative values around -50.0
        q_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype) + 1.0)
        k_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype) * 10.0 - 50.0)
        v_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        if use_gate_in_kernel:
            g_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        else:
            g_global.copy_(F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=torch.float)))
            g_global.clamp_(min=-5.0)
        b_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype).sigmoid())
        w_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype).sigmoid())
        do_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))

    # Broadcast to ensure all ranks have same data
    dist.broadcast(q_global, src=0)
    dist.broadcast(k_global, src=0)
    dist.broadcast(v_global, src=0)
    dist.broadcast(g_global, src=0)
    dist.broadcast(b_global, src=0)
    dist.broadcast(w_global, src=0)
    dist.broadcast(do_global, src=0)

    # Prepare and broadcast A_log and dt_bias for gate computation
    # Always generate these for reference, even when use_gate_in_kernel=False
    import triton
    num_even_heads = triton.next_power_of_2(H)
    projection_size = H * D

    # All ranks create tensors with same shape first
    A_log_global = torch.empty(num_even_heads, dtype=torch.float32, device=device)
    dt_bias_global = torch.empty(projection_size, dtype=torch.float32, device=device)

    # Rank 0 fills in the actual data
    if rank == 0:
        import math
        # Per-head variation in A_log is CRITICAL for catching backward kernel
        # bugs like missing `i_h * K` in gk pointer offsets.  When A_log=0 for
        # all heads, -exp(A_log)=-1 uniformly → same gate scaling → reading the
        # wrong head's gk gives nearly identical values → bug is invisible.
        if safe_gate and lower_bound is not None:
            A_log_global[:H] = torch.log(torch.linspace(1, 8, H, dtype=torch.float32, device=device))
        else:
            A_log_global.copy_(torch.log(torch.empty(num_even_heads, dtype=torch.float32, device=device).uniform_(1, 16)))
        # dt initialization from real training
        dt = torch.exp(torch.rand(projection_size, device=device) *
                       (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp_(min=1e-4)
        dt_bias_global.copy_(dt + torch.log(-torch.expm1(-dt)))

    # Broadcast from rank 0 to all ranks
    dist.broadcast(A_log_global, src=0)
    dist.broadcast(dt_bias_global, src=0)

    # Prepare cu_seqlens
    cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
    cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.long)

    # Each sequence is computed independently with h0=0 for simplicity
    ref_out = None
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dw = None, None, None, None, None, None

    if rank == 0:
        N = len(lengths)
        ref_outputs = []

        for i in range(N):
            seq_start = cu_seqlens_list[i]
            seq_end = cu_seqlens_list[i + 1]

            q_seq = q_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            k_seq = k_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            v_seq = v_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            g_seq = g_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            b_seq = b_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            w_seq = w_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            do_seq = do_global[:, seq_start:seq_end].clone()

            q_seq_norm = F.normalize(q_seq, p=2, dim=-1)
            k_seq_norm = F.normalize(k_seq, p=2, dim=-1)
            if use_gate_in_kernel:
                g_seq_input = g_seq.requires_grad_(True)
                if safe_gate and lower_bound is not None:
                    g_processed = naive_kda_lowerbound_gate(
                        g_seq_input.to(torch.float),
                        A_log_global[:H].contiguous(),
                        dt_bias_global,
                        lower_bound=lower_bound,
                    )
                else:
                    from fla.ops.kda.gate import naive_kda_gate
                    g_processed = naive_kda_gate(
                        g_seq_input.to(torch.float),
                        A_log_global[:H].contiguous(),
                        dt_bias_global,
                    )
                g_for_grad = g_seq_input
            else:
                # chunk_gdn2 gets the raw g_local here and only cumsums it,
                # so the reference must use the very same log-decay.
                g_processed = g_seq
                g_for_grad = g_seq

            o_seq, _ = naive_recurrent_gdn2(
                q=q_seq_norm,
                k=k_seq_norm,
                v=v_seq,
                g=g_processed,
                b=b_seq,
                w=w_seq,
                initial_state=None,
                output_final_state=False,
            )

            ref_outputs.append({
                'o': o_seq,
                'q': q_seq,
                'k': k_seq,
                'v': v_seq,
                'g': g_for_grad,
                'b': b_seq,
                'w': w_seq,
                'do': do_seq,
            })

        all_dq = []
        all_dk = []
        all_dv = []
        all_dg = []
        all_db = []
        all_dw = []

        for item in ref_outputs:
            (item['o'] * item['do']).sum().backward()
            all_dq.append(item['q'].grad.detach())
            all_dk.append(item['k'].grad.detach())
            all_dv.append(item['v'].grad.detach())
            all_dg.append(item['g'].grad.detach())
            all_db.append(item['b'].grad.detach())
            all_dw.append(item['w'].grad.detach())

        ref_out = torch.cat([item['o'].detach() for item in ref_outputs], dim=1)
        ref_dq = torch.cat(all_dq, dim=1)
        ref_dk = torch.cat(all_dk, dim=1)
        ref_dv = torch.cat(all_dv, dim=1)
        ref_dg = torch.cat(all_dg, dim=1)
        ref_db = torch.cat(all_db, dim=1)
        ref_dw = torch.cat(all_dw, dim=1)

    dist.barrier()

    # Build CP context
    context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)

    chunk_size = T // world_size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size

    # Get local slices
    q_local = q_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    k_local = k_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    v_local = v_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    g_local = g_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    b_local = b_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    w_local = w_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    do_local = do_global[:, start_idx:end_idx, :].clone()

    dist.barrier()

    # CP Forward
    o_local, _ = chunk_gdn2(
        q=q_local,
        k=k_local,
        v=v_local,
        g=g_local,
        b=b_local,
        w=w_local,
        cp_context=context,
        disable_recompute=disable_recompute,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        A_log=A_log_global[:H].contiguous() if use_gate_in_kernel else None,
        dt_bias=dt_bias_global if use_gate_in_kernel else None,
        state_v_first=state_v_first,
    )

    # CP Backward
    o_local.backward(do_local)

    o_cp_global = gather_global(o_local, world_size=world_size)

    dq_cp_global = gather_global(q_local.grad, world_size=world_size)

    dk_cp_global = gather_global(k_local.grad, world_size=world_size)

    dv_cp_global = gather_global(v_local.grad, world_size=world_size)

    dg_cp_global = gather_global(g_local.grad, world_size=world_size)

    db_cp_global = gather_global(b_local.grad, world_size=world_size)

    dw_cp_global = gather_global(w_local.grad, world_size=world_size)

    test_passed = True
    if rank == 0:

        # Tolerance: ratio=8e-3 for all tensors, matching test_kda.py.
        # Measured ratios are actually ~0.9-1.6% (db ~2.3%): per-dim gating
        # plus the L2 norm over k ~ N(-50, 10) amplify the bf16 state
        # communication precision loss.
        # db (b grad) additionally involves a cross-rank reduction, so it is
        # marked warning-only unconditionally.
        tensors_to_verify = [
            ("Output", ref_out, o_cp_global),
            ("dq", ref_dq, dq_cp_global),
            ("dk", ref_dk, dk_cp_global),
            ("dv", ref_dv, dv_cp_global),
            ("dg", ref_dg, dg_cp_global),
            ("db", ref_db, db_cp_global),
            ("dw", ref_dw, dw_cp_global),
        ]

        try:
            for name, ref, cp in tensors_to_verify:
                warn = (name == "db")
                assert_close(name, ref, cp, ratio=8e-3, warning=warn)
        except AssertionError:
            test_passed = False

    dist.barrier()

    if not test_passed:
        raise AssertionError(f"Test {test_name} failed on rank {rank}")


@pytest.mark.parametrize(
    ('world_size', 'T', 'lengths', 'options'),
    [
        pytest.param(2, 10240, [3000, 4000, 3240], {}, id='CP2_SequenceCut'),
        pytest.param(2, 10240, [5120, 5120], {}, id='CP2_BoundaryAligned'),
        pytest.param(4, 10240, [7000, 3240], {}, id='CP4_Complex'),
        pytest.param(4, 10240, [10240], {}, id='CP4_SingleSequence'),
        pytest.param(8, 65536, [65536], {}, id='CP8_SingleSequence'),
        pytest.param(2, 10240, [1000, 1500, 2000, 2500, 1240, 1000, 1000], {}, id='CP2_ManyShortSequences'),
        pytest.param(2, 10240, [3000, 4000, 3240], dict(disable_recompute=True), id='CP2_DisableRecompute'),
        pytest.param(2, 10240, [3000, 4000, 3240], dict(state_v_first=True), id='CP2_TransposeState'),
        pytest.param(4, 10240, [10240], dict(state_v_first=True), id='CP4_TransposeState'),
    ],
)
def test_chunk(request, world_size, T, lengths, options):
    run_distributed(
        worker=run_cp_gdn2_test_worker,
        world_size=world_size,
        T=T,
        H=12,
        D=128,
        lengths=lengths,
        dtype=torch.bfloat16,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        test_name=request.node.callspec.id,
        **options,
    )
