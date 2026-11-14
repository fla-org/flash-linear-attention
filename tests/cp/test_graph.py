# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Compare CP KDA graph replay with eager outputs and gradients.

Each rank captures a forward or training step (cp_context + use_graph) once,
replays it with several cu_seqlens layouts (different splits and zero-length tail padding),
and compares against the eager CP path on the same shard.
The recorded region includes cross-rank state all-gathers, so ranks must capture and replay in lockstep.
NPU eager-parity and global-reference checks use separate tests; the latter
also exposes existing backend precision discrepancies against the recurrent oracle.
"""

import inspect
import logging
from importlib import import_module

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from fla.ops.cp import FLACPContext, build_cp_context
from fla.ops.kda import chunk_kda
from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import IS_NPU, IS_TF32_SUPPORTED, assert_close, device, device_torch_lib
from tests.cp.utils import shard_tensor
from tests.ops.kda_graph import MAX_NUM_SEQS, VARLEN_CASES, D, H, T, make_inputs

pytestmark = pytest.mark.skipif(not (IS_TF32_SUPPORTED or IS_NPU), reason="requires NVIDIA Ampere or newer, or Ascend")
logger = logging.getLogger(__name__)

TOKEN_NAMES = ["q", "k", "v", "g", "beta"]
INPUT_NAMES = TOKEN_NAMES + ["A_log", "dt_bias"]
OUTPUT_NAMES = ["o", "dq", "dk", "dv", "dg", "db", "dA", "dbias"]


def _make_inputs(seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Same seed on every rank reproduces one logical global batch; ranks slice their shard."""
    generator = torch.Generator(device).manual_seed(seed)
    dtype = torch.bfloat16
    hv = H
    q = torch.randn(1, T, H, D, dtype=dtype, device=device, generator=generator)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32, device=device, generator=generator), p=2, dim=-1).to(dtype)
    v = torch.randn(1, T, hv, D, dtype=dtype, device=device, generator=generator)
    g = torch.randn(1, T, hv, D, dtype=dtype, device=device, generator=generator)
    A_log = torch.log(torch.empty(1, 1, hv, 1, device=device).uniform_(1, 16, generator=generator))
    dt_bias = torch.randn(hv * D, dtype=torch.float32, device=device, generator=generator)
    beta = torch.rand(1, T, hv, dtype=dtype, device=device, generator=generator).sigmoid()
    do = torch.randn(1, T, hv, D, dtype=dtype, device=device, generator=generator)
    return dict(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias, do=do)


def _run_graph(rank, world_size):
    device = torch.device('cuda', rank)
    part = T // world_size
    lo, hi = rank * part, (rank + 1) * part
    group = dist.group.WORLD

    cu_static = torch.zeros(MAX_NUM_SEQS + 1, dtype=torch.long, device=device)
    pre_dev = torch.zeros(1, dtype=torch.int32, device=device)
    post_dev = torch.zeros(1, dtype=torch.int32, device=device)

    def refresh(base):
        local = base.cu_seqlens
        cu_static[: len(local)] = local
        cu_static[len(local):] = local[-1]
        pre_dev.fill_(base.pre_num_ranks)
        post_dev.fill_(base.post_num_ranks)

    # one persistent context around the static buffers; host fields are only read eagerly
    base0 = build_cp_context(torch.tensor(VARLEN_CASES[0][1], dtype=torch.long, device=device), group=group)
    refresh(base0)
    ctx = FLACPContext(
        group=group,
        cu_seqlens=cu_static,
        cu_seqlens_cpu=None,
        is_last_rank=base0.is_last_rank,
        pre_num_ranks=base0.pre_num_ranks,
        is_first_rank=base0.is_first_rank,
        post_num_ranks=base0.post_num_ranks,
        pre_num_ranks_dev=pre_dev,
        post_num_ranks_dev=post_dev,
    )

    def shard_leaves(inputs):
        leaves = {n: inputs[n][:, lo:hi].detach().clone().requires_grad_() for n in TOKEN_NAMES}
        for n in ("A_log", "dt_bias"):
            leaves[n] = inputs[n].detach().clone().requires_grad_()
        return leaves

    initial_inputs = _make_inputs(0, device)
    leaves = shard_leaves(initial_inputs)
    do = initial_inputs["do"][:, lo:hi].detach().clone()

    def step():
        for t in leaves.values():
            if t.grad is not None:
                t.grad.zero_()
        o, _ = chunk_kda(
            q=leaves["q"],
            k=leaves["k"],
            v=leaves["v"],
            g=leaves["g"],
            beta=leaves["beta"],
            use_gate_in_kernel=True,
            cp_context=ctx,
            use_graph=True,
            max_num_seqs=MAX_NUM_SEQS,
            A_log=leaves["A_log"],
            dt_bias=leaves["dt_bias"],
        )
        (o * do).sum().backward()
        return o

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o_captured = step()

    for seed, (tag, sequence_offsets) in enumerate(VARLEN_CASES, start=1):
        inputs = _make_inputs(seed * 17, device)
        for n in TOKEN_NAMES:
            leaves[n].data.copy_(inputs[n][:, lo:hi])
        for n in ("A_log", "dt_bias"):
            leaves[n].data.copy_(inputs[n])
        do.copy_(inputs["do"][:, lo:hi])
        cu = torch.tensor(sequence_offsets, dtype=torch.long, device=device)
        refresh(build_cp_context(cu, group=group))

        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        got = [o_captured.clone()] + [leaves[n].grad.clone() for n in INPUT_NAMES]
        dist.barrier()

        # eager CP reference on the same shard
        ref_leaves = shard_leaves(inputs)
        o, _ = chunk_kda(
            q=ref_leaves["q"],
            k=ref_leaves["k"],
            v=ref_leaves["v"],
            g=ref_leaves["g"],
            beta=ref_leaves["beta"],
            use_gate_in_kernel=True,
            cp_context=build_cp_context(cu, group=group),
            A_log=ref_leaves["A_log"],
            dt_bias=ref_leaves["dt_bias"],
        )
        (o * inputs["do"][:, lo:hi]).sum().backward()
        ref = [o.detach()] + [ref_leaves[n].grad for n in INPUT_NAMES]
        dist.barrier()

        for name, r, g in zip(OUTPUT_NAMES, ref, got):
            assert_close(f"{tag}::{name}", r, g, ratio=8e-3)

    # destroying a communicator recorded in a CUDA graph can hang; process exit reaps it
    dist.barrier()


def test_chunk_kda(run_distributed):
    # a communicator recorded by a graph must outlive the graph; worker exit reaps it
    if IS_NPU:
        run_distributed(
            worker=_run_npu_graph,
            world_size=2,
            destroy_process_group=False,
            capture='direct',
            training=True,
            dtype=torch.bfloat16,
            options={'use_gate_in_kernel': True, 'chunk_size': 64},
            reference='eager',
        )
    else:
        run_distributed(worker=_run_graph, world_size=2, destroy_process_group=False)


def _make_npu_inputs(
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    shape: tuple[int, int, int, int, int] = (T, H, H, D, D),
) -> dict[str, torch.Tensor]:
    """Same seed on every rank reproduces one logical global batch; ranks slice their shard."""
    if dtype == torch.bfloat16 and shape == (T, H, H, D, D):
        return _make_inputs(seed, device)
    generator = torch.Generator(device).manual_seed(seed)
    tokens, heads, hv, key_dim, value_dim = shape
    q = torch.randn(1, tokens, heads, key_dim, dtype=dtype, device=device, generator=generator)
    k = F.normalize(
        torch.randn(1, tokens, heads, key_dim, dtype=torch.float32, device=device, generator=generator),
        p=2,
        dim=-1,
    ).to(dtype)
    v = torch.randn(1, tokens, hv, value_dim, dtype=dtype, device=device, generator=generator)
    g = torch.randn(1, tokens, hv, key_dim, dtype=dtype, device=device, generator=generator)
    A_log = torch.log(torch.empty(1, 1, hv, 1, device=device).uniform_(1, 16, generator=generator))
    dt_bias = torch.randn(hv * key_dim, dtype=torch.float32, device=device, generator=generator)
    beta = torch.rand(1, tokens, hv, dtype=dtype, device=device, generator=generator).sigmoid()
    do = torch.randn(1, tokens, hv, value_dim, dtype=dtype, device=device, generator=generator)
    return dict(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias, do=do)


def _npu_global_reference(inputs, offsets, training, options):
    leaves = {name: inputs[name].detach().cpu().requires_grad_(training) for name in INPUT_NAMES}
    q, k, v, g, beta = (leaves[name] for name in TOKEN_NAMES)
    if options.get('use_qk_l2norm_in_kernel', False):
        q, k = (F.normalize(x.float(), dim=-1).to(x.dtype) for x in (q, k))
    if options.get('use_beta_sigmoid_in_kernel', False):
        # fused beta activation stores FP32 even when its raw input is BF16.
        beta = beta.float().sigmoid() * (2 if options.get('allow_neg_eigval', False) else 1)
    if options.get('use_gate_in_kernel', False):
        if options.get('lower_bound') is None:
            g = naive_kda_gate(g=g, A_log=leaves['A_log'], dt_bias=leaves['dt_bias'])
        else:
            g = naive_kda_lowerbound_gate(
                g=g,
                A_log=leaves['A_log'],
                dt_bias=leaves['dt_bias'],
                lower_bound=options['lower_bound'],
            )
    parts = [
        naive_recurrent_kda(q[:, start:end], k[:, start:end], v[:, start:end], g[:, start:end], beta[:, start:end])[0]
        for start, end in zip(offsets[:-1], offsets[1:]) if end > start
    ]
    o = torch.cat(parts, dim=1)
    result = {'o': o.detach()}
    if training:
        (o * inputs['do'][:, :offsets[-1]].cpu()).sum().backward()
        result.update({name: leaves[name].grad for name in INPUT_NAMES if leaves[name].grad is not None})
    return result


def _run_npu_graph(
    rank, world_size, capture, training, dtype, options, layout='contiguous', cu_dtype=torch.long,
    shape=(T, H, H, D, D), reference='global',
):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    device = torch.device('npu', rank)
    gate = options.get('use_gate_in_kernel', False)
    total_tokens = shape[0]
    part = total_tokens // world_size
    varlen_cases = [(tag, [offset * total_tokens // T for offset in offsets]) for tag, offsets in VARLEN_CASES]
    group = dist.group.WORLD

    max_num_seqs = MAX_NUM_SEQS + (layout == 'zigzag')
    cu_static = torch.zeros(max_num_seqs + 1, dtype=cu_dtype, device=device)
    pre_dev = torch.zeros(2 if layout == 'zigzag' else 1, dtype=torch.int32, device=device)
    post_dev = torch.zeros_like(pre_dev)

    def refresh(base):
        local = base.cu_seqlens
        cu_static[: len(local)] = local
        cu_static[len(local):] = local[-1]
        if layout == 'zigzag':
            pre_dev.copy_(torch.tensor(base.pre_num_ranks_by_part, dtype=torch.int32, device=device))
            post_dev.copy_(torch.tensor(base.post_num_ranks_by_part, dtype=torch.int32, device=device))
        else:
            pre_dev.fill_(base.pre_num_ranks)
            post_dev.fill_(base.post_num_ranks)

    base0 = build_cp_context(torch.tensor(varlen_cases[0][1], dtype=torch.long, device=device), group=group, layout=layout)
    refresh(base0)
    active_names = INPUT_NAMES if gate else TOKEN_NAMES

    def call(leaves, context, use_graph):
        return chunk_kda(
            **{name: leaves[name] for name in TOKEN_NAMES},
            cp_context=context,
            use_graph=use_graph,
            max_num_seqs=max_num_seqs,
            A_log=leaves['A_log'] if gate else None,
            dt_bias=leaves['dt_bias'] if gate else None,
            **options,
        )[0]

    def shard_leaves(inputs, actual_part=part):
        leaves = {}
        for name in TOKEN_NAMES:
            local = shard_tensor(inputs[name], rank, world_size, layout).detach().clone()
            local[:, :actual_part].copy_(shard_tensor(inputs[name][:, :actual_part * world_size], rank, world_size, layout))
            leaves[name] = local.requires_grad_(training)
        for n in ("A_log", "dt_bias"):
            leaves[n] = inputs[n].detach().clone().requires_grad_(training)
        return leaves

    def make_inputs(seed):
        inputs = _make_npu_inputs(seed, device, dtype, shape)
        if not gate:
            inputs['g'] = F.logsigmoid(inputs['g'].float()) * 0.02
        return inputs

    initial_inputs = make_inputs(0)
    leaves = shard_leaves(initial_inputs)
    do = shard_tensor(initial_inputs['do'], rank, world_size, layout).detach().clone()

    forward_calls = 0

    def forward(*args):
        nonlocal forward_calls
        forward_calls += 1
        ctx = FLACPContext(
            group=group,
            cu_seqlens=args[-3],
            pre_num_ranks_dev=args[-2],
            post_num_ranks_dev=args[-1],
            layout=layout,
        )
        return call(dict(zip(INPUT_NAMES, args[:-3])), ctx, use_graph=True)

    def step():
        for t in leaves.values():
            if t.grad is not None:
                t.grad.zero_()
        o = forward(*(leaves[name] for name in INPUT_NAMES), cu_static, pre_dev, post_dev)
        if training:
            (o * do).sum().backward()
        return o

    logger.info('rank=%d: warmup (%s)', rank, capture)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            step()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    dist.barrier()
    logger.info('rank=%d: capture', rank)
    if capture == 'direct':
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            o_captured = step()
    else:
        # release warmup autograd nodes before capturing forward and backward on a different stream.
        for tensor in leaves.values():
            tensor.grad = None
        graphed = torch.npu.make_graphed_callables(
            forward,
            tuple(leaves[name] for name in INPUT_NAMES) + (cu_static, pre_dev, post_dev),
            num_warmup_iters=0,
            allow_unused_input=True,
        )
    torch.npu.synchronize()
    logger.info('rank=%d: capture complete', rank)
    captured_forward_calls = forward_calls
    capture_addresses = tuple(tensor.data_ptr() for tensor in (*leaves.values(), do))
    captured_output_address = o_captured.data_ptr() if capture == 'direct' else None
    captured_gradient_addresses = tuple(leaves[name].grad.data_ptr() for name in active_names) if capture == 'direct' else ()
    metadata_addresses = tuple(tensor.data_ptr() for tensor in (cu_static, pre_dev, post_dev))
    replay_cases = varlen_cases + [
        ('quarter-aligned', [0, total_tokens // 4, total_tokens // 2, 3 * total_tokens // 4, total_tokens]),
        ('one-token-boundary', [0, part - 1, part + 1, total_tokens, total_tokens]),
    ]
    if layout == 'zigzag':
        replay_cases += [
            ('four-front-sequences', [0, 1, 2, 3, total_tokens]),
            ('part-one-token-boundary', [0, part // 2 - 1, part // 2 + 1, total_tokens, total_tokens]),
        ]
    replay_cases += [
        ('half-capacity', [0, 1, part // 2 + 1, total_tokens // 2, total_tokens // 2]),
        varlen_cases[0],
    ]
    global_reference_errors = []
    for seed, (tag, sequence_offsets) in enumerate(replay_cases, start=1):
        inputs = make_inputs(seed * 17)
        actual_part = sequence_offsets[-1] // world_size
        actual_tokens = sequence_offsets[-1]
        live_leaves = shard_leaves(inputs, actual_part)
        if capture == 'direct':
            with torch.no_grad():
                for name in INPUT_NAMES:
                    leaves[name].copy_(live_leaves[name])
        else:
            leaves = live_leaves
        do.copy_(shard_tensor(inputs['do'], rank, world_size, layout))
        do[:, :actual_part].copy_(shard_tensor(inputs['do'][:, :actual_tokens], rank, world_size, layout))
        cu = torch.tensor(sequence_offsets, dtype=torch.long, device=device)
        refresh(build_cp_context(cu, group=group, layout=layout))

        dist.barrier()
        logger.info('rank=%d: replay %s', rank, tag)
        if capture == 'direct':
            graph.replay()
            o = o_captured
        else:
            o = graphed(*(leaves[name] for name in INPUT_NAMES), cu_static, pre_dev, post_dev)
            if training:
                (o * do).sum().backward()
        torch.npu.synchronize()
        assert forward_calls == captured_forward_calls, 'replay re-entered the Python operator'
        if capture == 'direct':
            assert tuple(tensor.data_ptr() for tensor in (*leaves.values(), do)) == capture_addresses
            assert o.data_ptr() == captured_output_address
            assert tuple(leaves[name].grad.data_ptr() for name in active_names) == captured_gradient_addresses
        got = {'o': o.detach().clone()}
        if training:
            got.update({name: leaves[name].grad.clone() for name in active_names})
        if actual_part < part:
            for name, tensor in got.items():
                if name == 'o' or name in TOKEN_NAMES:
                    padding = tensor[:, actual_part:]
                    torch.testing.assert_close(padding, torch.zeros_like(padding), rtol=0, atol=0)
                    got[name] = tensor[:, :actual_part]
        assert tuple(tensor.data_ptr() for tensor in (cu_static, pre_dev, post_dev)) == metadata_addresses
        dist.barrier()

        # eager CP reference on the same shard
        ref_leaves = {
            name: (shard_tensor(inputs[name][:, :actual_tokens], rank, world_size, layout)
                   if name in TOKEN_NAMES else inputs[name])
            .detach().clone().requires_grad_(training)
            for name in INPUT_NAMES
        }
        o = call(ref_leaves, build_cp_context(cu, group=group, layout=layout), use_graph=False)
        ref = {'o': o.detach()}
        if training:
            (o * shard_tensor(inputs['do'][:, :actual_tokens], rank, world_size, layout)).sum().backward()
            ref.update({name: ref_leaves[name].grad for name in active_names})
        dist.barrier()

        for name in got:
            assert torch.isfinite(got[name]).all(), f'{tag}::{name} graph result is not finite'
            assert torch.isfinite(ref[name]).all(), f'{tag}::{name} eager result is not finite'
            assert_close(f"{tag}::{name}", ref[name].float(), got[name].float(), ratio=8e-3)

        if reference == 'eager':
            continue
        global_reference = _npu_global_reference(inputs, sequence_offsets, training, options)
        for name, actual in got.items():
            expected = global_reference[name]
            if name == 'o' or name in TOKEN_NAMES:
                expected = shard_tensor(expected[:, :actual_tokens], rank, world_size, layout)
            else:
                dist.all_reduce(actual)
            assert torch.isfinite(expected).all(), f'{tag}::{name} global reference is not finite'
            # finish every replay layout before reporting global reference discrepancies.
            try:
                assert_close(f'{tag}::global::{name}', expected.to(device).float(), actual.float(), ratio=8e-3)
            except AssertionError as error:
                global_reference_errors.append(str(error))

    # captured communicators must outlive their graphs; process exit reaps them.
    dist.barrier()
    assert not global_reference_errors, '\n'.join(global_reference_errors)


@pytest.mark.parametrize(
    ('capture', 'training', 'dtype', 'options', 'cu_dtype', 'world_size', 'layout', 'shape'),
    [
        pytest.param(
            'callable', False, torch.float16, {'chunk_size': 32},
            torch.int32, 2, 'contiguous', (T, H, H, D, D),
            id='cp2-contiguous-fp16-pregated-forward-eager',
        ),
        pytest.param(
            'callable', True, torch.bfloat16,
            {
                'use_qk_l2norm_in_kernel': True,
                'use_gate_in_kernel': True,
                'use_beta_sigmoid_in_kernel': True,
                'allow_neg_eigval': True,
                'safe_gate': True,
                'lower_bound': -5.0,
                'state_v_first': True,
                'chunk_size': 64,
            },
            torch.int64, 4, 'zigzag', (T, H, H, D, D),
            id='cp4-zigzag-bf16-fused-safe-v-first-eager',
        ),
        pytest.param(
            'callable', True, torch.bfloat16,
            {'use_qk_l2norm_in_kernel': True, 'use_gate_in_kernel': True, 'chunk_size': 32},
            torch.int32, 8, 'zigzag', (1024, 2, 4, 64, 32),
            id='cp8-zigzag-bf16-gva-k64-v32-eager',
        ),
        pytest.param(
            'callable', True, torch.float16,
            {'use_gate_in_kernel': True, 'use_beta_sigmoid_in_kernel': True, 'disable_recompute': True},
            torch.int64, 4, 'contiguous', (1024, 2, 4, 64, 128),
            id='cp4-contiguous-fp16-gva-k64-v128-no-recompute-eager',
        ),
        pytest.param(
            'callable', True, torch.float16, {'chunk_size': 64},
            torch.int64, 2, 'zigzag', (8192, 2, 2, 128, 64),
            id='cp2-zigzag-fp16-long-t8192-eager',
        ),
    ],
)
@pytest.mark.skipif(not IS_NPU, reason='requires Ascend NPU graphs')
def test_chunk_kda_npu_graph(run_distributed, capture, training, dtype, options, cu_dtype, world_size, layout, shape):
    """Check replay invariants and eager parity across representative CP execution paths."""
    run_distributed(
        worker=_run_npu_graph,
        world_size=world_size,
        destroy_process_group=False,
        capture=capture,
        training=training,
        dtype=dtype,
        options=options,
        layout=layout,
        cu_dtype=cu_dtype,
        shape=shape,
        reference='eager',
    )


@pytest.mark.parametrize(
    ('capture', 'training', 'dtype', 'options', 'cu_dtype', 'world_size', 'layout', 'shape'),
    [
        pytest.param(
            'direct', True, torch.bfloat16,
            {'use_gate_in_kernel': True, 'chunk_size': 64},
            torch.int64, 2, 'contiguous', (T, H, H, D, D),
            id='cp2-contiguous-bf16-fused-global',
        ),
        pytest.param(
            'direct', True, torch.float16,
            {'use_gate_in_kernel': True, 'state_v_first': True, 'chunk_size': 32},
            torch.int32, 2, 'contiguous', (1024, 3, 3, 100, 80),
            id='cp2-contiguous-fp16-unaligned-k100-v80-global',
        ),
        pytest.param(
            'direct', True, torch.float16,
            {'use_gate_in_kernel': True, 'state_v_first': True, 'chunk_size': 32},
            torch.int32, 8, 'zigzag', (1024, 3, 3, 100, 80),
            id='cp8-zigzag-fp16-unaligned-k100-v80-global',
        ),
    ],
)
@pytest.mark.skipif(not IS_NPU, reason='requires Ascend NPU graphs')
def test_chunk_kda_npu_graph_reference(run_distributed, capture, training, dtype, options, cu_dtype, world_size, layout, shape):
    """Check cross-rank graph outputs and gradients against the independent CPU recurrence."""
    run_distributed(
        worker=_run_npu_graph,
        world_size=world_size,
        destroy_process_group=False,
        capture=capture,
        training=training,
        dtype=dtype,
        options=options,
        layout=layout,
        cu_dtype=cu_dtype,
        shape=shape,
        reference='global',
    )


def _eager_cp(inputs, sequence_offsets, gate):
    """Eager CP reference: world_size=1 crosses no rank, so only real tokens matter."""
    leaves = {n: inputs[n].detach().clone().requires_grad_() for n in ("q", "k", "v", "g", "beta")}
    extra = {}
    if gate:
        leaves["A_log"] = inputs["A_log"].detach().clone().requires_grad_()
        leaves["dt_bias"] = inputs["dt_bias"].detach().clone().requires_grad_()
        extra = dict(A_log=leaves["A_log"], dt_bias=leaves["dt_bias"])
    cu = torch.tensor(sequence_offsets, dtype=torch.long, device=device)
    o, _ = chunk_kda(
        q=leaves["q"],
        k=leaves["k"],
        v=leaves["v"],
        g=leaves["g"],
        beta=leaves["beta"],
        use_gate_in_kernel=gate,
        cp_context=build_cp_context(cu, group=dist.group.WORLD),
        **extra,
    )
    (o * inputs["do"]).sum().backward()
    out = [o.detach(), leaves["q"].grad, leaves["k"].grad, leaves["v"].grad, leaves["g"].grad, leaves["beta"].grad]
    out += [leaves["A_log"].grad, leaves["dt_bias"].grad] if gate else [None, None]
    return out


def test_chunk_kda_single_rank(single_rank_group, monkeypatch):
    # keep the eager reference on the same backend as graph capture
    monkeypatch.setenv("FLA_TILELANG", "0")
    gate = True
    cu = torch.tensor(VARLEN_CASES[0][1], dtype=torch.long, device=device)
    pre_dev = torch.zeros(1, dtype=torch.int32, device=device)
    post_dev = torch.zeros(1, dtype=torch.int32, device=device)
    base = build_cp_context(cu, group=dist.group.WORLD)
    ctx = FLACPContext(
        group=dist.group.WORLD,
        cu_seqlens=cu,
        cu_seqlens_cpu=base.cu_seqlens_cpu,
        is_last_rank=base.is_last_rank,
        pre_num_ranks=base.pre_num_ranks,
        is_first_rank=base.is_first_rank,
        post_num_ranks=base.post_num_ranks,
        pre_num_ranks_dev=pre_dev,
        post_num_ranks_dev=post_dev,
    )

    initial_inputs = make_inputs(seed=0, gate=gate, safe_gate=False)
    names = ["q", "k", "v", "g", "beta", "A_log", "dt_bias"]
    leaves = {n: initial_inputs[n].detach().clone().requires_grad_() for n in names}
    do = initial_inputs["do"].detach().clone()

    def step():
        for t in leaves.values():
            if t.grad is not None:
                t.grad.zero_()
        o, _ = chunk_kda(
            q=leaves["q"],
            k=leaves["k"],
            v=leaves["v"],
            g=leaves["g"],
            beta=leaves["beta"],
            use_gate_in_kernel=gate,
            cp_context=ctx,
            use_graph=True,
            max_num_seqs=MAX_NUM_SEQS,
            A_log=leaves["A_log"],
            dt_bias=leaves["dt_bias"],
        )
        (o * do).sum().backward()
        return o.detach().clone()

    stream = device_torch_lib.Stream()
    stream.wait_stream(device_torch_lib.current_stream())
    with device_torch_lib.stream(stream):
        for _ in range(3):
            step()
    device_torch_lib.current_stream().wait_stream(stream)

    graph = torch.npu.NPUGraph() if IS_NPU else torch.cuda.CUDAGraph()
    with device_torch_lib.graph(graph):
        o_captured = step()

    names_out = ["o", "dq", "dk", "dv", "dg", "db", "dA", "dbias"]
    for seed, (tag, sequence_offsets) in enumerate(VARLEN_CASES, start=1):
        inputs = make_inputs(seed=seed * 17, gate=gate, safe_gate=False)
        for n in names:
            leaves[n].data.copy_(inputs[n])
        do.copy_(inputs["do"])
        cu.copy_(torch.tensor(sequence_offsets, dtype=torch.long, device=device))

        graph.replay()
        device_torch_lib.synchronize()

        got = [o_captured] + [leaves[n].grad for n in names]
        ref = _eager_cp(inputs, sequence_offsets, gate)
        for name, r, g in zip(names_out, ref, got):
            assert_close(f"cp_world1::{tag}::{name}", r, g, 2e-3)


@pytest.mark.skipif(not IS_NPU, reason='requires Ascend NPU graphs')
@pytest.mark.parametrize(
    ('chunk_size', 'state_v_first', 'training'),
    [
        pytest.param(64, False, False, id='bt64-k-first-forward-tail'),
        pytest.param(64, False, True, id='bt64-k-first-backward-tail'),
        pytest.param(64, True, True, id='bt64-v-first-state-layout'),
        pytest.param(32, True, True, id='bt32-v-first-multi-chunk-tail'),
    ],
)
def test_chunk_kda_graph_cpu_snapshot(single_rank_group, monkeypatch, chunk_size, state_v_first, training):
    """Replay changing sequence boundaries without stale CPU bounds or out-of-range writes."""
    recurrence = import_module('fla.ops.common.backends.triton_ascend.chunk_delta_h')
    torch.manual_seed(42)
    tokens, width = 128, 64
    q = torch.randn(1, tokens, 1, width, device=device, dtype=torch.float16)
    k = F.normalize(torch.randn_like(q).float(), dim=-1).to(q.dtype)
    inputs = [q, k, torch.randn_like(q), torch.full_like(q, -0.01, dtype=torch.float32)]
    inputs.append(torch.full((1, tokens, 1), 0.5, device=device, dtype=q.dtype))
    leaves = [x.detach().requires_grad_(training) for x in inputs]
    do = torch.randn_like(q)
    cpu = torch.tensor([0, 64, tokens], dtype=torch.long)
    cu = cpu.to(device)
    context = build_cp_context(cu, group=dist.group.WORLD)
    cu = context.cu_seqlens
    context.pre_num_ranks_dev = torch.zeros(1, device=device, dtype=torch.int32)
    context.post_num_ranks_dev = torch.zeros(1, device=device, dtype=torch.int32)
    cpu_snapshot = context.cu_seqlens_cpu
    original_snapshot = cpu_snapshot.clone()
    options = dict(chunk_size=chunk_size, state_v_first=state_v_first)
    guards = []
    empty_like = torch.empty_like

    def guarded_empty_like(x, *args, **kwargs):
        frame = inspect.currentframe().f_back
        caller = frame.f_code.co_name
        guarded = (frame.f_globals.get('__name__') == recurrence.__name__ and x.shape == q.shape
                   and caller in ('chunk_gated_delta_rule_fwd_h_npu', 'chunk_gated_delta_rule_bwd_dhu_npu'))
        del frame
        if not guarded:
            return empty_like(x, *args, **kwargs)
        storage = torch.full((1, tokens + 2 * chunk_size, 1, width), -123., device=device, dtype=x.dtype)
        storage[:, chunk_size:chunk_size + tokens].fill_(float('nan'))
        guards.append((caller, storage))
        return storage[:, chunk_size:chunk_size + tokens]

    def step():
        for x in leaves:
            if x.grad is not None:
                x.grad.zero_()
        o, _ = chunk_kda(
            *leaves,
            cp_context=context,
            use_graph=True,
            max_num_seqs=2,
            **options,
        )
        if training:
            (o * do).sum().backward()
        return o

    with monkeypatch.context() as patch:
        patch.setattr(torch, 'empty_like', guarded_empty_like)
        for _ in range(3):
            step()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = step()
    guarded_callers = {caller for caller, _ in guards}
    assert 'chunk_gated_delta_rule_fwd_h_npu' in guarded_callers
    if training:
        assert 'chunk_gated_delta_rule_bwd_dhu_npu' in guarded_callers

    for offsets in ([0, 64, tokens], [0, 65, tokens], [0, tokens, tokens], [0, 64, tokens]):
        cu.copy_(torch.tensor(offsets, device=device))
        graph.replay()
        torch.npu.synchronize()
        for _, storage in guards:
            assert (storage[:, :chunk_size] == -123).all(), 'write before recurrence buffer'
            assert (storage[:, chunk_size + tokens:] == -123).all(), 'write after recurrence buffer'
        ref_inputs = [x.detach().clone().requires_grad_(training) for x in inputs]
        ref_context = build_cp_context(cu.clone(), group=dist.group.WORLD)
        ref, _ = chunk_kda(*ref_inputs, cp_context=ref_context, **options)
        if training:
            (ref * do).sum().backward()
        actual = [output] + ([x.grad for x in leaves] if training else [])
        expected = [ref] + ([x.grad for x in ref_inputs] if training else [])
        for name, r, g in zip(OUTPUT_NAMES, expected, actual):
            assert torch.isfinite(g).all(), name
            assert_close(f'cpu_snapshot::{offsets}::{name}', r.float(), g.float(), ratio=2e-3)
        assert context.cu_seqlens_cpu is cpu_snapshot
        torch.testing.assert_close(cpu_snapshot, original_snapshot, rtol=0, atol=0)
