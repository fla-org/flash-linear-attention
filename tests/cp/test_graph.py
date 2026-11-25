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
also exposes existing backend precision discrepancies against the CPU recurrent reference.
"""

import inspect
import logging
from importlib import import_module

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton

import fla.backends as backends
from fla.ops.cp import FLACPContext, build_cp_context
from fla.ops.cp.chunk_delta_h import _graph_metadata
from fla.ops.cp.chunk_merge import chunk_cp_merge
from fla.ops.kda import chunk_kda
from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import IS_NPU, IS_TF32_SUPPORTED, ascend_compile_kwargs, assert_close, device, device_torch_lib
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

    max_num_seqs = MAX_NUM_SEQS + 1
    cu_static = torch.zeros(max_num_seqs + 1, dtype=torch.long, device=device)
    pre_dev = torch.zeros(1, dtype=torch.int32, device=device)
    post_dev = torch.zeros(1, dtype=torch.int32, device=device)

    def refresh(base, empty_sequence=None):
        local = base.cu_seqlens
        if empty_sequence is not None:
            index = 0 if empty_sequence == 'leading' else 1
            local = torch.cat((local[:index], local[index:index + 1], local[index:]))
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
            max_num_seqs=max_num_seqs,
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

    replay_cases = list(VARLEN_CASES) + [
        ('leading-empty-sequence', [0, T]),
        ('internal-empty-sequence', [0, part // 4, part + part // 2, T]),
        VARLEN_CASES[0],
    ]
    for seed, (tag, sequence_offsets) in enumerate(replay_cases, start=1):
        inputs = _make_inputs(seed * 17, device)
        for n in TOKEN_NAMES:
            leaves[n].data.copy_(inputs[n][:, lo:hi])
        for n in ("A_log", "dt_bias"):
            leaves[n].data.copy_(inputs[n])
        do.copy_(inputs["do"][:, lo:hi])
        cu = torch.tensor(sequence_offsets, dtype=torch.long, device=device)
        empty_sequence = {'leading-empty-sequence': 'leading', 'internal-empty-sequence': 'internal'}.get(tag)
        refresh(build_cp_context(cu, group=group), empty_sequence)

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


@pytest.mark.skipif(IS_NPU, reason='Ascend graph coverage is provided by test_chunk_kda_npu_graph_reference')
def test_chunk_kda(run_distributed):
    # a communicator recorded by a graph must outlive the graph; worker exit reaps it
    run_distributed(worker=_run_graph, world_size=2, destroy_process_group=False)


@pytest.mark.parametrize(
    ('layout', 'state_v_first', 'native'),
    [
        pytest.param('contiguous', False, True, id='ascend-contiguous-k-first', marks=pytest.mark.skipif(
            not IS_NPU, reason='requires Ascend merge kernel',
        )),
        pytest.param('zigzag', True, True, id='ascend-zigzag-v-first', marks=pytest.mark.skipif(
            not IS_NPU, reason='requires Ascend merge kernel',
        )),
        pytest.param('contiguous', True, False, id='shared-contiguous-v-first'),
    ],
)
def test_chunk_cp_backward_merge_empty_graph(layout, state_v_first, native):
    """Preserve both guard rows when replay has no active boundary state to write."""
    HV, K, V = 1, 64, 32 if state_v_first else 64
    num_parts = 2 if layout == 'zigzag' else 1
    cu = torch.tensor([0, 32, 32, 64], device=device, dtype=torch.long)
    post = torch.ones(num_parts, device=device, dtype=torch.int32)
    state_shape = (HV, V, K) if state_v_first else (HV, K, V)
    storage = torch.full((5, *state_shape), 123., device=device, dtype=torch.float32)
    dht = storage[1:-1]
    summaries = torch.zeros((2, HV, K, V + K), device=device, dtype=torch.float32)
    summaries[1, ..., :V] = 2

    if native:
        from fla.ops.cp.backends.triton_ascend.chunk_delta_h import merge_fwd_bwd_kernel_npu, prepare_cp_graph_metadata
        graph_metadata = prepare_cp_graph_metadata
    else:
        graph_metadata = inspect.unwrap(_graph_metadata)

    def step():
        _, last, _, _ = graph_metadata(cu, layout=layout)
        for part in range(num_parts):
            h_seq_idx = last[part:part + 1]
            if native:
                merge_fwd_bwd_kernel_npu[(triton.cdiv(V, 32), HV)](
                    h=dht,
                    ag_hm=summaries,
                    pre_or_post_num_ranks=post[part:part + 1],
                    rank=0,
                    seq_offsets=None,
                    init_offsets=None,
                    h0_seq_ids=None,
                    h0=None,
                    h_seq_idx=h_seq_idx,
                    HV=HV,
                    K=K,
                    V=V,
                    BV=32,
                    BK=K,
                    FORWARD=False,
                    INTRACARD_MODE=False,
                    NUM_SEQ_ENTRIES=0,
                    STATE_V_FIRST=state_v_first,
                    AFFINE_CHAIN_PRECISION='ieee',
                    NUM_RANKS_ON_DEVICE=True,
                    **ascend_compile_kwargs(),
                )
            else:
                chunk_cp_merge(
                    h=dht,
                    ag_hm=summaries,
                    num_ranks=post,
                    rank=0,
                    state_v_first=state_v_first,
                    forward=False,
                    h_seq_idx=h_seq_idx,
                    use_graph=True,
                )

    stream = device_torch_lib.Stream()
    stream.wait_stream(device_torch_lib.current_stream())
    with device_torch_lib.stream(stream):
        step()
    device_torch_lib.current_stream().wait_stream(stream)
    device_torch_lib.synchronize()
    graph = torch.npu.NPUGraph() if IS_NPU else torch.cuda.CUDAGraph()
    with device_torch_lib.graph(graph):
        step()

    for offsets, active_rows in (
        ([0, 32, 32, 64], (0, 2) if num_parts == 2 else (2,)),
        ([0, 0, 0, 0], ()),
        ([0, 32, 64, 64], (0, 1) if num_parts == 2 else (1,)),
    ):
        cu.copy_(torch.tensor(offsets, device=device, dtype=cu.dtype))
        post.fill_(bool(active_rows))
        dht.fill_(-1)
        expected = torch.full_like(storage, 123.)
        expected[1:-1].fill_(-1)
        for row in active_rows:
            expected[row + 1].fill_(2)
        graph.replay()
        device_torch_lib.synchronize()
        torch.testing.assert_close(storage, expected, rtol=0, atol=0, msg=f'boundaries {offsets}')


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


def _npu_replay_cases(total_tokens, world_size, layout):
    """Exercise boundary changes and restoration within one captured graph."""
    part = total_tokens // world_size
    varlen_cases = [(tag, [offset * total_tokens // T for offset in offsets]) for tag, offsets in VARLEN_CASES]
    cases = varlen_cases + [
        ('quarter-aligned', [0, total_tokens // 4, total_tokens // 2, 3 * total_tokens // 4, total_tokens]),
        ('one-token-boundary', [0, part - 1, part + 1, total_tokens, total_tokens]),
    ]
    if layout == 'zigzag':
        cases += [
            ('four-front-sequences', [0, 1, 2, 3, total_tokens]),
            ('part-one-token-boundary', [0, part // 2 - 1, part // 2 + 1, total_tokens, total_tokens]),
            ('split-empty-sequence', [0, total_tokens]),
        ]
    return cases + [
        ('leading-empty-sequence', [0, total_tokens]),
        ('internal-empty-sequence', [0, part // 4, part + part // 2, total_tokens]),
        ('half-capacity', [0, 1, part // 2 + 1, total_tokens // 2, total_tokens // 2]),
        ('all-empty', [0]),
        varlen_cases[0],
    ]


def _npu_graph_results(o, leaves, active_names, *, training, actual_part, part):
    """Snapshot replay outputs and gradients, checking unused token capacity before trimming it."""
    result = {'o': o.detach().clone()}
    if training:
        result.update({name: leaves[name].grad.clone() for name in active_names})
    if actual_part < part:
        for name, tensor in result.items():
            if name == 'o' or name in TOKEN_NAMES:
                padding = tensor[:, actual_part:]
                torch.testing.assert_close(padding, torch.zeros_like(padding), rtol=0, atol=0)
                result[name] = tensor[:, :actual_part]
    return result


def _capture_npu_graph(forward, step, leaves, metadata, *, capture, rank):
    """Warm up and capture in rank lockstep, returning the stream to keep its lifetime with the graph."""
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
    output = None
    if capture == 'direct':
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured, stream=stream):
            output = step()
    else:
        # release warmup autograd nodes before capturing forward and backward on a different stream.
        for tensor in leaves.values():
            tensor.grad = None
        captured = torch.npu.make_graphed_callables(
            forward,
            tuple(leaves[name] for name in INPUT_NAMES) + metadata,
            num_warmup_iters=0,
            allow_unused_input=True,
        )
    torch.npu.synchronize()
    logger.info('rank=%d: capture complete', rank)
    return captured, output, stream


def _npu_eager_reference(inputs, cu, call, *, active_names, actual_tokens, rank, world_size, layout, training):
    """Run eager CP on the same shard with fresh leaves and current sequence boundaries."""
    leaves = {
        name: (shard_tensor(inputs[name][:, :actual_tokens], rank, world_size, layout)
               if name in TOKEN_NAMES else inputs[name])
        .detach().clone().requires_grad_(training)
        for name in INPUT_NAMES
    }
    context = build_cp_context(cu, group=dist.group.WORLD, layout=layout)
    o = call(leaves, context, use_graph=False)
    result = {'o': o.detach()}
    if training:
        (o * shard_tensor(inputs['do'][:, :actual_tokens], rank, world_size, layout)).sum().backward()
        result.update({name: leaves[name].grad for name in active_names})
    return result


def _check_npu_global_reference(tag, got, reference, *, rank, world_size, layout):
    """Compare against CPU recurrence, deferring tolerance failures until all replay layouts finish."""
    errors = []
    actual_tokens = reference['o'].shape[1]
    for name, actual in got.items():
        expected = reference[name]
        if name == 'o' or name in TOKEN_NAMES:
            expected = shard_tensor(expected[:, :actual_tokens], rank, world_size, layout)
        else:
            dist.all_reduce(actual)
        assert torch.isfinite(expected).all(), f'{tag}::{name} global reference is not finite'
        try:
            assert_close(f'{tag}::global::{name}', expected.to(actual.device).float(), actual.float(), ratio=8e-3)
        except AssertionError as error:
            errors.append(str(error))
    return errors


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
    replay_cases = _npu_replay_cases(total_tokens, world_size, layout)
    group = dist.group.WORLD

    max_num_seqs = MAX_NUM_SEQS + 1
    cu_static = torch.zeros(max_num_seqs + 1, dtype=cu_dtype, device=device)
    pre_dev = torch.zeros(2 if layout == 'zigzag' else 1, dtype=torch.int32, device=device)
    post_dev = torch.zeros_like(pre_dev)

    def refresh(base, empty_sequence=None):
        local = base.cu_seqlens
        if empty_sequence is not None:
            index = 0 if empty_sequence == 'leading' else base.front_num_seqs if empty_sequence == 'split' else 1
            local = torch.cat((local[:index], local[index:index + 1], local[index:]))
        cu_static[: len(local)] = local
        cu_static[len(local):] = local[-1]
        if layout == 'zigzag':
            pre_dev.copy_(torch.tensor(base.pre_num_ranks_by_part, dtype=torch.int32, device=device))
            post_dev.copy_(torch.tensor(base.post_num_ranks_by_part, dtype=torch.int32, device=device))
        else:
            pre_dev.fill_(base.pre_num_ranks)
            post_dev.fill_(base.post_num_ranks)

    base0 = build_cp_context(torch.tensor(replay_cases[0][1], dtype=torch.long, device=device), group=group, layout=layout)
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

    captured, o_captured, _capture_stream = _capture_npu_graph(
        forward=forward,
        step=step,
        leaves=leaves,
        metadata=(cu_static, pre_dev, post_dev),
        capture=capture,
        rank=rank,
    )
    captured_forward_calls = forward_calls
    capture_addresses = tuple(tensor.data_ptr() for tensor in (*leaves.values(), do))
    captured_output_address = o_captured.data_ptr() if capture == 'direct' else None
    captured_gradient_addresses = tuple(leaves[name].grad.data_ptr() for name in active_names) if capture == 'direct' else ()
    metadata_addresses = tuple(tensor.data_ptr() for tensor in (cu_static, pre_dev, post_dev))
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
        empty_sequence = {
            'leading-empty-sequence': 'leading', 'internal-empty-sequence': 'internal', 'split-empty-sequence': 'split',
        }.get(tag)
        if actual_tokens:
            refresh(build_cp_context(cu, group=group, layout=layout), empty_sequence)
        else:
            cu_static.zero_()
            pre_dev.zero_()
            post_dev.zero_()

        dist.barrier()
        logger.info('rank=%d: replay %s', rank, tag)
        if capture == 'direct':
            captured.replay()
            o = o_captured
        else:
            o = captured(*(leaves[name] for name in INPUT_NAMES), cu_static, pre_dev, post_dev)
            if training:
                (o * do).sum().backward()
        torch.npu.synchronize()
        assert forward_calls == captured_forward_calls, 'replay re-entered the Python operator'
        if capture == 'direct':
            assert tuple(tensor.data_ptr() for tensor in (*leaves.values(), do)) == capture_addresses
            assert o.data_ptr() == captured_output_address
            assert tuple(leaves[name].grad.data_ptr() for name in active_names) == captured_gradient_addresses
        got = _npu_graph_results(o, leaves, active_names, training=training, actual_part=actual_part, part=part)
        assert tuple(tensor.data_ptr() for tensor in (cu_static, pre_dev, post_dev)) == metadata_addresses
        dist.barrier()

        if not actual_tokens:
            # Token rows were checked before trimming; gate parameters also receive no gradient.
            for result in got.values():
                torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)
            continue

        ref = _npu_eager_reference(
            inputs,
            cu,
            call,
            active_names=active_names,
            actual_tokens=actual_tokens,
            rank=rank,
            world_size=world_size,
            layout=layout,
            training=training,
        )
        dist.barrier()

        for name in got:
            assert torch.isfinite(got[name]).all(), f'{tag}::{name} graph result is not finite'
            assert torch.isfinite(ref[name]).all(), f'{tag}::{name} eager result is not finite'
            assert_close(f"{tag}::{name}", ref[name].float(), got[name].float(), ratio=8e-3)

        if reference == 'eager':
            continue
        global_reference = _npu_global_reference(inputs, sequence_offsets, training, options)
        global_reference_errors.extend(
            _check_npu_global_reference(tag, got, global_reference, rank=rank, world_size=world_size, layout=layout),
        )

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
@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
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
        pytest.param(
            'direct', True, torch.float16, {'chunk_size': 64},
            torch.int64, 2, 'zigzag', (1024, 1, 1, 64, 64),
            id='cp2-zigzag-fp16-pregated-global',
        ),
    ],
)
@pytest.mark.skipif(not IS_NPU, reason='requires Ascend NPU graphs')
@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
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


@pytest.mark.skipif(IS_NPU and backends._DISPATCH_DISABLED, reason='Ascend CP graph replay requires backend dispatch')
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
@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
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


@pytest.mark.skipif(not IS_NPU, reason='requires Ascend NPU graphs')
@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Ascend CP graph replay requires backend dispatch')
@pytest.mark.parametrize(
    ('cu_dtype', 'cu_stride', 'state_v_first', 'disable_recompute', 'chunk_size'),
    [
        pytest.param(torch.int64, 1, False, False, 64, id='int64-k-first-recompute'),
        pytest.param(torch.int32, 2, True, True, 32, id='int32-strided-v-first-saved-state'),
    ],
)
@pytest.mark.parametrize('layout', ['contiguous', 'zigzag'])
def test_chunk_kda_graph_metadata(
    single_rank_group, monkeypatch, layout, cu_dtype, cu_stride, state_v_first, disable_recompute, chunk_size,
):
    """Keep packed metadata private to each autograd invocation and captured graph."""
    metadata_module = import_module('fla.ops.cp.backends.triton_ascend.chunk_delta_h')
    prepare = metadata_module.prepare_cp_graph_metadata
    created = []

    def track(cu_seqlens, layout):
        metadata = prepare(cu_seqlens, layout=layout)
        tensors = (metadata[0], metadata[1], *metadata[2], *metadata[3])
        bases = {x.untyped_storage().data_ptr() for x in tensors}
        assert len(bases) == 1
        created.append(next(iter(bases)))
        return metadata

    monkeypatch.setattr(metadata_module, 'prepare_cp_graph_metadata', track)
    cu = torch.tensor([0, 1, 64, 65, 128], device=device, dtype=cu_dtype)
    context = build_cp_context(cu, group=dist.group.WORLD, layout=layout)
    storage = torch.empty(context.cu_seqlens.numel() * cu_stride, device=device, dtype=cu_dtype)
    cu = storage[::cu_stride]
    cu.copy_(context.cu_seqlens)
    context.cu_seqlens = cu
    pre = context.pre_num_ranks_by_part if layout == 'zigzag' else [context.pre_num_ranks]
    post = context.post_num_ranks_by_part if layout == 'zigzag' else [context.post_num_ranks]
    context.pre_num_ranks_dev = torch.tensor(pre, device=device, dtype=torch.int32)
    context.post_num_ranks_dev = torch.tensor(post, device=device, dtype=torch.int32)
    options = dict(state_v_first=state_v_first, disable_recompute=disable_recompute, chunk_size=chunk_size)

    def inputs(seed):
        torch.manual_seed(seed)
        q = torch.randn(1, 128, 1, 64, device=device, dtype=torch.float16)
        k = F.normalize(torch.randn_like(q).float(), dim=-1).to(q.dtype)
        return [x.requires_grad_() for x in (
            q, k, torch.randn_like(q), torch.full_like(q, -0.01, dtype=torch.float32),
            torch.full((1, 128, 1), 0.5, device=device, dtype=q.dtype),
        )]

    def forward(*leaves):
        return chunk_kda(*leaves, cp_context=context, use_graph=True, max_num_seqs=len(cu) - 1, **options)[0]

    def check(leaves, output, sequence_offsets=None):
        reference_leaves = [x.detach().clone().requires_grad_() for x in leaves]
        boundaries = cu.clone() if sequence_offsets is None else torch.tensor(sequence_offsets, device=device)
        reference_context = build_cp_context(boundaries, group=dist.group.WORLD, layout=layout)
        reference, _ = chunk_kda(*reference_leaves, cp_context=reference_context, **options)
        reference.sum().backward()
        for name, got, want in zip(('o', 'dq', 'dk', 'dv', 'dg', 'db'),
                                   [output] + [x.grad for x in leaves],
                                   [reference] + [x.grad for x in reference_leaves]):
            assert torch.isfinite(got).all(), name
            assert_close('metadata::' + name, want.float(), got.float(), ratio=2e-3)

    first, second = inputs(1), inputs(2)
    first_output, second_output = forward(*first), forward(*second)
    assert len(created) == 2 and created[0] != created[1]
    second_output.sum().backward()
    first_output.sum().backward()
    assert len(created) == 2
    assert not hasattr(context, '_cp_graph_metadata')
    check(first, first_output)
    check(second, second_output)
    del first_output, second_output

    callables, live_inputs, graph_buffers = [], [], []
    for seed in (3, 4):
        capture = inputs(seed)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                for x in capture:
                    x.grad = None
                forward(*capture).sum().backward()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        for x in capture:
            x.grad = None
        before = len(created)
        callables.append(torch.npu.make_graphed_callables(
            forward, tuple(capture), num_warmup_iters=0, allow_unused_input=True,
        ))
        assert len(created) == before + 1
        graph_buffers.append(created[-1])
        live_inputs.append([x.detach().clone().requires_grad_() for x in capture])
    assert graph_buffers[0] != graph_buffers[1]
    before_replay = len(created)
    replay_cases = [(offsets, offsets) for offsets in ([0, 32, 64, 128, 128], [0, 64, 65, 128, 128], [0, 1, 64, 65, 128])]
    replay_cases += [(offsets, [0, 128] if layout == 'zigzag' else [0, 64, 128]) for offsets in (
        [0, 64, 128, 128, 128], [0, 0, 64, 128, 128], [0, 64, 64, 128, 128],
        [0, 0, 64, 64, 128], [0, 64, 128, 128, 128],
    )]
    for offsets, sequence_offsets in replay_cases:
        cu.copy_(torch.tensor(offsets, device=device, dtype=cu_dtype))
        reference_context = build_cp_context(
            cu_seqlens=torch.tensor(sequence_offsets, device=device),
            group=dist.group.WORLD,
            layout=layout,
        )
        pre = reference_context.pre_num_ranks_by_part if layout == 'zigzag' else [reference_context.pre_num_ranks]
        post = reference_context.post_num_ranks_by_part if layout == 'zigzag' else [reference_context.post_num_ranks]
        context.pre_num_ranks_dev.copy_(torch.tensor(pre, device=device, dtype=torch.int32))
        context.post_num_ranks_dev.copy_(torch.tensor(post, device=device, dtype=torch.int32))
        for index in (1, 0):
            leaves = live_inputs[index]
            for x in leaves:
                x.grad = None
            output = callables[index](*leaves)
            output.sum().backward()
            torch.npu.synchronize()
            check(leaves, output, sequence_offsets)
        assert len(created) == before_replay
        assert not hasattr(context, '_cp_graph_metadata')
