# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare context-parallel convolution outputs and gradients with a full-sequence reference."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from fla.modules.causal_conv1d import causal_conv1d
from fla.modules.causal_conv1d.backends.cuda import ops
from fla.ops.cp import FLACPContext, build_cp_context
from fla.utils import IS_NVIDIA, assert_close, device, device_torch_lib
from tests.cp.utils import gather_global, shard_tensor


def _run_causal_conv1d(
    rank: int,
    world_size: int,
    lengths: list[int],
    D: int,
    W: int,
    activation: str | None,
    has_bias: bool,
    has_residual: bool,
    non_contiguous: bool,
    dtype: torch.dtype,
    layout: str,
    conv_backend: str,
) -> None:
    torch.manual_seed(42)
    T = sum(lengths)
    assert T % world_size == 0
    assert layout != 'zigzag' or T % (2 * world_size) == 0
    x = torch.randn(1, T, D, device=device, dtype=dtype) * 100
    dy = torch.randn(1, T, D, device=device, dtype=dtype)
    weight = torch.randn(D, W, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype) if has_bias else None
    residual = torch.randn_like(x) if has_residual else None
    cu_seqlens_cpu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    cu_seqlens = cu_seqlens_cpu.to(device)

    reference = {}
    if rank == 0:
        x_ref = x.clone().requires_grad_()
        weight_ref = weight.clone().requires_grad_()
        bias_ref = bias.clone().requires_grad_() if bias is not None else None
        residual_ref = residual.clone().requires_grad_() if residual is not None else None
        y_ref, _ = causal_conv1d(
            x=x_ref,
            weight=weight_ref,
            bias=bias_ref,
            residual=residual_ref,
            activation=activation,
            backend='triton',
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
        y_ref.backward(dy)
        reference = {'y': y_ref.detach(), 'dx': x_ref.grad, 'dw': weight_ref.grad}
        if bias_ref is not None:
            reference['db'] = bias_ref.grad
        if residual_ref is not None:
            reference['dr'] = residual_ref.grad

    cp_context = build_cp_context(
        cu_seqlens=cu_seqlens,
        group=dist.group.WORLD,
        conv1d_kernel_size=W,
        cu_seqlens_cpu=cu_seqlens_cpu,
        layout=layout,
    )

    def local_input(tensor: torch.Tensor) -> torch.Tensor:
        local = shard_tensor(tensor, rank=rank, world_size=world_size, layout=layout).clone()
        if non_contiguous:
            local = torch.stack((local, torch.zeros_like(local)), dim=-1)[..., 0]
        return local.requires_grad_()

    x_local = local_input(x)
    weight_local = weight.clone().requires_grad_()
    bias_local = bias.clone().requires_grad_() if bias is not None else None
    residual_local = local_input(residual) if residual is not None else None
    dy_local = shard_tensor(dy, rank=rank, world_size=world_size, layout=layout).clone()
    if non_contiguous:
        dy_local = torch.stack((dy_local, torch.zeros_like(dy_local)), dim=-1)[..., 0]
    y, final_state = causal_conv1d(
        x=x_local,
        weight=weight_local,
        bias=bias_local,
        residual=residual_local,
        activation=activation,
        backend=conv_backend,
        cp_context=cp_context,
    )
    assert final_state is None
    y.backward(dy_local)
    results = {'y': y.detach(), 'dx': x_local.grad, 'dw': weight_local.grad}
    if bias_local is not None:
        results['db'] = bias_local.grad
    if residual_local is not None:
        results['dr'] = residual_local.grad

    for name, value in results.items():
        if name in ('dw', 'db'):
            value = value.float().contiguous()
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        else:
            value = gather_global(value, world_size=world_size, layout=layout)
        results[name] = value

    if rank == 0:
        for name, value in results.items():
            assert torch.isfinite(value).all(), name
            assert_close(name, reference[name].float(), value.float(), ratio=0.001 if dtype == torch.float32 else 0.005)


@pytest.mark.parametrize(
    ('world_size', 'lengths', 'backend', 'options'),
    [
        pytest.param(world_size, lengths, backend, {'layout': layout}, id=f'{name}-{backend}')
        for world_size, lengths, layout, backends, name in [
            (2, [300, 400, 324], 'contiguous', ('triton', 'cuda', 'mix'), 'cp2-sequence-cut'),
            (2, [512, 512], 'contiguous', ('triton', 'cuda'), 'cp2-boundary-aligned'),
            (4, [700, 324], 'contiguous', ('triton', 'cuda'), 'cp4-sequence-cut'),
            (4, [1024], 'contiguous', ('triton', 'cuda'), 'cp4-single-sequence'),
            (2, [100, 150, 200, 250, 124, 100, 100], 'contiguous', ('triton', 'cuda'), 'cp2-short-sequences'),
            # short local tails exercise backward masking when a sequence ends within W-1 tokens of a rank boundary
            (2, [513, 511], 'contiguous', ('triton', 'cuda'), 'cp2-tail-1'),
            (2, [514, 510], 'contiguous', ('triton', 'cuda'), 'cp2-tail-2'),
            (4, [257, 255, 257, 255], 'contiguous', ('triton', 'cuda'), 'cp4-alternating-short-tails'),
            (2, [200, 313, 511], 'contiguous', ('triton', 'cuda'), 'cp2-packed-short-tail'),
            (2, [512, 1, 511], 'contiguous', ('triton', 'cuda'), 'cp2-single-token-sequence'),
            (4, [257, 253, 257, 257], 'contiguous', ('triton', 'cuda'), 'cp4-short-tails'),
            (4, [257, 1, 253, 257, 1, 253, 2], 'contiguous', ('triton', 'cuda'), 'cp4-single-token-sequences'),
            (2, [1024], 'zigzag', ('triton',), 'cp2-zigzag-single-sequence'),
            (2, [300, 400, 324], 'zigzag', ('triton', 'mix'), 'cp2-zigzag-sequence-cut'),
            (4, [150, 260, 300, 314], 'zigzag', ('triton',), 'cp4-zigzag-sequence-cut'),
            (4, [256, 256, 256, 256], 'zigzag', ('triton',), 'cp4-zigzag-boundary-aligned'),
        ]
        for backend in backends
    ] + [
        pytest.param(
            2,
            lengths,
            'cuda',
            {**options, 'dtype': dtype},
            id=f'{name}-{dtype_name}',
        )
        for lengths, options, name in [
            ([126], dict(D=65, W=2, activation=None, has_bias=False), 'dense-width-2'),
            ([62, 1, 63], dict(D=127, W=3, activation='silu', has_residual=True, non_contiguous=True), 'packed-width-3-strided'),
            ([65, 61], dict(D=33, has_bias=False, has_residual=True, non_contiguous=True), 'packed-width-4-strided'),
            ([2, 4], dict(D=7, activation=None, has_residual=True), 'minimum-local-length'),
        ]
        for dtype, dtype_name in [(torch.float32, 'fp32'), (torch.float16, 'fp16'), (torch.bfloat16, 'bf16')]
    ],
)
def test_causal_conv1d(run_distributed, world_size: int, lengths: list[int], backend: str, options: dict) -> None:
    if device != 'cuda' or device_torch_lib.device_count() < world_size:
        pytest.skip(f'Requires {world_size} CUDA or ROCm GPUs')
    if backend == 'cuda':
        if not IS_NVIDIA:
            pytest.skip('CUDA backend requires an NVIDIA GPU')
        pytest.importorskip('causal_conv1d')
    options = dict(
        D=128,
        W=4,
        activation='swish',
        has_bias=True,
        has_residual=False,
        non_contiguous=False,
        dtype=torch.float32,
        layout='contiguous',
    ) | options
    run_distributed(
        worker=_run_causal_conv1d,
        world_size=world_size,
        lengths=lengths,
        conv_backend=backend,
        **options,
    )


@pytest.mark.parametrize(
    ('inputs', 'error', 'match'),
    [
        pytest.param({'layout': 'zigzag'}, NotImplementedError, 'contiguous CP', id='zigzag'),
        pytest.param({'width': 1}, NotImplementedError, 'kernel widths', id='width-1'),
        pytest.param({'shape': (2, 8, 8)}, ValueError, 'input shape', id='batched'),
        pytest.param({'shape': (1, 2, 8)}, ValueError, 'tokens per rank', id='short-rank'),
        pytest.param({'context_width': 3}, ValueError, 'kernel width must match', id='mismatched-width'),
        pytest.param({'activation': 'relu'}, NotImplementedError, 'activation must be', id='relu'),
    ],
)
def test_causal_conv1d_cuda_invalid(monkeypatch, inputs, error, match):
    def unexpected_cuda_call(**kwargs) -> None:
        pytest.fail('Invalid arguments must be rejected before CUDA execution')

    monkeypatch.setattr(ops, 'causal_conv1d_fn_cuda', unexpected_cuda_call)
    x = torch.zeros(inputs.get('shape', (1, 8, 8)))
    weight = torch.zeros(8, inputs.get('width', 4))
    cp_context = FLACPContext(
        cu_seqlens=torch.tensor([0, x.shape[1]], dtype=torch.int32),
        conv1d_kernel_size=inputs.get('context_width', weight.shape[-1]),
        layout=inputs.get('layout', 'contiguous'),
    )
    with pytest.raises(error, match=match):
        causal_conv1d(
            x=x,
            weight=weight,
            activation=inputs.get('activation'),
            backend='cuda',
            cp_context=cp_context,
        )


def test_causal_conv1d_cuda_missing_package(monkeypatch):
    monkeypatch.setattr(ops, 'causal_conv1d_fn_cuda', None)
    with pytest.raises(ImportError, match='causal-conv1d'):
        causal_conv1d(
            x=torch.zeros(1, 8, 8),
            weight=torch.zeros(8, 4),
            backend='cuda',
            cp_context=FLACPContext(cu_seqlens=torch.tensor([0, 8], dtype=torch.int32), conv1d_kernel_size=4),
        )
