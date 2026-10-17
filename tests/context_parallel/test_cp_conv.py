# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare context-parallel convolution outputs and gradients with a full-sequence reference."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fla.modules.causal_conv1d import causal_conv1d
from fla.modules.causal_conv1d.backends import cuda
from fla.ops.cp import FLACPContext, build_cp_context
from fla.utils import IS_NVIDIA, assert_close, device, device_torch_lib

if TYPE_CHECKING:
    from pathlib import Path


def _run_cp_conv_worker(
    rank: int,
    world_size: int,
    init_method: str,
    lengths: list[int],
    D: int,
    W: int,
    activation: str | None,
    has_bias: bool,
    has_residual: bool,
    non_contiguous: bool,
    dtype: torch.dtype,
    layout: str,
    backend: str,
) -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    device_torch_lib.set_device(rank)
    dist.init_process_group(
        backend='nccl',
        init_method=init_method,
        timeout=timedelta(seconds=180),
        world_size=world_size,
        rank=rank,
    )
    try:
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
        part_len = T // (2 * world_size) if layout == 'zigzag' else T // world_size

        def slice_local(tensor: torch.Tensor) -> torch.Tensor:
            front = tensor[:, rank * part_len:(rank + 1) * part_len]
            if layout == 'contiguous':
                return front
            back_start = (2 * world_size - 1 - rank) * part_len
            return torch.cat((front, tensor[:, back_start:back_start + part_len]), dim=1)

        def local_input(tensor: torch.Tensor) -> torch.Tensor:
            local = slice_local(tensor).clone()
            if non_contiguous:
                local = torch.stack((local, torch.zeros_like(local)), dim=-1)[..., 0]
            return local.requires_grad_()

        x_local = local_input(x)
        weight_local = weight.clone().requires_grad_()
        bias_local = bias.clone().requires_grad_() if bias is not None else None
        residual_local = local_input(residual) if residual is not None else None
        dy_local = slice_local(dy).clone()
        if non_contiguous:
            dy_local = torch.stack((dy_local, torch.zeros_like(dy_local)), dim=-1)[..., 0]
        y, final_state = causal_conv1d(
            x=x_local,
            weight=weight_local,
            bias=bias_local,
            residual=residual_local,
            activation=activation,
            backend=backend,
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
                gathered = [torch.empty_like(value, memory_format=torch.contiguous_format) for _ in range(world_size)]
                dist.all_gather(gathered, value.contiguous())
                if layout == 'zigzag':
                    gathered = [part[:, :part_len] for part in gathered] + [
                        part[:, part_len:] for part in reversed(gathered)
                    ]
                value = torch.cat(gathered, dim=1)
            results[name] = value

        if rank == 0:
            for name, value in results.items():
                assert torch.isfinite(value).all(), name
                assert_close(name, reference[name].float(), value.float(), ratio=0.001 if dtype == torch.float32 else 0.005)
    finally:
        dist.destroy_process_group()


def _run_cp_conv(
    tmp_path: Path,
    world_size: int,
    lengths: list[int],
    D: int = 128,
    W: int = 4,
    activation: str | None = 'swish',
    has_bias: bool = True,
    has_residual: bool = False,
    non_contiguous: bool = False,
    dtype: torch.dtype = torch.float32,
    layout: str = 'contiguous',
    backend: str = 'triton',
) -> None:
    if device != 'cuda' or device_torch_lib.device_count() < world_size:
        pytest.skip(f'Requires {world_size} CUDA or ROCm GPUs')
    if backend == 'cuda':
        if not IS_NVIDIA:
            pytest.skip('CUDA backend requires an NVIDIA GPU')
        pytest.importorskip('causal_conv1d')
    mp.spawn(
        fn=_run_cp_conv_worker,
        args=(
            world_size,
            f'file://{tmp_path / "rendezvous"}',
            lengths,
            D,
            W,
            activation,
            has_bias,
            has_residual,
            non_contiguous,
            dtype,
            layout,
            backend,
        ),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize('backend', ['triton', 'cuda'])
@pytest.mark.parametrize(
    ('world_size', 'lengths'),
    [
        pytest.param(2, [300, 400, 324], id='cp2-sequence-cut'),
        pytest.param(2, [512, 512], id='cp2-boundary-aligned'),
        pytest.param(4, [700, 324], id='cp4-sequence-cut'),
        pytest.param(4, [1024], id='cp4-single-sequence'),
        pytest.param(2, [100, 150, 200, 250, 124, 100, 100], id='cp2-short-sequences'),
        # short local tails exercise backward masking when a sequence ends within W-1 tokens of a rank boundary
        pytest.param(2, [513, 511], id='cp2-tail-1'),
        pytest.param(2, [514, 510], id='cp2-tail-2'),
        pytest.param(4, [257, 255, 257, 255], id='cp4-alternating-short-tails'),
        pytest.param(2, [200, 313, 511], id='cp2-packed-short-tail'),
        pytest.param(2, [512, 1, 511], id='cp2-single-token-sequence'),
        pytest.param(4, [257, 253, 257, 257], id='cp4-short-tails'),
        pytest.param(4, [257, 1, 253, 257, 1, 253, 2], id='cp4-single-token-sequences'),
    ],
)
def test_causal_conv1d_cp(tmp_path: Path, world_size: int, lengths: list[int], backend: str):
    _run_cp_conv(tmp_path=tmp_path, world_size=world_size, lengths=lengths, backend=backend)


@pytest.mark.parametrize(
    ('world_size', 'lengths'),
    [
        pytest.param(2, [1024], id='cp2-single-sequence'),
        pytest.param(2, [300, 400, 324], id='cp2-sequence-cut'),
        pytest.param(4, [150, 260, 300, 314], id='cp4-sequence-cut'),
        pytest.param(4, [256, 256, 256, 256], id='cp4-boundary-aligned'),
    ],
)
def test_causal_conv1d_cp_zigzag(tmp_path: Path, world_size: int, lengths: list[int]):
    _run_cp_conv(tmp_path=tmp_path, world_size=world_size, lengths=lengths, layout='zigzag')


@pytest.mark.parametrize('layout', ['contiguous', 'zigzag'])
def test_causal_conv1d_cp_mix(tmp_path: Path, layout: str):
    _run_cp_conv(tmp_path=tmp_path, world_size=2, lengths=[300, 400, 324], layout=layout, backend='mix')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=['fp32', 'fp16', 'bf16'])
@pytest.mark.parametrize(
    ('lengths', 'D', 'W', 'activation', 'has_bias', 'has_residual', 'non_contiguous'),
    [
        pytest.param([126], 65, 2, None, False, False, False, id='dense-width-2'),
        pytest.param([62, 1, 63], 127, 3, 'silu', True, True, True, id='packed-width-3-strided'),
        pytest.param([65, 61], 33, 4, 'swish', False, True, True, id='packed-width-4-strided'),
        pytest.param([2, 4], 7, 4, None, True, True, False, id='minimum-local-length'),
    ],
)
def test_causal_conv1d_cp_cuda(
    tmp_path: Path,
    lengths: list[int],
    D: int,
    W: int,
    activation: str | None,
    has_bias: bool,
    has_residual: bool,
    non_contiguous: bool,
    dtype: torch.dtype,
):
    _run_cp_conv(
        tmp_path=tmp_path,
        world_size=2,
        lengths=lengths,
        D=D,
        W=W,
        activation=activation,
        has_bias=has_bias,
        has_residual=has_residual,
        non_contiguous=non_contiguous,
        dtype=dtype,
        backend='cuda',
    )


@pytest.mark.parametrize(
    ('case', 'error', 'match'),
    [
        pytest.param('layout', NotImplementedError, 'contiguous CP', id='zigzag'),
        pytest.param('width', NotImplementedError, 'kernel widths', id='width-1'),
        pytest.param('shape', ValueError, 'input shape', id='batched'),
        pytest.param('length', ValueError, 'tokens per rank', id='short-rank'),
        pytest.param('context', ValueError, 'kernel width must match', id='mismatched-width'),
        pytest.param('activation', NotImplementedError, 'activation must be', id='relu'),
        pytest.param('package', ImportError, 'causal-conv1d', id='missing-package'),
    ],
)
def test_causal_conv1d_cp_cuda_invalid(monkeypatch: pytest.MonkeyPatch, case: str, error: type[Exception], match: str):
    def unexpected_cuda_call(**kwargs):
        pytest.fail('Invalid arguments must be rejected before CUDA execution')

    monkeypatch.setattr(cuda, 'causal_conv1d_fn_cuda', None if case == 'package' else unexpected_cuda_call)
    x = torch.zeros(2 if case == 'shape' else 1, 2 if case == 'length' else 8, 8)
    weight = torch.zeros(8, 1 if case == 'width' else 4)
    cp_context = FLACPContext(
        cu_seqlens=torch.tensor([0, x.shape[1]], dtype=torch.int32),
        conv1d_kernel_size=3 if case == 'context' else weight.shape[-1],
        layout='zigzag' if case == 'layout' else 'contiguous',
    )
    with pytest.raises(error, match=match):
        causal_conv1d(
            x=x,
            weight=weight,
            activation='relu' if case == 'activation' else None,
            backend='cuda',
            cp_context=cp_context,
        )
