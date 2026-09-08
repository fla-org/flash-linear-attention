# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from datetime import timedelta
from itertools import product

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from fla.modules import FusedLinearCrossEntropyLoss
from fla.modules.l2warp import l2_warp
from fla.utils import IS_NVIDIA, assert_close, device, device_torch_lib


def _check_parallel_case(rank, world_size, local_vocab, dtype, option, reduction):
    torch.manual_seed(42)
    N, H, V = 63, 64, world_size * local_vocab
    x = torch.randn(1, N, H, device=device, dtype=dtype).requires_grad_()
    weight = (torch.randn(V, H, device=device) / H ** 0.5).to(dtype).requires_grad_()
    with_bias = option in ('combined', 'l2', 'l2_tie')
    bias = torch.randn(V, device=device, dtype=dtype).requires_grad_() if with_bias else None
    target = torch.randint(V, (1, N), device=device)
    target[:, ::7] = -100
    target[0, :4] = torch.tensor([0, local_vocab - 1, min(local_vocab, V - 1), V - 1], device=device)
    if option == 'ignored':
        target.fill_(-100)
    if option == 'confident':
        with torch.no_grad():
            x.zero_()
            x[..., 0] = 1
            weight.zero_()
            weight[-1, 0] = 12
            target.fill_(V - 1)
    if option == 'l2_tie':
        with torch.no_grad():
            weight.zero_()
            bias.fill_(1)

    kwargs = dict(reduction=reduction)
    if option == 'combined':
        kwargs.update(label_smoothing=0.1, logit_scale=0.3, logit_softcapping=3.0)
    if option in ('l2', 'l2_tie'):
        kwargs.update(use_l2warp=True, l2_penalty_factor=0.7)
    raw = F.linear(x, weight, bias)
    logits = raw.float() * kwargs.get('logit_scale', 1)
    if 'logit_softcapping' in kwargs:
        logits = 3 * torch.tanh(logits / 3)
    if option == 'ignored':
        ref = logits.sum() * 0
    else:
        ref = F.cross_entropy(
            logits.flatten(0, 1),
            target.flatten(),
            label_smoothing=kwargs.get('label_smoothing', 0),
            reduction=reduction,
        )
    if kwargs.get('use_l2warp'):
        ref = l2_warp(ref, raw.float(), kwargs['l2_penalty_factor'])
    inputs = (x, weight, bias) if with_bias else (x, weight)
    ref_grads = torch.autograd.grad(ref * 2, inputs)

    start, end = rank * local_vocab, (rank + 1) * local_vocab
    local_x = x.detach().clone().requires_grad_()
    local_weight = weight[start:end].detach().clone().requires_grad_()
    local_bias = bias[start:end].detach().clone().requires_grad_() if with_bias else None
    local_inputs = (local_x, local_weight, local_bias) if with_bias else (local_x, local_weight)
    tri = FusedLinearCrossEntropyLoss(process_group=dist.group.WORLD, **kwargs)(
        local_x, target, local_weight, local_bias,
    )
    tri_grads = torch.autograd.grad(tri * 2, local_inputs)
    expected_grads = (ref_grads[0],) + tuple(grad[start:end] for grad in ref_grads[1:])
    for name, expected, actual in zip(('loss', 'dx', 'dw', 'db'), (ref, *expected_grads), (tri, *tri_grads)):
        assert torch.isfinite(actual).all(), name
        assert_close(name, expected.float(), actual.float(), ratio=1e-2, err_atol=0 if option == 'confident' else 1e-6)


def _parallel_worker(rank, world_size, init_file):
    device_torch_lib.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        backend='nccl',
        init_method=f'file://{init_file}',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5),
    )
    try:
        for local_vocab, dtype, option, reduction in product(
            (67, 4103, 65539),
            (torch.bfloat16, torch.float32),
            ('plain', 'combined', 'ignored', 'l2', 'l2_tie'),
            ('mean', 'sum'),
        ):
            try:
                _check_parallel_case(rank, world_size, local_vocab, dtype, option, reduction)
            except AssertionError as error:
                raise AssertionError(f'{local_vocab=}, {dtype=}, {option=}, {reduction=}: {error}') from error
        _check_parallel_case(rank, world_size, 67, torch.bfloat16, 'confident', 'mean')
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('world_size', [1, 2, 4])
@pytest.mark.skipif(not IS_NVIDIA, reason="Vocabulary-parallel fused linear CE requires the default GPU backend")
def test_fused_linear_cross_entropy_parallel(world_size, tmp_path):
    if device_torch_lib.device_count() < world_size:
        pytest.skip(f"Requires {world_size} devices")
    mp.spawn(
        fn=_parallel_worker,
        args=(world_size, str(tmp_path / 'init')),
        nprocs=world_size,
        join=True,
    )


def test_fused_linear_cross_entropy_parallel_ascend_rejection(monkeypatch):
    from fla.modules.backends.triton_ascend import TritonAscendBackend

    monkeypatch.setattr(dist, 'get_world_size', lambda group: 2)
    with pytest.raises(NotImplementedError, match="Vocabulary-parallel"):
        TritonAscendBackend().fused_linear_cross_entropy_fwd(x=None, target=None, weight=None, process_group=object())
