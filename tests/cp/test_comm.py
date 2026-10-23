# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.distributed as dist

from fla.ops.cp import comm
from fla.utils import device


def _run_send_recv(rank: int, world_size: int):
    group = dist.group.WORLD
    tensors = torch.arange(1, world_size * 6 + 1, dtype=torch.float32, device=device).view(world_size, 2, 3)
    send = tensors[rank].clone()
    previous = tensors[rank - 1] if rank > 0 else torch.zeros_like(send)
    following = tensors[rank + 1] if rank + 1 < world_size else torch.zeros_like(send)

    for op, kwargs, expected in (
        (comm.send_recv_fwd, {}, previous),
        (comm.send_recv_fwd, {'recv_from_prev': False}, following),
        (comm.send_recv_bwd, {}, following),
        (comm.send_recv_bwd, {'recv_from_next': False}, previous),
        (comm.conv_cp_send_recv_fwd, {}, previous),
        (comm.conv_cp_send_recv_bwd, {}, following),
    ):
        received = op(send, group=group, **kwargs)
        torch.testing.assert_close(received, expected, rtol=0, atol=0)
        # a neighbor's receive must not retain the full gathered allocation
        assert received.untyped_storage().nbytes() == send.numel() * send.element_size()
        received.fill_(-1)
        torch.testing.assert_close(send, tensors[rank], rtol=0, atol=0)


@pytest.mark.parametrize('world_size', [1, 3], ids=['single-rank', 'three-ranks'])
def test_send_recv(run_distributed, world_size: int):
    run_distributed(worker=_run_send_recv, world_size=world_size)
