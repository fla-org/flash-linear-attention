# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest.mock import Mock

import pytest
import torch

from fla.ops.cp import comm


@pytest.mark.parametrize(
    ('world_size', 'rank', 'previous', 'following'),
    [(1, 0, None, None), (3, 0, None, 1), (3, 1, 0, 2), (3, 2, 1, None)],
)
@pytest.mark.parametrize(
    ('op', 'kwargs', 'receive_previous'),
    [
        pytest.param(comm.send_recv_fwd, {'recv_from_prev': True}, True, id='forward-previous'),
        pytest.param(comm.send_recv_fwd, {'recv_from_prev': False}, False, id='forward-next'),
        pytest.param(comm.send_recv_bwd, {'recv_from_next': True}, False, id='backward-next'),
        pytest.param(comm.send_recv_bwd, {'recv_from_next': False}, True, id='backward-previous'),
        pytest.param(comm.conv_cp_send_recv_fwd, {}, True, id='conv-forward'),
        pytest.param(comm.conv_cp_send_recv_bwd, {}, False, id='conv-backward'),
    ],
)
def test_send_recv(monkeypatch, world_size, rank, previous, following, op, kwargs, receive_previous):
    group = object()
    gathered = torch.arange(world_size * 6, dtype=torch.float32).view(world_size, 2, 3)
    original = gathered.clone()
    send = gathered[rank].clone()
    gather = Mock(return_value=(gathered, None))
    monkeypatch.setattr(comm.dist, 'get_rank', Mock(return_value=rank))
    monkeypatch.setattr(comm.dist, 'get_world_size', Mock(return_value=world_size))
    monkeypatch.setattr(comm, 'all_gather_into_tensor', gather)

    received = op(send, group=group, **kwargs)
    source = previous if receive_previous else following
    expected = torch.zeros_like(send) if source is None else gathered[source]
    assert torch.equal(received, expected)
    gather.assert_called_once_with(inp=send, group=group)
    comm.dist.get_rank.assert_called_once_with(group=group)
    comm.dist.get_world_size.assert_called_once_with(group=group)
    received.fill_(-1)
    assert torch.equal(gathered, original)
    assert torch.equal(send, original[rank])
