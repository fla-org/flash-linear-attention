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


@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize(
    ('forward', 'flag', 'sources'),
    [
        pytest.param(True, True, [None, 0, 1], id='forward-previous'),
        pytest.param(True, False, [1, 2, None], id='forward-next'),
        pytest.param(False, True, [1, 2, None], id='backward-next'),
        pytest.param(False, False, [None, 0, 1], id='backward-previous'),
    ],
)
def test_send_recv(monkeypatch, rank, forward, flag, sources):
    group = object()
    gathered = torch.arange(18, dtype=torch.float32).view(3, 2, 3)
    send = gathered[rank].clone()
    gather = Mock(return_value=(gathered, None))
    monkeypatch.setattr(comm.dist, 'get_rank', Mock(return_value=rank))
    monkeypatch.setattr(comm.dist, 'get_world_size', Mock(return_value=3))
    monkeypatch.setattr(comm, 'all_gather_into_tensor', gather)

    if forward:
        received = comm.send_recv_fwd(send_tensor=send, group=group, recv_from_prev=flag)
    else:
        received = comm.send_recv_bwd(send_tensor=send, group=group, recv_from_next=flag)

    source = sources[rank]
    expected = torch.zeros_like(send) if source is None else gathered[source].clone()
    assert torch.equal(received, expected)
    gather.assert_called_once_with(inp=send, group=group)
    comm.dist.get_rank.assert_called_once_with(group=group)
    comm.dist.get_world_size.assert_called_once_with(group=group)
    received.fill_(-1)
    assert torch.equal(gathered, torch.arange(18, dtype=torch.float32).view(3, 2, 3))
