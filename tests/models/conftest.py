# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from transformers import AutoModelForCausalLM

from fla.layers import gla
from fla.models import GLAConfig, RWKV7Config
from fla.modules import mlp
from fla.ops.gla.naive import naive_recurrent_gla
from fla.utils import device_name


@pytest.fixture
def reference_gla_ops(reference_residual_ops, monkeypatch):
    """Run GLA API and checkpointing tests with explicit CPU operator references."""
    def recurrent(q, k, v, gk, initial_state=None, output_final_state=False, state_v_first=False, cu_seqlens=None):
        if initial_state is not None and state_v_first:
            initial_state = initial_state.transpose(-1, -2)
        if cu_seqlens is None:
            output, state = naive_recurrent_gla(q, k, v, gk, initial_state, output_final_state)
        else:
            outputs, states = [], []
            boundaries = cu_seqlens.tolist()
            for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                output, state = naive_recurrent_gla(
                    q=q[:, start:end],
                    k=k[:, start:end],
                    v=v[:, start:end],
                    gk=gk[:, start:end],
                    initial_state=None if initial_state is None else initial_state[i:i + 1],
                    output_final_state=output_final_state,
                )
                outputs.append(output)
                if output_final_state:
                    states.append(state)
            output = torch.cat(outputs, dim=1)
            state = torch.cat(states, dim=0) if output_final_state else None
        if state is not None and state_v_first:
            state = state.transpose(-1, -2)
        return output, state

    monkeypatch.setattr(gla, 'RMSNorm', reference_residual_ops)
    monkeypatch.setattr(gla, 'fused_recurrent_gla', recurrent)
    monkeypatch.setitem(gla.ACT2FN, 'swish', F.silu)
    monkeypatch.setattr(mlp, 'swiglu', lambda gate, value: F.silu(gate) * value)


@pytest.fixture
def make_model(request):
    """Construct a small model for API tests."""
    def create(config_class, **kwargs):
        if config_class is RWKV7Config and not torch._C.has_lapack:
            pytest.skip('RWKV7 initialization requires LAPACK.')
        if config_class is GLAConfig and device_name == 'cpu':
            request.getfixturevalue('reference_gla_ops')
        config = config_class(**{
            'hidden_size': 64,
            'num_heads': 2,
            'num_hidden_layers': 1 if config_class is GLAConfig else 0,
            'num_self_decoder_layers': 0,
            'intermediate_size': 128,
            'vocab_size': 64,
            'fuse_norm': False,
            'fuse_cross_entropy': False,
            **({'fuse_swiglu': False} if config_class is GLAConfig else {}),
            **kwargs,
        })
        return AutoModelForCausalLM.from_config(config)

    return create


@pytest.fixture(scope='session')
def cpu_mesh(tmp_path_factory):
    if not dist.is_gloo_available():
        pytest.skip('CPU DTensor initialization requires Gloo.')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    dist.init_process_group(
        backend='gloo',
        init_method='env://' if world_size > 1 else f'file://{tmp_path_factory.mktemp("gloo") / "store"}',
        rank=int(os.environ.get('RANK', '0')),
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        yield init_device_mesh('cpu', (world_size,))
    finally:
        dist.destroy_process_group()


@pytest.fixture(params=[None, Replicate(), Shard(0)], ids=['tensor', 'replicate', 'shard'])
def init_layout(request):
    if request.param is None:
        return None
    return request.getfixturevalue('cpu_mesh'), (request.param,)
