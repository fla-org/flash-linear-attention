# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Host eligibility boundaries; real launches are tested in test_swila_dispatch."""

from dataclasses import dataclass, replace

import pytest
import torch

from fla.ops.swila import dispatch, gated_bwd, gated_split_bwd, temporal_bwd


class AllocationReached(Exception):
    pass


class ScheduleReached(Exception):
    pass


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device("cuda", 0)
    contiguous: bool = True

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def is_contiguous(self):
        return self.contiguous

    def new_zeros(self, *shape, **kwargs):
        raise AllocationReached(shape)


def metadata_inputs(prefix=(1, 65, 1)):
    activation = TensorMetadata((*prefix, 64))
    router = TensorMetadata((*prefix, 64, 4))
    return {
        "q": activation, "k": activation, "v": activation,
        "beta": router, "pi_q": router, "pi_k": router,
        "log_decay": replace(activation, dtype=torch.float32), "cu_seqlens": None,
    }


def temporal_metadata(inputs):
    return {**inputs, "gate_kv": inputs["v"], "gate_q": inputs["v"]}


def mock_h100(monkeypatch, capability=(9, 0), name="NVIDIA H100 80GB HBM3"):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: name)


def gated_arguments(inputs):
    saved = replace(inputs["beta"], dtype=torch.float32)
    return {**inputs, "r": saved, "u": saved, "initial_state": None, "do": inputs["v"], "dht": None}


def temporal_arguments(inputs):
    saved = replace(inputs["beta"], dtype=torch.float32)
    return {
        **temporal_metadata(inputs), "r": saved, "u": saved, "pi_bar": saved, "rho": saved, "qS": saved,
        "initial_state": None, "final_state": None, "r_prev_init": None, "rho_prev_init": None,
        "do": inputs["v"], "dht": None, "scale": 1.0,
    }


def probe_gated_schedule(monkeypatch, arguments):
    def scheduled(*args, **kwargs):
        raise ScheduleReached(args[-1])

    with monkeypatch.context() as patch:
        patch.setattr(gated_bwd, "_fused_recurrent_gated_swila_bwd_scheduled", scheduled)
        try:
            gated_bwd.fused_recurrent_gated_swila_bwd(**arguments)
        except ScheduleReached as result:
            return result.args[0]
        except AllocationReached:
            return None
    raise AssertionError("host routing did not reach a schedule or its first allocation")


def probe_temporal_schedule(monkeypatch, arguments):
    fixed_tiles = []
    original = temporal_bwd.prune_temporal_value_configs

    def record_pruning(configs, named_args):
        fixed_tiles.extend((config.kwargs["BV"], config.num_warps, config.num_stages) for config in configs)
        return original(configs, named_args)

    with monkeypatch.context() as patch:
        patch.setattr(temporal_bwd, "prune_temporal_value_configs", record_pruning)
        with pytest.raises(AllocationReached) as result:
            temporal_bwd.fused_recurrent_swila_temporal_bwd(**arguments)
    return fixed_tiles, result.value.args[0][0]


def test_swila_gated_metadata_boundary():
    inputs = metadata_inputs()
    eligible = dispatch._use_dense_d64e4_key_bwd_fastpath
    assert eligible(**inputs)
    assert not eligible(**{**inputs, "cu_seqlens": (0, 65)})
    for name in ("q", "k", "v", "beta", "pi_q", "pi_k"):
        for changes in ({"dtype": torch.float32}, {"contiguous": False}, {"device": torch.device("cpu")}):
            assert not eligible(**{**inputs, name: replace(inputs[name], **changes)}), (name, changes)
    for changes in ({"dtype": torch.bfloat16}, {"contiguous": False}, {"device": torch.device("cpu")}):
        assert not eligible(**{**inputs, "log_decay": replace(inputs["log_decay"], **changes)})
    assert not eligible(**{**inputs, "q": replace(inputs["q"], shape=(1, 65, 1, 32))})
    assert not eligible(**{**inputs, "beta": replace(inputs["beta"], shape=(1, 65, 1, 64, 8))})


def test_swila_temporal_metadata_boundary():
    inputs = temporal_metadata(metadata_inputs())
    eligible = dispatch._use_temporal_dense_d64e4_key_bwd_fastpath
    assert eligible(**inputs)
    assert not eligible(**{**inputs, "log_decay": None})
    assert not eligible(**{**inputs, "cu_seqlens": (0, 65)})
    for name in ("gate_kv", "gate_q"):
        assert not eligible(**{**inputs, name: None})
        for changes in ({"dtype": torch.float32}, {"contiguous": False}, {"shape": (1, 65, 1, 32)}):
            assert not eligible(**{**inputs, name: replace(inputs[name], **changes)}), (name, changes)


def test_swila_gated_automatic_metadata_boundary(monkeypatch):
    mock_h100(monkeypatch)
    inputs = metadata_inputs((1, 4096, 16))
    arguments = gated_arguments(inputs)
    assert probe_gated_schedule(monkeypatch, arguments) == ((8, 1, 2), (8, 1, 1))
    assert probe_gated_schedule(monkeypatch, {**arguments, "dense_split_schedule": None}) is None
    for prefix in ((2, 4096, 16), (1, 4095, 16), (1, 4096, 8)):
        assert probe_gated_schedule(monkeypatch, gated_arguments(metadata_inputs(prefix))) is None
    for name, changes in (
        ("r", {"dtype": torch.bfloat16}), ("u", {"contiguous": False}),
        ("do", {"dtype": torch.float32}), ("log_decay", {"device": torch.device("cuda", 1)}),
    ):
        assert probe_gated_schedule(monkeypatch, {**arguments, name: replace(arguments[name], **changes)}) is None
    assert probe_gated_schedule(monkeypatch, {**arguments, "cu_seqlens": (0, 4096)}) is None
    for capability, name in (((8, 0), "NVIDIA A100"), ((9, 0), "NVIDIA H200"), ((9, 1), "NVIDIA H100")):
        mock_h100(monkeypatch, capability, name)
        assert probe_gated_schedule(monkeypatch, arguments) is None
    with pytest.raises(ValueError, match="dense_split_schedule must be"):
        gated_bwd.fused_recurrent_gated_swila_bwd(**arguments, dense_split_schedule="unknown")


def test_swila_temporal_automatic_metadata_boundary(monkeypatch):
    mock_h100(monkeypatch)
    arguments = temporal_arguments(metadata_inputs((1, 4096, 16)))
    assert probe_temporal_schedule(monkeypatch, arguments) == ([(8, 1, 2), (16, 1, 1)], 4)
    assert probe_temporal_schedule(monkeypatch, {**arguments, "split_schedule": None}) == ([], 4)
    assert probe_temporal_schedule(monkeypatch, {**arguments, "debug_intermediates": True}) == ([], 4)
    for prefix in ((2, 4096, 16), (1, 4095, 16), (1, 4096, 8)):
        assert probe_temporal_schedule(monkeypatch, temporal_arguments(metadata_inputs(prefix))) == ([], 4)
    for name in ("r_prev_init", "rho_prev_init", "d_r_prev_final", "d_rho_prev_final", "dht"):
        assert probe_temporal_schedule(monkeypatch, {**arguments, name: arguments["r"]}) == ([], 8)
    assert probe_temporal_schedule(monkeypatch, {**arguments, "log_decay": None}) == ([], 8)
    assert probe_temporal_schedule(monkeypatch, {**arguments, "cu_seqlens": (0, 4096)}) == ([], 8)
    for capability, name in (((8, 0), "NVIDIA A100"), ((9, 0), "NVIDIA H200"), ((9, 1), "NVIDIA H100")):
        mock_h100(monkeypatch, capability, name)
        assert probe_temporal_schedule(monkeypatch, arguments) == ([], 4)
    for invalid in ("unknown", ((8, 1, 2),), ((8, True, 2), (8, 1, 2), (16, 1, 1))):
        with pytest.raises(ValueError, match="split_schedule"):
            temporal_bwd.fused_recurrent_swila_temporal_bwd(**arguments, split_schedule=invalid)


def test_swila_deferred_forward_and_explicit_schedule_boundary(monkeypatch):
    mock_h100(monkeypatch)
    inputs = metadata_inputs((2, 4096, 8))
    assert dispatch._use_dense_d64e4_key_deferred_readout(**inputs)
    assert not dispatch._use_dense_d64e4_key_deferred_readout(**metadata_inputs((2, 4095, 8)))
    assert not dispatch._use_dense_d64e4_key_deferred_readout(**{**inputs, "cu_seqlens": (0, 8192)})
    schedule = ((8, 1, 2), (8, 1, 1))
    assert gated_split_bwd._validate_dense_split_schedule(schedule=schedule, **inputs) == schedule
    for invalid in (((8, 1, 2),), ((8, True, 2), (8, 1, 1)), ((32, 1, 2), (32, 1, 1))):
        with pytest.raises(ValueError, match="predeclared"):
            gated_split_bwd._validate_dense_split_schedule(schedule=invalid, **inputs)
    with pytest.raises(ValueError, match="SM90"):
        gated_split_bwd._validate_dense_split_schedule(schedule=schedule, **{**inputs, "cu_seqlens": (0, 8192)})
    mock_h100(monkeypatch, (8, 0), "NVIDIA A100")
    assert not dispatch._use_dense_d64e4_key_deferred_readout(**inputs)
    with pytest.raises(ValueError, match="SM90"):
        gated_split_bwd._validate_dense_split_schedule(schedule=schedule, **inputs)
