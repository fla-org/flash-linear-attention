# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import pytest
import torch

from fla.ops.generalized_delta_rule.dplr.backends import DPLRTileLangBackend
from fla.utils import device
from tests.cp.ops.dplr import run_cp_dplr_test_worker
from tests.cp.utils import run_distributed

pytestmark = pytest.mark.skipif(device != "cuda", reason="requires CUDA or ROCm GPUs")


@pytest.mark.parametrize(
    ('world_size', 'lengths', 'options'),
    [
        pytest.param(2, [400, 624], {}, id='CP2_SequenceCut'),
        pytest.param(2, [512, 512], {}, id='CP2_BoundaryAligned'),
        pytest.param(4, [700, 324], {}, id='CP4_Complex'),
        pytest.param(4, [1024], {}, id='CP4_SingleSequence'),
        pytest.param(2, [100, 200, 300, 424], {}, id='CP2_ManyShortSequences'),
        pytest.param(2, [400, 624], dict(safe_gate=True), id='CP2_SafeGate'),
    ],
)
def test_chunk(request, world_size, lengths, options):
    run_distributed(
        worker=run_cp_dplr_test_worker,
        world_size=world_size,
        T=1024,
        H=4,
        D=64,
        lengths=lengths,
        dtype=torch.bfloat16,
        test_name=request.node.callspec.id,
        **options,
    )


def test_chunk_tilelang(monkeypatch):
    """CP2 through the TileLang DPLR backend (FLA_TILELANG=1), with route assertion."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")
    if not DPLRTileLangBackend.is_available():
        pytest.skip("TileLang backend not available")

    monkeypatch.setenv("FLA_TILELANG", "1")
    run_distributed(
        worker=run_cp_dplr_test_worker,
        world_size=2,
        test_name="CP2_TileLang",
        T=1024, H=128, D=64,
        lengths=[400, 624],
        dtype=torch.bfloat16,
        op_chunk_size=32,
        assert_tilelang=True,
    )
