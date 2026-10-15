# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Run the installed AscendC wheel through FLA's public cumsum entry point."""

import json
import os
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

import fla
import fla.utils.ascendc as native
from fla.ops.utils import chunk_local_cumsum


def main():
    if os.getenv('FLA_ASCENDC', '0') == '0':
        raise RuntimeError('Start Python with FLA_ASCENDC=1')
    if not native.is_available():
        raise RuntimeError('Install the native wheel before running this check')
    torch.manual_seed(42)
    reports = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for chunk in (64, 128):
            x = torch.randn(2, 129, 3, dtype=dtype, device='npu')
            ref = torch.cat([part.cumsum(1) for part in x.cpu().double().split(chunk, dim=1)], dim=1).float()
            with patch.object(native, 'chunk_local_cumsum', wraps=native.chunk_local_cumsum) as launch:
                out = chunk_local_cumsum(x, chunk_size=chunk)
            torch.npu.synchronize()
            assert launch.call_count == 1, 'The public entry did not select AscendC'
            assert out.dtype == torch.float32 and out.shape == x.shape and out.device == x.device
            out = out.cpu()
            assert torch.isfinite(out).all()
            ratio = ((ref - out).square().mean().sqrt() / (ref.square().mean().sqrt() + 1e-8)).item()
            assert ratio < 1e-3, ratio
            reports.append({'dtype': str(dtype), 'chunk': chunk, 'rms_ratio': ratio,
                            'max_abs_error': (ref - out).abs().max().item()})
    print(json.dumps({'fla': fla.__file__, 'native': native.build_metadata(), 'cases': reports}, indent=2))


if __name__ == '__main__':
    main()
