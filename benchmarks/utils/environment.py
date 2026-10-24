# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Device synchronization and environment metadata for benchmark runs."""

import platform
import socket
import subprocess

import torch


def synchronize_device(device: str | None = None) -> None:
    """Synchronize the active accelerator before/after timed runs."""
    if device is None:
        from fla.utils import device_name as device
    dev = device
    dev_mod = getattr(torch, dev, None)
    if dev_mod is not None and hasattr(dev_mod, 'synchronize'):
        dev_mod.synchronize()


def get_git_label() -> str:
    """Return 'branch[short_sha]', e.g. 'main[abc1234]'."""
    try:
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return f"{branch}[{sha}]"
    except Exception:
        return 'unknown'


def get_machine_info() -> dict:
    from fla.utils import device_platform

    info = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda or 'N/A',
        'device_platform': device_platform,
        'git_label': get_git_label(),
    }
    try:
        import triton
        info['triton_version'] = triton.__version__
    except Exception:
        info['triton_version'] = 'N/A'

    if device_platform == 'npu' and hasattr(torch, 'npu') and torch.npu.is_available():
        props = torch.npu.get_device_properties(0)
        info['gpu_name'] = torch.npu.get_device_name(0)
        info['gpu_count'] = torch.npu.device_count()
        info['gpu_memory_gb'] = round(props.total_memory / (1024**3), 1)
        info['cann_version'] = getattr(torch.version, 'cann', None) or 'N/A'
    elif torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_memory_gb'] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
        )
    else:
        info['gpu_name'] = 'N/A'
        info['gpu_count'] = 0
        info['gpu_memory_gb'] = 0
    return info
