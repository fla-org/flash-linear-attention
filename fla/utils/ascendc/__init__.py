# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Package-local runtime for the optional AscendC backend."""

import ctypes
import json
import os
from functools import cache
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent / '_native'
_HANDLES = []


@cache
def build_metadata():
    path = _ROOT / 'build.json'
    return json.loads(path.read_text()) if path.is_file() else {}


def is_available():
    return (
        build_metadata().get('soc') == 'ascend950'
        and (_ROOT / 'libfla_ascendc.so').is_file()
        and (_ROOT / 'opp/vendors/fla_ascendc/op_api/lib/libcust_opapi.so').is_file()
    )


@cache
def load():
    """Load only the native libraries shipped with this FLA installation."""
    if not is_available():
        raise RuntimeError('FLA was not built with the AscendC cumsum backend')
    vendor = _ROOT / 'opp/vendors/fla_ascendc'
    paths = os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(os.pathsep)
    if str(vendor) not in paths:
        raise RuntimeError('Start a new Python process with FLA_ASCENDC=1 after installing the native wheel')
    opapi = vendor / 'op_api/lib/libcust_opapi.so'
    os.environ['FLA_ASCENDC_OP_API_LIB'] = str(opapi)
    _HANDLES.append(ctypes.CDLL('libopapi.so', mode=os.RTLD_LOCAL | os.RTLD_NOW))
    _HANDLES.append(ctypes.CDLL(str(opapi), mode=os.RTLD_LOCAL | os.RTLD_NOW))
    library = _ROOT / 'libfla_ascendc.so'
    torch.ops.load_library(str(library))
    handle = ctypes.CDLL(str(library))
    handle.fla_ascendc_source_hash.restype = ctypes.c_char_p
    if handle.fla_ascendc_source_hash().decode() != build_metadata()['source_hash']:
        raise RuntimeError('FLA AscendC launcher and build metadata have different source hashes')
    _HANDLES.append(handle)
    return torch.ops.fla_ascendc.npu_chunk_local_cumsum


def chunk_local_cumsum(g, chunk_size, scale):
    import torch_npu

    with torch.npu.device(g.device):
        op = load()
        # flushing before launch preserves ordering with torch_npu's task queue.
        stream = torch_npu._C._npu_getCurrentRawStream(torch.npu.current_device())
        return op(g, None, None, chunk_size, False, scale, True, 0, stream)
