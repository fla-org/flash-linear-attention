# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Register the packaged OPP before torch_npu can initialize CANN."""

import os
from pathlib import Path

if os.environ.get('FLA_ASCENDC', '0') != '0':
    vendor = Path(__file__).resolve().parent / 'fla/utils/ascendc/_native/opp/vendors/fla_ascendc'
    if vendor.is_dir():
        paths = os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(os.pathsep)
        os.environ['ASCEND_CUSTOM_OPP_PATH'] = os.pathsep.join(
            [str(vendor), *[path for path in paths if path and path != str(vendor)]]
        )
