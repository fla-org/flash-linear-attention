# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Approximate value of 1/ln(2), used for log/exp base conversion
# Best FP32 approximation: 1.4426950216 (hex 0x3FB8AA3B)
RCP_LN2 = 1.4426950216
