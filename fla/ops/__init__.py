# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.ops.abc import chunk_abc
from fla.ops.attn import parallel_attn
from fla.ops.attnres import fused_attnres
from fla.ops.based import fused_chunk_based, parallel_based
from fla.ops.comba import chunk_comba, fused_recurrent_comba
from fla.ops.cyfa import chunk_cyfa, fused_recurrent_cyfa
from fla.ops.delta_rule import chunk_delta_rule, fused_chunk_delta_rule, fused_recurrent_delta_rule
from fla.ops.forgetting_attn import parallel_forgetting_attn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, chunk_gdn, fused_recurrent_gated_delta_rule, fused_recurrent_gdn
from fla.ops.generalized_delta_rule import (
    chunk_dplr_delta_rule,
    chunk_iplr_delta_rule,
    fused_recurrent_dplr_delta_rule,
    fused_recurrent_iplr_delta_rule,
)
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.ops.gsa import chunk_gsa, fused_recurrent_gsa
from fla.ops.hc import fused_hc_post, fused_hc_pre
from fla.ops.hgrn import fused_recurrent_hgrn
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.lightning_attn import chunk_lightning_attn, fused_recurrent_lightning_attn
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn
from fla.ops.log_linear_attn import chunk_log_linear_attn
from fla.ops.mesa_net import chunk_mesa_net
from fla.ops.momentum_delta_rule import (
    chunk_momentum_delta_rule,
    fused_recurrent_momentum_delta_rule,
)
from fla.ops.nsa import parallel_nsa
from fla.ops.parallax import parallel_parallax
from fla.ops.path_attn import parallel_path_attn
from fla.ops.precond_gated_delta_rule import chunk_precond_gated_delta_rule, fused_recurrent_precond_gated_delta_rule
from fla.ops.precond_kda import chunk_precond_kda, fused_recurrent_precond_kda
from fla.ops.retention import chunk_retention, fused_chunk_retention, fused_recurrent_retention, parallel_retention
from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6
from fla.ops.rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7
from fla.ops.simple_gla import chunk_simple_gla, fused_chunk_simple_gla, fused_recurrent_simple_gla, parallel_simple_gla
from fla.ops.stickbreaking_attn import parallel_stickbreaking_attn
from fla.ops.wall_attn import parallel_wall_attn, parallel_wall_attn_decode

__all__ = [
    'chunk_abc',
    'chunk_comba',
    'chunk_cyfa',
    'chunk_delta_rule',
    'chunk_dplr_delta_rule',
    'chunk_gated_delta_rule',
    'chunk_gdn',
    'chunk_gla',
    'chunk_gsa',
    'chunk_iplr_delta_rule',
    'chunk_kda',
    'chunk_lightning_attn',
    'chunk_linear_attn',
    'chunk_log_linear_attn',
    'chunk_mesa_net',
    'chunk_momentum_delta_rule',
    'chunk_precond_gated_delta_rule',
    'chunk_precond_kda',
    'chunk_retention',
    'chunk_rwkv6',
    'chunk_rwkv7',
    'chunk_simple_gla',
    'fused_attnres',
    'fused_chunk_based',
    'fused_chunk_delta_rule',
    'fused_chunk_gla',
    'fused_chunk_linear_attn',
    'fused_chunk_retention',
    'fused_chunk_simple_gla',
    'fused_hc_post',
    'fused_hc_pre',
    'fused_recurrent_comba',
    'fused_recurrent_cyfa',
    'fused_recurrent_delta_rule',
    'fused_recurrent_dplr_delta_rule',
    'fused_recurrent_gated_delta_rule',
    'fused_recurrent_gdn',
    'fused_recurrent_gla',
    'fused_recurrent_gsa',
    'fused_recurrent_hgrn',
    'fused_recurrent_iplr_delta_rule',
    'fused_recurrent_kda',
    'fused_recurrent_lightning_attn',
    'fused_recurrent_linear_attn',
    'fused_recurrent_momentum_delta_rule',
    'fused_recurrent_precond_gated_delta_rule',
    'fused_recurrent_precond_kda',
    'fused_recurrent_retention',
    'fused_recurrent_rwkv6',
    'fused_recurrent_rwkv7',
    'fused_recurrent_simple_gla',
    'parallel_attn',
    'parallel_based',
    'parallel_forgetting_attn',
    'parallel_nsa',
    'parallel_parallax',
    'parallel_path_attn',
    'parallel_retention',
    'parallel_simple_gla',
    'parallel_stickbreaking_attn',
    'parallel_wall_attn',
    'parallel_wall_attn_decode',
]
