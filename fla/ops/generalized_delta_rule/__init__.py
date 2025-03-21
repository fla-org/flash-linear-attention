from .dplr import chunk_dplr_delta_rule, fused_recurrent_dplr_delta_rule
from .iplr import chunk_iplr_delta_rule, fused_recurrent_iplr_delta_rule

__all__ = [
    'chunk_dplr_delta_rule',
    'fused_recurrent_dplr_delta_rule',
    'chunk_iplr_delta_rule',
    'fused_recurrent_iplr_delta_rule'
]
