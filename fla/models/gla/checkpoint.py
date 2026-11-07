# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Legacy GLA weight names for PyTorch and Transformers loading."""

import re

from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming

GLA_WEIGHT_NAMES = {
    r'layers\.(\d+)\.attn_norm\.': r'layers.\1.attn_residual.norm.',
    r'layers\.(\d+)\.attn_res_proj\.': r'layers.\1.attn_residual.query.',
    r'layers\.(\d+)\.attn_res_norm\.': r'layers.\1.attn_residual.attnres_norm.',
    r'layers\.(\d+)\.mlp_norm\.': r'layers.\1.mlp_residual.norm.',
    r'layers\.(\d+)\.mlp_res_proj\.': r'layers.\1.mlp_residual.query.',
    r'layers\.(\d+)\.mlp_res_norm\.': r'layers.\1.mlp_residual.attnres_norm.',
    r'^(model\.|)norm\.': r'\1output_residual.norm.',
    r'^(model\.|)res_proj\.': r'\1output_residual.query.',
    r'^(model\.|)res_norm\.': r'\1output_residual.attnres_norm.',
}

if get_checkpoint_conversion_mapping('gla') is None:
    register_checkpoint_conversion_mapping('gla', [
        WeightRenaming(source_patterns=source, target_patterns=target) for source, target in GLA_WEIGHT_NAMES.items()
    ])


def load_legacy_checkpoint(module, state_dict, prefix, *args) -> None:
    """Translate old keys on direct PyTorch loads; Transformers uses the registered weight conversions."""
    for name in list(state_dict):
        if not name.startswith(prefix):
            continue
        converted = name[len(prefix):]
        for source, target in GLA_WEIGHT_NAMES.items():
            converted = re.sub(source, target, converted)
        converted = prefix + converted
        if converted != name:
            state_dict.setdefault(converted, state_dict.pop(name))
