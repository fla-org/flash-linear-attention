# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

from transformers.configuration_utils import PretrainedConfig
from typing_extensions import NotRequired, TypedDict

from fla.layers.attn import Attention


class AttentionSpec(TypedDict):
    """JSON-serializable settings for standard attention at selected model layers."""

    layers: list[int]
    num_heads: int
    num_kv_heads: NotRequired[int]
    qkv_bias: NotRequired[bool]
    window_size: NotRequired[int | None]
    rope_theta: NotRequired[float]


AttentionConfig = AttentionSpec | list[AttentionSpec] | None


def parse_attention_config(attn: AttentionConfig, *, num_hidden_layers: int) -> AttentionConfig:
    """Validate attention settings and fill defaults without modifying the input.

    Preserve dictionary/list form and unknown keys; reject overlapping layer assignments.
    """
    if attn is None:
        return None
    if isinstance(num_hidden_layers, bool) or not isinstance(num_hidden_layers, int) or num_hidden_layers < 0:
        raise ValueError(f"field 'num_hidden_layers' must be a non-negative integer; got {num_hidden_layers!r}")
    if not isinstance(attn, (dict, list)):
        raise ValueError(f"attn must be None, a dictionary, or a list of dictionaries; got {attn!r}")

    configs = []
    assigned_layers: dict[int, int] = {}
    for i, config in enumerate([attn] if isinstance(attn, dict) else attn):
        context = "attn specification" if isinstance(attn, dict) else f"attn specification at index {i}"
        if not isinstance(config, dict):
            raise ValueError(f"{context} must be a dictionary; got {config!r}")
        config = dict(config)
        for name in ('layers', 'num_heads'):
            if name not in config:
                raise ValueError(f"{context} field {name!r} is required; got <missing>")

        layers = config['layers']
        if not isinstance(layers, (list, tuple)):
            raise ValueError(f"{context} field 'layers' must be a list or tuple of integer layer indices; got {layers!r}")
        for layer_idx in layers:
            if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
                raise ValueError(f"{context} field 'layers' must contain only integer layer indices; got {layer_idx!r}")
            if layer_idx < 0 or layer_idx >= num_hidden_layers:
                raise ValueError(
                    f"{context} field 'layers' contains out-of-range layer {layer_idx!r}; "
                    f"expected a value in [0, {num_hidden_layers})",
                )
            if layer_idx in assigned_layers:
                if assigned_layers[layer_idx] == i:
                    raise ValueError(f"{context} field 'layers' contains duplicate layer {layer_idx!r}; got {layers!r}")
                raise ValueError(
                    f"{context} assigns conflicting layer {layer_idx!r}, "
                    f"which is already assigned by attn specification at index {assigned_layers[layer_idx]}",
                )
            assigned_layers[layer_idx] = i
        config['layers'] = list(layers)

        if config.get('num_kv_heads') is None:
            config['num_kv_heads'] = config['num_heads']
        for name in ('num_heads', 'num_kv_heads'):
            value = config[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{context} field {name!r} must be a positive integer; got {value!r}")

        qkv_bias = config.setdefault('qkv_bias', False)
        if not isinstance(qkv_bias, bool):
            raise ValueError(f"{context} field 'qkv_bias' must be a Boolean; got {qkv_bias!r}")

        window_size = config.setdefault('window_size', None)
        if window_size is not None:
            if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
                raise ValueError(f"{context} field 'window_size' must be a positive integer; got {window_size!r}")

        rope_theta = config.setdefault('rope_theta', 10000.)
        try:
            valid_rope_theta = (
                not isinstance(rope_theta, bool)
                and isinstance(rope_theta, (int, float))
                and math.isfinite(rope_theta)
                and rope_theta > 0
            )
        except OverflowError:
            valid_rope_theta = False
        if not valid_rope_theta:
            raise ValueError(f"{context} field 'rope_theta' must be positive and finite; got {rope_theta!r}")
        configs.append(config)

    return configs[0] if isinstance(attn, dict) else configs


class AttentionConfigBase:
    """Validate attention settings on construction and later assignments.

    Validation depends on ``self.num_hidden_layers`` for layer range checks,
    so subclasses must assign ``num_hidden_layers`` before ``self.attn`` in
    their ``__init__``.
    """

    @property
    def attn(self) -> AttentionConfig:
        return self.__dict__.get('attn')

    @attn.setter
    def attn(self, value: AttentionConfig) -> None:
        self.__dict__['attn'] = self._parse_attention_config(value)

    def _parse_attention_config(self, attn: AttentionConfig) -> AttentionConfig:
        return parse_attention_config(attn, num_hidden_layers=self.num_hidden_layers)


def get_attention_spec(
    attn: AttentionConfig,
    *,
    layer_idx: int,
) -> AttentionSpec | None:
    """Return the normalized attention specification assigned to ``layer_idx``.

    This lookup is intended for model block construction. Unassigned layers
    return ``None`` and retain their model's native mixer.
    """
    if attn is None:
        return None
    specs = [attn] if isinstance(attn, dict) else attn
    for spec in specs:
        if layer_idx in spec['layers']:
            return spec
    return None


def get_attention(config: PretrainedConfig, layer_idx: int) -> Attention | None:
    """Create standard attention for a selected layer, or return None for a native mixer."""
    attn_spec = get_attention_spec(config.attn, layer_idx=layer_idx)
    if attn_spec is None:
        return None
    return Attention(
        hidden_size=config.hidden_size,
        num_heads=attn_spec['num_heads'],
        num_kv_heads=attn_spec['num_kv_heads'],
        qkv_bias=attn_spec['qkv_bias'],
        window_size=attn_spec['window_size'],
        rope_theta=attn_spec['rope_theta'],
        max_position_embeddings=config.max_position_embeddings,
        layer_idx=layer_idx,
    )
