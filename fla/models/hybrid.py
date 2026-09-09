# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math
from typing import TypedDict


class _RequiredHybridAttentionSpec(TypedDict):
    layers: list[int]
    num_heads: int


class HybridAttentionSpec(_RequiredHybridAttentionSpec, total=False):
    """JSON-serializable settings for standard attention at selected model layers."""

    num_kv_heads: int
    qkv_bias: bool
    window_size: int | None
    rope_theta: float


HybridAttentionConfig = HybridAttentionSpec | list[HybridAttentionSpec] | None


def _spec_context(spec_index: int | None) -> str:
    if spec_index is None:
        return "attn specification"
    return f"attn specification at index {spec_index}"


def _positive_int(value: object, *, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} field {field!r} must be a positive integer; got {value!r}")
    return value


def _normalize_spec(
    spec: dict,
    *,
    num_hidden_layers: int,
    spec_index: int | None,
    assigned_layers: dict[int, int | None],
) -> HybridAttentionSpec:
    context = _spec_context(spec_index)
    normalized = dict(spec)

    for field in ('layers', 'num_heads'):
        if field not in normalized:
            raise ValueError(f"{context} field {field!r} is required; got <missing>")

    layers = normalized['layers']
    if not isinstance(layers, (list, tuple)):
        raise ValueError(
            f"{context} field 'layers' must be a list or tuple of integer layer indices; got {layers!r}",
        )

    normalized_layers = []
    seen_layers = set()
    for layer_idx in layers:
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
            raise ValueError(
                f"{context} field 'layers' must contain only integer layer indices; got {layer_idx!r}",
            )
        if layer_idx < 0 or layer_idx >= num_hidden_layers:
            raise ValueError(
                f"{context} field 'layers' contains out-of-range layer {layer_idx!r}; "
                f"expected a value in [0, {num_hidden_layers})",
            )
        if layer_idx in seen_layers:
            raise ValueError(f"{context} field 'layers' contains duplicate layer {layer_idx!r}; got {layers!r}")
        if layer_idx in assigned_layers:
            previous_index = assigned_layers[layer_idx]
            previous_context = _spec_context(previous_index)
            raise ValueError(
                f"{context} assigns conflicting layer {layer_idx!r}, which is already assigned by {previous_context}",
            )
        seen_layers.add(layer_idx)
        assigned_layers[layer_idx] = spec_index
        normalized_layers.append(layer_idx)

    normalized['layers'] = normalized_layers
    normalized['num_heads'] = _positive_int(normalized['num_heads'], field='num_heads', context=context)

    num_kv_heads = normalized.get('num_kv_heads')
    if num_kv_heads is None:
        num_kv_heads = normalized['num_heads']
    normalized['num_kv_heads'] = _positive_int(num_kv_heads, field='num_kv_heads', context=context)

    qkv_bias = normalized.get('qkv_bias', False)
    if not isinstance(qkv_bias, bool):
        raise ValueError(f"{context} field 'qkv_bias' must be a Boolean; got {qkv_bias!r}")
    normalized['qkv_bias'] = qkv_bias

    window_size = normalized.get('window_size')
    if window_size is not None:
        window_size = _positive_int(window_size, field='window_size', context=context)
    normalized['window_size'] = window_size

    rope_theta = normalized.get('rope_theta', 10000.)
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
    normalized['rope_theta'] = rope_theta

    return normalized


def normalize_hybrid_attention_config(
    attn: HybridAttentionConfig,
    *,
    num_hidden_layers: int,
) -> HybridAttentionConfig:
    """Validate and normalize a hybrid-attention configuration.

    The accepted forms are ``None``, one specification dictionary, or a list of
    specification dictionaries. Defaults are applied independently to copied
    dictionaries, unknown keys are preserved, and the input's outer dictionary
    or list representation is retained. A layer may appear in only one
    specification.
    """
    if attn is None:
        return None
    if isinstance(num_hidden_layers, bool) or not isinstance(num_hidden_layers, int) or num_hidden_layers < 0:
        raise ValueError(f"field 'num_hidden_layers' must be a non-negative integer; got {num_hidden_layers!r}")
    if not isinstance(attn, (dict, list)):
        raise ValueError(f"attn must be None, a dictionary, or a list of dictionaries; got {attn!r}")

    is_single_spec = isinstance(attn, dict)
    specs = [attn] if is_single_spec else attn
    assigned_layers: dict[int, int | None] = {}
    normalized_specs = []

    for spec_index, spec in enumerate(specs):
        current_index = None if is_single_spec else spec_index
        if not isinstance(spec, dict):
            context = _spec_context(current_index)
            raise ValueError(f"{context} must be a dictionary; got {spec!r}")
        normalized_specs.append(
            _normalize_spec(
                spec,
                num_hidden_layers=num_hidden_layers,
                spec_index=current_index,
                assigned_layers=assigned_layers,
            ),
        )

    if is_single_spec:
        return normalized_specs[0]
    return normalized_specs


class _HybridAttentionConfigMixin:
    """Apply hybrid-attention validation to constructor and later assignments.

    Validation depends on ``self.num_hidden_layers`` for layer range checks,
    so subclasses must assign ``num_hidden_layers`` before ``self.attn`` in
    their ``__init__``.
    """

    @property
    def attn(self) -> HybridAttentionConfig:
        return self.__dict__.get('attn')

    @attn.setter
    def attn(self, value: HybridAttentionConfig) -> None:
        self.__dict__['attn'] = self._normalize_hybrid_attention_config(value)

    def _normalize_hybrid_attention_config(self, attn: HybridAttentionConfig) -> HybridAttentionConfig:
        return normalize_hybrid_attention_config(attn, num_hidden_layers=self.num_hidden_layers)


def get_hybrid_attention_spec(
    attn: HybridAttentionConfig,
    *,
    layer_idx: int,
) -> HybridAttentionSpec | None:
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
