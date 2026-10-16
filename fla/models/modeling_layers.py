# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math
from functools import partial

from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing_extensions import NotRequired, TypedDict

from fla.layers.attn import Attention

logger = logging.get_logger(__name__)


class HybridAttentionSpec(TypedDict):
    """JSON-serializable settings for standard attention at selected model layers."""

    layers: list[int]
    num_heads: int
    num_kv_heads: NotRequired[int]
    qkv_bias: NotRequired[bool]
    window_size: NotRequired[int | None]
    rope_theta: NotRequired[float]


HybridAttentionConfig = HybridAttentionSpec | list[HybridAttentionSpec] | None


def _positive_int(value: object, *, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} field {field!r} must be a positive integer; got {value!r}")
    return value


def _normalize_hybrid_attention_spec(
    spec: dict,
    *,
    num_hidden_layers: int,
    context: str,
    assigned_layers: dict[int, str],
) -> HybridAttentionSpec:
    normalized = dict(spec)

    for field in ('layers', 'num_heads'):
        if field not in normalized:
            raise ValueError(f"{context} field {field!r} is required; got <missing>")

    layers = normalized['layers']
    if not isinstance(layers, (list, tuple)):
        raise ValueError(
            f"{context} field 'layers' must be a list or tuple of integer layer indices; got {layers!r}",
        )

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
        if assigned_layers.get(layer_idx) == context:
            raise ValueError(f"{context} field 'layers' contains duplicate layer {layer_idx!r}; got {layers!r}")
        if layer_idx in assigned_layers:
            raise ValueError(
                f"{context} assigns conflicting layer {layer_idx!r}, "
                f"which is already assigned by {assigned_layers[layer_idx]}",
            )
        assigned_layers[layer_idx] = context

    normalized['layers'] = list(layers)
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
    assigned_layers: dict[int, str] = {}
    normalized_specs = []

    for spec_index, spec in enumerate(specs):
        context = "attn specification" if is_single_spec else f"attn specification at index {spec_index}"
        if not isinstance(spec, dict):
            raise ValueError(f"{context} must be a dictionary; got {spec!r}")
        normalized_specs.append(
            _normalize_hybrid_attention_spec(
                spec,
                num_hidden_layers=num_hidden_layers,
                context=context,
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


def get_hybrid_attention(config: PretrainedConfig, layer_idx: int) -> Attention | None:
    """Create standard attention for a selected layer, or return None for a native mixer."""
    attn_spec = get_hybrid_attention_spec(config.attn, layer_idx=layer_idx)
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


class GradientCheckpointingLayer(nn.Module):
    """Disable caching while checkpointing a layer during training.

    The parent model supplies ``_gradient_checkpointing_func`` when checkpointing
    is enabled. Reentrant checkpointing requires gradient-bearing inputs to be
    passed positionally.
    """

    gradient_checkpointing = False

    def __call__(self, *args, **kwargs):
        if not (self.gradient_checkpointing and self.training):
            return super().__call__(*args, **kwargs)

        changes = []
        if kwargs.get('use_cache'):
            kwargs['use_cache'] = False
            changes.append('`use_cache=False`')
        for name in ('past_key_value', 'past_key_values', 'layer_past'):
            if kwargs.get(name) is not None:
                kwargs[name] = None
                changes.append(f'`{name}=None`')
        if changes:
            logger.warning_once(
                f"Caching is incompatible with gradient checkpointing in {self.__class__.__name__}. "
                f"Setting {', '.join(changes)}.",
            )

        return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
