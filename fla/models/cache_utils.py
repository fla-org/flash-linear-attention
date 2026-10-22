# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import Cache as HFCacheBase
from transformers.cache_utils import CacheLayerMixin


def update_attention_cache(
    cached_state: tuple[torch.Tensor, ...] | list[torch.Tensor],
    new_state: tuple[torch.Tensor, ...] | list[torch.Tensor],
    window_size: int | None,
) -> list[torch.Tensor]:
    input_length = new_state[0].shape[1]
    updated_state = []
    for cached, new in zip(cached_state, new_state, strict=False):
        if window_size is None or cached_state[0].shape[1] < window_size:
            updated = torch.cat([cached, new], dim=1)
            if window_size is not None and updated.shape[1] > window_size:
                updated = updated[:, -window_size:].contiguous()
        elif new.shape[1] >= window_size:
            updated = new[:, -window_size:].contiguous()
        else:
            if cached.shape[1] > window_size:
                cached = cached[:, -window_size:].contiguous()
            updated = cached.roll(-input_length, dims=1)
            updated[:, -new.shape[1]:] = new
        updated_state.append(updated)
    return updated_state


class FLALayer(CacheLayerMixin):
    """Store recurrent, attention, convolution, and auxiliary state for one layer."""

    is_compileable = False
    is_sliding = False

    def __init__(self):
        super().__init__()
        self.state = None
        self._seen_tokens = 0
        self.window_size = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor | None = None):
        self.device = key_states.device

    def update(
        self,
        *,
        recurrent_state: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        attn_state: tuple[torch.Tensor, ...] | None = None,
        conv_state: Any | None = None,
        ffn_state: Any | None = None,
        offset: int = 1,
        cache_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if cache_kwargs is None:
            cache_kwargs = {}
        window_size = cache_kwargs.get("window_size", self.window_size)
        self.window_size = window_size
        self.is_sliding = window_size is not None

        if attn_state is not None and not isinstance(attn_state, (tuple, list)):
            raise ValueError("`attn_state` must be a tuple/list of tensors")

        if self.state is None:
            self.state = {
                "recurrent_state": None,
                "attn_state": None,
                "conv_state": None,
                "ffn_state": None,
            }

        if recurrent_state is not None:
            self.state["recurrent_state"] = recurrent_state

        # count incoming tokens before truncating the attention window
        has_attn_state = attn_state and attn_state[0] is not None
        input_length = attn_state[0].shape[1] if has_attn_state else 0

        if has_attn_state:
            cached_attn_state = self.state["attn_state"]
            if cached_attn_state is None:
                if window_size is not None and input_length > window_size:
                    attn_state = tuple(state[:, -window_size:].contiguous() for state in attn_state)
                self.state["attn_state"] = tuple(attn_state)
            elif window_size is None or input_length > 0:
                updated_attn_state = update_attention_cache(
                    cached_state=cached_attn_state,
                    new_state=attn_state,
                    window_size=window_size,
                )
                self.state["attn_state"] = tuple(updated_attn_state)

        if conv_state is not None:
            self.state["conv_state"] = conv_state
        if ffn_state is not None:
            self.state["ffn_state"] = ffn_state

        if not hasattr(self, 'device'):
            self.device = 'cpu'
        for state in (recurrent_state, attn_state, conv_state, ffn_state):
            if state is not None:
                if isinstance(state, torch.Tensor):
                    self.device = state.device
                elif isinstance(state, (tuple, list)):
                    first_tensor = next((item for item in state if isinstance(item, torch.Tensor)), None)
                    if first_tensor is not None:
                        self.device = first_tensor.device
                elif hasattr(state, 'device'):
                    self.device = state.device
                else:
                    # custom states such as LogLinearAttentionState store their device on tensor attributes
                    for attribute in vars(state).values():
                        if isinstance(attribute, torch.Tensor):
                            self.device = attribute.device
                            break
                break

        self._seen_tokens += input_length if has_attn_state else offset
        self.is_initialized = True

        return self.state

    def get_seq_length(self, cache_position=None) -> int:
        return self._seen_tokens

    def get_max_length(self) -> int:
        return -1

    def get_max_cache_shape(self) -> int:
        return self.get_max_length()

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        cached_length = self._seen_tokens
        if self.window_size is not None:
            cached_length = min(cached_length, self.window_size - 1)
        return cached_length + query_length, self._seen_tokens - cached_length

    def _map_state(self, function):
        def apply(value):
            if isinstance(value, torch.Tensor):
                return function(value)
            if isinstance(value, dict):
                return {key: apply(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(apply(item) for item in value)
            if isinstance(value, list):
                return [apply(item) for item in value]
            if value is None or isinstance(value, (int, float, bool, str)):
                return value
            raise NotImplementedError(f"Cache transformations are not supported for {type(value).__name__} states.")

        return apply(self.state)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.state = self._map_state(lambda value: value.index_select(0, beam_idx.to(value.device)))

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.state = self._map_state(lambda value: value.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self.reorder_cache(indices)

    def _validate_crop(self, max_length: int) -> int:
        if max_length < 0:
            max_length = max(0, self._seen_tokens + max_length)
        if max_length >= self._seen_tokens:
            return self._seen_tokens
        if self.state is not None and any(self.state.get(key) is not None for key in (
            "recurrent_state", "conv_state", "ffn_state",
        )):
            raise NotImplementedError("Recurrent cache rollback requires saved intermediate states.")
        if self.window_size is not None and self._seen_tokens > self.window_size:
            raise NotImplementedError("Cannot roll back a sliding-window cache after earlier tokens were discarded.")
        return max_length

    def crop(self, max_length: int) -> None:
        max_length = self._validate_crop(max_length)
        if self.state is not None and self.state["attn_state"] is not None:
            self.state["attn_state"] = tuple(value[:, :max_length] for value in self.state["attn_state"])
        self._seen_tokens = max_length

    def offload(self):
        self.state = self._map_state(lambda value: value.to("cpu", non_blocking=True))

    def prefetch(self):
        if self.state is not None:
            self.state = self._map_state(lambda value: value.to(self.device, non_blocking=True))

    def reset(self):
        self.state = None
        self._seen_tokens = 0
        self.is_initialized = False


class LegacyFLACache(HFCacheBase):
    """Store layer states with a shared token counter for the legacy cache API.

    Recurrent state shapes follow the owning layer, including V-first layouts.
    """

    is_compileable = False

    def __init__(self, seen_tokens: int = 0) -> None:
        super().__init__(layers=[])

        self.states: list[dict[str, Any]] = []

        self._seen_tokens = seen_tokens

    def __getitem__(self, layer_idx: int) -> dict[str, Any]:
        if layer_idx >= len(self):
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
        return self.states[layer_idx]

    def __iter__(self):
        yield from self.states

    def __len__(self):
        return len(self.states)

    def update(
        self,
        recurrent_state: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        attn_state: tuple[torch.Tensor, ...] | None = None,
        conv_state: Any | None = None,
        ffn_state: Any | None = None,
        layer_idx: int = 0,
        offset: int | None = 1,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update one layer while retaining any state omitted from the call.

        Args:
            recurrent_state (torch.Tensor or tuple[torch.Tensor, ...], Optional):
                The new recurrent state. Default: `None`.
            attn_state (tuple[torch.Tensor, ...], Optional):
                The new attention key/value states. Default: `None`.
            conv_state (Any, Optional):
                The new convolution state. Default: `None`.
            ffn_state (Any, Optional):
                The new feed-forward state. Default: `None`.
            layer_idx (int, Optional):
                The layer to update. Default: 0.
            offset (int, Optional):
                The number of new tokens; `None` is treated as 1. Default: 1.
            cache_kwargs (dict[str, Any], Optional):
                Cache options, including the attention `window_size`. Default: `None`.

        Returns:
            dict[str, Any]:
                The updated layer state.
        """

        if cache_kwargs is None:
            cache_kwargs = {}
        offset = 1 if offset is None else offset
        if attn_state is not None:
            input_length = attn_state[0].shape[1]
            window_size = cache_kwargs.get('window_size')
            if not isinstance(attn_state, (tuple, list)):
                raise ValueError("`attn_state` must be a tuple of tensors for key/value states")
        if len(self.states) <= layer_idx:
            if layer_idx == 0:
                self._seen_tokens += offset
            if attn_state is not None:
                if window_size is not None and input_length > window_size:
                    attn_state = [state[:, -window_size:].contiguous() for state in attn_state]
            state = dict(
                recurrent_state=recurrent_state,
                attn_state=attn_state,
                conv_state=conv_state,
                ffn_state=ffn_state,
            )
            self.states.append(state)
        else:
            if layer_idx == len(self.states) - 1:
                self._seen_tokens += offset
            state = self.states[layer_idx]
            if recurrent_state is not None:
                state['recurrent_state'] = recurrent_state
            if attn_state is not None:
                cached_attn_state = state['attn_state']
                if cached_attn_state is None:
                    state['attn_state'] = [
                        new_state[:, -window_size:].contiguous()
                        if window_size is not None and new_state.shape[1] > window_size
                        else new_state
                        for new_state in attn_state
                    ]
                elif window_size is None or input_length > 0:
                    state['attn_state'] = update_attention_cache(
                        cached_state=cached_attn_state,
                        new_state=attn_state,
                        window_size=window_size,
                    )
            if conv_state is not None:
                state['conv_state'] = conv_state
            if ffn_state is not None:
                state['ffn_state'] = ffn_state

        return state

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.states) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_cache_shape(self) -> int | None:
        """Returns the maximum sequence length of the cached states. Cache does not have a maximum length."""
        return None

    def reset(self):
        self.states.clear()
        self._seen_tokens = 0

    def to_legacy_cache(self) -> tuple:
        return tuple(self.states)

    @classmethod
    @torch.compiler.disable
    def from_legacy_cache(
        cls,
        past_key_values: tuple | None = None,
        seen_tokens: int = 0,
    ) -> LegacyFLACache:
        """Converts a cache in the legacy cache format into an equivalent `Cache`."""

        cache = cls(seen_tokens=seen_tokens)
        if isinstance(past_key_values, (list, tuple)):
            cache.states.extend(past_key_values)
        return cache


class FLACache(HFCacheBase):
    """Store layer states and independent token counters using Transformers cache layers.

    Recurrent state shapes follow the owning layer, including V-first layouts.
    """

    is_compileable = False

    def __init__(self, seen_tokens: int = 0, **kwargs):
        if kwargs.get("offloading", False):
            raise NotImplementedError("Automatic cache offloading is not supported by FLA models.")
        super().__init__(layer_class_to_replicate=FLALayer, **kwargs)
        self._seen_tokens = int(seen_tokens)

    def update(
        self,
        recurrent_state: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        attn_state: tuple[torch.Tensor, ...] | None = None,
        conv_state: Any | None = None,
        ffn_state: Any | None = None,
        layer_idx: int = 0,
        offset: int | None = 1,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        while len(self.layers) <= layer_idx:
            self.layers.append(self.layer_class_to_replicate())

        return self.layers[layer_idx].update(
            recurrent_state=recurrent_state,
            attn_state=attn_state,
            conv_state=conv_state,
            ffn_state=ffn_state,
            offset=offset if offset is not None else 1,
            cache_kwargs=cache_kwargs,
        )

    def __getitem__(self, layer_idx: int) -> dict[str, Any]:
        if layer_idx >= len(self.layers):
            raise KeyError(f"Cache only have {len(self.layers)} layers, however accessed {layer_idx} out of bounds")
        return self.layers[layer_idx].state

    def __iter__(self):
        for layer_idx in range(len(self.layers)):
            yield self[layer_idx]

    def __len__(self):
        return super().__len__()

    def get_seq_length(self, layer_idx: int | None = 0, cache_position=None) -> int:
        if len(self.layers) <= (layer_idx or 0):
            return 0
        return self.layers[layer_idx or 0].get_seq_length()

    def get_max_length(self, layer_idx: int = 0) -> int:
        return -1

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.get_max_length(layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        if layer_idx >= len(self.layers):
            return query_length, 0
        return self.layers[layer_idx].get_mask_sizes(query_length)

    def _transform_states(self, function):
        states = [layer._map_state(function) for layer in self.layers]
        for layer, state in zip(self.layers, states, strict=True):
            layer.state = state

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self._transform_states(lambda value: value.index_select(0, beam_idx.to(value.device)))

    def batch_repeat_interleave(self, repeats: int) -> None:
        self._transform_states(lambda value: value.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self.reorder_cache(indices)

    def crop(self, max_length: int) -> None:
        lengths = [layer._validate_crop(max_length) for layer in self.layers]
        for layer, length in zip(self.layers, lengths, strict=True):
            layer.crop(length)

    def reset(self):
        # retain layer objects for Transformers cache compatibility
        for layer in self.layers:
            layer.reset()
        self._seen_tokens = 0

    def to_legacy_cache(self) -> tuple[dict[str, Any], ...]:
        return tuple(self[layer_idx] for layer_idx in range(len(self.layers)))

    @classmethod
    @torch.compiler.disable
    def from_legacy_cache(
        cls,
        past_key_values: tuple[dict[str, Any], ...] | None = None,
        seen_tokens: int = 0,
        **kwargs,
    ) -> FLACache:
        if isinstance(past_key_values, cls):
            return past_key_values
        if hasattr(past_key_values, "states"):
            seen_tokens = past_key_values.get_seq_length()
            past_key_values = past_key_values.states
        if past_key_values is not None and not isinstance(past_key_values, (list, tuple)):
            raise TypeError("Expected an FLA cache or a sequence of FLA state dictionaries.")
        cache = cls(seen_tokens=seen_tokens, **kwargs)
        for state in past_key_values or ():
            layer = FLALayer()
            if state is not None:
                layer.update(**state, offset=0)
                layer.state = dict(state)
            layer._seen_tokens = int(seen_tokens)
            cache.layers.append(layer)
        return cache


class Cache(FLACache):
    """Cache shared by FLA model and layer APIs."""
