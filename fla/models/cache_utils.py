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


class FLALayer(CacheLayerMixin):
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
        A_state: torch.Tensor | None = None,
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
                "A_state": None,
            }

        if recurrent_state is not None:
            self.state["recurrent_state"] = recurrent_state

        # Extract input_size from attn_state if available (before potential window truncation)
        has_attn_state = attn_state and attn_state[0] is not None
        input_size = attn_state[0].shape[1] if has_attn_state else 0

        if has_attn_state:
            if self.state["attn_state"] is None:
                if window_size is not None and input_size > window_size:
                    attn_state = tuple(x[:, -window_size:].contiguous() for x in attn_state)
                self.state["attn_state"] = tuple(attn_state)
            else:
                old = self.state["attn_state"]
                if window_size is not None and input_size == 0:
                    pass
                # if the incoming chunk covers the whole window then we can replace the cache with its tail
                # otherwise we roll the existing window and splice in the new tokens
                elif window_size is not None and old[0].shape[1] >= window_size:
                    new_tuple = []
                    for old_x, new_x in zip(old, attn_state, strict=False):
                        tail = new_x[:, -window_size:]
                        if tail.shape[1] >= window_size:
                            new_tuple.append(tail.contiguous())
                        else:
                            old_x = old_x[:, -window_size:].contiguous() if old_x.shape[1] > window_size else old_x
                            rolled = old_x.roll(-input_size, dims=1)
                            rolled[:, -tail.shape[1]:] = tail
                            new_tuple.append(rolled)
                    self.state["attn_state"] = tuple(new_tuple)
                else:
                    new_tuple = []
                    for old_x, new_x in zip(old, attn_state, strict=False):
                        updated = torch.cat([old_x, new_x], dim=1)
                        if window_size is not None and updated.shape[1] > window_size:
                            updated = updated[:, -window_size:].contiguous()
                        new_tuple.append(updated)
                    self.state["attn_state"] = tuple(new_tuple)

        if conv_state is not None:
            self.state["conv_state"] = conv_state
        if ffn_state is not None:
            self.state["ffn_state"] = ffn_state
        if A_state is not None:
            self.state["A_state"] = A_state

        if not hasattr(self, 'device'):
            self.device = 'cpu'
        for state in (recurrent_state, attn_state, conv_state, ffn_state, A_state):
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
                    # For custom state objects (e.g., LogLinearAttentionState),
                    # try to find a tensor attribute to get the device.
                    for attr in vars(state).values():
                        if isinstance(attr, torch.Tensor):
                            self.device = attr.device
                            break
                break

        # Track seen tokens from attn_state if available, otherwise use offset
        if has_attn_state:
            # Use input_size captured before potential window truncation
            self._seen_tokens += input_size
        else:
            # For layers without attn_state (e.g., rwkv7, gated_deltanet), use offset
            self._seen_tokens += offset

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
            "recurrent_state", "conv_state", "ffn_state", "A_state",
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
    """
    A cache used for storing hidden states produced by flash linear attention models.

    It stores the recurrent state of each layer; the exact state shape is layer-dependent
    (e.g. `[batch_size, key_dim, value_dim]`, or `[batch_size, value_dim, key_dim]` for
    layers using the V-first state layout).
    """

    is_compileable = False

    def __init__(
        self,
        seen_tokens: int = 0,
    ) -> LegacyFLACache:
        super().__init__(layers=[])

        self.states: list[dict[str, Any]] = []

        self._seen_tokens = seen_tokens  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> dict[str, Any]:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        yield from self.states

    def __len__(self):
        return len(self.states)

    def update(
        self,
        recurrent_state: tuple[torch.Tensor] | None = None,
        attn_state: tuple[torch.Tensor] | None = None,
        conv_state: tuple[torch.Tensor] | None = None,
        ffn_state: tuple[torch.Tensor] | None = None,
        layer_idx: int = 0,
        offset: int | None = 1,
        cache_kwargs: dict[str, Any] | None = None,
        A_state: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Args:
            recurrent_state (`torch.Tensor`):
                The new recurrent state to cache.
            attn_state (`tuple[torch.Tensor]`):
                The new attention key/value states to cache.
            conv_state (`tuple[torch.Tensor]`):
                The new convolution state to cache.
            ffn_state (`tuple[torch.Tensor]`):
                The new feed-forward state to cache.
            layer_idx (`int`, defaults to 0):
                The index of the layer to cache the states for.
            offset (`int`, defaults to 1):
                The number of new tokens being processed.
            cache_kwargs (`Dict[str, Any]`):
                Additional arguments for the cache subclass.
            A_state (`torch.Tensor`, optional):
                The new auxiliary preconditioner state to cache. Default: `None`.

        Return:
            Dictionary of the updated state.
        """

        if cache_kwargs is None:
            cache_kwargs = {}
        offset = 1 if offset is None else offset
        if attn_state is not None:
            input_size = attn_state[0].shape[1]
            window_size = cache_kwargs.get('window_size')
            if not isinstance(attn_state, (tuple, list)):
                raise ValueError("`attn_state` must be a tuple of tensors for key/value states")
        if len(self.states) <= layer_idx:
            # update the number of seen tokens
            if layer_idx == 0:
                self._seen_tokens += offset
            if attn_state is not None:
                if window_size is not None and input_size > window_size:
                    attn_state = [state[:, -window_size:].contiguous() for state in attn_state]
            state = dict(
                recurrent_state=recurrent_state,
                attn_state=attn_state,
                conv_state=conv_state,
                ffn_state=ffn_state,
                A_state=A_state,
            )
            self.states.append(state)
        else:
            # update the number of seen tokens
            if layer_idx == len(self.states) - 1:
                self._seen_tokens += offset
            state = self.states[layer_idx]
            if recurrent_state is not None:
                state['recurrent_state'] = recurrent_state
            if attn_state is not None:
                if state['attn_state'] is None:
                    state['attn_state'] = [
                        new_state[:, -window_size:].contiguous()
                        if window_size is not None and new_state.shape[1] > window_size
                        else new_state
                        for new_state in attn_state
                    ]
                elif window_size is not None and input_size == 0:
                    pass
                # mirror FLALayer's window semantics so legacy caches handle oversized decoding chunks
                elif window_size is not None and state['attn_state'][0].shape[1] >= window_size:
                    updated_attn_state = []
                    for old_state, new_state in zip(state['attn_state'], attn_state, strict=False):
                        tail = new_state[:, -window_size:]
                        if tail.shape[1] >= window_size:
                            updated_attn_state.append(tail.contiguous())
                        else:
                            # DO NOT allocate new memory if the cache is full
                            # roll the key/value states to the left by `input_size`
                            old_state = (
                                old_state[:, -window_size:].contiguous() if old_state.shape[1] > window_size else old_state
                            )
                            old_state = old_state.roll(-input_size, 1)
                            # replace the newest slots with the new key/value states
                            old_state[:, -tail.shape[1]:] = tail
                            updated_attn_state.append(old_state)
                    state['attn_state'] = updated_attn_state
                else:
                    updated_attn_state = []
                    for old_state, new_state in zip(state['attn_state'], attn_state, strict=False):
                        updated = torch.cat([old_state, new_state], 1)
                        if window_size is not None and updated.shape[1] > window_size:
                            updated = updated[:, -window_size:].contiguous()
                        updated_attn_state.append(updated)
                    state['attn_state'] = updated_attn_state
            if conv_state is not None:
                state['conv_state'] = conv_state
            if ffn_state is not None:
                state['ffn_state'] = ffn_state
            if A_state is not None:
                state['A_state'] = A_state

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

        cache = cls(seen_tokens)
        if isinstance(past_key_values, (list, tuple)):
            for layer_idx in range(len(past_key_values)):
                cache.states.append(past_key_values[layer_idx])
        return cache


class FLACache(HFCacheBase):
    """
    A cache used for storing hidden states produced by flash linear attention models.

    It stores the recurrent state of each layer; the exact state shape is layer-dependent
    (e.g. `[batch_size, key_dim, value_dim]`, or `[batch_size, value_dim, key_dim]` for
    layers using the V-first state layout).
    """

    is_compileable = False

    def __init__(self, seen_tokens: int = 0, **kwargs):
        if kwargs.get("offloading", False):
            raise NotImplementedError("Automatic cache offloading is not supported by FLA models.")
        super().__init__(layer_class_to_replicate=FLALayer, **kwargs)
        self._seen_tokens = int(seen_tokens)

    def update(
        self,
        recurrent_state: tuple[torch.Tensor] | None = None,
        attn_state: tuple[torch.Tensor] | None = None,
        conv_state: tuple[torch.Tensor] | None = None,
        ffn_state: tuple[torch.Tensor] | None = None,
        A_state: torch.Tensor | None = None,
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
            A_state=A_state,
            offset=offset if offset is not None else 1,
            cache_kwargs=cache_kwargs,
        )

    def __getitem__(self, layer_idx: int) -> dict[str, Any]:
        if layer_idx >= len(self.layers):
            raise KeyError(f"Cache only have {len(self.layers)} layers, however accessed {layer_idx} out of bounds")
        return self.layers[layer_idx].state

    def __iter__(self):
        for i in range(len(self.layers)):
            yield self[i]

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
        # keeps the layer objects allocated for the HF cache compatibility
        # drops cache state
        for layer in self.layers:
            layer.reset()
        self._seen_tokens = 0

    def to_legacy_cache(self) -> tuple[dict[str, Any], ...]:
        return tuple(self[i] for i in range(len(self.layers)))

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
