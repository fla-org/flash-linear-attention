# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import inspect
from typing import Any

import torch
import transformers
from packaging import version
from transformers.cache_utils import Cache as HFCacheBase

_TRANSFORMERS_VERSION = transformers.__version__
_LEGACY_CACHE_VERSION = "4.53.3"

if version.parse(_TRANSFORMERS_VERSION) > version.parse(_LEGACY_CACHE_VERSION):
    from transformers.cache_utils import CacheLayerMixin
else:
    CacheLayerMixin = object


class FLALayer(CacheLayerMixin):
    """Store recurrent, attention, convolution, and auxiliary state for one layer."""

    is_compileable = True
    is_sliding = False

    def __init__(self):
        super().__init__()
        self.state = None
        self._seen_tokens = 0

    def lazy_initialization(self, key_states: torch.Tensor):
        self.state = None

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
        window_size = cache_kwargs.get("window_size")

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
            if self.state["attn_state"] is None:
                if window_size is not None and input_length > window_size:
                    attn_state = tuple(x[:, -window_size:].contiguous() for x in attn_state)
                self.state["attn_state"] = tuple(attn_state)
            else:
                cached_attn_state = self.state["attn_state"]
                if window_size is not None and input_length == 0:
                    pass
                # a full-window chunk replaces the cache; shorter chunks retain the preceding tokens
                elif window_size is not None and cached_attn_state[0].shape[1] >= window_size:
                    updated_attn_state = []
                    for cached_state, new_state in zip(cached_attn_state, attn_state, strict=False):
                        tail = new_state[:, -window_size:]
                        if tail.shape[1] >= window_size:
                            updated_attn_state.append(tail.contiguous())
                        else:
                            if cached_state.shape[1] > window_size:
                                cached_state = cached_state[:, -window_size:].contiguous()
                            rolled = cached_state.roll(-input_length, dims=1)
                            rolled[:, -tail.shape[1]:] = tail
                            updated_attn_state.append(rolled)
                    self.state["attn_state"] = tuple(updated_attn_state)
                else:
                    updated_attn_state = []
                    for cached_state, new_state in zip(cached_attn_state, attn_state, strict=False):
                        updated = torch.cat([cached_state, new_state], dim=1)
                        if window_size is not None and updated.shape[1] > window_size:
                            updated = updated[:, -window_size:].contiguous()
                        updated_attn_state.append(updated)
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

        return self.state

    def get_seq_length(self, cache_position=None) -> int:
        return self._seen_tokens

    def get_max_length(self) -> int:
        return -1

    def get_max_cache_shape(self) -> int:
        return self.get_max_length()

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return 0, 0

    def offload(self):
        self._move_states(device='cpu')

    def prefetch(self):
        self._move_states(device=None)

    def _move_states(self, device: torch.device | str | None):
        if self.state is None:
            return

        def move_tensor(state):
            if isinstance(state, torch.Tensor):
                return state.to(device=self.device if device is None else device, non_blocking=True)
            return state

        for name in ('recurrent_state', 'attn_state', 'conv_state', 'ffn_state'):
            state = self.state.get(name)
            if state is None:
                continue
            if isinstance(state, (tuple, list)):
                self.state[name] = tuple(move_tensor(tensor) for tensor in state)
            else:
                self.state[name] = move_tensor(state)

    def reset(self):
        self.state = None
        self._seen_tokens = 0


class LegacyFLACache(HFCacheBase):
    """Store layer states with a shared token counter for the legacy cache API.

    Recurrent state shapes follow the owning layer, including V-first layouts.
    """

    is_compileable = True

    def __init__(self, seen_tokens: int = 0) -> None:
        super().__init__()

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
                if state['attn_state'] is None:
                    state['attn_state'] = [
                        new_state[:, -window_size:].contiguous()
                        if window_size is not None and new_state.shape[1] > window_size
                        else new_state
                        for new_state in attn_state
                    ]
                elif window_size is not None and input_length == 0:
                    pass
                # mirror FLALayer's window semantics so legacy caches handle oversized decoding chunks
                elif window_size is not None and state['attn_state'][0].shape[1] >= window_size:
                    updated_attn_state = []
                    for old_state, new_state in zip(state['attn_state'], attn_state, strict=False):
                        tail = new_state[:, -window_size:]
                        if tail.shape[1] >= window_size:
                            updated_attn_state.append(tail.contiguous())
                        else:
                            old_state = (
                                old_state[:, -window_size:].contiguous() if old_state.shape[1] > window_size else old_state
                            )
                            old_state = old_state.roll(-input_length, 1)
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

    is_compileable = True

    def __init__(self, seen_tokens: int = 0, **kwargs):
        parameters = inspect.signature(super().__init__).parameters
        if 'layer_class_to_replicate' in parameters:
            self.use_layer_class_to_replicate = True
            super().__init__(layer_class_to_replicate=FLALayer, **kwargs)
        elif 'layer_classes' in parameters:
            self.use_layer_class_to_replicate = False
            super().__init__(layer_classes=FLALayer, **kwargs)
        else:
            raise TypeError(
                "FLA cache initialization failed: HFCacheBase.__init__ accepts neither "
                "'layer_class_to_replicate' nor 'layer_classes'. This Transformers version does not support layer caches.",
            )
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
        if not self.use_layer_class_to_replicate:
            self.append_new_layers(layer_idx)
        else:
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

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        query_length = int(cache_position.shape[0]) if cache_position is not None else 0
        kv_length = int(self.get_seq_length(layer_idx)) + query_length
        return kv_length, 0

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
        cache = cls(seen_tokens=seen_tokens, **kwargs)
        if isinstance(past_key_values, (list, tuple)):
            for layer_idx, state in enumerate(past_key_values):
                while len(cache.layers) <= layer_idx:
                    if cache.use_layer_class_to_replicate:
                        cache.layers.append(cache.layer_class_to_replicate())
                    else:
                        cache.append_new_layers(layer_idx)
                cache.layers[layer_idx].state = dict(state)
                # legacy cache tracks seen token globally but FLACache stores per layer
                cache.layers[layer_idx]._seen_tokens = int(seen_tokens)
        return cache


if version.parse(_TRANSFORMERS_VERSION) > version.parse(_LEGACY_CACHE_VERSION):
    class Cache(FLACache):
        def __init__(self, seen_tokens: int = 0, **kwargs: Any) -> None:
            super().__init__(seen_tokens=seen_tokens, **kwargs)
else:
    class Cache(LegacyFLACache):
        def __init__(self, seen_tokens: int = 0, **kwargs: Any) -> None:
            super().__init__(seen_tokens=seen_tokens)
