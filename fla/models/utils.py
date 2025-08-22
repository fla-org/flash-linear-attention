# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import transformers
from packaging import version
from transformers.cache_utils import Cache as HFCacheBase

_TF_VERSION = transformers.__version__
_NEED_NEW = "4.53.3"

if version.parse(_TF_VERSION) > version.parse(_NEED_NEW):
    from transformers.cache_utils import CacheLayerMixin
else:
    CacheLayerMixin = object


class FlashLinearLayer(CacheLayerMixin):
    is_compileable = True
    is_sliding = False

    def __init__(self):
        super().__init__()
        self.state = None

    def update(
        self,
        *,
        recurrent_state: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        attn_state: Optional[tuple[torch.Tensor, ...]] = None,
        conv_state: Optional[Any] = None,
        ffn_state: Optional[Any] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
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

        if attn_state is not None:
            input_size = attn_state[0].shape[1]
            if self.state["attn_state"] is None:
                if window_size is not None and input_size > window_size:
                    attn_state = tuple(x[:, -window_size:].contiguous() for x in attn_state)
                self.state["attn_state"] = tuple(attn_state)
            else:
                old = self.state["attn_state"]
                if window_size is not None and old[0].shape[1] >= window_size:
                    new_tuple = []
                    for old_x, new_x in zip(old, attn_state):
                        rolled = old_x.roll(-input_size, dims=1)
                        tail = new_x[:, -window_size:]
                        rolled[:, -tail.shape[1]:] = tail
                        new_tuple.append(rolled)
                    self.state["attn_state"] = tuple(new_tuple)
                else:
                    self.state["attn_state"] = tuple(
                        torch.cat([old_x, new_x], dim=1) for old_x, new_x in zip(old, attn_state)
                    )

        if conv_state is not None:
            self.state["conv_state"] = conv_state
        if ffn_state is not None:
            self.state["ffn_state"] = ffn_state

        return self.state

    def get_seq_length(self, cache_position=None) -> int:
        # we do not store seen_tokens here
        return 0

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return 0, 0


class LegacyCache(HFCacheBase):
    """
    A cache used for storing hidden states produced by flash linear attention models.

    It stores the states of each layer as the tensor of shape `[batch_size, key_dim, value_dim]`.
    """

    is_compileable = True

    def __init__(
        self,
        seen_tokens: int = 0
    ) -> LegacyCache:
        super().__init__()

        self.states: List[Dict[str, Any]] = []

        self._seen_tokens = seen_tokens  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> Dict[str, Any]:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for state in self.states:
            yield state

    def __len__(self):
        return len(self.states)

    def update(
        self,
        recurrent_state: Optional[tuple[torch.Tensor]] = None,
        attn_state: Optional[tuple[torch.Tensor]] = None,
        conv_state: Optional[tuple[torch.Tensor]] = None,
        ffn_state: Optional[tuple[torch.Tensor]] = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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

        Return:
            Dictionary of the updated state.
        """

        if cache_kwargs is None:
            cache_kwargs = {}
        if attn_state is not None:
            input_size = attn_state[0].shape[1]
            window_size = cache_kwargs.get('window_size', None)
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
                ffn_state=ffn_state
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
                if window_size is not None and state['attn_state'][0].shape[1] == window_size:
                    for i, (old_state, new_state) in enumerate(zip(state['attn_state'], attn_state)):
                        # DO NOT allocate new memory if the cache is full
                        # roll the key/value states to the left by `input_size`
                        old_state = old_state.roll(-input_size, 1)
                        # replace the last `input_size` tokens with the new key/value states
                        old_state[:, -input_size:] = new_state
                        state['attn_state'][i] = old_state
                else:
                    attn_state = [
                        torch.cat([old_state, new_state], 1)
                        for old_state, new_state in zip(state['attn_state'], attn_state)
                    ]
                    state['attn_state'] = attn_state
            if conv_state is not None:
                state['conv_state'] = conv_state
            if ffn_state is not None:
                state['ffn_state'] = ffn_state

        return state

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.states) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. Cache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> tuple:
        return tuple(self.states)

    @classmethod
    @torch.compiler.disable
    def from_legacy_cache(
        cls,
        past_key_values: Optional[tuple] = None,
        seen_tokens: int = 0
    ) -> LegacyCache:
        """Converts a cache in the legacy cache format into an equivalent `Cache`."""

        cache = cls(seen_tokens)
        if isinstance(past_key_values, list):
            for layer_idx in range(len(past_key_values)):
                cache.states.append(past_key_values[layer_idx])
        return cache


class NewStyleCache(HFCacheBase):
    """
    A cache used for storing hidden states produced by flash linear attention models.

    It stores the states of each layer as the tensor of shape `[batch_size, key_dim, value_dim]`.
    """

    is_compileable = True

    def __init__(self, seen_tokens: int = 0, **kwargs):
        super().__init__(layer_classes=FlashLinearLayer, **kwargs)
        self._seen_tokens = int(seen_tokens)

    def update(
        self,
        recurrent_state: Optional[tuple[torch.Tensor]] = None,
        attn_state: Optional[tuple[torch.Tensor]] = None,
        conv_state: Optional[tuple[torch.Tensor]] = None,
        ffn_state: Optional[tuple[torch.Tensor]] = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.append_new_layers(layer_idx)
        if layer_idx == 0:
            self._seen_tokens += int(offset)

        return self.layers[layer_idx].update(
            recurrent_state=recurrent_state,
            attn_state=attn_state,
            conv_state=conv_state,
            ffn_state=ffn_state,
            cache_kwargs=cache_kwargs,
        )

    def __getitem__(self, layer_idx: int) -> Dict[str, Any]:
        if layer_idx >= len(self.layers):
            raise KeyError(f"Cache only have {len(self.layers)} layers, however accessed {layer_idx} out of bounds")
        return self.layers[layer_idx].state

    def __iter__(self):
        for i in range(len(self.layers)):
            yield self[i]

    def __len__(self):
        return super().__len__()

    def get_seq_length(self, layer_idx: Optional[int] = 0, cache_position=None) -> int:
        if len(self.layers) <= (layer_idx or 0):
            return 0
        return self._seen_tokens

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        # Respect your global seen_tokens semantics
        # kv_length = past_seen + current_query_length
        query_len = int(cache_position.shape[0]) if cache_position is not None else 0
        kv_length = int(self._seen_tokens) + query_len
        return kv_length, 0

    def to_legacy_cache(self) -> tuple[Dict[str, Any], ...]:
        return tuple(self[i] for i in range(len(self.layers)))

    @classmethod
    @torch.compiler.disable
    def from_legacy_cache(
        cls,
        past_key_values: Optional[tuple[Dict[str, Any], ...]] = None,
        seen_tokens: int = 0,
        **kwargs,
    ) -> NewStyleCache:
        cache = cls(seen_tokens=seen_tokens, **kwargs)
        if isinstance(past_key_values, (list, tuple)):
            for i, st in enumerate(past_key_values):
                cache.append_new_layers(i)
                cache.layers[i].state = dict(st)
        return cache


if version.parse(_TF_VERSION) > version.parse(_NEED_NEW):
    class Cache(NewStyleCache):
        def __init__(self, seen_tokens: int = 0, **kwargs: Any) -> None:
            super().__init__(seen_tokens=seen_tokens, **kwargs)
else:
    class Cache(LegacyCache):
        def __init__(self, seen_tokens: int = 0, **kwargs: Any) -> None:
            super().__init__(seen_tokens=seen_tokens)
