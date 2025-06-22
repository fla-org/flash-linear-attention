# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers


class Cache(transformers.cache_utils.Cache):
    """
    A cache used for storing hidden states produced by flash linear attention models.

    It stores the states of each layer as the tensor of shape `[batch_size, key_dim, value_dim]`.
    """

    is_compileable = True

    def __init__(
        self,
        seen_tokens: int = 0
    ) -> Cache:
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
        recurrent_state: Optional[Tuple[torch.Tensor]] = None,
        attn_state: Optional[Tuple[torch.Tensor]] = None,
        conv_state: Optional[Tuple[torch.Tensor]] = None,
        ffn_state: Optional[Tuple[torch.Tensor]] = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            recurrent_state (`torch.Tensor`):
                The new recurrent state to cache.
            attn_state (`Tuple[torch.Tensor]`):
                The new attention key/value states to cache.
            conv_state (`Tuple[torch.Tensor]`):
                The new convolution state to cache.
            ffn_state (`Tuple[torch.Tensor]`):
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
            if not (isinstance(attn_state, Tuple) or isinstance(attn_state, List)):
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

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. Cache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple:
        return tuple(self.states)

    @classmethod
    @torch.compiler.disable
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple] = None,
        seen_tokens: int = 0
    ) -> Cache:
        """Converts a cache in the legacy cache format into an equivalent `Cache`."""

        cache = cls(seen_tokens)
        if isinstance(past_key_values, list):
            for layer_idx in range(len(past_key_values)):
                cache.states.append(past_key_values[layer_idx])
        return cache
