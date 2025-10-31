
from __future__ import annotations

import inspect
from typing import Any

import torch
import transformers
from packaging import version
from transformers.cache_utils import Cache as HFCacheBase
from transformers.generation import GenerationMixin
from transformers.utils.deprecation import deprecate_kwarg

_TF_VERSION = transformers.__version__
_NEED_NEW = "4.53.3"
_IS_TRANSFORMERS_4_56_PLUS = version.parse(_TF_VERSION) >= version.parse("4.56.0")

if version.parse(_TF_VERSION) > version.parse(_NEED_NEW):
    from transformers.cache_utils import CacheLayerMixin
else:
    CacheLayerMixin = object


class FLALayer(CacheLayerMixin):
    is_compileable = True
    is_sliding = False

    def __init__(self):
        super().__init__()
        self.state = None

    def lazy_initialization(self, key_states: torch.Tensor):
        self.state = None

    def update(
        self,
        *,
        recurrent_state: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        attn_state: tuple[torch.Tensor, ...] | None = None,
        conv_state: Any | None = None,
        ffn_state: Any | None = None,
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
                    for old_x, new_x in zip(old, attn_state, strict=False):
                        rolled = old_x.roll(-input_size, dims=1)
                        tail = new_x[:, -window_size:]
                        rolled[:, -tail.shape[1]:] = tail
                        new_tuple.append(rolled)
                    self.state["attn_state"] = tuple(new_tuple)
                else:
                    self.state["attn_state"] = tuple(
                        torch.cat([old_x, new_x], dim=1) for old_x, new_x in zip(old, attn_state, strict=False)
                    )

        if conv_state is not None:
            self.state["conv_state"] = conv_state
        if ffn_state is not None:
            self.state["ffn_state"] = ffn_state

        if not hasattr(self, 'device'):
            self.device = 'cpu'
        for state in (recurrent_state, attn_state, conv_state, ffn_state):
            if state is not None:
                self.device = state.device if isinstance(state, torch.Tensor) else state[0].device
                break

        return self.state

    def get_seq_length(self, cache_position=None) -> int:
        # we do not store seen_tokens here
        return 0

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return 0, 0

    def offload(self):
        if self.state is None:
            return

        def to_cpu(x):
            return x.to("cpu", non_blocking=True) if isinstance(x, torch.Tensor) else x
        for k in ("recurrent_state", "attn_state", "conv_state", "ffn_state"):
            v = self.state.get(k, None)
            if v is None:
                continue
            if isinstance(v, (tuple, list)):
                self.state[k] = tuple(to_cpu(t) for t in v)
            else:
                self.state[k] = to_cpu(v)

    def prefetch(self):
        if self.state is None:
            return

        def to_dev(x):
            return x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x
        for k in ("recurrent_state", "attn_state", "conv_state", "ffn_state"):
            v = self.state.get(k, None)
            if v is None:
                continue
            if isinstance(v, (tuple, list)):
                self.state[k] = tuple(to_dev(t) for t in v)
            else:
                self.state[k] = to_dev(v)

    def reset(self):
        pass


class LegacyFLACache(HFCacheBase):
    """
    A cache used for storing hidden states produced by flash linear attention models.

    It stores the states of each layer as the tensor of shape `[batch_size, key_dim, value_dim]`.
    """

    is_compileable = True

    def __init__(
        self,
        seen_tokens: int = 0,
    ) -> LegacyFLACache:
        super().__init__()

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

        Return:
            Dictionary of the updated state.
        """

        if cache_kwargs is None:
            cache_kwargs = {}
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
                    for i, (old_state, new_state) in enumerate(zip(state['attn_state'], attn_state, strict=False)):
                        # DO NOT allocate new memory if the cache is full
                        # roll the key/value states to the left by `input_size`
                        old_state = old_state.roll(-input_size, 1)
                        # replace the last `input_size` tokens with the new key/value states
                        old_state[:, -input_size:] = new_state
                        state['attn_state'][i] = old_state
                else:
                    attn_state = [
                        torch.cat([old_state, new_state], 1)
                        for old_state, new_state in zip(state['attn_state'], attn_state, strict=False)
                    ]
                    state['attn_state'] = attn_state
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
        if isinstance(past_key_values, list):
            for layer_idx in range(len(past_key_values)):
                cache.states.append(past_key_values[layer_idx])
        return cache


class FLACache(HFCacheBase):
    """
    A cache used for storing hidden states produced by flash linear attention models.

    It stores the states of each layer as the tensor of shape `[batch_size, key_dim, value_dim]`.
    """

    is_compileable = True

    def __init__(self, seen_tokens: int = 0, **kwargs):
        parent_init = super().__init__
        sig = inspect.signature(parent_init)
        param_names = list(sig.parameters.keys())

        if 'layer_class_to_replicate' in param_names:
            self.use_layer_class_to_replicate = True
            super().__init__(layer_class_to_replicate=FLALayer, **kwargs)
        elif 'layer_classes' in param_names:
            self.use_layer_class_to_replicate = False
            super().__init__(layer_classes=FLALayer, **kwargs)
        else:
            raise TypeError(
                "FLA cache initialization failed: HFCacheBase.__init__ accepts neither "
                "'layer_class_to_replicate' nor 'layer_classes'. This might be caused by an incompatible "
                "transformers version. Please check your transformers>=4.36.0",
            )
        self._seen_tokens = int(seen_tokens)

    def update(
        self,
        recurrent_state: tuple[torch.Tensor] | None = None,
        attn_state: tuple[torch.Tensor] | None = None,
        conv_state: tuple[torch.Tensor] | None = None,
        ffn_state: tuple[torch.Tensor] | None = None,
        layer_idx: int = 0,
        offset: int | None = 1,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.use_layer_class_to_replicate:
            self.append_new_layers(layer_idx)
        else:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())
        if layer_idx == 0:
            self._seen_tokens += int(offset)

        return self.layers[layer_idx].update(
            recurrent_state=recurrent_state,
            attn_state=attn_state,
            conv_state=conv_state,
            ffn_state=ffn_state,
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
        return self._seen_tokens

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        # Respect your global seen_tokens semantics
        # kv_length = past_seen + current_query_length
        query_len = int(cache_position.shape[0]) if cache_position is not None else 0
        kv_length = int(self._seen_tokens) + query_len
        return kv_length, 0

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
        cache = cls(seen_tokens=seen_tokens, **kwargs)
        if isinstance(past_key_values, (list, tuple)):
            for i, st in enumerate(past_key_values):
                while len(cache.layers) <= i:
                    cache.layers.append(cache.layer_class_to_replicate())
                cache.layers[i].state = dict(st)
        return cache


class FLAGenerationMixin(GenerationMixin):
    """
    Flash Linear Attention Generation Mixin that provides version-compatible generation methods.
    This mixin handles transformers library version differences, particularly for prepare_inputs_for_generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: HFCacheBase | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = True,
        logits_to_keep: int | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        # Use pre-computed version comparison for performance
        if _IS_TRANSFORMERS_4_56_PLUS:
            # For transformers 4.56.0+, use cache_position-based logic
            model_inputs = {}

            # Handle cache-dependent input preparation
            if past_key_values is not None:
                model_inputs["past_key_values"] = past_key_values

                # Use the new cache-dependent input preparation method if available
                if hasattr(self, '_cache_dependant_input_preparation') and cache_position is not None:
                    inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                        input_ids, inputs_embeds, cache_position,
                    )
                elif cache_position is not None:
                    # Fallback: manually slice using cache_position
                    if input_ids is not None and input_ids.shape[1] != cache_position.shape[0]:
                        input_ids = input_ids[:, cache_position]
                elif hasattr(past_key_values, '__len__') and len(past_key_values) > 0:
                    # Ultimate fallback to old behavior
                    input_ids = input_ids[:, -1:]

            # Handle input format (similar to base class logic)
            if inputs_embeds is not None and (cache_position is None or len(cache_position) == inputs_embeds.shape[1]):
                model_inputs['inputs_embeds'] = inputs_embeds
                model_inputs['input_ids'] = None
            else:
                model_inputs['input_ids'] = input_ids.contiguous() if input_ids is not None else None
                model_inputs['inputs_embeds'] = None

            model_inputs['cache_position'] = cache_position

        else:
            # For older transformers versions, use the original logic
            model_inputs = {}
            # only last token for `inputs_ids` if the `past_key_values` is not empty.
            if past_key_values is not None and hasattr(past_key_values, '__len__') and len(past_key_values) > 0:
                input_ids = input_ids[:, -1:]
            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and hasattr(past_key_values, '__len__') and len(past_key_values) == 0:
                model_inputs = {'inputs_embeds': inputs_embeds}
            else:
                # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
                # recompiles graphs as the stride of the inputs is a guard.
                # Ref: https://github.com/huggingface/transformers/pull/29114
                # TODO: use `next_tokens` directly instead.
                model_inputs = {'input_ids': input_ids.contiguous()}

        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
        })
        return model_inputs


if version.parse(_TF_VERSION) > version.parse(_NEED_NEW):
    class Cache(FLACache):
        def __init__(self, seen_tokens: int = 0, **kwargs: Any) -> None:
            super().__init__(seen_tokens=seen_tokens, **kwargs)
else:
    class Cache(LegacyFLACache):
        def __init__(self, seen_tokens: int = 0, **kwargs: Any) -> None:
            super().__init__(seen_tokens=seen_tokens)
