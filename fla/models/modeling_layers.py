
from functools import partial

from torch import nn
from transformers.utils import logging

logger = logging.get_logger(__name__)


class GradientCheckpointingLayer(nn.Module):
    """Base class for layers with gradient checkpointing.

    This class enables gradient checkpointing functionality for a layer.
    By default, gradient checkpointing is disabled (`gradient_checkpointing = False`).
    When `model.set_gradient_checkpointing()` is called, gradient checkpointing is enabled
    by setting `gradient_checkpointing = True` and assigning a checkpointing function to `_gradient_checkpointing_func`.

    Important:

        When using gradient checkpointing with `use_reentrant=True`, inputs that require gradients (e.g. hidden states)
        must be passed as positional arguments (`*args`) rather than keyword arguments to properly propagate gradients.

        Example:

            ```python
            >>> # Correct - hidden_states passed as positional arg
            >>> out = self.layer(hidden_states, attention_mask=attention_mask)

            >>> # Incorrect - hidden_states passed as keyword arg
            >>> out = self.layer(hidden_states=hidden_states, attention_mask=attention_mask)
            ```
    """

    gradient_checkpointing = False

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            do_warn = False
            layer_name = self.__class__.__name__
            message = f"Caching is incompatible with gradient checkpointing in {layer_name}. Setting"

            if "use_cache" in kwargs and kwargs["use_cache"]:
                kwargs["use_cache"] = False
                message += " `use_cache=False`,"
                do_warn = True

            # different names for the same thing in different layers
            # TODO cyril: this one without `S` can be removed after deprection cycle
            if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
                kwargs["past_key_value"] = None
                message += " `past_key_value=None`,"
                do_warn = True

            if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
                kwargs["past_key_values"] = None
                message += " `past_key_values=None`,"
                do_warn = True

            if "layer_past" in kwargs and kwargs["layer_past"] is not None:
                kwargs["layer_past"] = None
                message += " `layer_past=None`,"
                do_warn = True

            # warn if anything was changed
            if do_warn:
                message = message.rstrip(",") + "."
                logger.warning_once(message)

            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)
