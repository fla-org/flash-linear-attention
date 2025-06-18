# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM

from fla.utils import device

# Models that do not yet support variable sequence lengths (for modeling tests)
MODELING_UNSUPPORTED_VAR_LEN = [
    "ABCConfig", "ForgettingTransformerConfig", "LinearAttentionConfig", "LightNetConfig",
    "Mamba2Config", "MambaConfig", "MesaNetConfig", "SambaConfig", "GatedDeltaProductConfig",
    "RodimusConfig",
]

# Models not yet ready for basic testing
NOT_READY_FOR_TESTING = ["PaTHAttentionConfig"]

# Models requiring specific hardware (e.g., NVIDIA Hopper)
HOPPER_EXCLUSIVE = ["CombaConfig", "GatedDeltaNetConfig"]

GENERATION_UNSUPPORTED = [
    "ABCConfig", "GatedDeltaProductConfig", "LinearAttentionConfig", "LightNetConfig",
    "Mamba2Config", "MambaConfig", "NSAConfig", "SambaConfig", "RWKV6Config", "RWKV7Config",
]


def create_model_and_config(config_class, L, H, D, dtype, **kwargs):
    """
    A helper function to create a model and its configuration.
    """
    config_params = {
        'hidden_size': H * D,
        'num_hidden_layers': L,
        **({'num_heads': H} if config_class.__name__ != 'NSAConfig' else {}),
        **kwargs
    }
    config = config_class(**config_params)
    model = AutoModelForCausalLM.from_config(config)
    model.to(dtype).to(device)
    return model, config
