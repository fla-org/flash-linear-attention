
import math

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from fla.utils import device

# Models that do not yet support variable sequence lengths (for modeling tests)
MODELING_UNSUPPORTED_VARLEN = [
    "ABCConfig", "ForgettingTransformerConfig", "LinearAttentionConfig", "LightNetConfig",
    "Mamba2Config", "MambaConfig", "MesaNetConfig", "SambaConfig",
    "RodimusConfig",
]

# Models not yet ready for basic testing
NOT_READY_FOR_TESTING = ['RodimusConfig']

# Models requiring specific hardware (e.g., NVIDIA Hopper)
HOPPER_EXCLUSIVE = []

GENERATION_UNSUPPORTED = [
    "ABCConfig", "LinearAttentionConfig", "LightNetConfig",
    "Mamba2Config", "MambaConfig", "NSAConfig", "SambaConfig", "RWKV6Config", "RWKV7Config",
    "DeltaFormerConfig",
]


def create_model_and_config(config_class, L, H, D, dtype, **kwargs):
    """
    A helper function to create a model and its configuration.
    """
    config_params = {
        'hidden_size': H * D,
        'num_hidden_layers': L,
        **({'num_heads': H} if config_class.__name__ != 'NSAConfig' else {}),
        **kwargs,
    }
    config = config_class(**config_params)
    model = AutoModelForCausalLM.from_config(config)
    model.apply(init_weights_recursively)
    model.to(dtype).to(device)
    return model, config


def init_weights_with_asymmetric_pattern(module):
    """Initialize weights with asymmetric patterns for debugging.

    Args:
        module: The module to initialize weights for.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(module.weight, a=math.sqrt(5))
        with torch.no_grad():
            shape = module.weight.shape
            if len(shape) > 1:
                quarter_size = shape[0] // 4
                module.weight[:quarter_size] *= 1.2
                module.weight[-quarter_size:] *= 0.8
                if shape[0] == shape[1]:
                    idx = torch.arange(min(shape[0], shape[1]))
                    module.weight[idx, idx] += 0.05
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)
            with torch.no_grad():
                module.bias[::3] *= 1.1
                module.bias[1::3] *= 0.9
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            vocab_size, dim = module.weight.shape
            pattern = 0.01 * torch.sin(torch.arange(dim) * (6.28 / dim))
            for i in range(min(100, vocab_size)):
                module.weight[i] += pattern * (1 + i % 5) * 0.2


def init_weights_recursively(module):
    if hasattr(module, 'weight'):
        init_weights_with_asymmetric_pattern(module)
    for submodule in module.children():
        init_weights_recursively(submodule)
