from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class BitNetVisionConfig(PretrainedConfig):

    model_type = 'bitnet_vision'

    def __init__(
        self,
        # BitNet core parameters
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int = None,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: int = 2048,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: Optional[bool] = True,
        norm_first: bool = False,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        attention_bias: bool = False,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        attn: Optional[Dict] = None,
         # Vision specific parameters
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        num_classes: int = 1000,
        qkv_bias: bool = True,
        hidden_dropout_prob: float = 0.0,
        use_mask_token: bool = False,
        layer_norm_eps: float = 1e-6,
        interpolate_pos_encoding: bool = False,
        mlp_dim: int = None,
        # FLA-for-vision-related parameters
        scan_type: str = "uni-scan", # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        **kwargs
    ):
        # Initialize BitNet core parameters
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_first = norm_first
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_norm = fuse_norm

        # Initialize vision specific parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_mask_token = use_mask_token
        self.layer_norm_eps = layer_norm_eps
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.scan_type = scan_type

        if attn is not None:
            if not isinstance(attn, Dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['window_size'] = attn.get('window_size', None)

        if mlp_dim is None:
            self.mlp_dim = 4 * hidden_size # default value set to 4 * hidden_size
        else:
            self.mlp_dim = mlp_dim
        
        super().__init__(**kwargs)