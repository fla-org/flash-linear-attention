from typing import Dict, Optional
from transformers.configuration_utils import PretrainedConfig

class GatedDeltaNetVisionConfig(PretrainedConfig):
    model_type = 'gated_deltanet_vision'

    def __init__(
        self,
        # GatedDeltaNet core parameters  
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        expand_v: int = 2,
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        head_dim: int = 256,
        num_heads: int = 6,
        max_position_embeddings: int = 2048,
        hidden_act: str = "swish",
        num_hidden_layers: int = 21,
        norm_first: bool = False,
        norm_eps: float = 1e-6,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        fuse_cross_entropy: bool = True,
        
        # Vision specific parameters
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        num_classes: int = 1000,
        hidden_dropout_prob: float = 0.0,
        use_mask_token: bool = False,
        layer_norm_eps: float = 1e-6,
        interpolate_pos_encoding: bool = False,
        mlp_dim: int = None,
        # FLA-for-vision-related parameters
        scan_type: str = "uni-scan",
        **kwargs
    ):
        # Initialize GatedDeltaNet core parameters
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv  
        self.conv_size = conv_size
        self.num_heads = num_heads
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_first = norm_first
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range 
        self.fuse_cross_entropy = fuse_cross_entropy
        self.attn = attn
        self.max_position_embeddings = max_position_embeddings

        # Initialize vision specific parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_classes = num_classes
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
            self.mlp_dim = 4 * hidden_size
        else:
            self.mlp_dim = mlp_dim
        
        super().__init__(**kwargs)
