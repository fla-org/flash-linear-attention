from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig

class GLAVisionConfig(PretrainedConfig):

    model_type = 'gla_vision'

    def __init__(
        self,
        # GLA core parameters
        hidden_size: int = 2048,
        expand_k: int = 0.5,
        expand_v: int = 1,
        num_hidden_layers: int = 24,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        feature_map: Optional[str] = None,
        attn_mode: str = "chunk",
        use_short_conv: bool = False,
        conv_size: int = 4,
        use_output_gate: bool = True,
        clamp_min: Optional[float] = None,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-6,
        use_gk: bool = True,
        use_gv: bool = False,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
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
        scan_type: str = "uni-scan", # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        **kwargs
    ):
        # Initialize DeltaNet core parameters
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.feature_map = feature_map
        self.attn_mode = attn_mode
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_output_gate = use_output_gate
        self.clamp_min = clamp_min
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_gk = use_gk
        self.use_gv = use_gv
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy

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

        super().__init__(**kwargs)
